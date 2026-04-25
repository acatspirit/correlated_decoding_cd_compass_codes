import numpy as np
from pymatching import Matching
import compass_codes as cc
import collections
import circuit_gen as cc_circuit
import stim


##############################################
#
# CorrelatedDecoder class
#
##############################################

class CorrelatedDecoder:
    def __init__(self, eta, d, l, corr_type, mem_type="X"):
        self.eta = eta # the noise bias
        self.d = d # the distance of the compass code
        self.l = l # the elongation parameter
        self.corr_type = corr_type # the type of correlation for decoder (directional)
        self.mem_type = mem_type
        self.edge_type_d = {} # dictionary of the edge types for each detector. Empty until populated by running method to populate. Type 0(1) use pauli X(Z) measurements

        self.code = cc.CompassCode(d=self.d, l=self.l)
        self.H_x, self.H_z = self.code.H['X'], self.code.H['Z'] # parity check matrices from compass code class
        self.log_x, self.log_z = self.code.logicals['X'], self.code.logicals['Z'] # logical operators from compass code class

    def bernoulli_prob(self, old_prob, p):
        """ Given an old probability and a new error probability, return the updated probability
            according to the bernoulli formula
        """
        new_prob = old_prob*(1-p) + p*(1 - old_prob)
        return new_prob  

    def get_dB_scaling(self, matching):
        edge = next(iter(matching.to_networkx().edges.values()))
        edge_w = edge['weight']
        edge_p = edge['error_probability']
        decibels_per_w = -np.log10(edge_p / (1 - edge_p)) * 10 / edge_w 
        return decibels_per_w

    def depolarizing_err(self, p):
        """Generates the error vector for one shot according to depolarizing noise model.
        Args:
        - p: Error probability.
        - num_qubits: Number of qubits.
        - eta: depolarizing channel bias. Recover unbiased depolarizing noise eta = 0.5. 
                    Px, py, pz are determined according to 2D Compass Codes paper (2019) defn of eta
        
        Returns:
        - A list containing error vectors for no error, X, Z, and Y errors.
        """
        H = self.H_x
        eta = self.eta

        num_qubits = H.shape[1]
        # Error vectors for I, X, Z, and Y errors
        errors = np.zeros((2, num_qubits), dtype=int)

        # p = px + py + pz, px=py, eta = pz/(px + py)
        px = 0.5*p/(1+eta)
        pz = p*(eta/(1+eta))
        probs = [1 - p, px, pz, px]  # Probabilities for I, X, Z, and Y errors

        # Randomly choose error types for all qubits
        # np.random.seed(10)
        choices = np.random.choice(4, size=num_qubits, p=probs)
        # Assign errors based on the chosen types
        errors[0] = np.where((choices == 1) | (choices == 3), 1, 0)  # X or Y error
        errors[1] = np.where((choices == 2) | (choices == 3), 1, 0)  # Z or Y error
        return errors
    
    def test_error(self, error_x, error_z):

        M_x = Matching.from_check_matrix(self.H_x)
        M_z = Matching.from_check_matrix(self.H_z)
        
        syndrome_x, syndrome_z = error_x @ self.H_z.T % 2, error_z @ self.H_x.T % 2
        print(f"syndrome for X errors {syndrome_x}")
        print(f"syndrome for Z errors {syndrome_z}")

        correction_x = M_z.decode(syndrome_x)
        correction_z = M_x.decode(syndrome_z)

        print(f"correction for X errors {correction_x}")
        print(f"correction for Z errors {correction_z}")

        
    
    #
    #
    # Decoder functions
    #
    #

    def decoding_failures(self,p, shots, error_type):
        """ finds the number of logical errors after decoding
            p - probability of error
            shots - number of shots
            error_type - the type of error that you hope to decode, X = 0, Z = 1
        """
        if error_type == "X": 
            H = self.H_x
            L = self.log_x
        elif error_type == "Z":
            H = self.H_z
            L = self.log_z
        M = Matching.from_check_matrix(H)
        # get the depolarizing error vector 
        err_vec = [self.depolarizing_err(p)[error_type] for _ in range(shots)]
        # generate the syndrome for each shot
        syndrome_shots = err_vec@H.T%2
        # the correction to the errors
        correction = M.decode_batch(syndrome_shots)
        num_errors = np.sum((correction+err_vec)@L%2)
        return num_errors

    def decoding_failures_correlated(self, p, shots):
        """ Finds the number of logical errors after decoding.
            p - probability of error
            shots - number of shots
            corr_type - CORR_XZ or CORR_ZX. Whether to return X then Z or Z then X correlated errors.
        """
        M_z = Matching.from_check_matrix(self.H_z)
        M_x = Matching.from_check_matrix(self.H_x)
        
        # Generate error vectors
        err_vec = [self.depolarizing_err(p) for _ in range(shots)]
        err_vec_x = np.array([err[0] for err in err_vec])
        err_vec_z = np.array([err[1] for err in err_vec])
        
        # Syndrome for X errors and decoding
        syndrome_x = err_vec_x @ self.H_z.T % 2
        correction_x = M_z.decode_batch(syndrome_x)
        num_errors_x = np.sum((correction_x + err_vec_x) @ self.log_z % 2)
        
        # Syndrome for Z errors and decoding
        syndrome_z = err_vec_z @ self.H_x.T % 2
        correction_z = M_x.decode_batch(syndrome_z)
        num_errors_z = np.sum((correction_z + err_vec_z) @ self.log_x % 2)

        
        # Decode Z errors correlated
        if self.corr_type == "CORR_XZ": # correct Z errors after correcting X errors
            cond_prob = 0.5 # the conditional probability of Z error given a X error
            new_weight = np.log((1-cond_prob)/cond_prob)
            
            # Prepare weights and syndrome for X errors
            updated_weights = np.ones(correction_x.shape)
            updated_weights[np.nonzero(correction_x)] = new_weight
            
            num_errors_xz_corr = 0

            for i in range(shots):
                M_xz_corr = Matching.from_check_matrix(self.H_x, weights=updated_weights[i]) # updated weights set erasure to 0
                correction_xz_corr = M_xz_corr.decode(syndrome_x[i])
                num_errors_xz_corr += np.sum((correction_xz_corr + err_vec_z[i]) @ self.log_x % 2)
            
            num_errors_corr = num_errors_xz_corr + num_errors_x
        
        # Decode X errors correlated
        if self.corr_type == "CORR_ZX": # correct X errors after correcting Z errors
            cond_prob = 1/(2*self.eta+1) # the conditional probability of X error given a Z error
            new_weight = np.log((1-cond_prob)/cond_prob)

            # Prepare weights and syndrome for X errors
            updated_weights = np.ones(correction_z.shape)
            updated_weights[np.nonzero(correction_z)] = new_weight
            num_errors_zx_corr = 0

            for i in range(shots):
                M_zx_corr = Matching.from_check_matrix(self.H_z, weights=updated_weights[i]) # updated weights set erasure to 0
                correction_zx_corr = M_zx_corr.decode(syndrome_z[i])
                num_errors_zx_corr += np.sum((correction_zx_corr + err_vec_x[i]) @ self.log_z % 2)
            
            num_errors_corr = num_errors_zx_corr + num_errors_z
        
        num_errors_tot = num_errors_x + num_errors_z # do I need to change this?

        return num_errors_x, num_errors_z, num_errors_corr, num_errors_tot

    def decoding_failures_uncorr(self,p, shots):
        """ Finds the number of logical errors after decoding.
            p - probability of error
            shots - number of shots
        """
        # create a matching graph
        M_z = Matching.from_check_matrix(self.H_z)
        M_x = Matching.from_check_matrix(self.H_x)
        
        # Generate error vectors
        err_vec = [self.depolarizing_err(p) for _ in range(shots)]
        err_vec_x = np.array([err[0] for err in err_vec])
        err_vec_z = np.array([err[1] for err in err_vec])
        
        # Syndrome for Z errors and decoding
        syndrome_z = err_vec_x @ self.H_z.T % 2
        correction_z = M_z.decode_batch(syndrome_z)
        num_errors_x = np.sum((correction_z + err_vec_x) @ self.L_z % 2)
        
        # Syndrome for X errors and decoding
        syndrome_x = err_vec_z @ self.H_x.T % 2
        correction_x = M_x.decode_batch(syndrome_x)
        num_errors_z = np.sum((correction_x + err_vec_z) @ self.L_x % 2)
        
        return num_errors_x, num_errors_z
    
    ########################################################################
    #
    # Circuit level correlated decoding functions
    #
    ########################################################################



    #
    # Graph labelling / edge tracking
    #

    def probability_edge_mapping(self, edge_dict):
        """ Maps the probabilities to the corresponding edge weight in the matching graph. Takes into
            account the 'type' of qubit, whether it is clifford deformed or not. CURRENTLY DOES NOT TAKE INTO 
            ACCOUNT THE TYPE OF QUBIT - NOT SURE WHAT THIS MEANS / HOW TO ACCOUNT CD
        """
        weights_dict = {}

        for edge_1 in edge_dict:

            adjacent_edge_dict = edge_dict.get(edge_1, {})

            # populate weight dictionary 
            for edge_2 in adjacent_edge_dict:

                p = edge_dict.get(edge_1, {}).get(edge_2,0)
                weight = np.log((1-p)/p)
                weights_dict.setdefault(edge_1, {})[edge_2] = weight
        
        return weights_dict
    
    def get_qubit_in_edge(self, edge_type, stab1, stab2) -> np.ndarray:
        """
        Return the qubits involved in the edge connecting two stabilizers.
        If the stabilizer is connected to a boundary (stab == -1),
        return empty for that side.
        """

        n_qubits = self.H_x.shape[1]   # <-- number of columns = number of qubits
        qubits_stab1 = sparse.csr_matrix(np.zeros(n_qubits, dtype=int))
        qubits_stab2 = sparse.csr_matrix(np.zeros(n_qubits, dtype=int))

        if edge_type == 1:
            # Z stabilizers
            if stab1 != -1:
                qubits_stab1 = self.H_z.getrow(stab1 - self.H_x.shape[0])

            if stab2 != -1:
                qubits_stab2 = self.H_z.getrow(stab2 - self.H_x.shape[0])

        elif edge_type == 0:
            # X stabilizers
            if stab1 != -1:
                qubits_stab1 = self.H_x.getrow(stab1)

            if stab2 != -1:
                qubits_stab2= self.H_x.getrow(stab2)

        qubits_in_edge = qubits_stab1.multiply(qubits_stab2).indices

        return qubits_in_edge
 
    def get_edge_type_from_detector(self, edge, mem_type, CD_data_transform) -> int:
        """
        Returns the edge type (0 or 1) for a given edge in the DEM. Type 0(1) connect Pauli X(Z) measurements.  
        """
        d1 = edge[0]
        d2 = edge[1]
        stab1 = self.get_stab_from_detector(d1, mem_type)
        stab2 = self.get_stab_from_detector(d2, mem_type)

        num_stabs_r1 = self.H_x.shape[0] if mem_type == "X" else self.H_z.shape[0]
        edge_type = 0

        if stab1 >= self.H_x.shape[0] and stab2 >= self.H_x.shape[0]: # Z type edge in SC
            edge_type = 1
        elif stab1 < self.H_x.shape[0] and stab2 < self.H_x.shape[0]:
            edge_type = 0
        elif stab1 == -1 or stab2 ==-1:
            edge_type = 0 if max(stab1,stab2) < self.H_x.shape[0] else 1
        else:
            edge_type = 2 # edge between X and Z types ... don't touch this - directly from DEM during perfect round
        
        # apply the deformation if necessary
        qubit_in_edge = self.get_qubit_in_edge(edge_type, stab1, stab2)
        if qubit_in_edge.size == 0:
            CD_applied = 0
        elif abs(d1 - d2) >= num_stabs_r1:
            CD_applied = 0
        else:
            CD_applied = CD_data_transform[qubit_in_edge[0]]

        
        if CD_applied == 2 and edge_type != 2: # if there is a Hadamard on that qubit, swap the edge type
            edge_type = (edge_type + 1)%2
        
        return edge_type
    
    def get_stab_from_detector(self, detector, mem_type) -> int:
        """
        Returns the stabilizer index for a given detector in the DEM. This is used to determine which stabilizer measurement type (X or Z) is associated with a given detector.

        Inputs:
        detector - (integer) the value of the detector of question
        
        Outputs:
        stab_index - (integer) the index of the stabilizer included in the detector. The full stabilizer list includes X and Z types.
        """
        stab_index=0
        curr_det_index = detector

        if detector == -1:
            return -1

        # should be d*(Hx.shape[0] + Hz.shape[0]) detectors

        # X detectors are always the first half
        if mem_type == "X":
            if detector < self.H_x.shape[0]: # the detector is in the first layer and is an X detector for sure but whatever
                stab_index = detector
            elif detector > self.H_x.shape[0] + (self.d-1)*(self.H_x.shape[0] + self.H_z.shape[0]): # last layer of checks
                stab_index = detector- (self.d-1)*(self.H_x.shape[0] + self.H_z.shape[0]) - self.H_x.shape[0]
            else:
                curr_det_index -= self.H_x.shape[0]
                stab_index = curr_det_index % (self.H_x.shape[0] + self.H_z.shape[0])
        else: # Z detectors
            if detector < self.H_z.shape[0]: 
                stab_index = detector + self.H_x.shape[0] # Z stabs are offset by X ones, check X first
            elif detector >= self.H_z.shape[0] + (self.d-1)*(self.H_x.shape[0] + self.H_z.shape[0]):
                stab_index = detector - (self.d-1)*(self.H_x.shape[0] + self.H_z.shape[0]) - self.H_z.shape[0] + self.H_x.shape[0]
            else:
                curr_det_index -= self.H_z.shape[0]
                stab_index = curr_det_index%(self.H_x.shape[0] + self.H_z.shape[0])
        
        return stab_index
    
    def get_LB_RB_nodes(self, DEM):
        """ Get a list of the LB and RB on the code (closed boundary for each stabilizer type)
            Inputs - (detector error model) the model representing errors in system of choice
            Outputs - (list)s the lists of the X measurement L/R stabilizers, and the Z measurement
                        T/B stabilizers. Orthogonal to that logical type
        """
        xlb_nodes = [] # the detectors that correspond to left X stabilizers
        xrb_nodes = [] # '' right X stabilizers
        ztb_nodes = [] # '' top Z stabilizers
        zbb_nodes = [] # '' bottom Z stabilizers


        # for detector in DEM ...
        detectors = DEM.num_detectors

        # get the stab index from the detector
        for d in range(detectors):
            stab_index = self.get_stab_from_detector(d, self.mem_type)
            # print("stab and d",stab_index, d)

            # Assign X left/right stabilizers
            if stab_index < self.H_x.shape[0]: # it is an X detector

                qubits_in_stab = sorted(self.H_x.getrow(stab_index).indices)
                # print(qubits_in_stab)

                if qubits_in_stab[0] % self.d == 0: # on the left
                    # print(qubits_in_stab[0] % self.d)
                    xlb_nodes += [d]
                elif (qubits_in_stab[-1] +1)% self.d == 0: # on the right 
                    xrb_nodes += [d]
                else:
                    pass
            # Check if Z are top/bottom
            else: # now these are Z detectors
                qubits_in_stab = sorted(self.H_z.getrow(stab_index - self.H_x.shape[0]).indices)
                # print(qubits_in_stab[0])
                if qubits_in_stab[0] < self.d:
                    ztb_nodes += [d]
                elif qubits_in_stab[-1] >= (self.d**2-self.d):
                    zbb_nodes += [d]

        return xlb_nodes,xrb_nodes,ztb_nodes,zbb_nodes


    def get_edge_type_d(self, dem, mem_type, CD_type) -> dict:
        """
        Returns the dictionary mapping edges in the DEM to stabilizer measurement types. Updates the edge_type_d attribute of the class.
        The dictionary is for marginal edges only: hyperedges are assumed decomposed.
        Inputs:
        dem - (stim.DetectorErrorModel) the dem noise model used for the code
        Outputs:
        edge_type_d - (dict) a dictionary mapping edges in the DEM to stabilizer measurement types. Type 0(1) use pauli X(Z) measurements
            eg. {(0,-1):0, (2,4):1, (3,5):0, ...}
        """
        if CD_type != "SC":
            CD_data_transform = cc.CD_data_func(self.code.qbit_dict.values(), special=CD_type, ell=self.l, size=self.d)
        else:
            CD_data_transform = cc.CD_data_func(self.code.qbit_dict.values(), special="I", ell=self.l, size=self.d)

        for inst in dem:
            if inst.type == "error":
                decomposed_inst = self.decompose_dem_instruction_stim(inst)

                for edge in decomposed_inst["detectors"]:
                    # print(edge)
                    if tuple(sorted(edge)) in self.edge_type_d:
                        pass
                    else:
                        self.edge_type_d[tuple(sorted(edge))] = self.get_edge_type_from_detector(edge, mem_type, CD_data_transform)
                    
        return self.edge_type_d
    

    #
    # Complementary Gap
    #

    def get_complementary_gap(self,circuit,syndrome,obs_flips, decoder_type="MWPM"):
        '''
        Credit: Eva Takou for code backbone. Minor style changes have been made. The original function
        calculates the complementary gap (MWPM soft information) in the style of arxiv:2312.04522

        Inputs: 
        matching: the pymatching graph
        syndrome: the detector syndrome
        obs_flips: Z(X) logical flipped in X(Z) memory
        b1_nodes: the X(Z) detector nodes to the left(top) boundary for X(Z) memory (list of ints)
        b2_nodes: the X(Z) detector nodes to the right(bottom) boundary for X(Z) memory (list of ints)

        Outputs:
        Gap:                complementary gap
        Signed_Gap:         signed complementary gap
        gap_conditioned_PL: gap conditioned logical error rate
        
        
        '''    
        # print(obs_flips.shape)
        dem = circuit.detector_error_model(decompose_errors=True, flatten_loops=True, approximate_disjoint_errors=True)
        num_shots = np.shape(syndrome)[0]
        comp_matching = Matching(enable_correlations=True)
        matching = Matching.from_detector_error_model(dem, enable_correlations=True)

        xlb_nodes, xrb_nodes, ztb_nodes, zbb_nodes = self.get_LB_RB_nodes(dem)

        if self.mem_type == "X":
            b1_nodes = xlb_nodes
            b2_nodes = xrb_nodes
        else:
            b1_nodes = ztb_nodes
            b2_nodes = zbb_nodes


        b1 = max(b2_nodes)+1
        b2 = b1+1
        
        for edge in matching.edges():
            node1 = edge[0]
            node2 = edge[1]


            # when the edge is not a boundary add to the graph normally
            if node2 is not None:
                
                comp_matching.add_edge(node1=node1,node2=node2,
                                fault_ids = edge[2]['fault_ids'],
                                weight=edge[2]['weight'],
                                error_probability=edge[2]['error_probability'])
            
            # if the edge is a boundary edge 
            else: 
                if node1 in b1_nodes: # match to the left/top node
                    node2 = b1 
                if node1 in b2_nodes: # match to the right/bottom node
                    node2 = b2 

                # if the stabilizer is not of the memory type, keep to normal boundaries
                if node2 is None:
                    comp_matching.add_boundary_edge(node=node1,
                                                    fault_ids = edge[2]['fault_ids'],
                                                    weight=edge[2]['weight'],
                                                    error_probability=edge[2]['error_probability'])

                else:
                    comp_matching.add_edge(node1=node1,node2=node2,
                                    fault_ids = edge[2]['fault_ids'],
                                    weight=edge[2]['weight'],
                                    error_probability=edge[2]['error_probability'])            
                
        
        comp_matching.set_boundary_nodes({b2})     


        # don't fire the b2
        new_array = np.zeros((num_shots,1),dtype=int)
        det0      = np.hstack((syndrome,new_array))
        
        # do fire b2
        new_array = np.ones((num_shots,1),dtype=int)
        det1      = np.hstack((syndrome,new_array))

        # the I_L / ERR_L complementary matchings. Return the total weights of the solutions for each shot
        if decoder_type == "MWPM":
            # decode to obtain the original matching
            pred_reg, W_reg = matching.decode_batch(syndrome,return_weights=True) #This is the regular matching

            # the node fixed matching
            pred0, W0 = comp_matching.decode_batch(det0,return_weights=True)
            pred1, W1 = comp_matching.decode_batch(det1,return_weights=True)
        elif decoder_type == "MY_CORR":
            # regular matching
            pred_reg, W_reg = self.decoding_failures_correlated_circuit_level(
                circuit, 
                shots=num_shots, 
                mem_type=self.mem_type, 
                CD_type=self.corr_type, 
                decompose_biased=True, 
                return_weights=True, 
                input_syndrome=syndrome, 
                input_obs_flips=obs_flips,
                comp_matching = matching,
                b_extra = None,
                )

            # node fixed matching
            pred0, W0 = self.decoding_failures_correlated_circuit_level(
                circuit, 
                shots=num_shots, 
                mem_type=self.mem_type, 
                CD_type=self.corr_type, 
                decompose_biased=True, 
                return_weights=True, 
                input_syndrome=det0, 
                # input_syndrome=None, 
                input_obs_flips=obs_flips,
                # input_obs_flips=None,
                comp_matching = comp_matching,
                b_extra = b2, # this is the node we want to fire for the complementary matching
                )
            pred1, W1 = self.decoding_failures_correlated_circuit_level(
                circuit, 
                shots=num_shots, 
                mem_type=self.mem_type, 
                CD_type=self.corr_type, 
                decompose_biased=True, 
                return_weights=True, 
                input_syndrome=det1, 
                input_obs_flips=obs_flips,
                comp_matching = comp_matching,
                b_extra = b2,
                )
        elif decoder_type == "PY_CORR":
            pred_reg, W_reg = matching.decode_batch(syndrome, enable_correlations=True, return_weights=True) #This is the regular matching
            pred0, W0 = comp_matching.decode_batch(det0, enable_correlations=True, return_weights=True)
            pred1, W1 = comp_matching.decode_batch(det1, enable_correlations=True, return_weights=True)

        # print(f"w0 {W0}, w1 {W1}, wreg {W_reg}")
        # print(f"pred0 {pred0}, pred1 {pred1}, pred_reg {pred_reg}")
        # print(pred0, W0)
        # print("now for the regular", pred_reg, W_reg)
        # scale by edge weight, get dB. Why do we do this? also do we assume all edges normalized by the weight of the first
        edge = next(iter(matching.to_networkx().edges.values()))
        edge_w = edge['weight']
        edge_p = edge['error_probability']
        decibels_per_w = -np.log10(edge_p / (1 - edge_p)) * 10 / edge_w                

        # Unsigned gap
        Gap = []
        for k in range(num_shots):
            if W1[k]<W0[k]:
                Gap.append( (W0[k]-W1[k]) * decibels_per_w)
            else:
                Gap.append( (W1[k]-W0[k]) * decibels_per_w)     

        
        # signed gap - negative indicates MWPM failed
        Signed_Gap = []
        W_min = np.zeros(W_reg.shape)
        W_comp = np.zeros(W_reg.shape)
        pred_min = np.zeros(pred0.shape)


        for k in range(num_shots):
            # if W_reg[k] == W0[k]: # not always the case for correlated matching ... haven't yet figured out why
            if pred_reg[k] == pred0[k]: # if the regular matching is the same as the node fixed one, then we know that the node fixed one is the min weight solution
                # print(f"running eq wreg and w0, shot {k}")
                W_min[k] = W0[k]
                pred_min[k] = pred0[k]
                W_comp[k] = W1[k]
            # elif W_reg[k] == W1[k]:
            elif pred_reg[k] == pred1[k]:
                # print(f"running eq wreg and w1, shot {k}")
                W_min[k] = W1[k]
                pred_min[k] = pred1[k]
                W_comp[k] = W0[k]

            if pred_min[k]==obs_flips[k]: 
                # print(f"appending signed gap for shot {k} positive")
                Signed_Gap.append( (W_comp[k]-W_min[k]) * decibels_per_w) 
            else:
                # print(f"appending signed gap for shot {k} negative")
                Signed_Gap.append( (W_min[k]-W_comp[k]) * decibels_per_w) 


        errors = np.any(pred_reg != obs_flips, axis=1)

        # Classify all shots by their error + gap.
        custom_counts = collections.Counter()
        Gap  = np.round(Gap).astype(dtype=np.int64)
        for k in range(len(Gap)):
            g = Gap[k]
            key = f'E{g}' if errors[k] else f'C{g}'
            custom_counts[key] += 1/num_shots

        # P_L(e | g) = E_g / (E_g + C_g)
 
        gap_conditioned_PL = {}

        # collect all gap values that appear
        gaps = set()
        for key in custom_counts:
            gaps.add(int(key[1:]))

        for g in gaps:
            E = custom_counts.get(f'E{g}', 0.0)
            C = custom_counts.get(f'C{g}', 0.0)

            if E + C > 0:
                gap_conditioned_PL[g] = E / (E + C)
            else:
                gap_conditioned_PL[g] = np.nan    


        return Gap,Signed_Gap,gap_conditioned_PL
    
    def get_complementary_correction(self, dem, syndrome, observable_flip, input_matching=None, return_predictions=False):
        """ For one shot at a time, get the unsigned gap, the matching and the complementary matching for one dem

            :param dem: (stim.DetectorErrorModel) the input detector error model to be used in matching
            :param syndrome: (numpy array) the detectors flipped in the experiment
            :param observable_flip: (numpy array) whether a logical observable was flipped
            :param matching: (Matching matching object) if you want to directly feed in a matching graph, use this instead of dem
            :param return_predictions: (bool) include the prediction in the return value

            :return unsigned_gap: (array) decoder confidence from comparing two matchings
            :return matching_correction: (array) the edges included in the min weight solution
            :return comp_matching_correction: (array) the edges in the solution to complementary solution
            :return pred_min: (bool) whether the min weight decoder solution flipped a logical
            :return pred_picked: (int) whether the solution is connected to boundaries(no boundaries) - 1(0)
        """
        
        comp_matching = Matching()
        if input_matching is None:
            matching = Matching.from_detector_error_model(dem)
        else:
            matching = input_matching

        syndrome = syndrome.reshape(1,syndrome.shape[0]) # I hope this is doing the right thing not sure it is
        xlb_nodes, xrb_nodes, ztb_nodes, zbb_nodes = self.get_LB_RB_nodes(dem)

        if self.mem_type == "X":
            b1_nodes = xlb_nodes
            b2_nodes = xrb_nodes
        else:
            b1_nodes = ztb_nodes
            b2_nodes = zbb_nodes


        b1 = max(b2_nodes)+1
        b2 = b1+1
        
        for edge in matching.edges():
            node1 = edge[0]
            node2 = edge[1]


            # when the edge is not a boundary add to the graph normally
            if node2 is not None:
                
                comp_matching.add_edge(node1=node1,node2=node2,
                                fault_ids = edge[2]['fault_ids'],
                                weight=edge[2]['weight'],
                                error_probability=edge[2]['error_probability'])
            
            # if the edge is a boundary edge 
            else: 
                if node1 in b1_nodes: # match to the left/top node
                    node2 = b1 
                if node1 in b2_nodes: # match to the right/bottom node
                    node2 = b2 

                # if the stabilizer is not of the memory type, keep to normal boundaries
                if node2 is None:
                    comp_matching.add_boundary_edge(node=node1,
                                                    fault_ids = edge[2]['fault_ids'],
                                                    weight=edge[2]['weight'],
                                                    error_probability=edge[2]['error_probability'])

                else:
                    comp_matching.add_edge(node1=node1,node2=node2,
                                    fault_ids = edge[2]['fault_ids'],
                                    weight=edge[2]['weight'],
                                    error_probability=edge[2]['error_probability'])            
                
        
        comp_matching.set_boundary_nodes({b2})     
                
        # decode to obtain the original matching
        pred_reg, W_reg = matching.decode(syndrome,return_weight=True) #This is the regular matching


        # don't fire the b1 - I_L coset
        new_array = np.zeros((1,1),dtype=int)
        det0      = np.hstack((syndrome,new_array))
        
        # do fire b1 - Z/X_L coset
        new_array = np.ones((1,1),dtype=int)
        det1      = np.hstack((syndrome,new_array))

        # pred0 crosses logical even number of times, pred1 crosses odd number. Return the total weights of the solutions for each shot
        pred0, W0 = comp_matching.decode(det0,return_weight=True)
        pred1, W1 = comp_matching.decode(det1,return_weight=True)

        edges_in_pred0 = np.array(comp_matching.decode_to_edges_array(det0))
        edges_in_pred1 = np.array(comp_matching.decode_to_edges_array(det1))


        # signed gap
        if W_reg == W0: # MWPM picked pred0 solution
            pred_picked = 0
            W_min = W0
            pred_min = pred0
            W_comp = W1
            edges_in_correction = np.where(np.logical_or((edges_in_pred0 == b1) ,(edges_in_pred0 == b2)), -1, edges_in_pred0)
            edges_in_comp_correction = np.where(np.logical_or((edges_in_pred1 == b1), (edges_in_pred1 == b2)), -1, edges_in_pred1)
        else: # MWPM picked pred 1 solution 
            pred_picked = 1
            W_min = W1
            pred_min = pred1
            W_comp = W0
            edges_in_correction = np.where(np.logical_or((edges_in_pred1 == b1), (edges_in_pred1 == b2)), -1, edges_in_pred1)
            edges_in_comp_correction = np.where(np.logical_or((edges_in_pred0 == b1), (edges_in_pred0 == b2)), -1, edges_in_pred0)
        

        # if pred_min == observable_flip: # MWPM was successful 
        #     signed_gap = W_comp - W_min
        # else:
        #     signed_gap = W_min - W_comp

        unsigned_gap = W_comp - W_min


        if return_predictions:
            return unsigned_gap, edges_in_correction, edges_in_comp_correction, pred_min, pred_picked
        else:
            return unsigned_gap, edges_in_correction, edges_in_comp_correction



    #
    # Hyperedge decomposition (only decompose_dem_instruction_stim used)
    #


    def decompose_dem_instruction_stim_auto(self, inst):
        """ Decomposes a stim DEM instruction into its component detectors and probability. Uses STIM's decompose_errors to determine hyperedge decomposition.
            Decomposed edge is in the form {probability: [detector1, detector2, ...]}. Logical operators are omitted, and single detector errors are merged to a pair if decomposed.
            We insert boundary edges to odd cardinality hyperedges. Edges are sorted such that boundary edges are always last in the tuple, and the detectors are in ascending order.

            eg. error(p) D0 ^ D1 L0 -> {p: [(0, 1)]}
                error(p) D0 D2 ^ D1 -> {p: [(0, 2), (1, "BOUNDARY")]}. 

            :param inst: stim.DEMInstruction object. The instruction to be decomposed.
            :return: decomp_inst: dict. A dictionary with the probability as the key and a list of edges as the value.
        """
        # get the edge probability and detectors for an instruction
        prob_err = inst.args_copy()[0]
        targets = np.array(inst.targets_copy())
        decomp_inst = {prob_err: []}

        
        seperator_indices = np.where([target.is_separator() for target in targets])[0]
        split_indices = seperator_indices + 1
        edges = np.split(targets, split_indices)
        edges = [[e.val for e in edge if e.is_relative_detector_id()] for edge in edges]
        total_num_detectors = sum([len(edge) for edge in edges])
        if total_num_detectors > 2:
            for edge in edges:
                if len(edge) % 2 == 1 and len(edges) > 1:
                    edge.append("BOUNDARY")
        
        # Convert edges to list of tuples
        if total_num_detectors <= 2:
            # Flatten and group into one tuple if <= 2 detectors total
            flattened = [e for edge in edges for e in edge]
            edges = [tuple(sorted(flattened, key=lambda x: (isinstance(x, str), x)))]
        else:
            edges = [tuple(sorted(edge, key=lambda x: (isinstance(x,str), x))) for edge in edges]

        # Store result
        decomp_inst[prob_err] = edges

        return decomp_inst
    
    def decompose_dem_instruction_stim(self, inst):
        """
        Decomposes a stim DEM instruction into pairwise detector edges and assigns observables
        to the edges based on which sub-block (separated by `^`) the observable appeared in. Use
        stim DEM instruction decomposition from decompose_errros=True to choose hyperedge 
        decomposition

        Example:
            error(p) D0 D1^D2 L0 -> {p:p, detectors: [(0, -1), (2, -1)], observables: [None, 0]}
            error(p) D0 D1 L0^D2 -> {p:p, detectors: [(0, 1), (-1, 2)], observables: [0, None]}
            error(p) D0 D2 ^ D3 -> {p:p, detectors: [(0, 2), (-1, 3)], observables:[None, None]}
            error(p) D0 -> {p: p, detectors: [(-1, 0)], observables: [None]} 

        Returns:
            {
                'p': float,
                'detectors': List[Tuple[int, int]],
                'observables': List[Optional[int]],
            }
        """
        targets = list(inst.targets_copy())
        p = inst.args_copy()[0]

        blocks = []  # Each block is a list of targets between separators (^)
        current_block = []

        for t in targets:
            if t.is_separator():
                if current_block:
                    blocks.append(current_block)
                    current_block = []
            else:
                current_block.append(t)

        if current_block:
            blocks.append(current_block)

        detector_edges = []
        edge_observables = []

        for block in blocks:
            dets = []
            obs = []

            for t in block:
                if t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id():
                    obs.append(t.val)

            # Handle detectors → edges
            if len(dets) == 0:
                continue  # no detector => no edge
            elif len(dets) == 1:
                edge = (-1, dets[0])  # boundary edge
                detector_edges.append(edge)
                edge_observables.append(obs[0] if obs else None)
            else:
                # Decompose pairwise through chain
                for i in range(len(dets) - 1):
                    edge = tuple(sorted((dets[i], dets[i+1])))
                    detector_edges.append(edge)
                    edge_observables.append(obs[0] if obs else None)

        return {
            "p": p,
            "detectors": detector_edges,
            "observables": edge_observables
        }

    def decompose_dem_instruction_pairwise(self, inst):
        """ Decomposes a stim DEM instruction into its component detectors and probability. Uses pairwise decomposition to determine hyperedge decomposition.
            Decomposed edge is in the form {probability: [detector1, detector2, ...]}. Logical operators are omitted, and single detector errors are merged to a pair if decomposed.
            We insert boundary edges to edges with one detector, boundary node value is -1. Edges are sorted such that boundary edges are always last in the tuple, and the detectors are in ascending order.


            eg. error(p) D0 D1 L0 -> {p: p, detectors: [(0, 1)], observables: [0]}
                error(p) D0 -> {p: p, detectors: [(-1, 0)], observables: []} single detector error gets boundary edge
                error(p) D0 D2 D1 -> {p:p, detectors: [(0, 2), (2, 1)], observables: []}. 
                error(p) D0 D2 ^ D3 -> {p:p, detectors: [(0, 2), (2, 3)], observables:[]} We choose to ignore ^. If we treated the ^ as already decomposing, we would get [(0,2), (3,-1)]
                error(p) D0 D2 D3 L0 -> {p:p, detectors: [(0, 2), (2, 3)], observables:[0]}. 

            :param inst: stim.DEMInstruction object. The instruction to be decomposed.
            :return: decomp_inst: dict. A dictionary recording the probability of the error for that DEM instruction, the edges included in the
            decomposition, and the logical observables included.
        """
        # get the edge probability and detectors for an instruction
        targets = list(inst.targets_copy())
        decomp_inst = {"p": inst.args_copy()[0], "detectors": [], "observables": []}

        # separate detectors, logical observables, and separators
        for t in targets:
            if t.is_separator():
                continue
            elif t.is_logical_observable_id():
                # logical observable: L#
                decomp_inst["observables"].append(t.val)
            elif t.is_relative_detector_id():
                # detector: D#
                decomp_inst["detectors"].append(t.val)
        
        total_num_detectors = len(decomp_inst["detectors"])

        # iterate through array and make pairwise edge tuples with probability prob_err
        detectors = decomp_inst["detectors"]
        edges = []

        if total_num_detectors == 1:
            edges = [(-1, detectors[0])] # include a boundary edge
        
        else: # pairwise decompose
            for i in range(total_num_detectors-1):
                edges.append(tuple(sorted([detectors[i], detectors[i+1]])))
        
        # store result
        decomp_inst["detectors"] = edges
        return decomp_inst
    
    def decompose_dem_instruction_star(self, inst):
        """ Decomposes a stim DEM instruction into its component detectors and probability. Uses star decomposition to determine hyperedge decomposition.
            Decomposed edge is in the form {probability: [detector1, detector2, ...]}. Logical operators are omitted, and single detector errors are merged to a pair if decomposed.
            We insert boundary edges to edges with one detector, boundary node value is -1. Edges are sorted such that boundary edges are always last in the tuple, and the detectors are in ascending order.
            PASS IN DEM with DECOMPOSE_ERRORS=FALSE - talk to ken about this


            eg. error(p) D0 D1 L0 -> {p: p, detectors: [(0, 1)], observables: [0]}
                error(p) D0 -> {p: p, detectors: [(0, -1)], observables: []} single detector error gets boundary edge
                error(p) D0 D2 D1 -> {p:p, detectors: [(0, 2), (0, 1)], observables: []}. 
                error(p) D0 D2 ^ D3 -> {p:p, detectors: [(0, 2), (2, 3)], observables:[]} We choose to ignore ^. If we treated the ^ as already decomposing, we would get [(0,2), (3,-1)]
                error(p) D0 D2 D3 L0 -> {p:p, detectors: [(0, 2), (0, 3)], observables:[0]}. 

            :param inst: stim.DEMInstruction object. The instruction to be decomposed.
            :return: decomp_inst: dict. A dictionary recording the probability of the error for that DEM instruction, the edges included in the
            decomposition, and the logical observables included.
        """
        # get the edge probability and detectors for an instruction
        targets = list(inst.targets_copy())
        decomp_inst = {"p": inst.args_copy()[0], "detectors": [], "observables": []}

        # separate detectors, logical observables, and separators
        for t in targets:
            if t.is_separator():
                continue
            elif t.is_logical_observable_id():
                # logical observable: L#
                decomp_inst["observables"].append(t.val)
            elif t.is_relative_detector_id():
                # detector: D#
                decomp_inst["detectors"].append(t.val)
        
        total_num_detectors = len(decomp_inst["detectors"])

        # iterate through array and make pairwise edge tuples with probability prob_err
        detectors = decomp_inst["detectors"]
        edges = []

        if total_num_detectors == 1:
            edges = [(-1, detectors[0])] # include a boundary edge
        
        else: # star decompose
            center_node = detectors[0]
            for i in range(total_num_detectors-1):
                edges.append(tuple(sorted([center_node, detectors[i+1]])))
        
        # store result
        decomp_inst["detectors"] = edges
        return decomp_inst

    # 
    # Edge decomposition tables 
    #

    def get_joint_prob(self, dem):
        """ Creates an array of joint probabilities representing edges in the DEM. Each entry [E][F] is the joint probability of edges E and detector F. 
            The diagonal entries [E][E] are the marginal probabilities of one graphlike error mechanism. The joint probabilities are calculated using the bernoulli formula for combining 
            probabilities when two detectors share more than one hyperedge.

            :param dem: stim.DetectorErrorModel object. The detector error model of the circuit to be used in decoding.
            :return: joint_probs: dictionary {[edge 1][edge 2]: joint probability} The joint probability matrix. Each cell is the joint probability of two detectors.
        """

        
        joint_probs = {} # each entry is the joint probability of two edges. [E][E] is a marginal probability
        fault_ids = {} # each entry is the fault id for that edge

        # iterate through each edge in the dem, add hyperedges
        for inst in dem:
            if inst.type == "error":
                decomposed_inst = self.decompose_dem_instruction_stim(inst) # used to be pairwise
                prob_err = decomposed_inst["p"]
                edges = decomposed_inst["detectors"]
                observables = decomposed_inst["observables"]

                # update hyperedges in joint probability table
                if len(edges) > 1:
                    a, b = edges[0], edges[1]
                    p01 = joint_probs.get(a, {}).get(b, 0)
                    p10 = joint_probs.get(b, {}).get(a, 0)

                    new_p01 = self.bernoulli_prob(p01, prob_err)
                    new_p10 = self.bernoulli_prob(p10, prob_err)

                    joint_probs.setdefault(a, {})[b] = new_p01
                    joint_probs.setdefault(b, {})[a] = new_p10

                # update marginal probabilities
                for i,edge in enumerate(edges):
                    p = joint_probs.get(edge, {}).get(edge, 0)
                    new_p = self.bernoulli_prob(p, prob_err)
                    joint_probs.setdefault(edge, {})[edge] = new_p
                    
                    # assign fault ids
                    obs = observables[i]
                    # obs = observables
                    fault_ids[edge] = fault_ids.get(edge) or obs
                
        return joint_probs, fault_ids 
    
    def get_conditional_prob(self, joint_prob_dict, decompose_biased):
        """ Given a joint probability dictionary, calculates the conditional probabilities for each hyperedge. The conditional probability is given by 
            P(A|B) = P(A^B)/P(A)
            Where A and B are edges from decomposed hyperedges. The marginal probability is P(A), and the joint probability is P(A^B). The maximum conditional probability is 0.5
            Only hyperedge components are present in final dictionary.

            :param joint_prob_dict: the joint probability of decomposed hyperedge between edges A and B
            :return: conditional probability nested dictionary. Of the same form as joint_prob_dict:
                    {edge tuple 1:{edge tuple 1: marginal probability, edge tuple two: conditional probability, P(edge 2 | edge 1), ...}, ...} 
        """

        cond_prob_dict = {}

        for edge_1 in joint_prob_dict:
            # find P(A)
            marginal_p = joint_prob_dict.get(edge_1, {}).get(edge_1,0)
            if marginal_p == 0:
                continue

            adjacent_edge_dict = joint_prob_dict.get(edge_1, {})

            # populate cond_prob dictionary 
            for edge_2 in adjacent_edge_dict: # in the other function, e1 is edge in correction and e2 is the edge affected. Here it is different

                if edge_1 == edge_2:  
                    continue 

                joint_p = joint_prob_dict.get(edge_1, {}).get(edge_2,0)
                edge_check_type = self.edge_type_d[edge_2] # have to make sure this is populated by the time I populate
                # print(edge_check_type, edge_2)

                scale = 1
                if decompose_biased:
                    if edge_check_type == "X": # edge_2 is a Z error since it's checks are X type.
                        scale = self.eta/(self.eta + 1)
                    elif edge_check_type == "Z": # edge_2 is an X error
                        scale = 1/2*(self.eta + 1)


                # conditional probability calculation. Min taken because weights cannot be negative, and eta=0.5 represents a full erasure channel
                # cond_p = min(1/(2*self.eta + 1), joint_p/marginal_p) # how do I do directionality here / I might have to think about it, will this actually work? Dont wanna fully erase edges...?
                cond_p = min(0.5, scale*joint_p/marginal_p) # trying to include the channel, not sure about directionaility still
                cond_prob_dict.setdefault(edge_1, {})[edge_2] = cond_p
        return cond_prob_dict

    #
    # Graph construction
    #

    def edit_dem(self, edges_in_correction, dem, cond_prob_dict):
        """ Given a stim DEM, updates the probabilities in error instructions with detectors given by cond_prob_dict based on detectors fired in correction.
            If a detector edge picked in the correction has a key in cond_prob_dict, it belonged to a hyperedge. The conditional probability then overwrites
            the original DEM probability for that hyperedge. Logical observables are distributed across new error instructions as in the original instruction.
        """
        # get a list of corrected edges from the first round
        edges_in_correction = [tuple(sorted(edge)) for edge in edges_in_correction]

        # iterate through the dem and fix the probabilities if they're in the cond_prob_dict
        # Create new DEM with updated probabilities
        new_dem = stim.DetectorErrorModel()

        for inst in dem:
            if inst.type == "error":
                old_prob = inst.args_copy()[0]
                decomposed_inst = self.decompose_dem_instruction_pairwise(inst)
                
                if len(decomposed_inst["detectors"]) > 1: # if the edge is a hyperedge
                    
                    for edge_1 in decomposed_inst["detectors"]: # break each hyperedge into sub-edges with their conditional prob
                        new_prob = old_prob
                        for edge_2 in edges_in_correction: # check which conditional probability is highest out of hyperedges
                            curr_prob = cond_prob_dict.get(edge_2, {}).get(edge_1,0)
                            new_prob = max(curr_prob, new_prob)
                        
                        targets = [stim.target_relative_detector_id(node) for node in edge_1] # will I have a problem with value -1?

                        if len(decomposed_inst["observables"]) > 0:
                            targets += [stim.target_logical_observable_id(l) for l in decomposed_inst["observables"]]
                        
                        new_inst = stim.DemInstruction("error", [new_prob], targets) # targets in edge_1 only
                        new_dem.append(new_inst)
                    
                    
                else: # if the edge is not a hyperedge, leave it be
                    new_dem.append(inst) 
            else:
                new_dem.append(inst)  # Preserve non-error instructions like detectors or shifts


        return new_dem

    def compute_edge_weights_from_conditional_probs(self, correction_edges, match_graph, cond_prob_dict, fault_ids_dict):
        weights = {}
        fault_ids = {}
        all_edges = match_graph.edges()
        # print(fault_ids_dict)
        edges_in_correction = [tuple(sorted(edge)) for edge in correction_edges]
        for u,v,data in all_edges:
            # print(u,v)
            e2 = tuple(sorted([-1 if x is None else x for x in (u, v)]))
            # print(e2)
            log_error = fault_ids_dict.get(e2, None)
            # print(log_error)
            p = max((cond_prob_dict.get(e1, {}).get(e2, 0) for e1 in edges_in_correction), default=0)
            if p > 0:
                weight = np.log((1-p)/p) 
                # print(f"updating edge {(u,v)} with conditional probability {p} and weight {weight}, from weight {data['weight']}")
            else:
                weight = data['weight']
            weights[(u, v)] = weight
            # fault_ids[(u, v)] = set([log_error]) if log_error is not None else set()
            fault_ids[(u, v)] = data['fault_ids']
            # this fault id has indices with (node, None): set(id)
        # print(fault_ids)
        return weights, fault_ids
    
    def compute_edge_weights_from_comp_gap(self, correction_edges, comp_correction_edges, matching, unsigned_gap, cutoff):
        """ Adjust the edge weights based on the complementary gap obtained during first pass matching.
            Use the signed gap to determine whether to use the min weight or complementary correction. 

            :param correction_edges(list): list of node pairs that represent the edges in the first MWPM pass
            :param comp_correction_edges(list): list of node pairs that represent the spatial complementary error in MWPM first pass
            :param matching(Matching): the matching graph to be updated
            :param unsigned_gap(float): magnitude represents first pass decoder confidence (sum of weights). 
            :param cutoff(float): the gap magnitude that is lower than the relative weights. Determines whether
                                we assign the gap to the complementary or minimum error path.
            :return: the weights and fault_ids dictionary recording the adjusted weight for each edge in the matchgraph
        """

        weights = {}
        fault_ids = {}

        sorted_edges_in_correction = [tuple(sorted(edge)) for edge in correction_edges]
        sorted_comp_correction_edges = [tuple(sorted(edge)) for edge in comp_correction_edges]

        mwpm_correction = [edge for edge in sorted_edges_in_correction if edge not in sorted_comp_correction_edges]
        comp_correction = [edge for edge in sorted_comp_correction_edges if edge not in sorted_edges_in_correction]

        edge_weight_dB_scale = self.get_dB_scaling(matching)
        
        for u,v,data in matching.edges():
            # fix the boundary nodes comparison because pymatching is inconsistent
            edge = tuple([u if (v is not None) else -1, v if (v is not None) else u])
            
            if np.abs(edge_weight_dB_scale*unsigned_gap) <= cutoff: # when the confidence is low choose the complementary path
                if edge in mwpm_correction:
                    weights[(u,v)] = 1e6
                else:
                    weights[(u,v)] = data['weight']
            else:
                weights[(u,v)] = data['weight'] # maybe try the other way later ... 
                # if edge in comp_correction:
                #     weights[(u,v)] = np.abs(edge_weight_dB_scale*signed_gap)
                # else:
                #     weights[(u,v)] = data['weight']
        
            fault_ids[(u,v)] = data['fault_ids']
        return weights, fault_ids
    
    def compute_edge_weights_all_correlated_info(self, correction_edges, matching, unsigned_gap, cond_prob_dict, fault_ids_dict):
        # definitely add stopping conditions if decoder is already right
        # when getting the hyperedge corrections, check if the comp decoder was right first too 
        
        weights = {}
        fault_ids = {}
        edges_in_correction = [tuple(sorted(edge)) for edge in correction_edges]
        for u,v,data in matching.edges():

            # edges in the correction get adjusted by unsigned gap
            if (u,v) in edges_in_correction:
                print(f"{u,v} weight adjusted by unsigned gap")
                weight = 1/unsigned_gap

            # edges not in the correction get hyperedge adjustments
            else:
                e2 = tuple(sorted([-1 if x is None else x for x in (u, v)])) # get (u,v) to the proper bndry format given my code
                
                # find the max conditional probability adjustment for this edge given the correction
                p = max((cond_prob_dict.get(e1, {}).get(e2, 0) for e1 in edges_in_correction), default=0) 

                if p > 0:
                    print(f"{u,v} weight adjusted by cond prob")
                    weight = np.log((1-p)/p) 
                else:
                    weight = data['weight']
            fault_ids[(u, v)] = data['fault_ids']
            weights[(u, v)] = weight

        return weights, fault_ids

    def build_matching_from_weights(self, weights_dict, fault_ids_dict, original_num_nodes, b_extra=None):
        match = Matching()
        for (u, v), weight in weights_dict.items():
            # fault_id = fault_ids_dict.get(tuple([u if v is not None else -1, v if v is not None else u]), None)
            # print(fault_id, u,v)
            fault_id = fault_ids_dict.get((u,v),None)
            if None in (u, v):
                match.add_boundary_edge(u if u is not None else v, weight=weight, fault_ids=fault_id)
            else:
                match.add_edge(u, v, weight=weight, fault_ids=fault_id)
        
        # Now detect which nodes were never added via any edge
        used_nodes = set()
        for (u, v) in weights_dict.keys():
            if u is not None:
                used_nodes.add(u)
            if v is not None:
                used_nodes.add(v)

        # Fill in unused detector nodes (not involved in any edge)
        all_nodes = set(range(original_num_nodes))
        missing_nodes = all_nodes - used_nodes

        for node in missing_nodes:
            # Use an extremely high weight to ensure these edges are not used
            match.add_boundary_edge(node, weight=1e6)
        
        if b_extra is not None:
            match.set_boundary_nodes({b_extra})

        return match


    #
    # Decoding
    #

    def decoding_failures_correlated_circuit_level(
            self, 
            circuit, 
            shots, 
            mem_type, 
            CD_type, 
            decompose_biased=True, 
            return_weights=False, 
            input_syndrome=None, 
            input_obs_flips=None, 
            comp_matching=None,
            b_extra=None,
            ):
        """
        Finds the number of logical errors given a circuit using correlated decoding. Uses pymatching's correlated decoding approach, inspired by
        papers cited in the README.
        :param circuit: stim.Circuit object, the circuit to decode
        :param p: physical error rate
        :param shots: number of shots to sample
        :param memtype: basis to run memory experiment
        :param CD_type: the clifford deformation type applied to the code
        :param decompose_biased: whether to decompose hyperedges with bias in mind or give equal weight to X and Z components
        :param return_weights: whether to return the weights of the total path
        :return: number of logical errors
        """

        # 
        # Get the edge data for correlated decoding
        #

        # get the DEM get the matching graph
        dem = circuit.detector_error_model(decompose_errors=True, flatten_loops=True, approximate_disjoint_errors=True)
        if comp_matching is not None:
            matchgraph = comp_matching
        else:
            matchgraph = Matching.from_detector_error_model(dem, enable_correlations=False)
        
        self.edge_type_d = self.get_edge_type_d(dem, mem_type, CD_type)

        # get the joint probabilities table of the dem hyperedges
        joint_prob_dict, fault_ids = self.get_joint_prob(dem)
        
        # calculate the conditional probabilities based on joint probablities and marginal probabilities 
        cond_prob_dict = self.get_conditional_prob(joint_prob_dict, decompose_biased)

        # instead of performing the first round of error correction and going based on this, create a MWPM graph based on hyperedges in joint_prob_dict

        # new_dem = edit_dem() 

        
        
        #
        # Decode the circuit
        #
        
        # first round of decoding
        # get the syndromes and observable flips
        if input_syndrome is None and input_obs_flips is None:
            seed = np.random.randint(0, 2**32 - 1)
            sampler = circuit.compile_detector_sampler(seed=seed)
            syndrome, observable_flips = sampler.sample(shots, separate_observables=True)
            # print(syndrome.shape, observable_flips.shape)
        else: 
            syndrome, observable_flips = input_syndrome, input_obs_flips
            # print(syndrome.shape, observable_flips.shape)
        # print("syndrome inside function:", syndrome )

        # from eva
        # change the logicals so that there is an observable for each qubit, change back to the code cap case to check whether the real logical flipped

        corrections = np.zeros((shots, 1)) # largest fault id is 1, len of correction = 2 .... i changed this to 1 bc i have no clue why it was 2
        weights = np.zeros(shots)
        for i in range(shots):

            # print(syndrome[i].shape)
            edges_in_correction = matchgraph.decode_to_edges_array(syndrome[i])
            # print("edges in correction inside function from mycorr", edges_in_correction)

            
            # update weights based on conditional probabilities
            # updated_dem = self.edit_dem(edges_in_correction, dem, cond_prob_dict) # is this DEM updated correctly? make sure that it is getting the right edges

            # second round of decoding with updated weights
            # matching_corr = Matching.from_detector_error_model(updated_dem, enable_correlations=False)
            updated_weights, fault_ids_dict = self.compute_edge_weights_from_conditional_probs(edges_in_correction, matchgraph, cond_prob_dict, fault_ids)
            matching_corr = self.build_matching_from_weights(updated_weights, fault_ids_dict, matchgraph.num_nodes, b_extra=b_extra)
            # print("updated edges inside function from mycorr", matching_corr.edges())
            # print(matching_corr.decode(syndrome[i]).shape, matching_corr.decode(syndrome[i]))
            if return_weights:
                corrections[i], weights[i] = matching_corr.decode(syndrome[i], return_weight=True)
            else:
                corrections[i] = matching_corr.decode(syndrome[i])

        
        # calculate the number of logical errors
        log_errors_array = np.any(np.array(observable_flips) != np.array(corrections), axis=1) # usual code
        if return_weights:
            return corrections, weights
        else:
            return log_errors_array

    def decoding_failures_correlated_gap(self, circuit, shots, mem_type, CD_type, cutoff=1):
        """
        Two stage decoding following arxiv:2312.04522., with the addition of a hyperedge decoding step.
        """

        # get the hyperedge data + set up original matching
        dem = circuit.detector_error_model(decompose_errors=True, flatten_loops=True, approximate_disjoint_errors=True)
        matchgraph = Matching.from_detector_error_model(dem, enable_correlations=False)
        self.edge_type_d = self.get_edge_type_d(dem, mem_type, CD_type)

        # get the joint probabilities table of the dem hyperedges
        joint_prob_dict, fault_ids = self.get_joint_prob(dem)
        
        # calculate the conditional probabilities based on joint probablities and marginal probabilities 
        cond_prob_dict = self.get_conditional_prob(joint_prob_dict, decompose_biased=False)



        #
        # Decode the circuit
        #
        
        # first round of decoding
        # get the syndromes and observable flips
        seed = np.random.randint(0, 2**32 - 1)
        sampler = circuit.compile_detector_sampler(seed=seed) # should I be passing in a seed instead so I am comparing LER of right shots?
        detection_events, observable_flips = sampler.sample(shots, separate_observables=True)

        
        corrections = np.zeros(observable_flips.shape)
        for shot in range(shots):
            us_gap, edges_in_correction, edges_in_comp_correction, pred_min, pred_picked = self.get_complementary_correction(dem, detection_events[shot], observable_flips[shot], return_predictions=True)

            
            # when the first pass of MWPM is not confident, get the complementary graph 
            if us_gap < cutoff:
                comp_weights,comp_fault_ids = self.compute_edge_weights_from_comp_gap(edges_in_correction,edges_in_comp_correction, matchgraph, us_gap, cutoff)
                comp_matching = self.build_matching_from_weights(comp_weights, comp_fault_ids, matchgraph.num_nodes)

                # hyperedge adjustment based on comp correction
                hyperedge_weights, hyperedge_fault_ids = self.compute_edge_weights_from_conditional_probs(edges_in_comp_correction,
                                                                                                                comp_matching,
                                                                                                                cond_prob_dict,
                                                                                                                comp_fault_ids)

            else: # the first correction is confident, just do regular hyperedge decomposition on correction
                hyperedge_weights, hyperedge_fault_ids = self.compute_edge_weights_from_conditional_probs(edges_in_correction,
                                                                                                                matchgraph,
                                                                                                                cond_prob_dict,
                                                                                                                fault_ids)
            hyperedge_matching = self.build_matching_from_weights(hyperedge_weights, hyperedge_fault_ids,matchgraph.num_nodes)
            
            corrections[shot] = hyperedge_matching.decode(detection_events[shot])
        
        log_errors_array = np.any(np.array(observable_flips) != np.array(corrections), axis=1)

        return log_errors_array


    #
    #
    # Circuit sampling functions
    #
    #

    def get_num_log_errors(self, circuit, num_shots):
        """
        Get the number of logical errors from a circuit phenom. model, not the detector error model
        :param circuit: stim.Circuit object
        :param num_shots: number of shots to sample
        :return: logical errors array. Sum of array is the number of logical errors
        """
        matching = Matching.from_stim_circuit(circuit)
        seed = np.random.randint(0, 2**32 - 1)
        sampler = circuit.compile_detector_sampler(seed=seed)
        detection_events, observable_flips = sampler.sample(num_shots, separate_observables=True)
        predictions = matching.decode_batch(detection_events)
        
        
        num_errors_array = np.zeros(num_shots)
        for shot in range(num_shots):
            actual_for_shot = observable_flips[shot]
            predicted_for_shot = predictions[shot]
            if not np.array_equal(actual_for_shot, predicted_for_shot):
                num_errors_array[shot] = 1
        return num_errors_array

    def get_num_log_errors_DEM(self, circuit, num_shots, enable_corr, enable_pymatch_corr, meas_type, CD_type="SC"):
        """
        Get the number of logical errors from the detector error model
        :param circuit: stim.Circuit object
        :param num_shots: number of shots to sample
        :param enable_corr: boolean whether to use house-made correlated decoder
        :param enable_pymatch_corr: boolean whether to use pymatching correlated decoder
        :return: number of logical errors
        """
        if enable_corr:
            # house-made circuit level correlated decoder
            log_errors_array = self.decoding_failures_correlated_circuit_level(circuit, num_shots, meas_type, CD_type)

        
        else: # no correlated decoding or pymatching correlated decoding
            dem = circuit.detector_error_model(decompose_errors=enable_pymatch_corr, approximate_disjoint_errors=True)
            matchgraph = Matching.from_detector_error_model(dem,enable_correlations=enable_pymatch_corr)
            seed = np.random.randint(0, 2**32 - 1)
            sampler = circuit.compile_detector_sampler(seed=seed) # double check that this randomness is doing the right thing, every shot should be random and compare
            syndrome, observable_flips = sampler.sample(num_shots, separate_observables=True) # do i need to set a seed here?
            predictions = matchgraph.decode_batch(syndrome, enable_correlations=enable_pymatch_corr) # had a weird recent error, should have thrown an error earlier when I passed in enable correlations
            log_errors_array = np.any(np.array(observable_flips) != np.array(predictions), axis=1)
        
        return log_errors_array

    def get_log_error_circuit_level(self, p_list, meas_type, num_shots, noise_model="code_cap", cd_type="SC", corr_decoding= False, pymatch_corr = False, fully_biased=False):
        """
        Get the logical error rate for a list of physical error rates of gates at the circuit level
        :param p_list: list of p values
        :param meas_type: type of stabilizers measured in memory experiment. Meas type X indicates ZL detection for Z errors
        :param num_shots: number of shots to sample
        :param noise_model: the noise model to use, either "code_cap", "phenom", or "circuit_level". Code cap has a biased depolarizing channel on data 
            qubits at the beginning of rounds. Phenominological model has a biased depolarizing channel on data qubits at the beginning of rounds and bit-flip noise on 
            measurement qubits before measurement. Circuit level has biased depolarizing channel at the beginning of rounds, bit-flip noise on measurement qubits before measurement, 
            and a two-qubit depolarizing channel after each two-qubit clifford gate.
        :param cd_type: the type of clifford defomation applied to the circuit. Either None, XZZXonSqu, or ZXXZonSqu.
        :return: list of logical error rates, opposite type of the measurement type (e.g. if meas_type is X, then Z logical errors are returned)
        """
        

        log_error_L = []
        for p in p_list:
            # make the circuit
            circuit_obj = cc_circuit.CDCompassCodeCircuit(self.d, self.l, self.eta, meas_type) # change list of ps dependent on model
            if noise_model == "code_cap":# change this based on the noise model you want
                circuit = circuit_obj.make_elongated_circuit_from_parity(0,0,0,p,0,0,CD_type=cd_type, memory=False)  
            elif noise_model == "phenom":
                circuit = circuit_obj.make_elongated_circuit_from_parity(p,0,0,p,0,0,CD_type=cd_type, phenom_meas=True) # check the plots that matched pymatching to get error model right, before meas flip and data qubit pauli between rounds
            elif noise_model == "circuit_level":
                circuit = circuit_obj.make_elongated_circuit_from_parity(before_measure_flip=p,before_measure_pauli_channel=0,after_clifford_depolarization=p,before_round_data_pauli_channel=0,between_round_idling_pauli_channel=p,idling_dephasing=0,CD_type=cd_type, fully_biased=fully_biased) # between round idling biased pauli on all qubits, measurement flip errors, 2-qubit gate depolarizing
            else:
                raise ValueError("Invalid noise model. Choose either 'code_cap', 'phenom', or 'circuit_level'.")
            
            log_errors_array = self.get_num_log_errors_DEM(circuit, num_shots, corr_decoding, pymatch_corr, meas_type, cd_type)
            log_error_L.append(log_errors_array)

        return log_error_L

    def get_log_error_p(self, p_list, meas_type, num_shots):
        """ 
        Get the logical error rate for a list of physical error rates of gates at code cap using a circuit
        :param p_list: list of p values
        :param meas_type: type of memory experiment(X or Z), stabilizers measured
        :param num_shots: number of shots to sample
        :return: list of logical error rates
        """
        log_error_L = []
        for p in p_list:
            # make the circuit
            circuit = cc_circuit.CDCompassCodeCircuit(self.d, self.l, self.eta, [0.003, 0.001, p], meas_type)
            log_errors = self.get_num_log_errors(circuit.circuit, num_shots)
            log_error_L += [log_errors/num_shots]
        return log_error_L






