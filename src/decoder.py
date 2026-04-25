from matplotlib.lines import Line2D
import numpy as np
from pymatching import Matching
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
from scipy import sparse, linalg
import compass_codes as cc
import csv
import pandas as pd
import os
import collections
from datetime import datetime
import sys
import glob
from scipy.optimize import curve_fit
import circuit_gen as cc_circuit
import itertools
import stim
# from lmfit import Minimizer, Parameters, report_fit


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



############################################
#
# Functions for getting data (from DCC too)
#
############################################

def shots_averaging(num_shots, l, eta, err_type, in_df, CD_type, file, d=None, noise_model=None, min_total_shots=300):
    """Weighted average of chunked data."""
    if in_df is None:
        in_data = pd.read_csv(file)
        data = in_data[
            (in_data['l'] == l) &
            (in_data['eta'] == eta) &
            (in_data['error_type'] == err_type) &
            (in_data['CD_type'] == CD_type)
        ]
        if d is not None:
            data = data[data['d'] == d]
        if noise_model is not None and 'noise_model' in data.columns:
            data = data[data['noise_model'] == noise_model]
    else:
        data = in_df.copy()

    if num_shots is not None:
        data = data[data['num_shots'] == num_shots]

    data['weighted_errors'] = data['num_log_errors'] * data['num_shots']
    data_mean = (
        data.groupby('p', as_index=False)
            .agg({'num_shots': 'sum', 'weighted_errors': 'sum'})
    )
    data_mean['num_log_errors'] = data_mean['weighted_errors'] / data_mean['num_shots']
    data_mean = data_mean.drop(columns='weighted_errors')

    if min_total_shots is not None:
        data_mean = data_mean[data_mean['num_shots'] >= min_total_shots]
    return data_mean

def get_data(
    total_num_shots,
    d_list,
    l,
    p_list,
    eta,
    corr_type,
    circuit_data,
    noise_model="circuit_level",
    cd_type="SC",
    corr_decoding=False,
    pymatch_corr=False,
    data_file=None,
    append=False,
    chunk_size=5000,
    resume=True,
    fully_biased=False,
):
    """Generate logical error-rate data in chunks, with resume support.

    For each (d, p), this function checks `data_file` for previously completed
    chunks and only runs the remaining shots.
    """
    err_type = {0: "X", 1: "Z", 2: corr_type, 3: "TOTAL"}

    if circuit_data:
        columns = [
            "d", "num_shots", "p", "l", "eta", "error_type",
            "noise_model", "CD_type", "num_log_errors", "time_stamp"
        ]
    else:
        columns = [
            "d", "num_shots", "p", "l", "eta", "error_type",
            "num_log_errors", "time_stamp"
        ]

    all_rows = []

    expected_error_types = _get_expected_error_types(
        corr_type=corr_type,
        circuit_data=circuit_data,
        corr_decoding=corr_decoding,
        pymatch_corr=pymatch_corr,
    )

    existing_df = _safe_read_csv(data_file) if resume else None

    def flush_rows(rows_to_write):
        """Append rows to CSV immediately and force flush to disk."""
        if not rows_to_write:
            return

        if append and data_file is not None:
            chunk_df = pd.DataFrame(rows_to_write, columns=columns)
            file_exists = os.path.isfile(data_file)
            chunk_df.to_csv(
                data_file,
                mode="a",
                header=not file_exists,
                index=False,
            )

            with open(data_file, "a") as f:
                f.flush()
                os.fsync(f.fileno())

    for d in d_list:
        decoder = CorrelatedDecoder(eta, d, l, corr_type)

        for p in p_list:
            completed_shots = 0
            if resume:
                completed_shots = _get_completed_shots_for_point(
                    existing_df=existing_df,
                    d=d,
                    p=p,
                    l=l,
                    eta=eta,
                    expected_error_types=expected_error_types,
                    circuit_data=circuit_data,
                    noise_model=noise_model,
                    cd_type=cd_type,
                )

            if completed_shots >= total_num_shots:
                print(
                    f"Skipping d={d}, p={p}, eta={eta}, l={l} "
                    f"because {completed_shots}/{total_num_shots} shots already exist."
                )
                continue

            shots_done = completed_shots

            print(
                f"Resuming d={d}, p={p}, eta={eta}, l={l} "
                f"from {completed_shots}/{total_num_shots} shots."
            )

            while shots_done < total_num_shots:
                curr_num_shots = min(chunk_size, total_num_shots - shots_done)

                print(
                    f"Running d={d}, p={p}, eta={eta}, l={l}, "
                    f"shots {shots_done} -> {shots_done + curr_num_shots}"
                )

                if circuit_data:
                    log_errors_z_array = decoder.get_log_error_circuit_level(
                        np.array([p]),
                        "Z",
                        curr_num_shots,
                        noise_model,
                        cd_type,
                        corr_decoding,
                        pymatch_corr,
                        fully_biased=fully_biased
                    )
                    log_errors_x_array = decoder.get_log_error_circuit_level(
                        np.array([p]),
                        "X",
                        curr_num_shots,
                        noise_model,
                        cd_type,
                        corr_decoding,
                        pymatch_corr,
                        fully_biased=fully_biased
                    )

                    log_errors_z = np.sum(log_errors_z_array, axis=1)[0]
                    log_errors_x = np.sum(log_errors_x_array, axis=1)[0]
                    log_errors_total = np.sum(
                        np.logical_or(log_errors_x_array, log_errors_z_array),
                        axis=1,
                    )[0]

                    x_err_type, z_err_type, total_err_type = _get_expected_error_types(corr_type, circuit_data=circuit_data, corr_decoding=corr_decoding, pymatch_corr=pymatch_corr)

                    rows_for_chunk = [
                        {
                            "d": d,
                            "num_shots": curr_num_shots,
                            "p": p,
                            "l": l,
                            "eta": eta,
                            "error_type": x_err_type,
                            "noise_model": noise_model,
                            "CD_type": cd_type,
                            "num_log_errors": log_errors_x / curr_num_shots,
                            "time_stamp": datetime.now(),
                        },
                        {
                            "d": d,
                            "num_shots": curr_num_shots,
                            "p": p,
                            "l": l,
                            "eta": eta,
                            "error_type": z_err_type,
                            "noise_model": noise_model,
                            "CD_type": cd_type,
                            "num_log_errors": log_errors_z / curr_num_shots,
                            "time_stamp": datetime.now(),
                        },
                        {
                            "d": d,
                            "num_shots": curr_num_shots,
                            "p": p,
                            "l": l,
                            "eta": eta,
                            "error_type": total_err_type,
                            "noise_model": noise_model,
                            "CD_type": cd_type,
                            "num_log_errors": log_errors_total / curr_num_shots,
                            "time_stamp": datetime.now(),
                        },
                    ]

                else:
                    errors = decoder.decoding_failures_correlated(p, curr_num_shots)

                    rows_for_chunk = []
                    for i in range(len(errors)):
                        rows_for_chunk.append({
                            "d": d,
                            "num_shots": curr_num_shots,
                            "p": p,
                            "l": l,
                            "eta": eta,
                            "error_type": err_type[i],
                            "num_log_errors": errors[i] / curr_num_shots,
                            "time_stamp": datetime.now(),
                        })

                all_rows.extend(rows_for_chunk)
                flush_rows(rows_for_chunk)

                shots_done += curr_num_shots

                print(
                    f"Saved d={d}, p={p}, eta={eta}, l={l}, "
                    f"chunk_shots={curr_num_shots}, total_done={shots_done}/{total_num_shots}"
                )

                # keep in-memory state updated too, so repeated points in one run
                # see the latest completed shots without rereading the file
                if existing_df is None:
                    existing_df = pd.DataFrame(rows_for_chunk, columns=columns)
                else:
                    existing_df = pd.concat(
                        [existing_df, pd.DataFrame(rows_for_chunk, columns=columns)],
                        ignore_index=True
                    )

    return pd.DataFrame(all_rows, columns=columns)

def write_data(
    total_num_shots,
    d_list,
    l,
    p_list,
    eta,
    ID,
    corr_type,
    circuit_data,
    noise_model="circuit_level",
    cd_type="SC",
    corr_decoding=False,
    pymatch_corr=False,
    chunk_size=5000,
    overwrite=False,
    resume=True,
    fully_biased=False
):
    """Write data incrementally to CSV while the job runs.

    Parameters
    ----------
    total_num_shots : int
        Total number of shots desired for each (d, p).
    chunk_size : int
        Number of shots to run before checkpointing to CSV.
    overwrite : bool
        If True, delete an existing file with the same ID before starting.
    """
    # if circuit_data:
    #     os.makedirs("circuit_data", exist_ok=True)
    #     if pymatch_corr:
    #         data_file = f"circuit_data/py_corr_{ID}.csv"
    #     else:
    #         data_file = f"circuit_data/circuit_level_{ID}.csv"
    # else:
    #     os.makedirs("corr_err_data", exist_ok=True)
    #     data_file = f"corr_err_data/code_cap_{ID}.csv"

    if circuit_data:
        os.makedirs("circuit_data", exist_ok=True)

        if pymatch_corr:
            prefix = "py_corr"
        elif corr_decoding:
            prefix = "corr"
        else:
            prefix = "circuit_level"

        d_tag = "_".join(str(d) for d in d_list)
        p_tag = "_".join(f"{float(p):.8f}" for p in p_list)

        data_file = (
            f"circuit_data/{prefix}"
            f"_l{l}_eta{eta}_cd{cd_type}_d{d_tag}_p{p_tag}.csv"
        )

    if overwrite and os.path.isfile(data_file):
        os.remove(data_file)

    data = get_data(
        total_num_shots=total_num_shots,
        d_list=d_list,
        l=l,
        p_list=p_list,
        eta=eta,
        corr_type=corr_type,
        circuit_data=circuit_data,
        noise_model=noise_model,
        cd_type=cd_type,
        corr_decoding=corr_decoding,
        pymatch_corr=pymatch_corr,
        data_file=data_file,
        append=True,
        chunk_size=chunk_size,
        resume=resume,
        fully_biased=fully_biased,
    )

    return data

def concat_csv(folder_path, circuit_data):
    """Combines all CSV files is in folder 'folder_path' and writes them to one common 
        'output_file'. The CSV files in folder_path are deleted.
        in: folder_path - the folder that stores all the csv files to be combined
            output_file - the file that the CSV files will be combined into
        out: no output. The folder_path files are deleted and the output_file has the files in folder_path added to it
    """
    data_files = glob.glob(os.path.join(folder_path, '*.csv'))
    df_list_XZ = []
    df_list_ZX = []
    df_list_CL = []
    
    for file in data_files:
        df = pd.read_csv(file)
        if not circuit_data: # the error types are X, Z, CORR_XZ, CORR_ZX, TOTAL, want to classify based on CORR_XZ and CORR_ZX
            if 'CORR_XZ' in df['error_type'].values:
                df_list_XZ.append(df)
            elif 'CORR_ZX' in df['error_type'].values:
                df_list_ZX.append(df)
        else:
            df_list_CL.append(df) # the error types are X_Mem and Z_Mem
    
    if circuit_data:
        new_data_CL = pd.concat(df_list_CL, ignore_index=True)
    else:
        new_data_XZ = pd.concat(df_list_XZ, ignore_index=True)
        new_data_ZX = pd.concat(df_list_ZX, ignore_index=True)
    
    output_file_XZ = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/xz_corr_err_data.csv'
    output_file_ZX = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/zx_corr_err_data.csv'
    output_file_CL = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/circuit_data.csv'
    
    all_data_XZ = pd.DataFrame()
    all_data_ZX = pd.DataFrame()
    all_data_CL = pd.DataFrame()

    xz_exists = os.path.exists(output_file_XZ)
    zx_exists = os.path.exists(output_file_ZX)
    cl_exists = os.path.exists(output_file_CL)

    # Check if the output file already exists
    if not circuit_data:
        if xz_exists:
            # If it exists, load the existing data
            existing_data = pd.read_csv(output_file_XZ)
            # Append the new data to the existing data
            all_data_XZ = pd.concat([existing_data, new_data_XZ], ignore_index=True)
        elif not xz_exists:
            # If the file doesn't exist, the new data is the combined data
            all_data_XZ = new_data_XZ

        if zx_exists:
            # If it exists, load the existing data
            existing_data = pd.read_csv(output_file_ZX)
            # Append the new data to the existing data
            all_data_ZX = pd.concat([existing_data, new_data_ZX], ignore_index=True)
        elif not zx_exists:
            # If the file doesn't exist, the new data is the combined data
            all_data_ZX = new_data_ZX
        all_data_XZ.to_csv(output_file_XZ, index=False)
        all_data_ZX.to_csv(output_file_ZX, index=False)

    else:
        if cl_exists:
            # If it exists, load the existing data
            existing_data = pd.read_csv(output_file_CL)
            # Append the new data to the existing data
            all_data_CL = pd.concat([existing_data, new_data_CL], ignore_index=True)
        else:
            # If the file doesn't exist, the new data is the combined data
            all_data_CL = output_file_CL
        
        all_data_CL.to_csv(output_file_CL, index=False)

    
    for file in data_files:
        os.remove(file)

def load_and_concat_csvs(folder, pattern="*.csv"):
    files = glob.glob(f"{folder}/{pattern}")
    
    if len(files) == 0:
        raise ValueError(f"No CSV files found in {folder}")
    
    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    full_df = pd.concat(df_list, ignore_index=True)
    return full_df

def combine_chunked_data(df):
    df = df.copy()

    # Convert rate → total errors (temporarily)
    df["weighted_errors"] = df["num_log_errors"] * df["num_shots"]

    # Define grouping columns depending on data type
    group_cols = ["d", "p", "l", "eta", "error_type"]

    if "noise_model" in df.columns:
        group_cols += ["noise_model", "CD_type"]

    # Aggregate
    combined = (
        df.groupby(group_cols, as_index=False)
          .agg({
              "num_shots": "sum",
              "weighted_errors": "sum"
          })
    )

    # Convert back to rate
    combined["num_log_errors"] = (
        combined["weighted_errors"] / combined["num_shots"]
    )

    combined = combined.drop(columns="weighted_errors")

    return combined

def load_and_combine(folder, pattern="*.csv"):
    df = load_and_concat_csvs(folder, pattern)
    combined_df = combine_chunked_data(df)
    return combined_df

# use this now to merge csvs
def append_task_csvs_into_master(
    folder="circuit_data",
    master_file="circuit_data.csv",
    pattern="*.csv",
    delete_after_merge=False,
):
    """
    Append raw task CSV rows into the master CSV without touching existing rows.

    - Old master contents are left untouched.
    - New task rows are reordered to match the master header before appending.
    - Keeps time_stamp.
    - Does NOT combine shots.
    """

    files = glob.glob(os.path.join(folder, pattern))

    # Exclude the master file itself if it lives in the same folder
    master_abs = os.path.abspath(master_file)
    files = [f for f in files if os.path.abspath(f) != master_abs]

    if len(files) == 0:
        raise ValueError(f"No CSV files found in {folder}")

    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if len(df_list) == 0:
        raise ValueError("No readable task CSV files found.")

    new_df = pd.concat(df_list, ignore_index=True)

    # If master exists, match its exact column order
    if os.path.isfile(master_file):
        master_header = pd.read_csv(master_file, nrows=0).columns.tolist()

        # Add any missing columns to new_df
        for col in master_header:
            if col not in new_df.columns:
                new_df[col] = np.nan

        # Keep only the master columns, in master order
        new_df = new_df[master_header]

        # Append only new rows; do not rewrite old rows
        new_df.to_csv(master_file, mode="a", header=False, index=False)

    else:
        # If master does not exist yet, create it using a canonical order
        preferred_cols = [
            "d", "num_shots", "p", "l", "eta",
            "error_type", "num_log_errors", "time_stamp",
            "noise_model", "CD_type"
        ]
        existing_cols = [c for c in preferred_cols if c in new_df.columns]
        remaining_cols = [c for c in new_df.columns if c not in existing_cols]
        new_df = new_df[existing_cols + remaining_cols]

        new_df.to_csv(master_file, index=False)

    if delete_after_merge:
        for f in files:
            os.remove(f)

    return new_df

def _get_expected_error_types(corr_type, circuit_data, corr_decoding=False, pymatch_corr=False):
    """Return the list of error_type strings expected for one completed chunk."""
    if circuit_data:
        if pymatch_corr:
            return ["X_MEM_PY", "Z_MEM_PY", "TOTAL_MEM_PY"]
        elif corr_decoding:
            return ["X_MEM_CORR", "Z_MEM_CORR", "TOTAL_MEM_CORR"]
        else:
            return ["X_MEM", "Z_MEM", "TOTAL_MEM"]
    else:
        return ["X", "Z", corr_type, "TOTAL"]

def _safe_read_csv(csv_file):
    """Read CSV if it exists and is nonempty; otherwise return None."""
    if csv_file is None or (not os.path.isfile(csv_file)):
        return None
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Warning: could not read {csv_file}: {e}")
        return None

def _get_completed_shots_for_point(
    existing_df,
    d,
    p,
    l,
    eta,
    expected_error_types,
    circuit_data,
    noise_model=None,
    cd_type=None,
):
    """
    Return the number of shots already safely completed for one (d,p,l,eta,...)
    point in an existing per-task CSV.

    We sum num_shots separately for each expected error_type and take the minimum.
    That way, if the last write was partial/corrupted, we only count the shots that
    are present for all required error types.
    """
    if existing_df is None or existing_df.empty:
        return 0

    df = existing_df.copy()

    mask = (
        (df["d"] == d) &
        (np.round(df["p"], 12) == round(float(p), 12)) &
        (df["l"] == l) &
        (df["eta"] == eta)
    )

    if circuit_data:
        mask = mask & (df["noise_model"] == noise_model) & (df["CD_type"] == cd_type)

    df = df[mask]

    if df.empty:
        return 0

    completed_per_err = []
    for err in expected_error_types:
        err_df = df[df["error_type"] == err]
        if err_df.empty:
            completed_per_err.append(0)
        else:
            completed_per_err.append(int(err_df["num_shots"].sum()))

    completed_shots = min(completed_per_err) if completed_per_err else 0
    # completed_p = max(df["p"].unique()) if not df.empty else None
    return completed_shots#, completed_p

def get_data_DCC_chat(
    circuit_data,
    corr_decoding,
    noise_model,
    d_list,
    l_list,
    eta_list,
    cd_list,
    corr_list,
    total_num_shots,
    p_list=None,
    p_th_init_d=None,
    p_range=0.001,
    n_p=20,
    pymatch_corr=False,
    fully_biased=False,
    chunk_size=1000,
    overwrite=False,
    resume=True,
    shots_per_task=None,
):
    """
    Smaller-granularity SLURM array launcher.

    Circuit-level:
        one task = one (l, eta, cd_type, d, p)

    Code-cap correlated:
        one task = one (l, eta, corr_type, d, p)

    Parameters
    ----------
    total_num_shots : int
        Target total shots you eventually want per point.
    shots_per_task : int or None
        If set, each task contributes this many shots to its one point.
        This is strongly recommended for walltime-limited runs.
    """

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    if "SLURM_ARRAY_TASK_COUNT" in os.environ:
        slurm_array_size = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    else:
        slurm_array_size = int(os.environ["SLURM_ARRAY_TASK_MAX"]) + 1

    print(f"Task ID: {task_id}")
    print(f"SLURM Array Size: {slurm_array_size}")

    if circuit_data:
        # Build p-grid first
        circuit_param_base = list(itertools.product(l_list, eta_list, cd_list, d_list))
        param_arr = []

        for l, eta, cd_type, d in circuit_param_base:
            if p_th_init_d is not None:
                if pymatch_corr:
                    err_key = "TOTAL_MEM_PY"
                elif corr_decoding:
                    err_key = "TOTAL_MEM_PY" # CHANGE THIS LATER IF MORE DATA
                else:
                    err_key = "TOTAL_MEM"

                p_th_init = p_th_init_d[(l, eta, err_key, cd_type, noise_model)]
                p_list_local = np.linspace(
                    max(p_th_init - p_range, 0.0),
                    min(p_th_init + p_range, 1.0),
                    n_p,
                )
            else:
                p_list_local = p_list

            for p in p_list_local:
                param_arr.append((l, eta, cd_type, d, float(p)))

        num_param_points = len(param_arr)

        if task_id >= num_param_points:
            raise ValueError(
                f"Task ID {task_id} exceeds number of parameter points {num_param_points}"
            )

        l, eta, cd_type, d, p = param_arr[task_id]
        corr_type = "None"

        if shots_per_task is None:
            # fallback behavior, but not recommended
            reps = max(1, slurm_array_size // num_param_points)
            num_shots = int(total_num_shots // reps)
        else:
            num_shots = int(shots_per_task)

        print("Running circuit-level point:")
        print(f"l={l}, eta={eta}, cd_type={cd_type}, d={d}, p={p}")
        print(f"shots this task = {num_shots}")

        write_data(
            total_num_shots=num_shots,
            d_list=[d],
            l=l,
            p_list=[p],
            eta=eta,
            ID=task_id,
            corr_type=corr_type,
            circuit_data=circuit_data,
            noise_model=noise_model,
            cd_type=cd_type,
            corr_decoding=corr_decoding,
            pymatch_corr=pymatch_corr,
            chunk_size=chunk_size,
            overwrite=overwrite,
            resume=resume,
            fully_biased=fully_biased,
        )

    elif corr_decoding and not circuit_data:
        codecap_param_base = list(itertools.product(l_list, eta_list, corr_list, d_list))
        param_arr = []

        for l, eta, corr_type, d in codecap_param_base:
            if p_th_init_d is not None:
                p_th_init = p_th_init_d[(l, eta, corr_type)]
                p_list_local = np.linspace(
                    max(p_th_init - 0.03, 0.0),
                    min(p_th_init + 0.03, 1.0),
                    n_p,
                )
            else:
                p_list_local = p_list

            for p in p_list_local:
                param_arr.append((l, eta, corr_type, d, float(p)))

        num_param_points = len(param_arr)

        if task_id >= num_param_points:
            raise ValueError(
                f"Task ID {task_id} exceeds number of parameter points {num_param_points}"
            )

        l, eta, corr_type, d, p = param_arr[task_id]
        cd_type = "SC"
        noise_model_local = "code_cap"

        if shots_per_task is None:
            reps = max(1, slurm_array_size // num_param_points)
            num_shots = int(total_num_shots // reps)
        else:
            num_shots = int(shots_per_task)

        print("Running code-cap point:")
        print(f"l={l}, eta={eta}, corr_type={corr_type}, d={d}, p={p}")
        print(f"shots this task = {num_shots}")

        write_data(
            total_num_shots=num_shots,
            d_list=[d],
            l=l,
            p_list=[p],
            eta=eta,
            ID=task_id,
            corr_type=corr_type,
            circuit_data=circuit_data,
            noise_model=noise_model_local,
            cd_type=cd_type,
            corr_decoding=corr_decoding,
            pymatch_corr=pymatch_corr,
            chunk_size=chunk_size,
            overwrite=overwrite,
            resume=resume,
            fully_biased=fully_biased,
        )

def get_data_DCC(
        circuit_data, 
        corr_decoding, 
        noise_model, 
        d_list, 
        l_list, 
        eta_list, 
        cd_list, 
        corr_list, 
        total_num_shots, 
        p_list=None, 
        p_th_init_d=None, 
        p_range = 0.001,
        pymatch_corr=False,
        chunk_size=5000,
        overwrite=False,
        resume=True,
    ):
    """ Function to get the data from the DCC using parallel SLURM arrays. Each array task will get data for a specific (l, eta, corr_type) or (l, eta, cd_type) combo.
        The total number of shots will be split evenly across the array tasks so that the total number of shots is reached upon averaging. 
        in: circuit_data - boolean, whether to get data from circuit or vector code cap
            corr_decoding - boolean, whether to get data from correlated decoding or not
            noise_model - string, the noise model to use for circuit data, either "code_cap", "phenom", or "circuit_level"
            d_list - list of distances to run
            l_list - list of elongations to run
            eta_list - list of noise biases to run
            cd_list - list of clifford deformations to run, either "SC", "XZZXonSqu", or "ZXXZonSqu". Not to be used if corr_decoding is True and circuit_data is False.
            corr_list - list of correlation types to run, either "CORR_XZ" or "CORR_ZX". Only to be used if corr_decoding is True and circuit_data is False.
            total_num_shots - int, the total number of shots after averaging. Each SLURM array task will run total_num_shots/reps shots.
            p_list - list of physical error rates to scan over. If None, will be set based on p_th_init_d
            p_th_init_d - dictionary with keys (l, eta, corr_type) or (l, eta, cd_type) and values the initial guess for the threshold. If None, will use a default value based on eta
            pymatch_corr - boolean, whether to use pymatching correlated decoder for circuit data
        out: no output, but will write data to a CSV file for each SLURM array task. Run concat_csv after all tasks are complete to combine the CSV files into output_file.
    """


    task_id = int(os.environ['SLURM_ARRAY_TASK_ID']) # will iter over the total slurm array size and points to where you are 
    slurm_array_size = int(os.environ['SLURM_ARRAY_TASK_MAX']) # the size of the slurm array, used to determine how many tasks to run, currently 1000

    print(f"Task ID: {task_id}")
    print(f"SLURM Array Size: {slurm_array_size}")


    if circuit_data and not (corr_decoding or pymatch_corr): # change this to get different data for circuit level plot
        l_eta_cd_type_arr = list(itertools.product(l_list,eta_list,cd_list))
        reps = slurm_array_size//len(l_eta_cd_type_arr) # how many times to run file, num_shots each time
        ind = task_id%len(l_eta_cd_type_arr) # get the index of the task_id in the l_eta__corr_type_arr
        l, eta, cd_type = l_eta_cd_type_arr[ind] # get the l and eta from the task_id
        num_shots = int(total_num_shots//reps) # number of shots to sample
        print("l,eta,cd_type", l,eta, cd_type)
        corr_type = "None"
        if p_th_init_d is not None:
            p_th_init = p_th_init_d[(l, eta, "TOTAL_MEM", cd_type, noise_model)]
            p_list = np.linspace(max(p_th_init - p_range, 0.0), min(p_th_init + p_range, 1.0), 40)
        
        write_data(total_num_shots=num_shots, 
                   d_list=d_list, 
                   l=l, 
                   p_list=p_list, 
                   eta=eta, 
                   ID=task_id, 
                   corr_type=corr_type, 
                   circuit_data=circuit_data, 
                   noise_model=noise_model, 
                   cd_type=cd_type, 
                   corr_decoding=corr_decoding, 
                   pymatch_corr=pymatch_corr,
                   chunk_size=chunk_size,
                   overwrite=overwrite,
                   resume=resume)
    if circuit_data and (pymatch_corr or corr_decoding):
        l_eta_cd_type_arr = list(itertools.product(l_list,eta_list,cd_list))
        reps = slurm_array_size//len(l_eta_cd_type_arr) # how many times to run file, num_shots each time
        ind = task_id%len(l_eta_cd_type_arr) # get the index of the task_id in the l_eta__corr_type_arr
        l, eta, cd_type = l_eta_cd_type_arr[ind] # get the l and eta from the task_id, pymatching corr should be doing an erasure channel this whole time, see what happens
        num_shots = int(total_num_shots//reps) # number of shots to sample
        print("l,eta,cd_type", l,eta, cd_type)
        corr_type = "None"
        if p_th_init_d is not None:
            p_th_init = p_th_init_d[(l, eta, "TOTAL_MEM_PY", cd_type,noise_model)] # add the mem type somehow
            p_list = np.linspace(p_th_init - p_range, p_th_init + p_range, 18)
        write_data(total_num_shots=num_shots, 
                   d_list=d_list, 
                   l=l, 
                   p_list=p_list, 
                   eta=eta, 
                   ID=task_id, 
                   corr_type=corr_type, 
                   circuit_data=circuit_data, 
                   noise_model=noise_model, 
                   cd_type=cd_type, 
                   corr_decoding=corr_decoding, 
                   pymatch_corr=pymatch_corr,
                   chunk_size=chunk_size,
                   overwrite=overwrite,
                   resume=resume)

    if corr_decoding and not circuit_data: # change this to get different data for eta plot
        l_eta_corr_type_arr = list(itertools.product(l_list, eta_list, corr_list)) # list of tuples (l, eta, corr_type), currently 40
        reps = slurm_array_size//len(l_eta_corr_type_arr) # how many times to run file, num_shots each time
        ind = task_id%len(l_eta_corr_type_arr) # get the index of the task_id in the l_eta__corr_type_arr
        l, eta, corr_type = l_eta_corr_type_arr[ind] # get the l and eta from the task_id
        if p_th_init_d is not None:
            p_th_init = p_th_init_d[(l, eta, corr_type)]
            p_list = np.linspace(p_th_init - 0.03, p_th_init + 0.03, 40)
        num_shots = int(total_num_shots//reps) # number of shots to sample
        cd_type = "SC"
        noise_model = "code_cap"
        print("l,eta,corr_type", l,eta, corr_type)
        write_data(total_num_shots=num_shots, 
                   d_list=d_list, 
                   l=l, 
                   p_list=p_list, 
                   eta=eta, 
                   ID=task_id, 
                   corr_type=corr_type, 
                   circuit_data=circuit_data, 
                   noise_model=noise_model, 
                   cd_type=cd_type, 
                   corr_decoding=corr_decoding, 
                   pymatch_corr=pymatch_corr,
                   chunk_size=chunk_size,
                   overwrite=overwrite,
                   resume=resume)

    print("reps", reps)
    print("ind", ind)
    print("num_shots", num_shots)



######################
#
# plotting functions
#
######################


def full_error_plot(full_df, curr_eta, curr_l, curr_num_shots, noise_model, CD_type, file, corr_decoding=False, py_corr=False, loglog=False, averaging=True, circuit_level=False, plot_by_l=False):
    """Make a plot of all errors given a df with unedited contents of an entire CSV.
        :param full_df: pandas DataFrame with unedited contents from CSV
        :param curr_eta: current noise bias to filter DataFrame
        :param curr_l: current elongation parameter to filter DataFrame
        :param curr_num_shots: current number of shots to filter DataFrame
        :param noise_model: the type of simulation, either "code_cap", "phenom", or "circuit_level"
        :param CD_type: the type of clifford deformation used, from a list ["SC", "XZZXonSqu", "ZXXZonSqu"]
        :param py_corr: boolean whether pymatching correlated decoding was used, chooses from last of list ["CORR_XZ", "CORR_ZX", "X_MEM", "Z_MEM", "TOTAL_MEM", "X_MEM_PY", "Z_MEM_PY", "TOTAL_MEM_PY"]
        :param file: the CSV file path, used for averaging shots if in_df is None
        :param loglog: boolean whether to use loglog scale for plotting
        :param averaging: boolean whether to average shots over the number of jobs
        :param circuit_level: boolean whether the data is from circuit level simulations. Alternative is vector simulation.
        :param plot_by_l: boolean whether to plot by elongation parameter l instead of error type

        :return: no return, shows a matplotlib plot
    """

    # prob_scale = get_prob_scale(corr_type, curr_eta)

    # Filter the DataFrame based on the input parameters
    # filtered_df = full_df[(full_df['l'] == curr_l) & (full_df['eta'] == curr_eta) & (full_df['num_shots'] == curr_num_shots)] 
                    # & (df['time_stamp'].apply(lambda x: x[0:10]) == datetime.today().date())
    
    filtered_df = full_df[
    (full_df['l'] == curr_l) &
    (full_df['eta'] == curr_eta) &
    (full_df['noise_model'] == noise_model) &
    (full_df['CD_type'] == CD_type)
    ] # got rid num_shots

    if py_corr and not corr_decoding: 
        filtered_df = filtered_df[filtered_df['error_type'].isin(['X_MEM_PY', 'Z_MEM_PY', 'TOTAL_MEM_PY'])]
    elif corr_decoding:
        filtered_df = filtered_df[filtered_df['error_type'].isin(['X_MEM_CORR', 'Z_MEM_CORR', 'TOTAL_MEM_CORR'])]
    else:
        if circuit_level:
            filtered_df = filtered_df[filtered_df['error_type'].isin(['X_MEM', 'Z_MEM', 'TOTAL_MEM'])]
        else:
            filtered_df = filtered_df[filtered_df['error_type'].isin(['X', 'Z', 'TOTAL', 'CORR_XZ', 'CORR_ZX'])]

    # Get unique error types and unique d values
    error_types = filtered_df['error_type'].unique()

    d_values = filtered_df['d'].unique()


    # Create a figure with subplots for each error type
    if len(error_types)%2 == 0:
        fig, axes = plt.subplots(len(error_types)//2, 2, figsize=(15, 5*len(error_types)//2))
    else:
        fig, axes = plt.subplots((len(error_types)//2)+1, 2, figsize=(15, 5*((len(error_types)//2)+1)))
    axes = axes.flatten()
    

    # Plot each error type in a separate subplot
    for i, error_type in enumerate(error_types):
        ax = axes[i]
        ax.tick_params(axis='both', which='major', labelsize=16)  # Change major tick label size
        ax.tick_params(axis='x', which='major', labelsize=10)
        error_type_df = filtered_df[filtered_df['error_type'] == error_type]
        prob_scale = get_prob_scale(error_type, curr_eta)
        # Plot each d value
        for d in d_values:
            d_df = error_type_df[error_type_df['d'] == d]
            if averaging:
                # to check that this is working, figure out how big this DF is
                d_df_mean = shots_averaging(
                                num_shots=None,
                                l=curr_l,
                                eta=curr_eta,
                                err_type=error_type,
                                in_df=d_df,
                                CD_type=CD_type,
                                file=file
                            )
                d_df_mean = d_df_mean.sort_values("p").reset_index(drop=True)
                d_df_mean = d_df_mean[d_df_mean["num_shots"] >= 300]
                if loglog:
                    ax.loglog(d_df_mean['p']*prob_scale, d_df_mean['num_log_errors'],  label=f'd={d}')
                    error_bars = 10**(-6)*np.ones(len(d_df_mean['num_log_errors'])) #error bars are wrong
                    ax.fill_between(d_df_mean['p']*prob_scale, d_df_mean['num_log_errors'] - error_bars, d_df_mean['num_log_errors'] + error_bars, alpha=0.2)
                else:
                    ax.plot(d_df_mean['p']*prob_scale, d_df_mean['num_log_errors'],  label=f'd={d}')
            else:
                ax.scatter(d_df['p']*prob_scale, d_df['num_log_errors'], s=2, label=f'd={d}')

        
        ax.set_title(f'Error Type: {error_type}', fontsize=20)
        ax.set_xlabel('p', fontsize=14)
        ax.set_ylabel('num_log_errors', fontsize=20)
        ax.legend()

    if circuit_level:
        fig.suptitle(f'Logical Error Rates for eta = {curr_eta}, l = {curr_l}, Deformation = {CD_type}')
    else:
        fig.suptitle(f'Logical Error Rates for eta = {curr_eta} and l = {curr_l}')
    plt.tight_layout()
    plt.show()

def threshold_plot_old(full_df, p_th0, p_range, curr_eta, curr_l, curr_num_shots, corr_type, CD_type, noise_model, file, circuit_level=False, py_corr = False, corr_decoding=False, loglog=False, averaging=True, show_threshold=True, show_fit=False):
    """Make a plot of all 4 errors given a df with unedited contents"""

    prob_scale = get_prob_scale(corr_type, curr_eta)

    # Filter the DataFrame based on the input parameters
    filtered_df = full_df[(full_df['p'] > p_th0 - p_range)&(full_df['p'] < p_th0 + p_range)&(full_df['l'] == curr_l) & (full_df['eta'] == curr_eta) & (full_df['noise_model'] == noise_model) & (full_df['CD_type'] == CD_type)]
    
    if py_corr: 
        filtered_df = filtered_df[filtered_df['error_type'].isin(['X_MEM_PY', 'Z_MEM_PY', 'TOTAL_MEM_PY'])]
    elif corr_decoding:
        filtered_df = filtered_df[filtered_df['error_type'].isin(['X_MEM_CORR', 'Z_MEM_CORR', 'TOTAL_MEM_CORR'])]
    else:
        if circuit_level:
            filtered_df = filtered_df[filtered_df['error_type'].isin(['X_MEM', 'Z_MEM', 'TOTAL_MEM'])]
        else:
            filtered_df = filtered_df[filtered_df['error_type'].isin(['X', 'Z', 'TOTAL', 'CORR_XZ', 'CORR_ZX'])]
    
    
    d_values = filtered_df['d'].unique()
    num_lines = len(d_values)
    cmap = colormaps['Blues_r']
    color_values = np.linspace(0.1, 0.8, num_lines)
    colors = [cmap(val) for val in color_values]

    # Create a figure with subplots for each error type
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot each d value
    for i,d in enumerate(d_values):
        d_df = full_df[(full_df['d'] == d)&(full_df['error_type'] == corr_type)&(full_df['l'] == curr_l) & (full_df['eta'] == curr_eta)]
        if averaging:
            d_df_mean = shots_averaging(None, curr_l, curr_eta, corr_type, d_df, CD_type, file)
            if loglog:
                ax.loglog(d_df_mean['p']*prob_scale, d_df_mean['num_log_errors'],  label=f'd={d}', color=colors[i])
                error_bars = 10**(-6)*np.ones(len(d_df_mean['num_log_errors']))
                ax.fill_between(d_df_mean['p']*prob_scale, d_df_mean['num_log_errors'] - error_bars, d_df_mean['num_log_errors'] + error_bars, alpha=0.2, color=colors[i])
            else:
                ax.plot(d_df_mean['p']*prob_scale, d_df_mean['num_log_errors'],  label=f'd={d}', color=colors[i])
        else:
            ax.scatter(d_df['p']*prob_scale, d_df['num_log_errors'], s=2, label=f'd={d}',color=colors[i])

    popt, pcov = get_threshold(filtered_df, p_th0, p_range, curr_l, curr_eta, corr_type,curr_num_shots, CD_type)
    pth = popt[0]
    pth_error = np.sqrt(np.diag(pcov))[0]
    
    if show_threshold:
        ax.vlines(pth, ymin=0, ymax=max(filtered_df['num_log_errors']), color='red', linestyles='--', label=f'pth = {pth:.3f} +/- {pth_error:.3f}')
    if show_fit:
        for d in d_values:
            y_fit = []
            p_list = []
            for p in sorted(filtered_df['p'].unique()):
                x = (p, d)
                y_fit += [threshold_fit(x, *popt)]
                p_list += [p]
            if loglog:
                ax.loglog(np.array(p_list)*prob_scale, y_fit, linestyle='--', color='red')
            else:
                # print(p_list, y_fit)
                ax.plot(np.array(p_list)*prob_scale, y_fit, linestyle='--', color='red')

    
    ax.set_title(f'Error Type: {corr_type}', fontsize=20)
    ax.set_xlabel('p', fontsize=14)
    ax.set_ylabel('num_log_errors', fontsize=20)
    ax.legend()

    fig.suptitle(f'Logical Error Rates for eta = {curr_eta} and l = {curr_l}')
    plt.tight_layout()
    plt.show()

def threshold_plot(
    full_df,
    p_th0,
    p_range,
    curr_eta,
    curr_l,
    curr_num_shots,
    corr_type,
    CD_type,
    noise_model,
    file,
    circuit_level=False,
    py_corr=False,
    corr_decoding=False,
    loglog=False,
    averaging=True,
    show_threshold=True,
    show_fit=False,
):
    """Make a single threshold plot for one error type, using weighted averaging over raw chunk rows."""

    prob_scale = get_prob_scale(corr_type, curr_eta)

    # Filter the raw dataframe to the relevant physics slice and p-window
    filtered_df = full_df[
        (full_df["p"] > p_th0 - p_range) &
        (full_df["p"] < p_th0 + p_range) &
        (full_df["l"] == curr_l) &
        (full_df["eta"] == curr_eta) &
        (full_df["noise_model"] == noise_model) &
        (full_df["CD_type"] == CD_type)
    ].copy()

    if py_corr and not corr_decoding:
        filtered_df = filtered_df[
            filtered_df["error_type"].isin(["X_MEM_PY", "Z_MEM_PY", "TOTAL_MEM_PY"])
        ]
    elif corr_decoding:
        filtered_df = filtered_df[
            filtered_df["error_type"].isin(["X_MEM_CORR", "Z_MEM_CORR", "TOTAL_MEM_CORR"])
        ]
    else:
        if circuit_level:
            filtered_df = filtered_df[
                filtered_df["error_type"].isin(["X_MEM", "Z_MEM", "TOTAL_MEM"])
            ]
        else:
            filtered_df = filtered_df[
                filtered_df["error_type"].isin(["X", "Z", "TOTAL", "CORR_XZ", "CORR_ZX"])
            ]

    # Keep only the requested plotted error type
    filtered_df = filtered_df[filtered_df["error_type"] == corr_type].copy()

    if filtered_df.empty:
        print("No data found for this threshold plot.")
        return

    d_values = np.sort(filtered_df["d"].unique())
    num_lines = len(d_values)

    cmap = colormaps["Blues_r"]
    color_values = np.linspace(0.1, 0.8, max(num_lines, 1))
    colors = [cmap(val) for val in color_values]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot each d value
    for i, d in enumerate(d_values):
        d_df = filtered_df[filtered_df["d"] == d].copy()

        if d_df.empty:
            continue

        if averaging:
            # Temporary weighted combine over raw rows at equal p
            d_df_mean = shots_averaging(
                num_shots=None,   # important for new raw-row workflow
                l=curr_l,
                eta=curr_eta,
                err_type=corr_type,
                in_df=d_df,
                CD_type=CD_type,
                file=file,
            )

            # Keep only the plotting window just in case
            d_df_mean = d_df_mean[
                (d_df_mean["p"] > p_th0 - p_range) &
                (d_df_mean["p"] < p_th0 + p_range)
            ].copy()

            if d_df_mean.empty:
                continue

            if loglog:
                ax.loglog(
                    d_df_mean["p"] * prob_scale,
                    d_df_mean["num_log_errors"],
                    label=f"d={d}",
                    color=colors[i],
                )
                error_bars = 10**(-6) * np.ones(len(d_df_mean["num_log_errors"]))
                ax.fill_between(
                    d_df_mean["p"] * prob_scale,
                    d_df_mean["num_log_errors"] - error_bars,
                    d_df_mean["num_log_errors"] + error_bars,
                    alpha=0.2,
                    color=colors[i],
                )
            else:
                ax.plot(
                    d_df_mean["p"] * prob_scale,
                    d_df_mean["num_log_errors"],
                    label=f"d={d}",
                    color=colors[i],
                )
        else:
            # Raw scatter of chunk rows
            ax.scatter(
                d_df["p"] * prob_scale,
                d_df["num_log_errors"],
                s=2,
                label=f"d={d}",
                color=colors[i],
            )

    # Threshold fit on weighted-combined temporary data
    popt, pcov = get_threshold(
        full_df=filtered_df,
        pth0=p_th0,
        p_range=p_range,
        l=curr_l,
        eta=curr_eta,
        error_type=corr_type,
        num_shots=None,      # important: do not filter by per-row chunk size
        CD_type=CD_type,
        noise_model=noise_model,
    )

    if isinstance(popt, int) and popt == 0:
        print("Threshold fit failed or insufficient data.")
        pth = None
        pth_error = None
    else:
        pth = popt[0]
        pth_error = np.sqrt(np.diag(pcov))[0]

    if show_threshold and pth is not None:
        # Use weighted-combined y scale for a cleaner threshold line height
        temp_df = filtered_df.copy()
        temp_df["weighted_errors"] = temp_df["num_log_errors"] * temp_df["num_shots"]
        temp_avg = (
            temp_df.groupby(["d", "p"], as_index=False)
                  .agg({"num_shots": "sum", "weighted_errors": "sum"})
        )
        temp_avg["num_log_errors"] = temp_avg["weighted_errors"] / temp_avg["num_shots"]

        ymax = temp_avg["num_log_errors"].max() if not temp_avg.empty else filtered_df["num_log_errors"].max()

        ax.vlines(
            pth*prob_scale,
            ymin=0,
            ymax=ymax,
            color="red",
            linestyles="--",
            label=f"pth = {pth:.3f} +/- {pth_error:.3f}",
        )

    if show_fit and pth is not None:
        for d in d_values:
            # Build fit curve over the weighted-averaged p grid for that d
            d_df = filtered_df[filtered_df["d"] == d].copy()

            if averaging:
                d_df_mean = shots_averaging(
                    num_shots=None,
                    l=curr_l,
                    eta=curr_eta,
                    err_type=corr_type,
                    in_df=d_df,
                    CD_type=CD_type,
                    file=file,
                )
                p_vals = np.sort(d_df_mean["p"].unique())
            else:
                p_vals = np.sort(d_df["p"].unique())

            if len(p_vals) == 0:
                continue

            y_fit = [threshold_fit((p, d), *popt) for p in p_vals]

            if loglog:
                ax.loglog(
                    np.array(p_vals) * prob_scale,
                    y_fit,
                    linestyle="--",
                    color="red",
                )
            else:
                ax.plot(
                    np.array(p_vals) * prob_scale,
                    y_fit,
                    linestyle="--",
                    color="red",
                )

    ax.set_title(f"Error Type: {corr_type}", fontsize=20)
    ax.set_xlabel("p", fontsize=14)
    ax.set_ylabel("num_log_errors", fontsize=20)
    ax.legend()

    fig.suptitle(f"Logical Error Rates for eta = {curr_eta} and l = {curr_l}")
    plt.tight_layout()
    plt.show()

def eta_threshold_plot(eta_df, cd_type, corr_type_list, noise_model):
    """One subplot per corr_type, all l values overlaid, shaded error bands,
    single shared legend, deformation in title."""
    
    eta_df = eta_df.copy()

    eta_df['CD_type'] = eta_df['CD_type'].astype(str).str.strip()
    eta_df['noise_model'] = eta_df['noise_model'].astype(str).str.strip()

    cd_type = cd_type.strip()
    noise_model = noise_model.strip()

    df = eta_df[
        (eta_df['CD_type'] == cd_type) &
        (eta_df['noise_model'] == noise_model)
    ]

    l_values = sorted(df['l'].unique())
    num_cols = len(corr_type_list)

    # Colors
    cmap = colormaps['Blues_r']
    color_values = np.linspace(0.1, 0.8, len(l_values))
    l_colors = [cmap(val) for val in color_values]

    fig, axes = plt.subplots(
        1, num_cols,
        figsize=(8.5 * num_cols, 4.8),
        sharex=True,
        sharey=True
    )

    if num_cols == 1:
        axes = [axes]

    # Store handles for shared legend
    legend_handles = []
    legend_labels = []

    

    for col_idx, error_type in enumerate(corr_type_list):
        ax = axes[col_idx]

        for l_idx, l in enumerate(l_values):
            mask = (
                (df['l'] == l) &
                (df['error_type'] == error_type)
            )
            df_filtered = df[mask].sort_values(by='eta')
            
            if df_filtered.empty:
                continue

            eta_vals = df_filtered['eta'].to_numpy()
            pth = df_filtered['pth'].to_numpy()
            err = df_filtered['stderr'].to_numpy()

            color = l_colors[l_idx]

            # Plot line
            line, = ax.plot(
                eta_vals,
                pth,
                label=f'l = {l}',
                color=color,
                marker='o'
            )

            # Shaded error
            ax.fill_between(
                eta_vals,
                pth - err,
                pth + err,
                color=color,
                alpha=0.2
            )

            # Only collect legend entries once
            if col_idx == 0:
                legend_handles.append(line)
                legend_labels.append(f'l = {l}')


        parts = error_type.split("_")
        if len(parts) >= 2:
            title = rf"$\mathrm{{{parts[0]}}}_{{{parts[1]}}}$"
        else:
            title = rf"$\mathrm{{{error_type}}}$"

        ax.set_title(title, fontsize=16)
        # ax.set_title(f"{error_type}", fontsize=16)
        ax.set_xlabel("Noise Bias ($\\eta$)", fontsize=12)
        ax.grid(True)

    axes[0].set_ylabel("Threshold $p_{th}$", fontsize=12)

        # Global title
    fig.suptitle(
        f"Threshold vs Bias Pymatching Correlated Decoder (Deformation: {cd_type})",
        fontsize=18,
        y=0.98
    )

    # Shared legend
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=len(l_values),
        fontsize=11,
        frameon=False
    )

    # Manually leave vertical space for suptitle + legend
    fig.subplots_adjust(top=0.78, wspace=0.12)

    plt.show()

def eta_threshold_plot_totalmem_compare_deformations(
    eta_df,
    cd_type_list,
    noise_model,
    error_type="TOTAL_MEM"
):
    """
    Compare TOTAL_MEM threshold vs bias for multiple deformation types.
    One subplot per deformation type, all l values overlaid, shaded error bands,
    with one shared legend.
    """

    eta_df = eta_df.copy()

    eta_df['CD_type'] = eta_df['CD_type'].astype(str).str.strip()
    eta_df['noise_model'] = eta_df['noise_model'].astype(str).str.strip()
    eta_df['error_type'] = eta_df['error_type'].astype(str).str.strip()

    cd_type_list = [cd.strip() for cd in cd_type_list]
    noise_model = noise_model.strip()
    error_type = error_type.strip()

    df = eta_df[
        (eta_df['CD_type'].isin(cd_type_list)) &
        (eta_df['noise_model'] == noise_model) &
        (eta_df['error_type'] == error_type)
    ]

    l_values = sorted(df['l'].unique())
    num_cols = len(cd_type_list)

    # Colors for different l values
    cmap = plt.get_cmap("Paired")

    # Each ℓ gets a pair: (light, dark)
    num_l = len(l_values)

    # Paired has 12 colors → 6 pairs
    if num_l > 6:
        raise ValueError("Paired colormap supports up to 6 ℓ values (12 colors total).")

    l_color_pairs = [
        (cmap(2*i), cmap(2*i + 1))  # (light, dark)
        for i in range(num_l)
]

    fig, axes = plt.subplots(
        1, num_cols,
        figsize=(7 * num_cols, 5),
        sharex=True,
        sharey=True
    )

    if num_cols == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []

    # Pretty subplot titles
    title_map = {
        "SC": "CSS",
        "ZXXZonSqu": "ZXXZ\u2610",
        "Z": "V",
        "X": "H",
    }

    for col_idx, cd_type in enumerate(cd_type_list):
        ax = axes[col_idx]

        df_cd = df[df['CD_type'] == cd_type]

        for l_idx, l in enumerate(l_values):
            df_filtered = df_cd[df_cd['l'] == l].sort_values(by='eta')

            if df_filtered.empty:
                continue

            eta_vals = df_filtered['eta'].to_numpy()
            pth = df_filtered['pth'].to_numpy()
            err = df_filtered['stderr'].to_numpy()

            light_color, dark_color = l_color_pairs[l_idx]

            line, = ax.plot(
                eta_vals,
                pth,
                label=rf"$\ell = {l}$",
                color=dark_color,
                marker='o'
            )

            ax.fill_between(
                eta_vals,
                pth - err,
                pth + err,
                color=dark_color,
                alpha=0.2
            )

            if col_idx == 0:
                legend_handles.append(line)
                legend_labels.append(rf"$\ell = {l}$")

        subplot_title = title_map.get(cd_type, cd_type)
        ax.set_title(subplot_title, fontsize=16)
        ax.set_xlabel("Noise Bias ($\\eta$)", fontsize=12)
        ax.grid(True)

    axes[0].set_ylabel("Threshold $p_{th}$", fontsize=12)

    mem_type = " " if error_type.startswith("TOTAL") else title_map[error_type[0]]
    fig.suptitle(
        f"Threshold vs Bias {mem_type} Memory",
        fontsize=18,
        y=0.98
    )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=len(l_values),
        fontsize=11,
        frameon=False
    )

    fig.subplots_adjust(top=0.78, wspace=0.12)

    plt.show()

def eta_threshold_plot_compare_error_types(
    eta_df,
    cd_type,
    error_type_list,
    noise_model
):
    """
    Compare threshold vs bias across four error types for one deformation type.
    Uses a 2x2 grid of subplots, with all l values overlaid and shaded error bands.
    Uses one shared legend across all subplots.
    """

    eta_df = eta_df.copy()

    if len(error_type_list) != 4:
        raise ValueError("error_type_list must contain exactly 4 error types.")

    # Be robust to either 'cd_type' or 'CD_type'
    if 'CD_type' in eta_df.columns:
        cd_col = 'CD_type'
    elif 'cd_type' in eta_df.columns:
        cd_col = 'cd_type'
    else:
        raise ValueError("DataFrame must contain either 'CD_type' or 'cd_type'.")

    eta_df[cd_col] = eta_df[cd_col].astype(str).str.strip()
    eta_df['noise_model'] = eta_df['noise_model'].astype(str).str.strip()
    eta_df['error_type'] = eta_df['error_type'].astype(str).str.strip()

    cd_type = cd_type.strip()
    noise_model = noise_model.strip()
    error_type_list = [et.strip() for et in error_type_list]

    df = eta_df[
        (eta_df[cd_col] == cd_type) &
        (eta_df['noise_model'] == noise_model) &
        (eta_df['error_type'].isin(error_type_list))
    ]

    l_values = sorted(df['l'].unique())

    # Plasma colormap: one color per ell
    cmap = plt.get_cmap("plasma")
    color_values = np.linspace(0.1, 0.9, max(len(l_values), 1))
    l_colors = [cmap(val) for val in color_values]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 9),
        sharex=True,
        sharey=True
    )
    axes = axes.flatten()

    legend_handles = []
    legend_labels = []

    # Pretty titles
    title_map = {
        "CORR_XZ": r"$\mathrm{CORR}_{XZ}$",
        "CORR_ZX": r"$\mathrm{CORR}_{ZX}$",
        "TOTAL": r"$\mathrm{MWPM}$",
        "TOTAL_MEM": r"$\mathrm{TOTAL}_{MEM}$",
        "X_MEM": r"$X_{MEM}$",
        "Z_MEM": r"$Z_{MEM}$",
        "TOTAL_MEM_CORR": r"$\mathrm{TOTAL}_{MEM,CORR}$",
        "TOTAL_PY_CORR": r"$\mathrm{CORR}_{PY}$",
        "X_MEM_PY": r"$X_{MEM,PY}$",
        "Z_MEM_PY": r"$Z_{MEM,PY}$",
        "TOTAL_MEM_PY": r"$\mathrm{TOTAL}_{MEM,PY}$",
        "SC": "CSS",
        "ZXXZonSqu": "ZXXZ\u2610",
    }

    for idx, error_type in enumerate(error_type_list):
        ax = axes[idx]
        df_err = df[df['error_type'] == error_type]

        for l_idx, l in enumerate(l_values):
            df_filtered = df_err[df_err['l'] == l].sort_values(by='eta')

            if df_filtered.empty:
                continue

            eta_vals = df_filtered['eta'].to_numpy()
            pth = df_filtered['pth'].to_numpy()
            err = df_filtered['stderr'].to_numpy()

            color = l_colors[l_idx]

            line, = ax.plot(
                eta_vals,
                pth,
                label=rf"$\ell = {l}$",
                color=color,
                marker='o'
            )

            ax.fill_between(
                eta_vals,
                pth - err,
                pth + err,
                color=color,
                alpha=0.2
            )

            if idx == 0:
                legend_handles.append(line)
                legend_labels.append(rf"$\ell = {l}$")

        subplot_title = title_map.get(error_type, error_type)
        ax.set_title(subplot_title, fontsize=16)
        ax.grid(True)

        # Panel labels (a), (b), (c), (d)
        panel_labels = ['(a)', '(b)', '(c)', '(d)']
        ax.text(
            0.02, 0.95,
            panel_labels[idx],
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            va='top',
            ha='left'
        )

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # fig.suptitle(
    #     f"Threshold vs Bias ({title_map[cd_type]} Deformation)",
    #     fontsize=18,
    #     y=0.94
    # )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=len(l_values),
        fontsize=11,
        frameon=False
    )

    fig.supxlabel(r"Noise Bias ($\eta$)", fontsize=12, y=0.05)
    fig.supylabel(r"Threshold $p_{th}$", fontsize=12, x=0.05)

    fig.subplots_adjust(top=0.84, wspace=0.2, hspace=0.2)

    plt.show()

def eta_threshold_plot_compare_deformations_and_decoder_2x2(
    eta_df,
    cd_type_list,
    noise_model,
    error_type="TOTAL_MEM_PY",
    suffix_to_remove="_PY"
):
    """
    Compare threshold vs bias in a 2x2 grid:
      rows = deformation type
      cols = decoder type

    Left column: baseline decoder (MWPM)
    Right column: main decoder (correlated / PY)

    All l values are overlaid in each subplot.
    """

    eta_df = eta_df.copy()

    eta_df['CD_type'] = eta_df['CD_type'].astype(str).str.strip()
    eta_df['noise_model'] = eta_df['noise_model'].astype(str).str.strip()
    eta_df['error_type'] = eta_df['error_type'].astype(str).str.strip()

    cd_type_list = [cd.strip() for cd in cd_type_list]
    noise_model = noise_model.strip()
    error_type = error_type.strip()

    if len(cd_type_list) != 2:
        raise ValueError("cd_type_list must contain exactly 2 deformation types for the 2x2 layout.")

    if error_type.endswith(suffix_to_remove):
        baseline_error_type = error_type[:-len(suffix_to_remove)]
    else:
        raise ValueError(f"error_type must end with '{suffix_to_remove}', got {error_type}")

    df = eta_df[
        (eta_df['CD_type'].isin(cd_type_list)) &
        (eta_df['noise_model'] == noise_model) &
        (eta_df['error_type'].isin([error_type, baseline_error_type]))
    ].copy()

    l_values = sorted(df['l'].unique())

    cmap = colormaps["plasma"]
    color_values = np.linspace(0.1, 0.8, max(len(l_values), 1))
    colors = [cmap(val) for val in color_values]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 9),
        sharex=True,
        sharey=True
    )

    legend_handles = []
    legend_labels = []

    title_map = {
        "SC": "CSS",
        "ZXXZonSqu": "ZXXZ\u2610",
        "Z": "V",
        "X": "H",
    }

    decoder_titles = ["MWPM", "Correlated MWPM"]

    for row_idx, cd_type in enumerate(cd_type_list):
        df_cd = df[df['CD_type'] == cd_type]

        for col_idx, decoder_type in enumerate([baseline_error_type, error_type]):
            ax = axes[row_idx, col_idx]

            for l_idx, l in enumerate(l_values):
                color = colors[l_idx]

                df_plot = df_cd[
                    (df_cd['l'] == l) &
                    (df_cd['error_type'] == decoder_type)
                ].sort_values(by='eta')

                if df_plot.empty:
                    continue

                eta_vals = df_plot['eta'].to_numpy()
                pth = df_plot['pth'].to_numpy()
                err = df_plot['stderr'].to_numpy()

                line, = ax.plot(
                    eta_vals,
                    pth,
                    label=rf"$\ell = {l}$",
                    color=color,
                    marker='o',
                    linestyle='-',
                    linewidth=1.8
                )

                ax.fill_between(
                    eta_vals,
                    pth - err,
                    pth + err,
                    color=color,
                    alpha=0.2
                )

                if row_idx == 0 and col_idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(rf"$\ell = {l}$")

            row_title = title_map.get(cd_type, cd_type)
            col_title = decoder_titles[col_idx]
            ax.set_title(f"{row_title}, {col_title}", fontsize=16)

            # Panel labels
            panel_labels = ['(a)', '(b)', '(c)', '(d)']
            panel_idx = row_idx * 2 + col_idx

            ax.text(
                0.02, 0.95,
                panel_labels[panel_idx],
                transform=ax.transAxes,
                fontsize=14,
                fontweight='bold',
                va='top',
                ha='left'
            )
            ax.set_xscale("log")
            ax.grid(True)

            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(formatter)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # mem_type = "" if baseline_error_type.startswith("TOTAL") else title_map.get(baseline_error_type[0], baseline_error_type[0]) + " Memory"

    # fig.suptitle(
    #     f"Threshold vs Bias {mem_type}",
    #     fontsize=18,
    #     y=0.97
    # )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=len(l_values),
        fontsize=11,
        frameon=False
    )

    fig.supxlabel("Noise Bias ($\\eta$)", fontsize=13)
    fig.supylabel("Threshold $p_{th}$", fontsize=13)

    fig.subplots_adjust(top=0.83, wspace=0.10, hspace=0.18)

    plt.show()

# plotter for plasma colormap
def eta_threshold_plot_compare_deformations_and_decoder(
    eta_df,
    cd_type_list,
    noise_model,
    error_type="TOTAL_MEM_PY",
    suffix_to_remove="_PY"
):
    """
    Compare threshold vs bias for multiple deformation types.
    One subplot per deformation type, all l values overlaid.

    For each l:
      - solid line: error_type
      - dashed line: baseline error type with '{suffix_to_remove}' removed

    Example:
      error_type='TOTAL_MEM_CORR'  -> dashed comparison is 'TOTAL_MEM'
    """

    eta_df = eta_df.copy()

    eta_df['CD_type'] = eta_df['CD_type'].astype(str).str.strip()
    eta_df['noise_model'] = eta_df['noise_model'].astype(str).str.strip()
    eta_df['error_type'] = eta_df['error_type'].astype(str).str.strip()

    cd_type_list = [cd.strip() for cd in cd_type_list]
    noise_model = noise_model.strip()
    error_type = error_type.strip()

    if error_type.endswith(suffix_to_remove):
        baseline_error_type = error_type[:-len(suffix_to_remove)]   # removes the suffix
    else:
        raise ValueError(f"error_type must end with '{suffix_to_remove}', got {error_type}")

    df = eta_df[
        (eta_df['CD_type'].isin(cd_type_list)) &
        (eta_df['noise_model'] == noise_model) &
        (eta_df['error_type'].isin([error_type, baseline_error_type]))
    ]

    l_values = sorted(df['l'].unique())
    num_cols = len(cd_type_list)

    cmap = colormaps["plasma"]
    color_values = np.linspace(0.1, 0.8, max(len(l_values), 1))
    colors = [cmap(val) for val in color_values]


    fig, axes = plt.subplots(
        1, num_cols,
        figsize=(7 * num_cols, 5),
        sharex=True,
        sharey=True
    )

    if num_cols == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []

    # Pretty subplot titles
    title_map = {
        "SC": "CSS",
        "ZXXZonSqu": "ZXXZ\u2610",
        "Z": "V",
        "X": "H",
    }

    for col_idx, cd_type in enumerate(cd_type_list):
        ax = axes[col_idx]

        df_cd = df[df['CD_type'] == cd_type]

        for l_idx, l in enumerate(l_values):
            color=colors[l_idx]

            # Correlated decoder is main error type - dashed
            df_main = df_cd[
                (df_cd['l'] == l) &
                (df_cd['error_type'] == error_type)
            ].sort_values(by='eta')

            if not df_main.empty:
                eta_vals = df_main['eta'].to_numpy()
                pth = df_main['pth'].to_numpy()
                err = df_main['stderr'].to_numpy()

                line, = ax.plot(
                    eta_vals,
                    pth,
                    label=rf"$\ell = {l}$",
                    color=color,
                    marker='o',
                    linestyle='--'
                )

                ax.fill_between(
                    eta_vals,
                    pth - err,
                    pth + err,
                    color=color,
                    alpha=0.2
                )

                if col_idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(rf"$\ell = {l}$")

            # MWPM baseline - solid
            df_base = df_cd[
                (df_cd['l'] == l) &
                (df_cd['error_type'] == baseline_error_type)
            ].sort_values(by='eta')

            if not df_base.empty:
                eta_vals_base = df_base['eta'].to_numpy()
                pth_base = df_base['pth'].to_numpy()
                err_base = df_base['stderr'].to_numpy()

                ax.plot(
                    eta_vals_base,
                    pth_base,
                    color=color,
                    marker='o',
                    linestyle='-',
                    linewidth=1.8
                )

                ax.fill_between(
                    eta_vals_base,
                    pth_base - err_base,
                    pth_base + err_base,
                    color=color,
                    alpha=0.10
                )

        subplot_title = title_map.get(cd_type, cd_type)
        ax.set_title(subplot_title, fontsize=16)
        ax.set_xlabel("Noise Bias ($\\eta$)", fontsize=12)
        ax.set_xscale("log")
        ax.grid(True)

    axes[0].set_ylabel("Threshold $p_{th}$", fontsize=12)

    mem_type = "" if baseline_error_type.startswith("TOTAL") else title_map[baseline_error_type[0]] + " Memory"
    fig.suptitle(
        f"Threshold vs Bias {mem_type}",
        fontsize=18,
        y=0.98
    )

    # Shared l legend
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=len(l_values),
        fontsize=11,
        frameon=False
    )
    
    # Style legend for decoder type
    style_handles = [
        Line2D([0], [0], color="black", linestyle="--",  label="Correlated MWPM"),
        Line2D([0], [0], color="black", linestyle="-",  label="MWPM"),
    ]
    fig.legend(
        handles=style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.84),
        ncol=2,
        fontsize=11,
        frameon=False
    )

    fig.subplots_adjust(top=0.74, wspace=0.12)

    plt.show()

def eta_delta_threshold_gap_plot_compare_deformations_and_decoder(
    eta_df,
    cd_type_list,
    noise_model,
    error_type="TOTAL_MEM_PY",
    suffix_to_remove="_PY"
):
    """
    Compare threshold-vs-bias gap between two decoder types for multiple deformations.
    One subplot per deformation type, all l values overlaid.

    For each l, plots:
        delta_pth = pth(error_type) - pth(baseline_error_type)

    where baseline_error_type is formed by removing `suffix_to_remove`.

    Example:
        error_type='TOTAL_MEM_PY' -> baseline='TOTAL_MEM'

    The shaded band uses quadrature-combined stderr:
        sigma_delta = sqrt(stderr_main^2 + stderr_base^2)
    """

    eta_df = eta_df.copy()

    eta_df['CD_type'] = eta_df['CD_type'].astype(str).str.strip()
    eta_df['noise_model'] = eta_df['noise_model'].astype(str).str.strip()
    eta_df['error_type'] = eta_df['error_type'].astype(str).str.strip()

    cd_type_list = [cd.strip() for cd in cd_type_list]
    noise_model = noise_model.strip()
    error_type = error_type.strip()

    if error_type.endswith(suffix_to_remove):
        baseline_error_type = error_type[:-len(suffix_to_remove)] # MWPM with no correlations
    else:
        raise ValueError(
            f"error_type must end with '{suffix_to_remove}', got {error_type}"
        )

    df = eta_df[
        (eta_df['CD_type'].isin(cd_type_list)) &
        (eta_df['noise_model'] == noise_model) &
        (eta_df['error_type'].isin([error_type, baseline_error_type]))
    ].copy()

    l_values = sorted(df['l'].unique())
    num_cols = len(cd_type_list)

    cmap = colormaps["plasma"]
    color_values = np.linspace(0.1, 0.8, max(len(l_values), 1))
    colors = [cmap(val) for val in color_values]

    fig, axes = plt.subplots(
        1, num_cols,
        figsize=(7 * num_cols, 5),
        sharex=True,
        sharey=True
    )

    if num_cols == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []

    title_map = {
        "SC": "CSS",
        "ZXXZonSqu": "ZXXZ\u2610",
        "Z": "V",
        "X": "H",
    }

    for col_idx, cd_type in enumerate(cd_type_list):
        ax = axes[col_idx]
        df_cd = df[df['CD_type'] == cd_type]

        for l_idx, l in enumerate(l_values):
            color = colors[l_idx]

            df_main = df_cd[
                (df_cd['l'] == l) &
                (df_cd['error_type'] == error_type)
            ][['eta', 'pth', 'stderr']].rename(
                columns={'pth': 'pth_main', 'stderr': 'stderr_main'}
            )

            df_base = df_cd[
                (df_cd['l'] == l) &
                (df_cd['error_type'] == baseline_error_type)
            ][['eta', 'pth', 'stderr']].rename(
                columns={'pth': 'pth_base', 'stderr': 'stderr_base'}
            )

            df_gap = pd.merge(df_main, df_base, on='eta', how='inner').sort_values('eta')

            if df_gap.empty:
                continue

            eta_vals = df_gap['eta'].to_numpy()
            delta_pth = ((df_gap['pth_main'] - df_gap['pth_base'])/ df_gap['pth_base']).to_numpy()
            delta_err = np.sqrt(
                df_gap['stderr_main'].to_numpy()**2 +
                df_gap['stderr_base'].to_numpy()**2
            )

            line, = ax.plot(
                eta_vals,
                delta_pth,
                label=rf"$\ell = {l}$",
                color=color,
                marker='o',
                linestyle='-'
            )

            ax.fill_between(
                eta_vals,
                delta_pth - delta_err,
                delta_pth + delta_err,
                color=color,
                alpha=0.2
            )

            if col_idx == 0:
                legend_handles.append(line)
                legend_labels.append(rf"$\ell = {l}$")

        subplot_title = title_map.get(cd_type, cd_type)
        ax.set_title(subplot_title, fontsize=16)
        ax.set_xlabel("Noise Bias ($\\eta$)", fontsize=12)
        ax.set_xscale("log")
        ax.grid(True)

    axes[0].set_ylabel(r"$\frac{\Delta_{\mathrm{corr}}}{p^{\mathrm{MWPM}}_{\mathrm{th}}}$", fontsize=12)

    mem_type = "" if baseline_error_type.startswith("TOTAL") else title_map.get(
        baseline_error_type[0], baseline_error_type[0]
    ) + " Memory"

    fig.suptitle(
        f"Threshold Gap vs Bias {mem_type}",
        fontsize=18,
        y=0.98
    )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=len(l_values),
        fontsize=11,
        frameon=False
    )

    style_handles = [
        Line2D(
            [0], [0],
            color="black",
            linestyle="-",
            label=rf"$\Delta_{{\mathrm{{corr}}}} = p_{{th}}^{{\mathrm{{PY}}}} - p_{{th}}^{{\mathrm{{MWPM}}}}$"
        ),
    ]
    fig.legend(
        handles=style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.84),
        ncol=1,
        fontsize=11,
        frameon=False
    )

    fig.subplots_adjust(top=0.74, wspace=0.12)

    plt.show()

def eta_delta_threshold_gap_grid_compare_deformations_and_decoder(
    eta_df,
    cd_type_list,
    noise_model,
    error_type="TOTAL_MEM_PY",
    suffix_to_remove="_PY"
):
    """
    Make a 2x2 grid:
      columns = deformation types
      top row = Delta_corr
      bottom row = Delta_corr / p_th^MWPM
    """

    eta_df = eta_df.copy()

    eta_df['CD_type'] = eta_df['CD_type'].astype(str).str.strip()
    eta_df['noise_model'] = eta_df['noise_model'].astype(str).str.strip()
    eta_df['error_type'] = eta_df['error_type'].astype(str).str.strip()

    cd_type_list = [cd.strip() for cd in cd_type_list]
    noise_model = noise_model.strip()
    error_type = error_type.strip()

    if len(cd_type_list) != 2:
        raise ValueError("cd_type_list must contain exactly 2 deformation types.")

    if error_type.endswith(suffix_to_remove):
        baseline_error_type = error_type[:-len(suffix_to_remove)]
    else:
        raise ValueError(
            f"error_type must end with '{suffix_to_remove}', got {error_type}"
        )

    df = eta_df[
        (eta_df['CD_type'].isin(cd_type_list)) &
        (eta_df['noise_model'] == noise_model) &
        (eta_df['error_type'].isin([error_type, baseline_error_type]))
    ].copy()

    l_values = sorted(df['l'].unique())

    cmap = colormaps["plasma"]
    color_values = np.linspace(0.1, 0.8, max(len(l_values), 1))
    colors = [cmap(val) for val in color_values]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 8),
        sharex=True,
        sharey="row"
    )

    legend_handles = []
    legend_labels = []

    title_map = {
        "SC": "CSS",
        "ZXXZonSqu": "ZXXZ\u2610",
        "Z": "V",
        "X": "H",
    }

    for col_idx, cd_type in enumerate(cd_type_list):
        df_cd = df[df['CD_type'] == cd_type]

        for l_idx, l in enumerate(l_values):
            color = colors[l_idx]

            df_main = df_cd[
                (df_cd['l'] == l) &
                (df_cd['error_type'] == error_type)
            ][['eta', 'pth', 'stderr']].rename(
                columns={'pth': 'pth_main', 'stderr': 'stderr_main'}
            )

            df_base = df_cd[
                (df_cd['l'] == l) &
                (df_cd['error_type'] == baseline_error_type)
            ][['eta', 'pth', 'stderr']].rename(
                columns={'pth': 'pth_base', 'stderr': 'stderr_base'}
            )

            df_gap = pd.merge(
                df_main,
                df_base,
                on='eta',
                how='inner'
            ).sort_values('eta')

            if df_gap.empty:
                continue

            eta_vals = df_gap['eta'].to_numpy()

            p_py = df_gap['pth_main'].to_numpy()
            p_mwpm = df_gap['pth_base'].to_numpy()

            err_py = df_gap['stderr_main'].to_numpy()
            err_mwpm = df_gap['stderr_base'].to_numpy()

            delta_corr = p_py - p_mwpm
            delta_corr_err = np.sqrt(err_py**2 + err_mwpm**2)

            frac_delta = delta_corr / p_mwpm

            frac_delta_err = np.sqrt(
                (err_py / p_mwpm)**2 +
                ((p_py * err_mwpm) / (p_mwpm**2))**2
            )

            # Top row: absolute delta
            ax_top = axes[0, col_idx]
            line, = ax_top.plot(
                eta_vals,
                delta_corr,
                color=color,
                marker='o',
                linestyle='-',
                label=rf"$\ell = {l}$"
            )
            ax_top.fill_between(
                eta_vals,
                delta_corr - delta_corr_err,
                delta_corr + delta_corr_err,
                color=color,
                alpha=0.2
            )

            # Bottom row: normalized delta
            ax_bottom = axes[1, col_idx]
            ax_bottom.plot(
                eta_vals,
                frac_delta,
                color=color,
                marker='o',
                linestyle='-'
            )
            ax_bottom.fill_between(
                eta_vals,
                frac_delta - frac_delta_err,
                frac_delta + frac_delta_err,
                color=color,
                alpha=0.2
            )

            if col_idx == 0:
                legend_handles.append(line)
                legend_labels.append(rf"$\ell = {l}$")

        axes[0, col_idx].set_title(title_map.get(cd_type, cd_type), fontsize=16)

    # Formatting all axes
    for row_idx in range(2):
        for col_idx in range(2):
            ax = axes[row_idx, col_idx]
            ax.set_xscale("log")
            ax.grid(True)

            # Scientific notation with multiplier shown at top of y-axis
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(formatter)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[0, 0].set_ylabel(r"$\Delta_{\mathrm{CORR}}$", fontsize=12)
    axes[1, 0].set_ylabel(
        r"$\Delta_{\mathrm{CORR}} / p^{\mathrm{MWPM}}_{\mathrm{th}}$",
        fontsize=12
    )

    fig.supxlabel(r"Noise Bias ($\eta$)", fontsize=13)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(l_values),
        fontsize=11,
        frameon=False
    )

    definition_handle = [
        Line2D(
            [0], [0],
            color="black",
            linestyle="-",
            label=(
                r"$\Delta_{\mathrm{CORR}} = "
                r"p_{\mathrm{th}}^{\mathrm{CORR}} - "
                r"p_{\mathrm{th}}^{\mathrm{MWPM}}$"
            )
        )
    ]

    fig.legend(
        handles=definition_handle,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=1,
        fontsize=11,
        frameon=False
    )

    fig.subplots_adjust(
        top=0.84,
        bottom=0.10,
        wspace=0.20,
        hspace=0.20
    )

    plt.show()

def threshold_fit(x, pth, nu, a,b,c):
    p,d = x
    X = (d**(1/nu))*(p-pth)
    return a + b*X + c*X**2

def get_threshold_old(full_df, pth0, p_range, l, eta, error_type, num_shots, CD_type):
    """ returns the threshold and confidence given a df 
        in: df - the dataframe containing all data, filtered for one error_type, l eta, and probability range
        out: p_thr - a float, the probability where intersection of different lattice distances occurred
    """
    print(f"Getting threshold for l = {l}, eta = {eta}, error type = {error_type}, num_shots = {num_shots}, CD = {CD_type}")
    df = full_df[(full_df['p'] < pth0 + p_range) & ( full_df['p'] > pth0 - p_range) & (full_df['l'] == l) & (full_df['eta'] == eta) & (full_df['error_type'] == error_type) & (full_df['num_shots'] == num_shots) & (full_df['CD_type'] == CD_type)]
    # print(df.head)
    # df = full_df
    if df.empty:
        return 0, 0

    # get the p_list and d_list from the dataframe
    p_list = df['p'].to_numpy().flatten()
    d_list = df['d'].to_numpy().flatten()
    error_list = df['num_log_errors'].to_numpy().flatten()

    # run the fitting function
    # popt, pcov = curve_fit(threshold_fit, (p_list, d_list), error_list, p0=[pth0, 0.5, 1, 1, 1])
    popt, pcov = curve_fit(threshold_fit, (p_list, d_list), error_list, p0=[pth0, 0.5, 0,0,0])
    
    # pth = popt[0] # the threshold probability
    # pth_error = np.sqrt(np.trace(pcov))
    # overfitting = np.linalg.cond(pcov)
    # print(f"Overfitting condition number: {overfitting}")
    # print(f"diag of covariance matrix: {np.diag(pcov)}")
    return popt, pcov

def get_threshold(
    full_df,
    pth0,
    p_range,
    l,
    eta,
    error_type,
    CD_type,
    noise_model="circuit_level",
    num_shots=None,
):
    """
    Return the threshold fit parameters and covariance using a temporary
    weighted average over chunked/raw rows.

    Parameters
    ----------
    full_df : pd.DataFrame
        Raw dataframe containing possibly many chunk rows per (d, p).
    pth0 : float
        Initial threshold guess.
    p_range : float
        Fit only points with p in (pth0 - p_range, pth0 + p_range).
    l, eta, error_type, CD_type : filters for the desired dataset.
    num_shots : int or None
        Optional exact-row filter. Usually leave as None for the new workflow.
    noise_model : str or None
        Optional filter if you want to restrict to one noise model.
    """
    print(
        f"Getting threshold for l = {l}, eta = {eta}, error type = {error_type}, "
        f"num_shots = {num_shots}, CD = {CD_type}"
    )

    df = full_df[
        (full_df["p"] < pth0 + p_range) &
        (full_df["p"] > pth0 - p_range) &
        (full_df["l"] == l) &
        (full_df["eta"] == eta) &
        (full_df["error_type"] == error_type) &
        (full_df["CD_type"] == CD_type)
    ].copy()

    if noise_model is not None and "noise_model" in df.columns:
        df = df[df["noise_model"] == noise_model]

    # For the new chunked/raw-row workflow, this should usually be None.
    if num_shots is not None:
        df = df[df["num_shots"] == num_shots]

    if df.empty:
        return 0, 0

    # Weighted combine over repeated rows at the same (d, p)
    df["weighted_errors"] = df["num_log_errors"] * df["num_shots"]

    df_avg = (
        df.groupby(["d", "p"], as_index=False)
          .agg({
              "num_shots": "sum",
              "weighted_errors": "sum"
          })
    )

    df_avg["num_log_errors"] = df_avg["weighted_errors"] / df_avg["num_shots"]
    df_avg = df_avg.drop(columns="weighted_errors")

    if df_avg.empty:
        return 0, 0

    # Need enough points to fit 5 parameters robustly
    if len(df_avg) < 6:
        print("Not enough averaged (d, p) points to fit threshold.")
        return 0, 0

    p_list = df_avg["p"].to_numpy().flatten()
    d_list = df_avg["d"].to_numpy().flatten()
    error_list = df_avg["num_log_errors"].to_numpy().flatten()

    popt, pcov = curve_fit(
        threshold_fit,
        (p_list, d_list),
        error_list,
        p0=[pth0, 0.5, 0, 0, 0],
    )
    # print("popt =", popt)
    # print("pcov =", pcov)


    return popt, pcov

def get_prob_scale(corr_type, eta):
    """ extract the amount to be scaled by given a noise bias and the type of error
    """
    prob_scale = {'X': 0.5/(1+eta), 'Z': (1+2*eta)/(2*(1+eta)), 'CORR_XZ': 1, 'CORR_ZX':1, 'TOTAL':1, 'TOTAL_MEM':1, 'X_MEM':  1, 'Z_MEM': 1, 'TOTAL_MEM_PY':1, 'X_MEM_PY':1, 'Z_MEM_PY':1,'TOTAL_MEM_CORR':1, 'X_MEM_CORR':1, 'Z_MEM_CORR':1} # TOTAL_MEM 4/3 factor of total mem is due to code_cap pauli channel scalling factor in stim, remove this?
    return prob_scale[corr_type]

def get_thresholds_from_data_exactish(p_th_init_dict, p_range, output_file):
    """
    Given a dictionary of thresholds, get the thresholds from the data files and add them to the dictionary
    in: num_shots - the number of shots to sample
        p_th_init_dict - a dictionary of initial guesses for the threshold, only the entries you want to make exactish, with keys (l, eta, corr_type)
    out: threshold_d - the updated dictionary of thresholds
    """
    all_thresholds_df = pd.read_csv('/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/threshold_exactish_per_eta.csv')
    # print("thresholds_df 1", all_thresholds_df)


    for key in p_th_init_dict.keys():
        # print("key", key)
        l, eta, corr_type, CD_type, noise_model = key
        print("l,eta,corr_type, CD_type, noise_model", l,eta, corr_type, CD_type, noise_model)

        df = pd.read_csv(output_file)
        # threshold_d = {}

        p_th_init = p_th_init_dict[key]
        
        pop,pcov = get_threshold(df, p_th_init,p_range, l, eta, corr_type, CD_type)
        if type(pop) == int:
            print(f"Threshold fit failed for l={l}, eta={eta}, corr_type={corr_type}, CD_type={CD_type}, noise_model={noise_model}. Skipping.")
            continue
        print(p_th_init, pop)
        threshold = pop[0]
        std_error = np.sqrt(np.diag(pcov))[0] # should it be np.sqrt(np.trace(pcov)) instead to get the overall error?
        # threshold_d[key] = threshold
        all_thresholds_df = pd.concat([all_thresholds_df,pd.DataFrame({'l':l,'eta':eta, 'error_type':corr_type, 'CD_type':CD_type, 'noise_model':noise_model, 'pth':threshold, 'stderr':std_error}, index=[0])], ignore_index=True)
        # print("thresholds_df 2", all_thresholds_df)

    all_thresholds_df.to_csv('/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/threshold_exactish_per_eta.csv', index=False)


def get_thresholds_from_data_exactish_chat(
    p_th_init_dict,
    p_range,
    output_file,
    threshold_csv='/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/threshold_exactish_per_eta.csv',
    save_every_iteration=False,
):
    """
    Given a dictionary of threshold guesses, fit thresholds from the data file and append them
    to threshold_csv.

    This version is robust to using PY-labeled keys with CORR-labeled data:
        X_MEM_PY     <-> X_MEM_CORR
        Z_MEM_PY     <-> Z_MEM_CORR
        TOTAL_MEM_PY <-> TOTAL_MEM_CORR

    Behavior:
    - If no matching data exists, prints:
        "didn't get data for l={l}, eta={eta}, ..."
    - If matching data exists but threshold fit fails, prints:
        "can't get threshold for l={l}, eta={eta}, ..."
    """

    def candidate_error_types(err_type):
        """
        Return a list of candidate error_type labels to try in the data.
        First entry is the preferred label to fit with.
        """
        mapping = {
            "X_MEM_PY": ["X_MEM_PY", "X_MEM_CORR"],
            "Z_MEM_PY": ["Z_MEM_PY", "Z_MEM_CORR"],
            "TOTAL_MEM_PY": ["TOTAL_MEM_PY", "TOTAL_MEM_CORR"],
            "X_MEM_CORR": ["X_MEM_CORR", "X_MEM_PY"],
            "Z_MEM_CORR": ["Z_MEM_CORR", "Z_MEM_PY"],
            "TOTAL_MEM_CORR": ["TOTAL_MEM_CORR", "TOTAL_MEM_PY"],
        }
        return mapping.get(err_type, [err_type])

    def get_matching_subset(df, l, eta, CD_type, noise_model, err_type_candidates):
        """
        Return the subset of df matching l, eta, CD_type, noise_model, and any candidate error type.
        """
        subset = df[
            (df["l"] == l) &
            (df["eta"] == eta) &
            (df["CD_type"] == CD_type)
        ].copy()

        if "noise_model" in subset.columns and noise_model is not None:
            subset = subset[subset["noise_model"] == noise_model]

        subset = subset[subset["error_type"].isin(err_type_candidates)]
        return subset

    def choose_error_type_present(subset, err_type_candidates):
        """
        Pick the first candidate error type that is actually present in the subset.
        """
        present = set(subset["error_type"].unique())
        for err_type in err_type_candidates:
            if err_type in present:
                return err_type
        return None

    # Read files once, not inside the loop.
    df = pd.read_csv(output_file)
    all_thresholds_df = pd.read_csv(threshold_csv)

    new_rows = []

    for key, p_th_init in p_th_init_dict.items():
        l, eta, corr_type, CD_type, noise_model = key
        print(f"Trying l={l}, eta={eta}, error_type={corr_type}, CD_type={CD_type}, noise_model={noise_model}")

        err_type_candidates = candidate_error_types(corr_type)

        subset = get_matching_subset(
            df=df,
            l=l,
            eta=eta,
            CD_type=CD_type,
            noise_model=noise_model,
            err_type_candidates=err_type_candidates,
        )

        if subset.empty:
            print(
                f"didn't get data for l={l}, eta={eta}, "
                f"error_type={corr_type}, CD_type={CD_type}, noise_model={noise_model}"
            )
            continue

        fit_error_type = choose_error_type_present(subset, err_type_candidates)

        if fit_error_type is None:
            print(
                f"didn't get data for l={l}, eta={eta}, "
                f"error_type={corr_type}, CD_type={CD_type}, noise_model={noise_model}"
            )
            continue

        try:
            pop, pcov = get_threshold(
                df,
                p_th_init,
                p_range,
                l,
                eta,
                fit_error_type,
                CD_type,
            )
        except Exception as e:
            print(
                f"can't get threshold for l={l}, eta={eta}, "
                f"error_type={corr_type}, CD_type={CD_type}, noise_model={noise_model}. "
                f"Error: {e}"
            )
            continue

        if isinstance(pop, int) or pop is None or pcov is None:
            print(
                f"can't get threshold for l={l}, eta={eta}, "
                f"error_type={corr_type}, CD_type={CD_type}, noise_model={noise_model}"
            )
            continue

        try:
            threshold = pop[0]
            std_error = np.sqrt(np.diag(pcov))[0]
        except Exception as e:
            print(
                f"can't get threshold for l={l}, eta={eta}, "
                f"error_type={corr_type}, CD_type={CD_type}, noise_model={noise_model}. "
                f"Bad fit output: {e}"
            )
            continue

        print(f"Initial guess: {p_th_init}, fitted threshold: {threshold}")

        new_rows.append({
            "l": l,
            "eta": eta,
            "error_type": corr_type,   # preserve original dict key label
            "CD_type": CD_type,
            "noise_model": noise_model,
            "pth": threshold,
            "stderr": std_error,
        })

        if save_every_iteration:
            temp_df = pd.concat([all_thresholds_df, pd.DataFrame(new_rows)], ignore_index=True)
            temp_df.to_csv(threshold_csv, index=False)

    if new_rows:
        all_thresholds_df = pd.concat([all_thresholds_df, pd.DataFrame(new_rows)], ignore_index=True)

    all_thresholds_df.to_csv(threshold_csv, index=False)
    return all_thresholds_df

#
# for generating a threshold graph for Z/X too 
#

if __name__ == "__main__":

    p_th_init_dict = {(2,0.5, "CORR_ZX"):0.157, (2,1, "CORR_ZX"):0.149, (2,5, "CORR_ZX"):0.110,
                      (3,0.5, "CORR_ZX"):0.177, (3,1, "CORR_ZX"):0.178, (3,5, "CORR_ZX"):0.155,
                      (4,0.5, "CORR_ZX"):0.146, (4,1, "CORR_ZX"):0.173, (4,5, "CORR_ZX"):0.187,
                      (5,0.5, "CORR_ZX"):0.120, (5,1, "CORR_ZX"):0.148, (5,5, "CORR_ZX"):0.210,
                      (6,0.5, "CORR_ZX"):0.093, (6,1, "CORR_ZX"):0.109, (6,5, "CORR_ZX"):0.235,
                      (2,0.5, "CORR_XZ"):0.160, (2,1, "CORR_XZ"):0.167, (2,5, "CORR_XZ"):0.120,
                      (3,0.5, "CORR_XZ"):0.128, (3,1, "CORR_XZ"):0.165, (3,5, "CORR_XZ"):0.160,
                      (4,0.5, "CORR_XZ"):0.090, (4,1, "CORR_XZ"):0.145, (4,5, "CORR_XZ"):0.190,
                      (5,0.5, "CORR_XZ"):0.075, (5,1, "CORR_XZ"):0.110, (5,5, "CORR_XZ"):0.210,
                      (6,0.5, "CORR_XZ"):0.065, (6,1, "CORR_XZ"):0.090, (6,5, "CORR_XZ"):0.230,
                      (2,0.75,"CORR_XZ"): 0.149, (2,0.75,"CORR_ZX"):0.155, (2,2,"CORR_XZ"): 0.139,
                        (2,2,"CORR_ZX"): 0.122, (2,3,"CORR_XZ"): 0.127, (2,3,"CORR_ZX"): 0.115,
                        (2,4,"CORR_XZ"): 0.121, (2,4,"CORR_ZX"): 0.112, (3,0.75,"CORR_XZ"): 0.149,
                        (3,0.75,"CORR_ZX"): 0.176, (3,2,"CORR_XZ"): 0.177, (3,2,"CORR_ZX"): 0.175,
                        (3,3,"CORR_XZ"): 0.167, (3,3,"CORR_ZX"): 0.165, (3,4,"CORR_XZ"): 0.160,
                        (3,4,"CORR_ZX"): 0.160, (4,0.75,"CORR_XZ"): 0.114, (4,0.75,"CORR_ZX"): 0.159,
                        (4,2,"CORR_XZ"): 0.187, (4,2,"CORR_ZX"): 0.189, (4,3,"CORR_XZ"): 0.196,
                        (4,3,"CORR_ZX"): 0.196, (4,4,"CORR_XZ"): 0.192, (4,4,"CORR_ZX"): 0.192,
                        (5,0.75,"CORR_XZ"): 0.009, (5,0.75,"CORR_ZX"): 0.118, (5,2,"CORR_XZ"): 0.188,
                        (5,2,"CORR_ZX"): 0.189, (5,3,"CORR_XZ"): 0.206,(5,3,"CORR_ZX"): 0.205,
                        (5,4,"CORR_XZ"): 0.209,(5,4,"CORR_ZX"): 0.210,(6,0.75,"CORR_XZ"): 0.07,
                        (6,0.75,"CORR_ZX"): 0.092,(6,2,"CORR_XZ"): 0.185,(6,2,"CORR_ZX"): 0.180,
                        (6,3,"CORR_XZ"): 0.210,(6,3,"CORR_ZX"): 0.212,(6,4,"CORR_XZ"): 0.222,
                        (6,4,"CORR_ZX"): 0.222,
                        (2,1.5,"CORR_XZ"): 0.152, (2,1.5,"CORR_ZX"):0.130, (2,2.5,"CORR_XZ"):0.131, (2,2.5,"CORR_ZX"):0.118,
                        (2,3.5,"CORR_XZ"): 0.123, (2,3.5,"CORR_ZX"): 0.113, (2,4.5,"CORR_XZ"): 0.118, (2,4.5,"CORR_ZX"): 0.111,
                        (2,6,"CORR_XZ"): 0.114, (2,6,"CORR_ZX"): 0.108, (2,7,"CORR_XZ"): 0.112, (2,7,"CORR_ZX"): 0.107,
                        (3,1.5,"CORR_XZ"): 0.175, (3,1.5,"CORR_ZX"): 0.173, (3,2.5,"CORR_XZ"): 0.174, (3,2.5,"CORR_ZX"): 0.170,
                        (3,3.5,"CORR_XZ"): 0.163, (3,3.5,"CORR_ZX"): 0.160, (3,4.5,"CORR_XZ"): 0.159, (3,4.5,"CORR_ZX"): 0.158,
                        (3,6,"CORR_XZ"): 0.154, (3,6,"CORR_ZX"): 0.152, (3,7,"CORR_XZ"): 0.153, (3,7,"CORR_ZX"): 0.151,
                        (4,1.5,"CORR_XZ"): 0.176, (4,1.5,"CORR_ZX"): 0.177, (4,2.5,"CORR_XZ"): 0.193, (4,2.5,"CORR_ZX"): 0.194,
                        (4,3.5,"CORR_XZ"): 0.194, (4,3.5,"CORR_ZX"): 0.194, (4,4.5,"CORR_XZ"): 0.189, (4,4.5,"CORR_ZX"): 0.189,
                        (4,6,"CORR_XZ"): 0.183, (4,6,"CORR_ZX"): 0.184, (4,7,"CORR_XZ"): 0.181, (4,7,"CORR_ZX"): 0.180,
                        (5,1.5,"CORR_XZ"): 0.169, (5,1.5,"CORR_ZX"): 0.163, (5,2.5,"CORR_XZ"): 0.198, (5,2.5,"CORR_ZX"): 0.201,
                        (5,3.5,"CORR_XZ"): 0.209,(5,3.5,"CORR_ZX"): 0.210, (5,4.5,"CORR_XZ"): 0.209,(5,4.5,"CORR_ZX"): 0.209,
                        (5,6,"CORR_XZ"): 0.203, (5,6,"CORR_ZX"): 0.205, (5,7,"CORR_XZ"): 0.200, (5,7,"CORR_ZX"): 0.202,
                        (6,1.5,"CORR_XZ"): 0.135, (6,1.5,"CORR_ZX"): 0.118, (6,2.5,"CORR_XZ"): 0.20, (6,2.5,"CORR_ZX"): 0.202,
                        (6,3.5,"CORR_XZ"): 0.217, (6,3.5,"CORR_ZX"): 0.224, (6,4.5,"CORR_XZ"): 0.224, (6,4.5,"CORR_ZX"): 0.227,
                        (6,6,"CORR_XZ"): 0.224, (6,6,"CORR_ZX"): 0.227, (6,7,"CORR_XZ"): 0.222, (6,7,"CORR_ZX"): 0.225,
                        (2,1.67, "CORR_XZ"): 0.127, (2,1.67, "CORR_ZX"):0.148, (2,3, "CORR_XZ"):0.127, (2,3, "CORR_ZX"):0.116,
                        (2,4.26, "CORR_XZ"):0.121, (2,4.26, "CORR_ZX"):0.113, (2,5.89, "CORR_XZ"):0.116, (2,5.89, "CORR_ZX"):0.109,
                        (3,1.67, "CORR_XZ"): 0.176, (3,1.67, "CORR_ZX"):0.179, (3,3, "CORR_XZ"):0.169, (3,3, "CORR_ZX"):0.164,
                        (3,4.26, "CORR_XZ"):0.160, (3,4.26, "CORR_ZX"):0.158, (3,5.89, "CORR_XZ"):0.156, (3,5.89, "CORR_ZX"):0.152,
                        (4,1.67, "CORR_XZ"): 0.181, (4,1.67, "CORR_ZX"):0.183, (4,3, "CORR_XZ"):0.196, (4,3, "CORR_ZX"):0.197,
                        (4,4.26, "CORR_XZ"):0.192, (4,4.26, "CORR_ZX"):0.192, (4,5.89, "CORR_XZ"):0.185, (4,5.89, "CORR_ZX"):0.185,
                        (5,1.67, "CORR_XZ"): 0.178, (5,1.67, "CORR_ZX"):0.176, (5,3, "CORR_XZ"):0.206, (5,3, "CORR_ZX"):0.208,
                        (5,4.26, "CORR_XZ"):0.211,(5,4.26, "CORR_ZX"):0.212, (5,5.89, "CORR_XZ"):0.205,(5,5.89, "CORR_ZX"):0.207,
                        (6,1.67, "CORR_XZ"): 0.161, (6,1.67, "CORR_ZX"):0.144, (6,3, "CORR_XZ"): 0.212, (6,3, "CORR_ZX"):0.215,
                        (6,4.26, "CORR_XZ"): 0.225, (6,4.26, "CORR_ZX"):0.226, (6,5.89, "CORR_XZ"): 0.227, (6,5.89, "CORR_ZX"):0.229
                        }

    p_th_init_dict_C_CC = {(2, 0.5, "X_MEM", "XZZXonSqu", "code_cap"):0.11, (2, 0.5, "Z_MEM", "XZZXonSqu", "code_cap"):0.11,
                         (2, 5, "X_MEM", "XZZXonSqu", "code_cap"):0.17, (2, 5, "Z_MEM", "XZZXonSqu", "code_cap"):0.16,
                         (2, 10, "X_MEM", "XZZXonSqu", "code_cap"):0.19, (2, 10, "Z_MEM", "XZZXonSqu", "code_cap"):0.19,
                         (2, 50, "X_MEM", "XZZXonSqu", "code_cap"):0.27, (2, 50, "Z_MEM", "XZZXonSqu", "code_cap"):0.23,
                         (2, 100, "X_MEM", "XZZXonSqu", "code_cap"):0.29, (2, 100, "Z_MEM", "XZZXonSqu", "code_cap"):0.24,
                         (2, 500, "X_MEM", "XZZXonSqu", "code_cap"):0.3, (2, 500, "Z_MEM", "XZZXonSqu", "code_cap"):0.22,
                         (2, 1000, "X_MEM", "XZZXonSqu", "code_cap"):0.35, (2, 1000, "Z_MEM", "XZZXonSqu", "code_cap"):0.21,
                         (3, 0.5, "X_MEM", "XZZXonSqu", "code_cap"):0.15, (3, 0.5, "Z_MEM", "XZZXonSqu", "code_cap"):0.082,
                         (3, 5, "X_MEM", "XZZXonSqu", "code_cap"):0.245, (3, 5, "Z_MEM", "XZZXonSqu", "code_cap"):0.105,
                         (3, 10, "X_MEM", "XZZXonSqu", "code_cap"):0.277, (3, 10, "Z_MEM", "XZZXonSqu", "code_cap"):0.115,
                         (3, 50, "X_MEM", "XZZXonSqu", "code_cap"):0.33, (3, 50, "Z_MEM", "XZZXonSqu", "code_cap"):0.1376,
                         (3, 100, "X_MEM", "XZZXonSqu", "code_cap"):0.33, (3, 100, "Z_MEM", "XZZXonSqu", "code_cap"):0.135,
                         (3, 500, "X_MEM", "XZZXonSqu", "code_cap"):0.35, (3, 500, "Z_MEM", "XZZXonSqu", "code_cap"):0.133,
                         (3, 1000, "X_MEM", "XZZXonSqu", "code_cap"):0.35, (3, 1000, "Z_MEM", "XZZXonSqu", "code_cap"):0.135,
                         (4, 0.5, "X_MEM", "XZZXonSqu", "code_cap"):0.186, (4, 0.5, "Z_MEM", "XZZXonSqu", "code_cap"):0.061,
                         (4, 5, "X_MEM", "XZZXonSqu", "code_cap"):0.255, (4, 5, "Z_MEM", "XZZXonSqu", "code_cap"):0.098,
                         (4, 10, "X_MEM", "XZZXonSqu", "code_cap"):0.282, (4, 10, "Z_MEM", "XZZXonSqu", "code_cap"):0.11,
                         (4, 50, "X_MEM", "XZZXonSqu", "code_cap"):0.33, (4, 50, "Z_MEM", "XZZXonSqu", "code_cap"):0.133,
                         (4, 100, "X_MEM", "XZZXonSqu", "code_cap"):0.34, (4, 100, "Z_MEM", "XZZXonSqu", "code_cap"):0.136,
                         (4, 500, "X_MEM", "XZZXonSqu", "code_cap"):0.35, (4, 500, "Z_MEM", "XZZXonSqu", "code_cap"):0.132,
                         (4, 1000, "X_MEM", "XZZXonSqu", "code_cap"):0.36, (4, 1000, "Z_MEM", "XZZXonSqu", "code_cap"):0.135,
                         (5, 0.5, "X_MEM", "XZZXonSqu", "code_cap"):0.21, (5, 0.5, "Z_MEM", "XZZXonSqu", "code_cap"):0.055,
                         (5, 5, "X_MEM", "XZZXonSqu", "code_cap"):0.26, (5, 5, "Z_MEM", "XZZXonSqu", "code_cap"):0.090,
                         (5, 10, "X_MEM", "XZZXonSqu", "code_cap"):0.287, (5, 10, "Z_MEM", "XZZXonSqu", "code_cap"):0.10,
                         (5, 50, "X_MEM", "XZZXonSqu", "code_cap"):0.33, (5, 50, "Z_MEM", "XZZXonSqu", "code_cap"):0.127,
                         (5, 100, "X_MEM", "XZZXonSqu", "code_cap"):0.346, (5, 100, "Z_MEM", "XZZXonSqu", "code_cap"):0.133,
                         (5, 500, "X_MEM", "XZZXonSqu", "code_cap"):0.36, (5, 500, "Z_MEM", "XZZXonSqu", "code_cap"):0.133,
                         (5, 1000, "X_MEM", "XZZXonSqu", "code_cap"):0.36, (5, 1000, "Z_MEM", "XZZXonSqu", "code_cap"):0.132,
                         (6, 0.5, "X_MEM", "XZZXonSqu", "code_cap"):0.23, (6, 0.5, "Z_MEM", "XZZXonSqu", "code_cap"):0.05,
                         (6, 5, "X_MEM", "XZZXonSqu", "code_cap"):0.265, (6, 5, "Z_MEM", "XZZXonSqu", "code_cap"):0.090,
                         (6, 10, "X_MEM", "XZZXonSqu", "code_cap"):0.29, (6, 10, "Z_MEM", "XZZXonSqu", "code_cap"):0.10,
                         (6, 50, "X_MEM", "XZZXonSqu", "code_cap"):0.33, (6, 50, "Z_MEM", "XZZXonSqu", "code_cap"):0.125,
                         (6, 100, "X_MEM", "XZZXonSqu", "code_cap"):0.35, (6, 100, "Z_MEM", "XZZXonSqu", "code_cap"):0.133,
                         (6, 500, "X_MEM", "XZZXonSqu", "code_cap"):0.35, (6, 500, "Z_MEM", "XZZXonSqu", "code_cap"):0.13,
                         (6, 1000, "X_MEM", "XZZXonSqu", "code_cap"):0.36, (6, 1000, "Z_MEM", "XZZXonSqu", "code_cap"):0.126,
                         (2,0.5,"TOTAL_MEM_PY", "XZZXonSqu", "code_cap"):0.00, (2,0.5,"TOTAL_MEM_PY", "ZXXZonSqu", "code_cap"):0.00,
                         (2,0.5,"TOTAL_MEM_PY", "SC", "code_cap"):0.00, (3,0.5,"TOTAL_MEM_PY", "XZZXonSqu", "code_cap"):0.00, 
                         (3,0.5,"TOTAL_MEM_PY", "ZXXZonSqu", "code_cap"):0.00, (3,0.5,"TOTAL_MEM_PY", "SC", "code_cap"):0.00,
                         (4,0.5,"TOTAL_MEM_PY", "XZZXonSqu", "code_cap"):0.00, (4,0.5,"TOTAL_MEM_PY", "ZXXZonSqu", "code_cap"):0.00,
                         (4,0.5,"TOTAL_MEM_PY", "SC", "code_cap"):0.00, (5,0.5,"TOTAL_MEM_PY", "XZZXonSqu", "code_cap"):0.00, 
                         (5,0.5,"TOTAL_MEM_PY", "ZXXZonSqu", "code_cap"):0.00, (5,0.5,"TOTAL_MEM_PY", "SC", "code_cap"):0.00,
                         (6,0.5,"TOTAL_MEM_PY", "XZZXonSqu", "code_cap"):0.00, (6,0.5,"TOTAL_MEM_PY", "ZXXZonSqu", "code_cap"):0.00,
                         (6,0.5,"TOTAL_MEM_PY", "SC", "code_cap"):0.00
                         }
    
    p_th_init_CL = {
                    (2,0.5,"X_MEM", "SC","circuit_level"):0.00721, (2,0.5,"Z_MEM", "SC","circuit_level"):0.00736, (2,0.5,"TOTAL_MEM", "SC","circuit_level"):0.00728,
                    (4,0.5,"X_MEM","SC", "circuit_level"):0.00935, (4,0.5,"Z_MEM", "SC","circuit_level"):0.00466, (4,0.5,"TOTAL_MEM", "SC","circuit_level"):0.00474,
                    (6,0.5,"X_MEM", "SC","circuit_level"):0.01027, (6,0.5,"Z_MEM", "SC","circuit_level"):0.00354, (6,0.5,"TOTAL_MEM", "SC","circuit_level"):0.00348,
                    (2,5,"X_MEM","SC", "circuit_level"):0.00734, (2,5,"Z_MEM", "SC","circuit_level"):0.01051, (2,5,"TOTAL_MEM","SC", "circuit_level"):0.00745,
                    (4,5,"X_MEM", "SC","circuit_level"):0.01057, (4,5,"Z_MEM", "SC","circuit_level"):0.00648, (4,5,"TOTAL_MEM","SC", "circuit_level"):0.00721,
                    (6,5,"X_MEM", "SC","circuit_level"):0.01219, (6,5,"Z_MEM", "SC","circuit_level"):0.00491, (6,5,"TOTAL_MEM", "SC","circuit_level"):0.00500,
                    (2,10,"X_MEM","SC", "circuit_level"):0.00740, (2,10,"Z_MEM","SC", "circuit_level"):0.01116, (2,10,"TOTAL_MEM", "SC","circuit_level"):0.00747,
                    (4,10,"X_MEM", "SC","circuit_level"):0.01101, (4,10,"Z_MEM", "SC","circuit_level"):0.00679, (4,10,"TOTAL_MEM", "SC","circuit_level"):0.00802,
                    (6,10,"X_MEM", "SC","circuit_level"):0.01279, (6,10,"Z_MEM", "SC","circuit_level"):0.00527, (6,10,"TOTAL_MEM", "SC","circuit_level"):0.00550,
                    (2,25,"X_MEM", "SC","circuit_level"):0.00745, (2,25,"Z_MEM","SC", "circuit_level"):0.01196, (2,25,"TOTAL_MEM", "SC","circuit_level"):0.00753,
                    (4,25,"X_MEM", "SC","circuit_level"):0.01129, (4,25,"Z_MEM", "SC","circuit_level"):0.00726, (4,25,"TOTAL_MEM", "SC","circuit_level"):0.00871,
                    (6,25,"X_MEM", "SC","circuit_level"):0.01329, (6,25,"Z_MEM", "SC","circuit_level"):0.00572, (6,25,"TOTAL_MEM","SC", "circuit_level"):0.00593,
                    (2,50,"X_MEM", "SC","circuit_level"):0.00745, (2,50,"Z_MEM", "SC","circuit_level"):0.01230, (2,50,"TOTAL_MEM", "SC","circuit_level"):0.00757,
                    (4,50,"X_MEM", "SC","circuit_level"):0.01150, (4,50,"Z_MEM", "SC","circuit_level"):0.00751, (4,50,"TOTAL_MEM", "SC","circuit_level"):0.00886,
                    (6,50,"X_MEM","SC", "circuit_level"):0.01346, (6,50,"Z_MEM", "SC","circuit_level"):0.00584, (6,50,"TOTAL_MEM", "SC","circuit_level"):0.00603,
                    (2,0.5,"X_MEM", "ZXXZonSqu","circuit_level"):0.00734, (2,0.5,"Z_MEM","ZXXZonSqu", "circuit_level"):0.00726, (2,0.5,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00732,
                    (4,0.5,"X_MEM","ZXXZonSqu", "circuit_level"):0.00941, (4,0.5,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00455, (4,0.5,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00458,
                    (6,0.5,"X_MEM", "ZXXZonSqu","circuit_level"):0.01034, (6,0.5,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00354, (6,0.5,"TOTAL_MEM","ZXXZonSqu", "circuit_level"):0.00352,
                    (2,5,"X_MEM","ZXXZonSqu", "circuit_level"):0.00863, (2,5,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00873, (2,5,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00873,
                    (4,5,"X_MEM","ZXXZonSqu", "circuit_level"):0.01232, (4,5,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00548, (4,5,"TOTAL_MEM","ZXXZonSqu", "circuit_level"):0.00555,
                    (6,5,"X_MEM", "ZXXZonSqu","circuit_level"):0.01367, (6,5,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00436, (6,5,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00441,
                    (2,10,"X_MEM","ZXXZonSqu", "circuit_level"):0.00911, (2,10,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00913, (2,10,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00911,
                    (4,10,"X_MEM","ZXXZonSqu", "circuit_level"):0.01314, (4,10,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00574, (4,10,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00590,
                    (6,10,"X_MEM", "ZXXZonSqu","circuit_level"):0.01473, (6,10,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00460, (6,10,"TOTAL_MEM","ZXXZonSqu", "circuit_level"):0.00455,
                    (2,25,"X_MEM", "ZXXZonSqu","circuit_level"):0.00928, (2,25,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00932, (2,25,"TOTAL_MEM","ZXXZonSqu", "circuit_level"):0.00935,
                    (4,25,"X_MEM", "ZXXZonSqu","circuit_level"):0.01384, (4,25,"Z_MEM","ZXXZonSqu", "circuit_level"):0.00607, (4,25,"TOTAL_MEM","ZXXZonSqu", "circuit_level"):0.00612,
                    (6,25,"X_MEM", "ZXXZonSqu","circuit_level"):0.01570, (6,25,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00481, (6,25,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00477,
                    (2,50,"X_MEM", "ZXXZonSqu","circuit_level"):0.00949, (2,50,"Z_MEM","ZXXZonSqu", "circuit_level"):0.00954, (2,50,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00949,
                    (4,50,"X_MEM","ZXXZonSqu", "circuit_level"):0.01411, (4,50,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00603, (4,50,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00616,
                    (6,50,"X_MEM", "ZXXZonSqu","circuit_level"):0.01610, (6,50,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00468, (6,50,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00493,
                    (3,0.5,"X_MEM","SC", "circuit_level"):0.00835, (3,0.5,"Z_MEM", "SC","circuit_level"):0.00569, (3,0.5,"TOTAL_MEM", "SC","circuit_level"):0.00595,
                    (5,0.5,"X_MEM", "SC","circuit_level"):0.00985, (5,0.5,"Z_MEM", "SC","circuit_level"):0.00411, (5,0.5,"TOTAL_MEM", "SC","circuit_level"):0.00409,
                    (3,5,"X_MEM", "SC","circuit_level"):0.00937, (3,5,"Z_MEM", "SC","circuit_level"):0.00789, (3,5,"TOTAL_MEM","SC", "circuit_level"):0.00873,
                    (5,5,"X_MEM", "SC","circuit_level"):0.01143, (5,5,"Z_MEM", "SC","circuit_level"):0.00565, (5,5,"TOTAL_MEM", "SC","circuit_level"):0.00586,
                    (3,10,"X_MEM", "SC","circuit_level"):0.00964, (3,10,"Z_MEM", "SC","circuit_level"):0.00842, (3,10,"TOTAL_MEM", "SC","circuit_level"):0.00922,
                    (5,10,"X_MEM", "SC","circuit_level"):0.01194, (5,10,"Z_MEM", "SC","circuit_level"):0.00603, (5,10,"TOTAL_MEM", "SC","circuit_level"):0.00645,
                    (3,25,"X_MEM", "SC","circuit_level"):0.00985, (3,25,"Z_MEM", "SC","circuit_level"):0.00899, (3,25,"TOTAL_MEM", "SC","circuit_level"):0.00960,
                    (5,25,"X_MEM", "SC","circuit_level"):0.01249, (5,25,"Z_MEM", "SC","circuit_level"):0.00637, (5,25,"TOTAL_MEM","SC", "circuit_level"):0.00700,
                    (3,50,"X_MEM", "SC","circuit_level"):0.00987, (3,50,"Z_MEM", "SC","circuit_level"):0.00916, (3,50,"TOTAL_MEM", "SC","circuit_level"):0.00970,
                    (5,50,"X_MEM","SC", "circuit_level"):0.01255, (5,50,"Z_MEM", "SC","circuit_level"):0.00654, (5,50,"TOTAL_MEM", "SC","circuit_level"):0.00719,
                    (3,0.5,"X_MEM","ZXXZonSqu", "circuit_level"):0.00852, (3,0.5,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00559, (3,0.5,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00574,
                    (5,0.5,"X_MEM", "ZXXZonSqu","circuit_level"):0.00987, (5,0.5,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00389, (5,0.5,"TOTAL_MEM","ZXXZonSqu", "circuit_level"):0.00394,
                    (3,5,"X_MEM","ZXXZonSqu", "circuit_level"):0.01097, (3,5,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00669, (3,5,"TOTAL_MEM","ZXXZonSqu", "circuit_level"):0.00698,
                    (5,5,"X_MEM", "ZXXZonSqu","circuit_level"):0.01298, (5,5,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00483, (5,5,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00481,
                    (3,10,"X_MEM","ZXXZonSqu", "circuit_level"):0.01158, (3,10,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00688, (3,10,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00734,
                    (5,10,"X_MEM", "ZXXZonSqu","circuit_level"):0.01390, (5,10,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00508, (5,10,"TOTAL_MEM","ZXXZonSqu", "circuit_level"):0.00517,
                    (3,25,"X_MEM", "ZXXZonSqu","circuit_level"):0.01297, (3,25,"Z_MEM","ZXXZonSqu", "circuit_level"):0.00715, (3,25,"TOTAL_MEM","ZXXZonSqu", "circuit_level"):0.00759,
                    (5,25,"X_MEM", "ZXXZonSqu","circuit_level"):0.01466, (5,25,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00525, (5,25,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00538,
                    (3,50,"X_MEM","ZXXZonSqu", "circuit_level"):0.01288, (3,50,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00713, (3,50,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00764,
                    (5,50,"X_MEM", "ZXXZonSqu","circuit_level"):0.01502, (5,50,"Z_MEM", "ZXXZonSqu","circuit_level"):0.00529, (5,50,"TOTAL_MEM", "ZXXZonSqu","circuit_level"):0.00538,
                    #################### SC High Eta #######################
                    # eta = 75
                    (2,75,"X_MEM","SC","circuit_level"):0.00750,
                    (2,75,"Z_MEM","SC","circuit_level"):0.01243,
                    (2,75,"TOTAL_MEM","SC","circuit_level"):0.00752,

                    (3,75,"X_MEM","SC","circuit_level"):0.00998,
                    (3,75,"Z_MEM","SC","circuit_level"):0.00919,
                    (3,75,"TOTAL_MEM","SC","circuit_level"):0.00974,

                    (4,75,"X_MEM","SC","circuit_level"):0.01154,
                    (4,75,"Z_MEM","SC","circuit_level"):0.00759,
                    (4,75,"TOTAL_MEM","SC","circuit_level"):0.00892,

                    (5,75,"X_MEM","SC","circuit_level"):0.01276,
                    (5,75,"Z_MEM","SC","circuit_level"):0.00649,
                    (5,75,"TOTAL_MEM","SC","circuit_level"):0.00725,

                    (6,75,"X_MEM","SC","circuit_level"):0.01373,
                    (6,75,"Z_MEM","SC","circuit_level"):0.00575,
                    (6,75,"TOTAL_MEM","SC","circuit_level"):0.00614,
                    
                    # eta = 100
                    (2,100,"X_MEM","SC","circuit_level"):0.00743,
                    (2,100,"Z_MEM","SC","circuit_level"):0.01241,
                    (2,100,"TOTAL_MEM","SC","circuit_level"):0.00754,

                    (3,100,"X_MEM","SC","circuit_level"):0.00998,
                    (3,100,"Z_MEM","SC","circuit_level"):0.00926,
                    (3,100,"TOTAL_MEM","SC","circuit_level"):0.00974,

                    (4,100,"X_MEM","SC","circuit_level"):0.01157,
                    (4,100,"Z_MEM","SC","circuit_level"):0.00758,
                    (4,100,"TOTAL_MEM","SC","circuit_level"):0.00882,

                    (5,100,"X_MEM","SC","circuit_level"):0.01265,
                    (5,100,"Z_MEM","SC","circuit_level"):0.00649,
                    (5,100,"TOTAL_MEM","SC","circuit_level"):0.00729,

                    (6,100,"X_MEM","SC","circuit_level"):0.01378,
                    (6,100,"Z_MEM","SC","circuit_level"):0.00578,
                    (6,100,"TOTAL_MEM","SC","circuit_level"):0.00632,
                    
                    # eta = 500
                    (2,500,"X_MEM","SC","circuit_level"):0.00749,
                    (2,500,"Z_MEM","SC","circuit_level"):0.01260,
                    (2,500,"TOTAL_MEM","SC","circuit_level"):0.00756,

                    (3,500,"X_MEM","SC","circuit_level"):0.01001,
                    (3,500,"Z_MEM","SC","circuit_level"):0.00936,
                    (3,500,"TOTAL_MEM","SC","circuit_level"):0.00988,

                    (4,500,"X_MEM","SC","circuit_level"):0.01173,
                    (4,500,"Z_MEM","SC","circuit_level"):0.00761,
                    (4,500,"TOTAL_MEM","SC","circuit_level"):0.00916,

                    (5,500,"X_MEM","SC","circuit_level"):0.01282,
                    (5,500,"Z_MEM","SC","circuit_level"):0.00653,
                    (5,500,"TOTAL_MEM","SC","circuit_level"):0.00635,

                    (6,500,"X_MEM","SC","circuit_level"):0.01383,
                    (6,500,"Z_MEM","SC","circuit_level"):0.00584,
                    (6,500,"TOTAL_MEM","SC","circuit_level"):0.00635,

                    # eta = 1000
                    (2,1000,"X_MEM","SC","circuit_level"):0.00750,
                    (2,1000,"Z_MEM","SC","circuit_level"):0.01258,
                    (2,1000,"TOTAL_MEM","SC","circuit_level"):0.00753,

                    (3,1000,"X_MEM","SC","circuit_level"):0.01008,
                    (3,1000,"Z_MEM","SC","circuit_level"):0.00929,
                    (3,1000,"TOTAL_MEM","SC","circuit_level"):0.00986,

                    (4,1000,"X_MEM","SC","circuit_level"):0.01173,
                    (4,1000,"Z_MEM","SC","circuit_level"):0.00761,
                    (4,1000,"TOTAL_MEM","SC","circuit_level"):0.00916,

                    (5,1000,"X_MEM","SC","circuit_level"):0.01300,
                    (5,1000,"Z_MEM","SC","circuit_level"):0.00658,
                    (5,1000,"TOTAL_MEM","SC","circuit_level"):0.00737,

                    (6,1000,"X_MEM","SC","circuit_level"):0.01397,
                    (6,1000,"Z_MEM","SC","circuit_level"):0.00584,
                    (6,1000,"TOTAL_MEM","SC","circuit_level"):0.00622,

                    #################### ZXXZonSqu High Eta #######################
                    # eta = 75
                    (2,75,"X_MEM","ZXXZonSqu","circuit_level"):0.00948,
                    (2,75,"Z_MEM","ZXXZonSqu","circuit_level"):0.00962,
                    (2,75,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00954,

                    (3,75,"X_MEM","ZXXZonSqu","circuit_level"):0.01240,
                    (3,75,"Z_MEM","ZXXZonSqu","circuit_level"):0.00727,
                    (3,75,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00777,

                    (4,75,"X_MEM","ZXXZonSqu","circuit_level"):0.01427,
                    (4,75,"Z_MEM","ZXXZonSqu","circuit_level"):0.00607,
                    (4,75,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00619,

                    (5,75,"X_MEM","ZXXZonSqu","circuit_level"):0.01523,
                    (5,75,"Z_MEM","ZXXZonSqu","circuit_level"):0.00534,
                    (5,75,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00545,

                    (6,75,"X_MEM","ZXXZonSqu","circuit_level"):0.01647,
                    (6,75,"Z_MEM","ZXXZonSqu","circuit_level"):0.00475,
                    (6,75,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00484,
                    
                    # eta = 100
                    (2,100,"X_MEM","ZXXZonSqu","circuit_level"):0.00950,
                    (2,100,"Z_MEM","ZXXZonSqu","circuit_level"):0.00966,
                    (2,100,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00955,

                    (3,100,"X_MEM","ZXXZonSqu","circuit_level"):0.01238,
                    (3,100,"Z_MEM","ZXXZonSqu","circuit_level"):0.00728,
                    (3,100,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00764,

                    (4,100,"X_MEM","ZXXZonSqu","circuit_level"):0.01425,
                    (4,100,"Z_MEM","ZXXZonSqu","circuit_level"):0.00608,
                    (4,100,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00629,

                    (5,100,"X_MEM","ZXXZonSqu","circuit_level"):0.01547,
                    (5,100,"Z_MEM","ZXXZonSqu","circuit_level"):0.00533,
                    (5,100,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00546,

                    (6,100,"X_MEM","ZXXZonSqu","circuit_level"):0.01659,
                    (6,100,"Z_MEM","ZXXZonSqu","circuit_level"):0.00480,
                    (6,100,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00482,
                    
                    # eta = 500
                    (2,500,"X_MEM","ZXXZonSqu","circuit_level"):0.00950,
                    (2,500,"Z_MEM","ZXXZonSqu","circuit_level"):0.00962,
                    (2,500,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00961,

                    (3,500,"X_MEM","ZXXZonSqu","circuit_level"):0.01238,
                    (3,500,"Z_MEM","ZXXZonSqu","circuit_level"):0.00725,
                    (3,500,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00778,

                    (4,500,"X_MEM","ZXXZonSqu","circuit_level"):0.01439,
                    (4,500,"Z_MEM","ZXXZonSqu","circuit_level"):0.00608,
                    (4,500,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00629,

                    (5,500,"X_MEM","ZXXZonSqu","circuit_level"):0.01566,
                    (5,500,"Z_MEM","ZXXZonSqu","circuit_level"):0.00542,
                    (5,500,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00555,

                    (6,500,"X_MEM","ZXXZonSqu","circuit_level"):0.01697,
                    (6,500,"Z_MEM","ZXXZonSqu","circuit_level"):0.00479,
                    (6,500,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00488,

                    # eta = 1000
                    (2,1000,"X_MEM","ZXXZonSqu","circuit_level"):0.00961,
                    (2,1000,"Z_MEM","ZXXZonSqu","circuit_level"):0.00972,
                    (2,1000,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00966,

                    (3,1000,"X_MEM","ZXXZonSqu","circuit_level"):0.01258,
                    (3,1000,"Z_MEM","ZXXZonSqu","circuit_level"):0.00731,
                    (3,1000,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00781,

                    (4,1000,"X_MEM","ZXXZonSqu","circuit_level"):0.01452,
                    (4,1000,"Z_MEM","ZXXZonSqu","circuit_level"):0.00616,
                    (4,1000,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00629,

                    (5,1000,"X_MEM","ZXXZonSqu","circuit_level"):0.01577,
                    (5,1000,"Z_MEM","ZXXZonSqu","circuit_level"):0.00540,
                    (5,1000,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00552,

                    (6,1000,"X_MEM","ZXXZonSqu","circuit_level"):0.01697,
                    (6,1000,"Z_MEM","ZXXZonSqu","circuit_level"):0.00488,
                    (6,1000,"TOTAL_MEM","ZXXZonSqu","circuit_level"):0.00492,

    }
                    
    p_th_init_CL_pycorr = {                # CHAT DATA In the middle

                    # ===================== SC =====================
                    # eta = 0.5
                    (2,0.5,"X_MEM_PY","SC","circuit_level"):0.00852,
                    (2,0.5,"Z_MEM_PY","SC","circuit_level"):0.00871,
                    (2,0.5,"TOTAL_MEM_PY","SC","circuit_level"):0.00859,

                    (3,0.5,"X_MEM_PY","SC","circuit_level"):0.00897,
                    (3,0.5,"Z_MEM_PY","SC","circuit_level"):0.00721,
                    (3,0.5,"TOTAL_MEM_PY","SC","circuit_level"):0.00768,

                    (4,0.5,"X_MEM_PY","SC","circuit_level"):0.00949,
                    (4,0.5,"Z_MEM_PY","SC","circuit_level"):0.00620,
                    (4,0.5,"TOTAL_MEM_PY","SC","circuit_level"):0.00652,

                    (5,0.5,"X_MEM_PY","SC","circuit_level"):0.00996,
                    (5,0.5,"Z_MEM_PY","SC","circuit_level"):0.00546,
                    (5,0.5,"TOTAL_MEM_PY","SC","circuit_level"):0.00565,

                    (6,0.5,"X_MEM_PY","SC","circuit_level"):0.01034,
                    (6,0.5,"Z_MEM_PY","SC","circuit_level"):0.00485,
                    (6,0.5,"TOTAL_MEM_PY","SC","circuit_level"):0.00493,


                    # # eta = 5
                    (2,5,"X_MEM_PY","SC","circuit_level"):0.00801,
                    (2,5,"Z_MEM_PY","SC","circuit_level"):0.01167,
                    (2,5,"TOTAL_MEM_PY","SC","circuit_level"):0.00871,

                    (3,5,"X_MEM_PY","SC","circuit_level"):0.01002,
                    (3,5,"Z_MEM_PY","SC","circuit_level"):0.00962,
                    (3,5,"TOTAL_MEM_PY","SC","circuit_level"):0.00989,

                    (4,5,"X_MEM_PY","SC","circuit_level"):0.01097,
                    (4,5,"Z_MEM_PY","SC","circuit_level"):0.00814,
                    (4,5,"TOTAL_MEM_PY","SC","circuit_level"):0.00916,

                    (5,5,"X_MEM_PY","SC","circuit_level"):0.01109,
                    (5,5,"Z_MEM_PY","SC","circuit_level"):0.00719,
                    (5,5,"TOTAL_MEM_PY","SC","circuit_level"):0.00768,

                    (6,5,"X_MEM_PY","SC","circuit_level"):0.01226,
                    (6,5,"Z_MEM_PY","SC","circuit_level"):0.00624,
                    (6,5,"TOTAL_MEM_PY","SC","circuit_level"):0.00606,


                    # eta = 10
                    (2,10,"X_MEM_PY","SC","circuit_level"):0.00863,
                    (2,10,"Z_MEM_PY","SC","circuit_level"):0.01241,
                    (2,10,"TOTAL_MEM_PY","SC","circuit_level"):0.00871,

                    (3,10,"X_MEM_PY","SC","circuit_level"):0.01029,
                    (3,10,"Z_MEM_PY","SC","circuit_level"):0.01008,
                    (3,10,"TOTAL_MEM_PY","SC","circuit_level"):0.01029,

                    (4,10,"X_MEM_PY","SC","circuit_level"):0.01137,
                    (4,10,"Z_MEM_PY","SC","circuit_level"):0.00861,
                    (4,10,"TOTAL_MEM_PY","SC","circuit_level"):0.00977,

                    (5,10,"X_MEM_PY","SC","circuit_level"):0.01215,
                    (5,10,"Z_MEM_PY","SC","circuit_level"):0.00789,
                    (5,10,"TOTAL_MEM_PY","SC","circuit_level"):0.00852,

                    (6,10,"X_MEM_PY","SC","circuit_level"):0.01285,
                    (6,10,"Z_MEM_PY","SC","circuit_level"):0.00669,
                    (6,10,"TOTAL_MEM_PY","SC","circuit_level"):0.00732,

                    # eta = 25
                    (2,25,"X_MEM_PY","SC","circuit_level"):0.00871,
                    (2,25,"Z_MEM_PY","SC","circuit_level"):0.01306,
                    (2,25,"TOTAL_MEM_PY","SC","circuit_level"):0.00871,

                    (3,25,"X_MEM_PY","SC","circuit_level"):0.01053,
                    (3,25,"Z_MEM_PY","SC","circuit_level"):0.01076,
                    (3,25,"TOTAL_MEM_PY","SC","circuit_level"):0.01063,

                    (4,25,"X_MEM_PY","SC","circuit_level"):0.01169,
                    (4,25,"Z_MEM_PY","SC","circuit_level"):0.00922,
                    (4,25,"TOTAL_MEM_PY","SC","circuit_level"):0.01044,

                    (5,25,"X_MEM_PY","SC","circuit_level"):0.01257,
                    (5,25,"Z_MEM_PY","SC","circuit_level"):0.00795,
                    (5,25,"TOTAL_MEM_PY","SC","circuit_level"):0.00909,

                    (6,25,"X_MEM_PY","SC","circuit_level"):0.01340,
                    (6,25,"Z_MEM_PY","SC","circuit_level"):0.00726,
                    (6,25,"TOTAL_MEM_PY","SC","circuit_level"):0.00776,

                    # eta = 50
                    (2,50,"X_MEM_PY","SC","circuit_level"):0.00867,
                    (2,50,"Z_MEM_PY","SC","circuit_level"):0.01340,
                    (2,50,"TOTAL_MEM_PY","SC","circuit_level"):0.00865,

                    (3,50,"X_MEM_PY","SC","circuit_level"):0.01061,
                    (3,50,"Z_MEM_PY","SC","circuit_level"):0.01095,
                    (3,50,"TOTAL_MEM_PY","SC","circuit_level"):0.01067,

                    (4,50,"X_MEM_PY","SC","circuit_level"):0.01186,
                    (4,50,"Z_MEM_PY","SC","circuit_level"):0.00939,
                    (4,50,"TOTAL_MEM_PY","SC","circuit_level"):0.01061,

                    (5,50,"X_MEM_PY","SC","circuit_level"):0.01281,
                    (5,50,"Z_MEM_PY","SC","circuit_level"):0.00814,
                    (5,50,"TOTAL_MEM_PY","SC","circuit_level"):0.00935,

                    (6,50,"X_MEM_PY","SC","circuit_level"):0.01363,
                    (6,50,"Z_MEM_PY","SC","circuit_level"):0.00740,
                    (6,50,"TOTAL_MEM_PY","SC","circuit_level"):0.00802,

                    # eta = 75
                    (2,75,"X_MEM_PY","SC","circuit_level"):0.00865,
                    (2,75,"Z_MEM_PY","SC","circuit_level"):0.01352,
                    (2,75,"TOTAL_MEM_PY","SC","circuit_level"):0.00872,

                    (3,75,"X_MEM_PY","SC","circuit_level"):0.01063,
                    (3,75,"Z_MEM_PY","SC","circuit_level"):0.01113,
                    (3,75,"TOTAL_MEM_PY","SC","circuit_level"):0.01073,

                    (4,75,"X_MEM_PY","SC","circuit_level"):0.01194,
                    (4,75,"Z_MEM_PY","SC","circuit_level"):0.00947,
                    (4,75,"TOTAL_MEM_PY","SC","circuit_level"):0.01073,

                    (5,75,"X_MEM_PY","SC","circuit_level"):0.01287,
                    (5,75,"Z_MEM_PY","SC","circuit_level"):0.00828,
                    (5,75,"TOTAL_MEM_PY","SC","circuit_level"):0.00952,

                    (6,75,"X_MEM_PY","SC","circuit_level"):0.01385,
                    (6,75,"Z_MEM_PY","SC","circuit_level"):0.00723,
                    (6,75,"TOTAL_MEM_PY","SC","circuit_level"):0.00803,

                    # eta = 100
                    (2,100,"X_MEM_PY","SC","circuit_level"):0.00872,
                    (2,100,"Z_MEM_PY","SC","circuit_level"):0.01356,
                    (2,100,"TOTAL_MEM_PY","SC","circuit_level"):0.00871,

                    (3,100,"X_MEM_PY","SC","circuit_level"):0.01065,
                    (3,100,"Z_MEM_PY","SC","circuit_level"):0.01111,
                    (3,100,"TOTAL_MEM_PY","SC","circuit_level"):0.01074,

                    (4,100,"X_MEM_PY","SC","circuit_level"):0.01191,
                    (4,100,"Z_MEM_PY","SC","circuit_level"):0.00952,
                    (4,100,"TOTAL_MEM_PY","SC","circuit_level"):0.01082,

                    (5,100,"X_MEM_PY","SC","circuit_level"):0.01289,
                    (5,100,"Z_MEM_PY","SC","circuit_level"):0.00829,
                    (5,100,"TOTAL_MEM_PY","SC","circuit_level"):0.00945,

                    (6,100,"X_MEM_PY","SC","circuit_level"):0.01371,
                    (6,100,"Z_MEM_PY","SC","circuit_level"):0.00735,
                    (6,100,"TOTAL_MEM_PY","SC","circuit_level"):0.00789,

                    # eta = 500
                    (2,500,"X_MEM_PY","SC","circuit_level"):0.00866,
                    (2,500,"Z_MEM_PY","SC","circuit_level"):0.01371,
                    (2,500,"TOTAL_MEM_PY","SC","circuit_level"):0.00877,

                    (3,500,"X_MEM_PY","SC","circuit_level"):0.01073,
                    (3,500,"Z_MEM_PY","SC","circuit_level"):0.01127,
                    (3,500,"TOTAL_MEM_PY","SC","circuit_level"):0.01078,

                    (4,500,"X_MEM_PY","SC","circuit_level"):0.01202,
                    (4,500,"Z_MEM_PY","SC","circuit_level"):0.00957,
                    (4,500,"TOTAL_MEM_PY","SC","circuit_level"):0.01082,

                    (5,500,"X_MEM_PY","SC","circuit_level"):0.01305,
                    (5,500,"Z_MEM_PY","SC","circuit_level"):0.00831,
                    (5,500,"TOTAL_MEM_PY","SC","circuit_level"):0.00911,

                    (6,500,"X_MEM_PY","SC","circuit_level"):0.01392,
                    (6,500,"Z_MEM_PY","SC","circuit_level"):0.00737,
                    (6,500,"TOTAL_MEM_PY","SC","circuit_level"):0.00783,


                    # eta = 1000
                    (2,1000,"X_MEM_PY","SC","circuit_level"):0.00872,
                    (2,1000,"Z_MEM_PY","SC","circuit_level"):0.01368,
                    (2,1000,"TOTAL_MEM_PY","SC","circuit_level"):0.00874,

                    (3,1000,"X_MEM_PY","SC","circuit_level"):0.01067,
                    (3,1000,"Z_MEM_PY","SC","circuit_level"):0.01131,
                    (3,1000,"TOTAL_MEM_PY","SC","circuit_level"):0.01084,

                    (4,1000,"X_MEM_PY","SC","circuit_level"):0.01206,
                    (4,1000,"Z_MEM_PY","SC","circuit_level"):0.00957,
                    (4,1000,"TOTAL_MEM_PY","SC","circuit_level"):0.01091,

                    (5,1000,"X_MEM_PY","SC","circuit_level"):0.01303,
                    (5,1000,"Z_MEM_PY","SC","circuit_level"):0.00844,
                    (5,1000,"TOTAL_MEM_PY","SC","circuit_level"):0.00967,

                    (6,1000,"X_MEM_PY","SC","circuit_level"):0.01412,
                    (6,1000,"Z_MEM_PY","SC","circuit_level"):0.00737,
                    (6,1000,"TOTAL_MEM_PY","SC","circuit_level"):0.00798,

                    # ===================== ZXXZonSqu =====================
                    # eta = 0.5
                    (2,0.5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.00850,
                    (2,0.5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00852,
                    (2,0.5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00959,

                    (3,0.5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.00910,
                    (3,0.5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00721,
                    (3,0.5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00751,

                    (4,0.5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.00948,
                    (4,0.5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00610,
                    (4,0.5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00635,

                    (5,0.5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01015,
                    (5,0.5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00527,
                    (5,0.5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00540,

                    (6,0.5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01048,
                    (6,0.5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00477,
                    (6,0.5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00496,


                    # eta = 5
                    (2,5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.00998,
                    (2,5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.0100,
                    (2,5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00974,

                    (3,5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01146,
                    (3,5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00823,
                    (3,5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00871,

                    (4,5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01245,
                    (4,5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00678,
                    (4,5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00728,

                    (5,5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01323,
                    (5,5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00614,
                    (5,5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00631,

                    (6,5,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01378,
                    (6,5,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00557,
                    (6,5,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00569,

                    # eta = 10 

                    (2,10,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01038,
                    (2,10,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.01038,
                    (2,10,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.01052,

                    (3,10,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01205,
                    (3,10,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00850,
                    (3,10,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00922,

                    (4,10,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01325,
                    (4,10,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00740,
                    (4,10,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00766,

                    (5,10,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01401,
                    (5,10,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00641,
                    (5,10,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00658,

                    (6,10,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01479,
                    (6,10,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00570,
                    (6,10,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00586,

                    # eta = 25
                    (2,25,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01076,
                    (2,25,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.01008,
                    (2,25,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.01070,

                    (3,25,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01251,
                    (3,25,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00882,
                    (3,25,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00945,

                    (4,25,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01397,
                    (4,25,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00751,
                    (4,25,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00772,

                    (5,25,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01487,
                    (5,25,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00677,
                    (5,25,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00688,

                    (6,25,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01574,
                    (6,25,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00603,
                    (6,25,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00620,

                    # eta = 50
                    (2,50,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01082,
                    (2,50,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.01087,
                    (2,50,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.01089,

                    (3,50,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01285,
                    (3,50,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00897,
                    (3,50,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00960,

                    (4,50,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01420,
                    (4,50,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00772,
                    (4,50,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00802,

                    (5,50,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01536,
                    (5,50,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00683,
                    (5,50,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00704,

                    (6,50,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01629,
                    (6,50,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00618,
                    (6,50,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00641, 


                    # eta 75
                    (2,75,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01084,
                    (2,75,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.01086,
                    (2,75,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.01087,

                    (3,75,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01291,
                    (3,75,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00893,
                    (3,75,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00966,

                    (4,75,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01440,
                    (4,75,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00770,
                    (4,75,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00810,

                    (5,75,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01541,
                    (5,75,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00674,
                    (5,75,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00695,

                    (6,75,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01650,
                    (6,75,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00605,
                    (6,75,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00615,

                    # eta 100
                    (2,100,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01088,
                    (2,100,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.01087,
                    (2,100,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.01086,

                    (3,100,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01291,
                    (3,100,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00903,
                    (3,100,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00950,

                    (4,100,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01442,
                    (4,100,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00766,
                    (4,100,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00796,

                    (5,100,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01552,
                    (5,100,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00675,
                    (5,100,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00702,

                    (6,100,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01656,
                    (6,100,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00604,
                    (6,100,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00619,


                    # eta 500
                    (2,500,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01097,
                    (2,500,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.01099,
                    (2,500,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.01099,

                    (3,500,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01301,
                    (3,500,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00905,
                    (3,500,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00961,

                    (4,500,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01465,
                    (4,500,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00770,
                    (4,500,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00809,

                    (5,500,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01555,
                    (5,500,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00679,
                    (5,500,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00698,

                    (6,500,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01671,
                    (6,500,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00607,
                    (6,500,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00619,

                    # eta 1000
                    (2,1000,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01097,
                    (2,1000,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.01103,
                    (2,1000,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.01101,

                    (3,1000,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01319,
                    (3,1000,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00907,
                    (3,1000,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00969,

                    (4,1000,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01465,
                    (4,1000,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00777,
                    (4,1000,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00810,

                    (5,1000,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01580,
                    (5,1000,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00677,
                    (5,1000,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00702,

                    (6,1000,"X_MEM_PY","ZXXZonSqu","circuit_level"):0.01697,
                    (6,1000,"Z_MEM_PY","ZXXZonSqu","circuit_level"):0.00618,
                    (6,1000,"TOTAL_MEM_PY","ZXXZonSqu","circuit_level"):0.00629,
    }
                    # (2,0.5,"X_MEM_PY", "SC","circuit_level"):0.00852, (2,0.5,"Z_MEM_PY", "SC","circuit_level"):0.00869, (2,0.5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00859,
                    # (3,0.5,"X_MEM_PY","SC", "circuit_level"):0.00897, (3,0.5,"Z_MEM_PY", "SC","circuit_level"):0.00721, (3,0.5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00768,
                    # (4,0.5,"X_MEM_PY", "SC","circuit_level"):0.00949, (4,0.5,"Z_MEM_PY", "SC","circuit_level"):0.00620, (4,0.5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00652,
                    # (5,0.5,"X_MEM_PY", "SC","circuit_level"):0.00996, (5,0.5,"Z_MEM_PY", "SC","circuit_level"):0.00546, (5,0.5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00,
                    # (6,0.5,"X_MEM_PY","SC", "circuit_level"):0.00935, (6,0.5,"Z_MEM_PY", "SC","circuit_level"):0.00466, (6,0.5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00474,
                    # (2,5,"X_MEM_PY", "SC","circuit_level"):0.00721, (2,5,"Z_MEM_PY", "SC","circuit_level"):0.00736, (2,5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00728,
                    # (3,5,"X_MEM_PY","SC", "circuit_level"):0.00935, (3,5,"Z_MEM_PY", "SC","circuit_level"):0.00466, (3,5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00474,
                    # (4,5,"X_MEM_PY", "SC","circuit_level"):0.01027, (4,5,"Z_MEM_PY", "SC","circuit_level"):0.00354, (4,5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00348,
                    # (5,5,"X_MEM_PY", "SC","circuit_level"):0.00721, (5,5,"Z_MEM_PY", "SC","circuit_level"):0.00736, (5,5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00728,
                    # (6,5,"X_MEM_PY","SC", "circuit_level"):0.00935, (6,5,"Z_MEM_PY", "SC","circuit_level"):0.00466, (6,5,"TOTAL_MEM_PY", "SC","circuit_level"):0.00474,
                    # (2,10,"X_MEM_PY", "SC","circuit_level"):0.00721, (2,10,"Z_MEM_PY", "SC","circuit_level"):0.00736, (2,10,"TOTAL_MEM_PY", "SC","circuit_level"):0.00728,
                    # (3,10,"X_MEM_PY","SC", "circuit_level"):0.00935, (3,10,"Z_MEM_PY", "SC","circuit_level"):0.00466, (3,10,"TOTAL_MEM_PY", "SC","circuit_level"):0.00474,
                    # (4,10,"X_MEM_PY", "SC","circuit_level"):0.01027, (4,10,"Z_MEM_PY", "SC","circuit_level"):0.00354, (4,10,"TOTAL_MEM_PY", "SC","circuit_level"):0.00348,
                    # (5,10,"X_MEM_PY", "SC","circuit_level"):0.00721, (5,10,"Z_MEM_PY", "SC","circuit_level"):0.00736, (5,10,"TOTAL_MEM_PY", "SC","circuit_level"):0.00728,
                    # (6,10,"X_MEM_PY","SC", "circuit_level"):0.00935, (6,10,"Z_MEM_PY", "SC","circuit_level"):0.00466, (6,10,"TOTAL_MEM_PY", "SC","circuit_level"):0.00474,
                    # (2,25,"X_MEM_PY", "SC","circuit_level"):0.00721, (2,25,"Z_MEM_PY", "SC","circuit_level"):0.00736, (2,25,"TOTAL_MEM_PY", "SC","circuit_level"):0.00728,
                    # (3,25,"X_MEM_PY","SC", "circuit_level"):0.00935, (3,25,"Z_MEM_PY", "SC","circuit_level"):0.00466, (3,25,"TOTAL_MEM_PY", "SC","circuit_level"):0.00474,
                    # (4,25,"X_MEM_PY", "SC","circuit_level"):0.01027, (4,25,"Z_MEM_PY", "SC","circuit_level"):0.00354, (4,25,"TOTAL_MEM_PY", "SC","circuit_level"):0.00348,
                    # (5,25,"X_MEM_PY", "SC","circuit_level"):0.00721, (5,25,"Z_MEM_PY", "SC","circuit_level"):0.00736, (5,25,"TOTAL_MEM_PY", "SC","circuit_level"):0.00728,
                    # (6,25,"X_MEM_PY","SC", "circuit_level"):0.00935, (6,25,"Z_MEM_PY", "SC","circuit_level"):0.00466, (6,25,"TOTAL_MEM_PY", "SC","circuit_level"):0.00474,
                    # (2,50,"X_MEM_PY", "SC","circuit_level"):0.00721, (2,50,"Z_MEM_PY", "SC","circuit_level"):0.00736, (2,50,"TOTAL_MEM_PY", "SC","circuit_level"):0.00728,
                    # (3,50,"X_MEM_PY","SC", "circuit_level"):0.00935, (3,50,"Z_MEM_PY", "SC","circuit_level"):0.00466, (3,50,"TOTAL_MEM_PY", "SC","circuit_level"):0.00474,
                    # (4,50,"X_MEM_PY", "SC","circuit_level"):0.01027, (4,50,"Z_MEM_PY", "SC","circuit_level"):0.00354, (4,50,"TOTAL_MEM_PY", "SC","circuit_level"):0.00348,
                    # (5,50,"X_MEM_PY", "SC","circuit_level"):0.00721, (5,50,"Z_MEM_PY", "SC","circuit_level"):0.00736, (5,50,"TOTAL_MEM_PY", "SC","circuit_level"):0.00728,
                    # (6,50,"X_MEM_PY","SC", "circuit_level"):0.00935, (6,50,"Z_MEM_PY", "SC","circuit_level"):0.00466, (6,50,"TOTAL_MEM_PY", "SC","circuit_level"):0.00474,
                    # (2,0.5,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (2,0.5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (2,0.5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (3,0.5,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (3,0.5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (3,0.5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474,
                    # (4,0.5,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.01027, (4,0.5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00354, (4,0.5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00348,
                    # (5,0.5,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (5,0.5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (5,0.5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (6,0.5,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (6,0.5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (6,0.5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474,
                    # (2,5,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (2,5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (2,5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (3,5,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (3,5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (3,5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474,
                    # (4,5,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.01027, (4,5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00354, (4,5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00348,
                    # (5,5,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (5,5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (5,5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (6,5,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (6,5,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (6,5,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474,
                    # (2,10,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (2,10,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (2,10,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (3,10,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (3,10,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (3,10,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474,
                    # (4,10,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.01027, (4,10,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00354, (4,10,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00348,
                    # (5,10,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (5,10,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (5,10,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (6,10,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (6,10,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (6,10,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474,
                    # (2,25,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (2,25,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (2,25,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (3,25,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (3,25,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (3,25,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474,
                    # (4,25,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.01027, (4,25,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00354, (4,25,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00348,
                    # (5,25,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (5,25,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (5,25,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (6,25,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (6,25,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (6,25,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474,
                    # (2,50,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (2,50,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (2,50,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (3,50,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (3,50,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (3,50,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474,
                    # (4,50,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.01027, (4,50,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00354, (4,50,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00348,
                    # (5,50,"X_MEM_PY", "ZXXZonSqu","circuit_level"):0.00721, (5,50,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00736, (5,50,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00728,
                    # (6,50,"X_MEM_PY","ZXXZonSqu", "circuit_level"):0.00935, (6,50,"Z_MEM_PY", "ZXXZonSqu","circuit_level"):0.00466, (6,50,"TOTAL_MEM_PY", "ZXXZonSqu","circuit_level"):0.00474}

    #### parameters


    
        

    # simulation

    # if getting threshold specific data
    # p_th_init = p_th_init_dict[(l,eta,corr_type)]
    # p_th_init = 0.158
    # p_list = np.linspace(p_th_init-0.03, p_th_init + 0.03, 40)

    # otherwise p_list is range of probabilities
    # p_list = np.logspace(-2.5, -1.5, 40)
    # p_list = None

    l_list = [2,4,6] # elongation params, do 3 and 5 in another batch
    d_list = [11,13,15,17,19] # code distances
    eta_list = [100,1000] # noise bias
    cd_list = ["SC", "ZXXZonSqu"] # clifford deformation types
    corr_type = "TOTAL_MEM" # which type of correlation to use, depending on the type of decoder. Choose from ['CORR_XZ', 'CORR_ZX', 'TOTAL', 'TOTAL_MEM', 'TOTAL_PY_CORR', 'TOTAL_MEM_CORR']
    error_type = "TOTAL_MEM" # which type of error to plot
    # num_shots = 66666
    corr_list = ['CORR_XZ', 'CORR_ZX']
    corr_type_list = ['X_MEM', 'Z_MEM', 'TOTAL_MEM']  
    noise_model = "circuit_level"
    # py_corr = False # whether to use pymatching correlated decoder for circuit data
    py_corr_list = [True, False] # whether to use pymatching correlated decoder for circuit data, do both in separate batches
    circuit_data = True # whether circuit level or code cap data is desired
    corr_decoding = False # whether to get data for correlated decoding (corrxz or corrzx), or circuit level (X/Z mem or X/Z mem py)
    total_num_shots = 10**6
    chunk_size=10**3
    n_p = 20
    p_range=0.00125
    p_list = np.logspace(-2.5,-1.5,n_p)

    if circuit_data:
        folder_path = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/circuit_data/'
        if noise_model == "circuit_level":
            # output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/circuit_data_clean.csv'
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/circuit_biased_data.csv'
        elif noise_model == "code_cap":
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/code_cap_circuit_data.csv'
        elif noise_model == "phenom":
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/phenom_data.csv'
    #     elif corr_type == "CORR_XZ":
    #         output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/xz_circuit_data.csv'
    else:
        folder_path = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/corr_err_data/'
        if corr_type == "CORR_ZX":
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/zx_corr_err_data.csv'
        elif corr_type == "CORR_XZ":
            output_file = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/xz_corr_err_data.csv'

    # num_points = len(l_list) * len(eta_list) * len(cd_list) * len(d_list) * n_p
    # repeats_per_submission = 1000 // num_points
    # shots_added_per_submission = repeats_per_submission * shots_per_task
    
    # run this to get data from the dcc
    # for py_corr in py_corr_list:
    #     if py_corr == True:
    #         p_init_d_temp = p_th_init_CL_pycorr
    #     else:
    #         p_init_d_temp = p_th_init_CL
    #     get_data_DCC_chat(circuit_data=circuit_data,
    #                     corr_decoding=corr_decoding,
    #                     noise_model=noise_model,
    #                     d_list=d_list,
    #                     l_list=l_list,
    #                     eta_list=eta_list,
    #                     cd_list=cd_list,
    #                     corr_list=corr_list,
    #                     total_num_shots=total_num_shots,
    #                     p_list=p_list,
    #                     p_th_init_d=None,
    #                     pymatch_corr=py_corr,
    #                     fully_biased=True,
    #                     n_p = n_p,
    #                     p_range=p_range,
    #                     chunk_size=chunk_size,
    #                     resume=True,
    #                     shots_per_task=None,
    #                     )
    # get_data_DCC(circuit_data, corr_decoding, noise_model, d_list, l_list, eta_list, cd_list, corr_list, total_num_shots, p_list=None, p_th_init_d=p_th_init_CL_pycorr, pymatch_corr=py_corr)

    # run this once you have data and want to combo it to one csv
    # append_task_csvs_into_master(master_file=output_file)

    # concat_csv(folder_path, circuit_data)


    # plot the threshold results

    # eta - 0.5, 5, 10, 25, 50, retake for lower ranges at lower eta 
    # l - 2,3,4,5,6
    # no corr
    # num_shots - 30303, total - 1e6
    # d - 11-19 odd 
    # CD_type - SC, ZXXZonSqu



    # params to plot
    # eta = 0.5
    # l = 3

    # curr_num_shots = chunk_size # the file has 20408 for the 3,5 and 30303 for the 2,4,6 and 52631 for pycorr
    # noise_model = "circuit_level"
    # CD_type = "SC"
    # py_corr = False # whether to use pymatching correlated decoder for circuit data
    # corr_decoding = False # whether to get data for correlated decoding using my decoder
    # error_type = "TOTAL_MEM" # which type of error to plot, choose from ['X_MEM', 'Z_MEM', 'TOTAL_MEM', 'TOTAL_PY_MEM', 'TOTAL_MEM_PY_CORR']
    # p_range = 0.00125

    



    # df = pd.read_csv(output_file)
    
    # plot_df = full_df[
    # (full_df['l'] == 2) &
    # (full_df['eta'] == 10) &
    # (full_df['noise_model'] == 'circuit_level') &
    # (full_df['CD_type'] == 'ZXXZonSqu') &
    # (full_df['error_type'].isin(['X_MEM_CORR', 'Z_MEM_CORR', 'TOTAL_MEM_CORR']))
    # ].copy()

    # print(plot_df.head())
    # print(plot_df["num_log_errors"].describe())

    # for err in ['X_MEM_CORR', 'Z_MEM_CORR', 'TOTAL_MEM_CORR']:
    #     for d in sorted(plot_df['d'].unique()):
    #         tmp = plot_df[(plot_df['error_type'] == err) & (plot_df['d'] == d)].copy()
    #         tmp['weighted_errors'] = tmp['num_log_errors'] * tmp['num_shots']
    #         agg = (
    #             tmp.groupby('p', as_index=False)
    #             .agg(
    #                 total_shots=('num_shots', 'sum'),
    #                 total_errors=('weighted_errors', 'sum')
    #             )
    #         )
    #         agg['logical_error_rate'] = agg['total_errors'] / agg['total_shots']
    #         agg = agg.sort_values('p').reset_index(drop=True)

    #         print(f"\nerr={err}, d={d}")
    #         print(agg[['p', 'total_shots', 'logical_error_rate']].to_string(index=False))

    # full_error_plot(
    #     full_df=df,
    #     curr_eta=eta,
    #     curr_l=l,
    #     curr_num_shots=None,
    #     noise_model=noise_model,
    #     CD_type=CD_type,
    #     file=output_file,
    #     corr_decoding=corr_decoding,
    #     py_corr=py_corr,
    #     circuit_level=circuit_data,
    #     loglog=True,
    # )


    # make a plot for specific thresholds
    # pth0 = p_th_init_CL_pycorr[(l, eta, "Z_MEM_PY", CD_type, noise_model)]
    # popt, pcov = get_threshold(df, pth0, p_range, l, eta, error_type, CD_type)
    # p_th = popt[0]
    # pth_error = np.sqrt(pcov[0][0])
    # print(p_th, pth_error)

    # get_thresholds_from_data_exactish(chunk_size, p_th_init_CL_pycorr, p_range, output_file)
    # eta_df = pd.read_csv("/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/threshold_exactish_per_eta.csv")
    eta_df_code_cap = pd.read_csv("/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/all_thresholds_per_eta_elongated.csv")
    # eta_threshold_plot_totalmem_compare_deformations(
    # eta_df,
    # cd_type_list=["SC", "ZXXZonSqu"],
    # noise_model="circuit_level",
    # error_type = "TOTAL_MEM_PY"
# )
    # eta_threshold_plot_compare_deformations_and_decoder(
    # eta_df,
    # cd_type_list=["SC", "ZXXZonSqu"],
    # noise_model="circuit_level",
    # error_type = "TOTAL_MEM_PY",
    # suffix_to_remove="_PY"
    # )

    # eta_threshold_plot_compare_deformations_and_decoder_2x2(
    # eta_df,
    # cd_type_list=["SC", "ZXXZonSqu"],
    # noise_model="circuit_level",
    # error_type = "TOTAL_MEM_PY",
    # suffix_to_remove="_PY"
    # )

    # eta_delta_threshold_gap_plot_compare_deformations_and_decoder(
    # eta_df,
    # cd_type_list=["SC", "ZXXZonSqu"],
    # noise_model="circuit_level",
    # error_type = "TOTAL_MEM_PY",
    # suffix_to_remove="_PY"
    # )

    eta_threshold_plot_compare_error_types(
    eta_df_code_cap,
    cd_type="SC",
    error_type_list=["CORR_XZ", "CORR_ZX", "TOTAL_PY_CORR", "TOTAL"],
    noise_model="code_cap"
    )
    
    # eta_delta_threshold_gap_grid_compare_deformations_and_decoder(
    # eta_df,
    # cd_type_list=["SC", "ZXXZonSqu"],
    # noise_model="circuit_level",
    # error_type = "TOTAL_MEM_PY",
    # )
    
    # p_range_df = df[(df['p'] <= pth0 + p_range) & (df["p"] >= pth0 - p_range)]
    # print(len(p_range_df))
    # threshold_plot(df, pth0, p_range, eta, l, curr_num_shots, error_type, CD_type, noise_model, file=output_file, circuit_level=True, py_corr = py_corr, corr_decoding=corr_decoding, loglog=False, averaging=True, show_threshold=True, show_fit=True)
    # eta_threshold_plot(eta_df, CD_type,corr_type_list, noise_model)
    # get_thresholds_from_data_exactish(curr_num_shots, p_th_init_CL,p_range, output_file)
    # make eta plot
    # eta_df = pd.read_csv("/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/all_thresholds_per_eta_elongated.csv")
    # corr_type_list = ['TOTAL', "TOTAL_PY_CORR"]
    # eta_threshold_plot(eta_df, "XZZXonSqu", corr_type_list, "code_cap")



