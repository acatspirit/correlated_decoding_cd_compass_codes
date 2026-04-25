import numpy as np
import compass_codes as cc
import stim 


class CDCompassCodeCircuit:
    def __init__(self, d, l, eta, mem_type,memory=True):
        self.d = d
        self.l = l
        self.eta = eta
        self.memory = memory
        
        self.code = cc.CompassCode(d=d, l=l)
        self.H_x, self.H_z = self.code.H['X'], self.code.H['Z']
        self.log_x, self.log_z = self.code.logicals['X'], self.code.logicals['Z']
        self.type = mem_type # str "X" or "Z", indicates the type of memory experiment / which stabilizer you measure, also which logical you want to measure

        self.qubit_order_d = self.check_order_d_elongated()
    
    
    

    def get_circuit(self, before_measure_flip, after_clifford_depolarization, before_round_data_depolarization, idling_dephasing):
        """ 
        Returns the circuit object for the compass code with the specified parameters.
        Inputs:
            before_measure_depolarization - (float) the probability of a depolarizing error before measurement
            after_clifford_depolarization - (float) the probability of a depolarizing error after clifford deformation
            before_round_data_depolarization - (float) the probability of a depolarizing error before each round of data qubit measurements
            idling_dephasing - (float) the probability of a dephasing error on idling qubits
        Returns:
            circuit - (stim.Circuit) the circuit object for the compass code with the specified parameters
        """
        self.circuit = self.make_elongated_circuit_from_parity(before_measure_flip, after_clifford_depolarization, before_round_data_depolarization, idling_dephasing)

    #
    # Helper functions to make surface code circuits
    #

    def check_order_d(self):
        """ Change this for longer codes
            Right now (from STIM ex): 
            HX: 0 - TR, 1 - TL, 2 - BR, 3 - BL
            HZ: 0 - TR, 1 - BR, 2 - TL, 3 - BL
        """
        if self.type == "X": H = self.H_x
        elif self.type == "Z": H = self.H_z

        plaq_d = self.convert_sparse_to_d(H)
        order_d = {0:[], 1:[], 2:[], 3:[]}
        d = int(np.sqrt(H.shape[1]))

        for plaq in plaq_d:
            q_list = plaq_d[plaq] # the list of qubits in the plaquette
            if type == "X":
                if len(q_list) == 2 and max(q_list) <= H.shape[1]//2: # if the two qubit stabilizer is on the top bndry
                    order_d[2] += [(q_list[1], plaq)]
                    order_d[3] += [(q_list[0], plaq)]
                elif len(q_list) == 2 and max(q_list) >= H.shape[1]//2: # if the two qubit stabilizer is on bottom 
                    order_d[0] += [(q_list[1], plaq)]
                    order_d[1] += [(q_list[0], plaq)]
                else: # length 4 plaquette
                    order_d[0] += [(q_list[1], plaq)]
                    order_d[1] += [(q_list[0], plaq)]
                    order_d[2] += [(q_list[3], plaq)]
                    order_d[3] += [(q_list[2], plaq)]
            if type == "Z": 
                if len(plaq_d[plaq]) == 2 and (q_list[0]-q_list[1])%d == 0 and q_list[0]%d == 0: # if the two qubit stabilizer is on the left bndry
                    order_d[0] += [(q_list[0], plaq)]
                    order_d[1] += [(q_list[1], plaq)]
                elif len(plaq_d[plaq]) == 2 and (q_list[0]-q_list[1])%d == 0 and q_list[0]%d != 0: # if the two qubit stabilizer is on right bndry 
                    order_d[2] += [(q_list[0], plaq)]
                    order_d[3] += [(q_list[1], plaq)]
                else: # length 4 plaquette
                    order_d[0] += [(q_list[1], plaq)]
                    order_d[1] += [(q_list[3], plaq)]
                    order_d[2] += [(q_list[0], plaq)]
                    order_d[3] += [(q_list[2], plaq)]
                    
        return order_d

    def check_order_d_elongated(self):
        """ New order based on zigzag pattern in PRA (101)042312, 2020.
            Z stabilizers 
            #    #
            |  / |
            1 2  3 4 .....
            |/   |/
            #    #
            X stabilizers
            #--1--#
                /
               2    .....
             /
            #--3--#

        """
        stab_size = (self.l)*2# the size of the largest stabilizer

        # create the order dictionary to store the qubit ordering for each plaquette
        order_d_x = {}
        order_d_z = {}

        # create the order dictionary to store the qubit ordering for each plaquette
        for row in range(self.H_x.shape[0]):
            order_d_x[row] = []
        for row in range(self.H_z.shape[0]):
            order_d_z[row] = []

        # qubit ordering for Z stabilizers
        for row in range(self.H_z.shape[0]):
            start = self.H_z.indptr[row]
            end = self.H_z.indptr[row+1]
            qubits = sorted(self.H_z.indices[start:end]) # the qubits in the plaquette

            
            for i in range(len(qubits)//2):
                match_qubit_ind = np.where(qubits == (qubits[i] + self.d))[0][0]
                order_d_z[row] += [(qubits[i], row)]
                order_d_z[row] += [(qubits[match_qubit_ind], row)]
        
        
        # qubit ordering for X stabilizers
        for row in range(self.H_x.shape[0]):
            start = self.H_x.indptr[row]
            end = self.H_x.indptr[row+1]
            qubits = sorted(self.H_x.indices[start:end]) # the qubits in the plaquette

            for qubit in qubits:
                order_d_x[row] += [(qubit, row)]
        return order_d_x, order_d_z

    def qubit_to_stab_d(self):  
        """ Given a parity check matrix, returns a dictionary of qubits to stabs
            Returns: (dict) qubit to stab mapping
        """
        rows_x, cols_x, values = sparse.find(self.H_x)
        rows_z, cols_z, values = sparse.find(self.H_z)
        d_x = {}
        d_z = {}
        for i in range(len(cols_x)):
            q = cols_x[i]
            plaq = rows_x[i]

            if q not in d_x:
                d_x[q] = [plaq]
            else:
                d_x[q] += [plaq]
        
        for i in range(len(cols_z)):
            q = cols_z[i]
            plaq = rows_z[i]

            if q not in d_z:
                d_z[q] = [plaq]
            else:
                d_z[q] += [plaq]

        return d_x, d_z
    
    
    def convert_sparse_to_d(self):

        rows_x, cols_x, values = sparse.find(self.H_x)
        rows_z, cols_z, values = sparse.find(self.H_z)
        d_x = {}
        d_z = {}

        for i in range(len(rows_x)):
            plaq = rows_x[i]
            qubit = cols_x[i]

            if plaq not in d_x:
                d_x[plaq] = [cols_x[i]]
            else:
                d_x[plaq] += [cols_x[i]]
        sorted_d_x = dict(sorted(zip(d_x.keys(),d_x.values())))


        for i in range(len(rows_z)):
            plaq = rows_z[i]
            qubit = cols_z[i]

            if plaq not in d_z:
                d_z[plaq] = [cols_z[i]]
            else:
                d_z[plaq] += [cols_z[i]]
        sorted_d_z = dict(sorted(zip(d_z.keys(),d_z.values())))
        return sorted_d_x, sorted_d_z

    

    def isolate_observables_DEM(self) -> stim.DetectorErrorModel:
        """Returns two new DEMs containing only the detector faults for specific logical observables included in the measurement circuit."""
        circuit, detectors_d = self.make_elongated_MPP_circuit_from_parity()
        dem = circuit.detector_error_model(flatten_loops=True) # DEM for both measurements 

        # Create two new DetectorErrorModels for X and Z observables
        dem_x = stim.DetectorErrorModel()
        dem_z = stim.DetectorErrorModel()
        dets_x = {}
        dets_z = {}
        detector_coords = []

        # Deconstruct the DEM and split into X and Z detector dictionaries. Each instruction is stored in the dictionary with probability as the value.
        # Hyperedges are decomposed based on detector type
        for inst in dem:
            if inst.type == "error":
                # for each inst check whether its a detector or a logic is_relative_detector_id 
                prob_err = inst.args_copy()[0]
                targets = inst.targets_copy()
                dets_x_list = []
                dets_z_list = []
                for target in targets:
                    if target.is_relative_detector_id():
                        det = "D" + str(target.val)
                        if det in detectors_d["X"]:
                            dets_x_list.append(det)
                        elif det in detectors_d["Z"]:
                            dets_z_list.append(det)
                    elif target.is_logical_observable_id():
                        observable_id = "L" + str(target.val)
                        if observable_id == "L0":
                            dets_x_list.append(observable_id)
                        elif observable_id == "L1":
                            dets_z_list.append(observable_id)
                if dets_x_list:

                    key = tuple(dets_x_list)
                    if key not in dets_x:
                        dets_x[key] = prob_err
                    else:
                        curr_p = dets_x[key]
                        dets_x[key] = curr_p + prob_err - 2 * curr_p * prob_err  # combine probabilities if multiple error mechanisms detected
                if dets_z_list:
                    key = tuple(dets_z_list)
                    if key not in dets_z:
                        dets_z[key] = prob_err
                    else:
                        curr_p = dets_z[key]
                        dets_z[key] = curr_p + prob_err - 2 * curr_p * prob_err
            else:
                detector_coords.append(inst) # not sure if I need this to be seperate for X and Z qubits, but I will just keep it and add it to both DEMs
        
        # Construct the new DEMs for X and Z observables
        for key in dets_x.keys():
            prob = dets_x[key]
            targets = [stim.target_relative_detector_id(int(det[1:])) if det[0] == "D" else stim.target_logical_observable_id(0) for det in key]
            dem_x.append("error", prob, targets=targets)
        for key in dets_z.keys():
            prob = dets_z[key]
            targets = [stim.target_relative_detector_id(int(det[1:])) if det[0] == "D" else stim.target_logical_observable_id(1) for det in key]
            dem_z.append("error", prob, targets=targets)

        for inst in detector_coords:
            dem_z.append(inst)
            dem_x.append(inst)

        return dem_x, dem_z
    


    ############################################
    # 
    # Functions to make the circuits
    # 
    ############################################  

    def make_circuit_from_parity(self, p_err):
        """ Given a parity check matrix pair, generates a STIM circuit and detectors to implement the outlined code.
            Inputs:
                H - (scipy sparse mat) the parity check matrix
                Type - str - type == "X" is the X parity check matrix and produces those stabilizers
                            type == "Z" ' ' 
            Returns: (Stim circuit object) the circuit corresponding to the checks of the specified code

            Note - the X and Z circuits can be seperated in code-capacity model. Circuit level model will
                    require integrating the circuits to avoid hook errors. See fig 3. of this paper: 
                    https://arxiv.org/pdf/1404.3747
            TODO: change this function for non-CSS codes - how do I do this generally
            TODO: add elongation
        """
        H_x, H_z = self.H_x, self.H_z
        # make the circuit
        circuit = stim.Circuit()

        # get the qubit ordering
        plaq_d_x = self.convert_sparse_to_d()
        plaq_d_z = self.convert_sparse_to_d()
        
        order_d_x = self.check_order_d()
        order_d_z = self.check_order_d()
        
        qubit_d_x = self.qubit_to_plaq_d()
        qubit_d_z = self.qubit_to_plaq_d()
        
        # general parameters
        num_ancillas = len(plaq_d_x) + len(plaq_d_z)
        num_qubits = len(qubit_d_x)
        d = self.d

        data_q_x_list = [num_ancillas + q for q in list(qubit_d_x.keys())]
        data_q_z_list = [num_ancillas + q for q in list(qubit_d_z.keys())]
        data_q_list = data_q_x_list

        # convention - X plaqs first, then Z plaqs starting with 0
        full_plaq_L = range(len(plaq_d_x) + len(plaq_d_z))
        
        # reset the ancillas
        circuit.append("R", full_plaq_L)
        circuit.append("H", plaq_d_x)

        # reset the qubits
        for q in range(len(qubit_d_x)):
            if type == "X":
                circuit.append("RX", q + num_ancillas)
            if type == "Z":
                circuit.append("R", q + num_ancillas)
    
        
        for order in order_d_x: # go thru the qubits in order of gates
            q_x_list = order_d_x[order] # (qubit, ancilla)
            q_z_list = order_d_z[order]
            
            for q,p in q_x_list:
                circuit.append("CX", [p, q + num_ancillas])
            
            for q,p in q_z_list:
                circuit.append("CX", [q + num_ancillas, p + len(plaq_d_x)])
            
            circuit.append("TICK")
        
        circuit.append("H", plaq_d_x)
        circuit.append("X_ERROR", full_plaq_L, p_err)
        circuit.append("MR", full_plaq_L)

        # for X mem measure X plaqs
        if self.type == "X":
            for i in range(len(plaq_d_x)):
                circuit.append("DETECTOR", stim.target_rec(-num_ancillas + i))
        
            circuit.append("Z_ERROR", data_q_list, p_err)
            circuit.append("MX", data_q_list)

            # time to reconstruct each plaquette
            for i in plaq_d_x: 
                
                q_x_list = plaq_d_x[i] # get the qubits in the plaq
                anc = i 
                detector_list =  [-num_qubits + q for q in q_x_list] + [-num_ancillas + anc - num_qubits]
                
                circuit.append("DETECTOR", [stim.target_rec(d) for d in detector_list])
        
            # construct the logical observable to include - pick the top line of qubits since this is an X meas
            circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(- num_qubits + d*q) for q in range(d)], 0) # parity of the whole line needs to be the same
        
        # Z mem measure Z plaqs
        if self.type == "Z":
            for i in range(len(plaq_d_z)):
                circuit.append("DETECTOR", stim.target_rec(-num_ancillas + i + len(plaq_d_x)))
        
            circuit.append("X_ERROR", data_q_list, p_err)
            circuit.append("M", data_q_list)

            # time to reconstruct each plaquette
            for i in plaq_d_z: 
                
                q_z_list = plaq_d_z[i] # get the qubits in the plaq
                anc = i 
                detector_list =  [-num_qubits + q for q in q_z_list] + [-num_ancillas +len(plaq_d_x)+ anc - num_qubits]
                
                circuit.append("DETECTOR", [stim.target_rec(d) for d in detector_list])
        
            # construct the logical observable to include - pick the top line of qubits since this is an X meas
            circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(- num_qubits + q) for q in range(d)], 0)
        # print(repr(circuit))
        return circuit
    
    def add_meas_round(self, curr_circuit, stab_d_x, stab_d_z, order_d_x, order_d_z, qubit_d_x, qubit_d_z, num_ancillas, num_qubits_x, num_qubits_z, CD_data, p_i, p_gate, p_i_round, CD_type, fully_biased=False):
        """
        Add a measurement round to the circuit. Construct the gates with error model 
        for one round of stabilizer construction.
        """
        circuit = curr_circuit

        px = 0.5*p_i_round/(1+self.eta)
        pz = p_i_round*(self.eta/(1+self.eta))


        p_x_gate = p_gate/(12*(1+self.eta))
        p_z_gate = self.eta*p_gate/(3*(1+self.eta))
        

        circuit.append("PAULI_CHANNEL_1",range(num_ancillas + num_qubits_x), [px,px,pz]) # idling error on all qubits in between measurement rounds
        circuit.append("H", range(num_ancillas))
        if fully_biased:
            circuit.append("PAULI_CHANNEL_1", range(num_ancillas), [p_x_gate, p_x_gate, p_z_gate]) # try fully biased gates after H
        else:
            circuit.append("DEPOLARIZE1", range(num_ancillas), p_gate) # depolarizing error on the ancillas after H

        
        if p_i > 0: circuit.append("Z_ERROR", [num_ancillas + q for q in list(qubit_d_x.keys())], p_i) # idling error on the data qubits during round

        # go through each stabilizer in order, X stabilizers first
        for order in order_d_x:
            q_x_list = order_d_x[order] # (qubit, ancilla) in each stabilizer, not offset for x ancillas for the z ancillas, or ancillas for the data qubits

            if p_i > 0:
                active_qubits = {q for q,_ in q_x_list} # the dummy list for qubits that are idling
                active_ancilla = order

                # keep track of the idling qubits outside the stabilizer
                inactive_ancillas = [anc for anc in range(num_ancillas) if anc != active_ancilla]
                inactive_qubits = [q + num_ancillas for q in range(num_qubits_z) if q not in active_qubits]
                full_inactive_list = inactive_ancillas + inactive_qubits
            
            # apply a CX to each qubit in the stabilizer in the correct order
            for q,anc in q_x_list:
                ctrl = anc
                target = q + num_ancillas

                gate = "CX" if CD_type == "SC" else ("CZ" if CD_data[q] == 2 else "CX")

                circuit.append(gate, [ctrl, target]) # apply the gate gate

                # add 2-qubit depolarizing (biased or not depending on CD type) after the gate
                if gate == "CX":
                    if fully_biased:
                        circuit.append("PAULI_CHANNEL_2", [ctrl, target], [p_x_gate, p_x_gate, p_z_gate, p_x_gate, p_x_gate, p_x_gate, p_x_gate, p_x_gate,p_x_gate, p_x_gate, p_x_gate, p_z_gate,p_x_gate, p_x_gate, p_z_gate]) # try fully biased gates after 2 qubit
                    else:
                        circuit.append("DEPOLARIZE2", [ctrl, target], p_gate)
                else:
                    
                    circuit.append("PAULI_CHANNEL_2", [ctrl, target], [p_x_gate, p_x_gate, p_z_gate, p_x_gate, p_x_gate, p_x_gate, p_x_gate, p_x_gate,p_x_gate, p_x_gate, p_x_gate, p_z_gate,p_x_gate, p_x_gate, p_z_gate]) # Z error only after CZ gate

                if p_i > 0:
                    # apply idling errors to the qubits in the stabilizer without CX
                    for other_q in active_qubits - {q}:
                        circuit.append("Z_ERROR", [other_q + num_ancillas], p_i) # Idling error on the X qubits
                    circuit.append("Z_ERROR", full_inactive_list, p_i) # Idling error on the ancillas and qubits outside the stabilizer


        if p_i > 0: circuit.append("Z_ERROR", [num_ancillas + q for q in list(qubit_d_x.keys())], p_i)# idling error on the data qubits during round
        circuit.append("TICK")

        # now do the Z stabilizers
        for order in order_d_z: 
            q_z_list = order_d_z[order] # (qubit, ancilla) in each stabilizer, not offset for x ancillas for the z ancillas, or ancillas for the data qubits

            if p_i > 0:
                active_qubits = {q for q,_ in q_z_list} # the dummy list for qubits that are idling
                active_ancilla = order + len(stab_d_x) # the ancilla for this stabilizer, shifted to account for X stabs

                # keep track of the idling qubits outside the stabilizer
                inactive_ancillas = [anc for anc in range(num_ancillas) if anc != active_ancilla]
                inactive_qubits = [q + num_ancillas for q in range(num_qubits_z) if q not in active_qubits]
                full_inactive_list = inactive_ancillas + inactive_qubits

            # apply a CX to each qubit in the stabilizer in the correct order
            for q,anc in q_z_list:
                ctrl = anc + len(stab_d_x) # ancillas are shifted to account for X stabs
                target = q + num_ancillas

                gate = "CZ" if CD_type == "SC" else ("CX" if CD_data[q] == 2 else "CZ")

                circuit.append(gate, [ctrl, target]) # apply the CX gate
                # add 2-qubit depolarizing (biased or not depending on CD type) after the gate
                if gate == "CX":
                    if fully_biased:
                        circuit.append("PAULI_CHANNEL_2", [ctrl, target], [p_x_gate, p_x_gate, p_z_gate, p_x_gate, p_x_gate, p_x_gate, p_x_gate, p_x_gate,p_x_gate, p_x_gate, p_x_gate, p_z_gate,p_x_gate, p_x_gate, p_z_gate]) # try fully biased gates after 2 qubit
                    else:
                        circuit.append("DEPOLARIZE2", [ctrl, target], p_gate)
                else:
                    circuit.append("PAULI_CHANNEL_2", [ctrl, target], [p_x_gate, p_x_gate, p_z_gate, p_x_gate, p_x_gate, p_x_gate, p_x_gate, p_x_gate,p_x_gate, p_x_gate, p_x_gate, p_z_gate,p_x_gate, p_x_gate, p_z_gate]) # Z error only after CZ gate


                if p_i > 0:
                    # apply idling errors to the qubits in the stabilizer without CX
                    for other_q in active_qubits - {q}:
                        circuit.append("Z_ERROR", [other_q + num_ancillas], p_i) # Idling error on the X qubits
                    circuit.append("Z_ERROR", full_inactive_list, p_i) # Idling error on the ancillas and qubits outside the stabilizer

        circuit.append("H", range(num_ancillas))
        if fully_biased:
            circuit.append("PAULI_CHANNEL_1", range(num_ancillas), [p_x_gate, p_x_gate, p_z_gate]) # try fully biased gates after H
        else:
            circuit.append("DEPOLARIZE1", range(num_ancillas), p_gate) # depolarizing error on the ancillas after H
        circuit.append("TICK")
        
        
        # circuit.append("Z_ERROR", [q for q in range(len(stab_d_x), num_ancillas + num_qubits_x)], p_i) # idling error on the ancillas
        return circuit

    

    def make_elongated_circuit_from_parity(self, before_measure_flip, before_measure_pauli_channel, after_clifford_depolarization, before_round_data_pauli_channel,
                                            between_round_idling_pauli_channel, idling_dephasing, phenom_meas=False, CD_type = "SC", num_rounds = None, fully_biased=False):
        """ 
        create a surface code memory experiment circuit from a parity check matrix
        Inputs:
                after_clifford_depolarization - (float) the probability of a gate error
                before_measure_flip - (float) the probability of a measurement error
                before_measure_pauli_channel - (float) the probability of a biased pauli error before measurement applied to data qubits
                before_round_data_pauli_channel - (float) the probability of error in a biased depolarizing error channel before each round, biased towards Z
                between_round_idling_pauli_channel - (float) the probability of a biased pauli channel on all qubits between rounds, biased towards Z
                idling_dephasing - (float) the probability of a dephasing error on idling qubits during rounds
                phenom_meas - (bool) whether to use phenomenological measurement errors (True) or circuit-level measurement errors (False). Phenom meas errors are (p_meas_x + p_meas_z)*stabilizer weight/ 4
                CD_circuit - (bool) whether to apply clifford deformation to the circuit, ZXXZonSqu is the only option right now 
                CD_type - (str) the type of clifford deformation to apply, only ZXXZonSqu and XZZXonSq are valid, otherwise None which indicates no clifford deformation
                memory - (bool) whether or not to run multiple time slices / do a full memory experiment
            Returns: (stim.Circuit) the circuit with noise added

            The error model is the biased noise model from the paper: PRA (101)042312, 2020
            - 2-qubit gates are followed by 2-qubit depolarizing channel with p = p_gate (x)
            - measurement outcomes are preceded by a bit flip with probability p_meas (x)
            - idling qubits are between rounds, biased pauli channel with probability p_i_round (x)

            Z memory - measuring X stabs first time is random, don't add detectors to these, just the second round
        """
        p_gate = after_clifford_depolarization # gate error on two-qubit and single qubit cliffords (H, CNOT, CZ)
        p_meas = before_measure_flip # measurement bit/phase flip error before measurements unweighted
        p_data_meas = before_measure_pauli_channel # apply biased depolarizing error on DATA qubits before measurement
        p_data_dep = before_round_data_pauli_channel # apply biased depolarizing error on data qubits before each round
        p_i_round = between_round_idling_pauli_channel # idling error on all qubits between the measurement rounds
        p_i = idling_dephasing # idling error on all qubits during rounds

        if num_rounds == None:
            num_rounds = self.d

        memory = self.memory

        px_data = 0.5*p_data_dep/(1+self.eta) # biased depolarizing error on data qubits before round
        pz_data = p_data_dep*(self.eta/(1+self.eta)) # biased depolarizing error on data qubits before round
        py_data = px_data

        px_meas = 0.5*p_data_meas/(1+self.eta) # biased depolarizing error on data qubits before measurement
        pz_meas = p_data_meas*(self.eta/(1+self.eta)) # biased depolarizing error on data qubits before measurement
        py_meas = px_meas

        p_phenom_meas = (0.5*p_meas/(1+self.eta) + p_meas*(self.eta/(1+self.eta)))/4 # reweight by stabilizer weight preparation with phenom circuit
        

        # make the circuit
        circuit = stim.Circuit()

        # get the qubit ordering
        stab_d_x,stab_d_z = self.convert_sparse_to_d()
        
        # get the qubit ordered properly for each stabilizer
        order_d_x, order_d_z = self.check_order_d_elongated()
        
        # get the stabilizer that belong to each qubit
        qubit_d_x,qubit_d_z = self.qubit_to_stab_d()

        # get the data for the clifford deformation for the basis setup
        if CD_type != "SC":
            CD_data_transform = cc.CD_data_func(self.code.qbit_dict.values(), special=CD_type, ell=self.l, size=self.d) # data for which qubits have a transformation applied, dictionary 
        else:
            CD_data_transform = cc.CD_data_func(self.code.qbit_dict.values(), special="I", ell=self.l, size=self.d)
            
        
        # general parameters
        num_ancillas = len(stab_d_x) + len(stab_d_z) # total number of stabilizer to initialize
        num_qubits_x = len(qubit_d_x)
        num_qubits_z = len(qubit_d_z)
        
        data_q_x_list = [num_ancillas + q for q in list(qubit_d_x.keys())] # all the x data qubits
        data_q_z_list = [num_ancillas + q for q in list(qubit_d_z.keys())] # all the z data qubits
        data_q_list = [num_ancillas + q for q in range(self.d**2)] # change this later when wanna do X and Z seperately


        # convention - X stabs first, then Z stabs starting with 0
        full_stab_L = range(num_ancillas)
        
        # reset the ancillas
        circuit.append("R", full_stab_L)

        # reset the data qubits
        if self.type == "X":
        
            if CD_type != "SC":
                circuit.append("R", [q + num_ancillas for q in CD_data_transform if CD_data_transform[q] == 2]) # put code into 0L of the CD code
                circuit.append("RX", [q + num_ancillas for q in CD_data_transform if CD_data_transform[q] == 0])
            else: 
                circuit.append("RX", data_q_list)

            circuit.append("PAULI_CHANNEL_1", data_q_list, [px_data, py_data, pz_data]) # biased pauli channel on data qubits before the round
            circuit.append("Z_ERROR", [anc for anc in range(num_ancillas)], p_i) # idling error on the ancillas
        elif self.type == "Z":
            
            if CD_type != "SC":
                circuit.append("RX", [q + num_ancillas for q in CD_data_transform if CD_data_transform[q] == 2])
                circuit.append("R", [q + num_ancillas for q in CD_data_transform if CD_data_transform[q] == 0]) # put the code into the 1L of the CD code
            else:
                circuit.append("R", data_q_list)
                

            circuit.append("PAULI_CHANNEL_1", data_q_list, [px_data, py_data, pz_data])
            circuit.append("Z_ERROR", [anc for anc in range(num_ancillas)], p_i) # idling error on the ancillas
            
        #
        # start the for loop to repeat for d rounds - memory experiment round 1
        #

        # Round 0 - t=0 measurements
        circuit.append("TICK")
        # print(p_gate)
        circuit = self.add_meas_round(circuit, stab_d_x, stab_d_z, order_d_x, order_d_z, qubit_d_x, qubit_d_z, num_ancillas, num_qubits_x, num_qubits_z, CD_data=CD_data_transform,p_i=p_i, p_gate=p_gate, p_i_round=0, CD_type=CD_type, fully_biased=fully_biased) # set the idling error between rounds to 0 on first round

        # idling errors on the data qubits during round 
        circuit.append("Z_ERROR", data_q_z_list, p_i)
        
        # add the measurement error to the ancillas before the measurements, phenom model scale with the size of the stabilizer
        for anc in range(len(stab_d_x)):
            if phenom_meas:
                circuit.append("X_ERROR", anc, min(p_phenom_meas*len(stab_d_x[anc]),1))
            else:
                circuit.append("X_ERROR", anc, p_meas)
        for anc in range(len(stab_d_z)):
            if phenom_meas:
                circuit.append("X_ERROR", anc + len(stab_d_x), min(1,p_phenom_meas*len(stab_d_z[anc]))) 
            else:
                circuit.append("X_ERROR", anc + len(stab_d_x), p_meas) 
        
    
        circuit.append("MR", full_stab_L) # measure the ancillas at t=0

        # if memory:
        # initialize the t=0 detectors for the X or Z stabilizers
        if self.type == "X": # the Z stabilizers will be indeterministic at t=0
            for i in range(len(stab_d_x)):
                circuit.append("DETECTOR", stim.target_rec(-num_ancillas + i))
        elif self.type == "Z": # the X stabilizers will be indeterministic at t=0
            for i in range(len(stab_d_z)):
                circuit.append("DETECTOR", stim.target_rec(-num_ancillas + len(stab_d_x) + i ))
        
        circuit.append("TICK") # add a tick to the circuit to mark the end of the t=0 measurements

        #
        # start the for loop to repeat for d rounds - memory experiment rounds 2-d
        #
        if num_rounds > 1:
            loop_circuit = stim.Circuit() # create a loop circuit to repeat the following for d-1 rounds
            # All other d rounds - t>0 measurements

            # add error to the data qubits
            loop_circuit.append("PAULI_CHANNEL_1", data_q_list, [px_data, py_data, pz_data])
        
            loop_circuit = self.add_meas_round(loop_circuit, stab_d_x, stab_d_z, order_d_x, order_d_z, qubit_d_x, qubit_d_z, num_ancillas, num_qubits_x, num_qubits_z, CD_data=CD_data_transform, p_i=p_i, p_gate=p_gate, p_i_round=p_i_round, CD_type=CD_type, fully_biased=fully_biased)


            # idling errors on the data qubits, measure the ancillas, bit flip errors on measurements
            loop_circuit.append("Z_ERROR", data_q_z_list, p_i)
            
            # add the error to the ancillas before the ancilla measurement, phenom model
            for anc in range(len(stab_d_x)):
                if phenom_meas:
                    loop_circuit.append("X_ERROR", anc, min(p_phenom_meas*len(stab_d_x[anc]),1))
                else:
                    loop_circuit.append("X_ERROR", anc, p_meas)
            for anc in range(len(stab_d_z)):
                if phenom_meas:
                    loop_circuit.append("X_ERROR", anc + len(stab_d_x), min(p_phenom_meas*len(stab_d_z[anc]),1)) 
                else:
                    loop_circuit.append("X_ERROR", anc + len(stab_d_x), p_meas) 

            loop_circuit.append("MR", full_stab_L) # measure the ancillas at t>0

            # timelike detectors for the X or Z stabilizers
            for i in range(num_ancillas):
                loop_circuit.append("DETECTOR", [stim.target_rec(-num_ancillas + i), stim.target_rec(-2*num_ancillas+ i)]) # anc round d tied to anc round d=0

            loop_circuit.append("TICK") # add a tick to the circuit to mark the end of the t>0 iteration
            
            if memory:
                # repeat the loop circuit d-1 times - circuit level only
                circuit.append(stim.CircuitRepeatBlock(repeat_count=(num_rounds-1), body=loop_circuit))# end the repeat block


        #
        # Stabilizer measurement reconstruction
        #


        # reconstruct the stabilizers and measure the data qubits
        # for X mem measure X stabs
        if self.type == "X":
            # measure all the data qubits in the X stabilizers
            # circuit.append("Z_ERROR", full_stab_L, p_meas) 
            circuit.append("PAULI_CHANNEL_1", data_q_list, [px_meas, py_meas, pz_meas]) # apply biased depolarizing error on data qubits before measurement

            if CD_type != "SC":
                for q in CD_data_transform:
                    if CD_data_transform[q] == 2:
                        circuit.append("M", q + num_ancillas)
                    elif CD_data_transform[q] == 0:
                        circuit.append("MX", q + num_ancillas) # apply H to the qubits that have a transformation applied
            else:
                circuit.append("MX", data_q_list)

            # reconstruct each X stabilizer with a detector
            for anc in stab_d_x:
                q_x_list = stab_d_x[anc] # get the qubits in the stab
                detector_list =  [-num_qubits_x + q for q in q_x_list] + [-num_ancillas + anc - num_qubits_x]
                circuit.append("DETECTOR", [stim.target_rec(d) for d in detector_list])
            
            
            # construct the logical observable to include - pick the top line of qubits since this is an X meas
            circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-num_qubits_x + self.d*q) for q in range(self.d)], 0)
        
        # Z mem measure Z stabs
        if self.type == "Z":
            # measure all the data qubits in the Z stabilizers
            # circuit.append("X_ERROR", full_stab_L, p_meas) # add the error to the data qubits
            circuit.append("PAULI_CHANNEL_1", data_q_list, [px_meas, py_meas, pz_meas]) # apply biased depolarizing error on data qubits before measurement

            if CD_type != "SC":
                for q in CD_data_transform:
                    if CD_data_transform[q] == 2:
                        circuit.append("MX", q+num_ancillas)
                    elif CD_data_transform[q] == 0:
                        circuit.append("M", q + num_ancillas) # apply H to the qubits that have a transformation applied
            else:
                circuit.append("M", data_q_list)

            # reconstruct each stabilizer with a detector
            for anc in stab_d_z: 
                
                q_z_list = stab_d_z[anc] # get the qubits in the stab
                detector_list =  [-num_qubits_z + q for q in q_z_list] + [-num_ancillas +len(stab_d_x)+ anc - num_qubits_z]
                circuit.append("DETECTOR", [stim.target_rec(d) for d in detector_list])
        
            # construct the logical observable to include - pick the top line of qubits since this is an X meas
            circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-num_qubits_z + q) for q in range(self.d)], 0)
        return circuit

    def MBQC_round_helper(self, order_d_x, order_d_z,  p_gate,p_meas,p_i):
        """ Helper function to add a round of measurements to the circuit for the MBQC model.
            """

        circuit = ""

        # X measurements 
        for order in order_d_x:
            q_x_list = order_d_x[order]

            circuit += f"MPP({p_meas}) " # if measurements have nonzero error, this circuit gets d_eff = 2 for all lattice d
            for q,anc in q_x_list:
                circuit += f"X{q}"
                if q != q_x_list[-1][0]: # if not the last qubit in the stabilizer
                    circuit += "*"
                
            circuit += "\n"
        # Z measurements
        for order in order_d_z:
            q_z_list = order_d_z[order]

            circuit += f"MPP({p_meas}) "
            for q,anc in q_z_list:
                circuit += f"Z{q}"
                if q != q_z_list[-1][0]: # if not the last qubit in the stabilizer
                    circuit += "*"
            circuit += "\n"

        return circuit

    def make_elongated_MPP_circuit_from_parity(self, p_gate,p_meas,p_i):
        
        # get the qubit ordering
        stab_d_x,stab_d_z = self.convert_sparse_to_d()
        
        # get the qubit ordered properly for each stabilizer
        order_d_x, order_d_z = self.check_order_d_elongated()
        
        # get the stabilizer that belong to each qubit
        qubit_d_x,qubit_d_z = self.qubit_to_stab_d()

        detector_d = {"X":[], "Z":[]} # dictionary to store the detectors for each stabilizer type
        
        # general parameters
        num_ancillas = len(stab_d_x) + len(stab_d_z) # total number of stabilizer to initialize
        num_qubits_x = len(qubit_d_x)
        num_qubits_z = len(qubit_d_z)
        num_data_qubits = num_qubits_x

        circuit = ""

        # label the qubits
        row = 0
        for i in range(num_data_qubits):
            circuit += f"QUBIT_COORDS({row},{i%self.d}) {i} \n" # add the data qubits to the circuit
            if i % self.d == self.d - 1:
                row += 1
        
        # add observables for the data qubits
        circuit += f"OBSERVABLE_INCLUDE(0) "
        for i in range(self.d):
            circuit += f"X{i*self.d} "
        
        circuit += "\n"
        circuit += f"OBSERVABLE_INCLUDE(1) "
        for i in range(self.d):
            circuit += f"Z{i} "
        circuit += "\n"
        
        # add the noise
        circuit += f"DEPOLARIZE1({p_i}) "
        for i in range(num_data_qubits):
            circuit += f"{i} "
        # circuit += f"DEPOLARIZE1({p_i}) "
        # for i in range(num_data_qubits):
        #     circuit += f"{i} "
        
        circuit += "\n"

        # do the measurements 
        circuit += self.MBQC_round_helper(order_d_x, order_d_z,  p_gate,p_meas,p_i)

        for round_count in range(self.d):
            # add the noise
            circuit += f"X_ERROR({p_i}) "
            for i in range(num_data_qubits):
                circuit += f"{i} "
            
            circuit += "\n"

            circuit += self.MBQC_round_helper(order_d_x, order_d_z,  p_gate,p_meas,p_i) # add the measurements again to the circuit

            # add detectors for each stabilizer
            # double check that the detectors are added correctly to the dictionary
            for anc in stab_d_x:
                # circuit += f"DETECTOR({anc},{anc},{anc + round_count*num_ancillas}) rec[{-num_ancillas + anc}] rec[{-2*num_ancillas + anc}]\n"
                circuit += f"DETECTOR rec[{-num_ancillas + anc}] rec[{-2*num_ancillas + anc}]\n"
                detector_d["X"] += [f"D{anc + round_count*num_ancillas}"] # fix this to add the detectors for every round 
            for anc in stab_d_z:
                # circuit += f"DETECTOR({anc},{anc},{anc + len(stab_d_x)  + round_count*num_ancillas}) rec[{-num_ancillas + len(stab_d_x) + anc}] rec[{-2*num_ancillas + anc + len(stab_d_x)}]\n"
                circuit += f"DETECTOR rec[{-num_ancillas + len(stab_d_x) + anc}] rec[{-2*num_ancillas + anc + len(stab_d_x)}]\n"
                detector_d["Z"] += [f"D{anc + len(stab_d_x)  + round_count*num_ancillas}"] # here too

        # add observables for the data qubits
        circuit += f"OBSERVABLE_INCLUDE(0) "
        for i in range(self.d):
            circuit += f"X{i*self.d} "
        
        circuit += "\n"
        circuit += f"OBSERVABLE_INCLUDE(1) "
        for i in range(self.d):
            circuit += f"Z{i} "
        circuit += "\n"
        
        stim_circuit = stim.Circuit(circuit)
        return stim_circuit, detector_d
        # return circuit

    def make_clifford_deformed_circuit_from_parity(self):
        """ Given a parity check matrix pair, generates a STIM circuit and detectors to implement the outlined code.
            Inputs:
                H - (scipy sparse mat) the parity check matrix
                Type - str - type == "X" is the X parity check matrix and produces those stabilizers
                            type == "Z" ' ' 
            Returns: (Stim circuit object) the circuit corresponding to the checks of the specified code

            Note - the X and Z circuits can be seperated in code-capacity model. Circuit level model will
                    require integrating the circuits to avoid hook errors. See fig 3. of this paper: 
                    https://arxiv.org/pdf/1404.3747
        """
        circuit = stim.Circuit()
        


        return circuit


# for d rounds
# 3 sets of detectors:
# before the for loop, measure the ancillas that I want to track (ie X mem measure X stab ancillas)
# during the for loop, for each round, make a detector between each stabilizer and itself at time 0. That is, measure the ancilla at round d and make a detector with the same ancilla outside the for loop
# after the for loop, reconstruct the stabilizers in mem of interest, and measure the data qubits in each stabilizer (ie tie an anc for an X stab in the last iteration of the for loop to the data qubits in the X stab)




# p_list = np.linspace(0,0.5, 30)
# d_dict = {5:[], 7:[], 9:[], 11:[]}
# num_shots = 100000
# l = 2
# type_d = {0:"X", 1:"Z"}
# type=type_d[0]
# eta = 1.67
# prob_scale = {'X': 0.5/(1+eta), 'Z': (1+2*eta)/(2*(1+eta)), 'CORR_XZ': 1, 'TOTAL':1}



# circuit = CDCompassCodeCircuit(5, l, eta, [0.003, 0.001, 0.05], type)



# order_d = circuit.check_order_d_elongated() # looks mostly good
# print(circuit.H_x)
# print(order_d)
# for d in list(d_dict.keys()):
#     circuit = CDCompassCodeCircuit(d, l, eta, 0.05, type)
#     # print('circuit:')
    # my_c = make_elongated_circuit_from_parity(H_x, H_z, d, 0.05, 0.5, type)
    # print(repr(my_c)) same between files
    # d_dict[d] = circuit.get_log_error_circuit_level(p_list, H_x,H_z, type, eta, d, num_shots)


# matching = Matching.from_stim_circuit(my_c)
# matching.draw()
# for d in d_dict:
#     plt.plot(p_list*prob_scale[type], d_dict[d], label=f"d={d}")
# plt.xlabel("p")
# plt.ylabel(f"{[v for k, v in type_d.items() if v != type][0]} Logical Errors ")
# plt.legend()
# # plt.savefig("l3_eta1.67_zmem.png", dpi=300)
# plt.show()