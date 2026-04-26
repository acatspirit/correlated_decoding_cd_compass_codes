import numpy as np
import compass_codes as cc
import stim 
import scipy.sparse as sparse


class CDCompassCodeCircuit:
    def __init__(self, d, l, eta, mem_type,memory=True):
        """
        Implements the syndrome extraction circuit for Clifford-deformed elongated compass codes.

        Author: Arianna Meinking
        Date: April 25, 2026
        """
        self.d = d # code distance
        self.l = l # elongation parameter
        self.eta = eta # bias parameter
        self.memory = memory # whether or not to run d rounds of memory
        
        self.code = cc.CompassCode(d=d, l=l) # initialize the code object to get the parity check matrices and logical operators
        self.H_x, self.H_z = self.code.H['X'], self.code.H['Z'] # get the parity check matrices for the code
        self.log_x, self.log_z = self.code.logicals['X'], self.code.logicals['Z'] # get the logical operators for the code
        self.type = mem_type # str "X" or "Z", indicates the type of memory experiment / which stabilizer you measure, also which logical you want to measure

        self.qubit_order_d = self.check_order_d_elongated() # the check order to measure qubits in each stabilizer
    
    

    #
    # Helper functions to make surface code circuits
    #

    def check_order_d_elongated(self):
        """ Stabilizer measurement order based on zigzag pattern in PRA (101)042312, 2020.
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
        """ Given a parity check matrix, returns a dictionary of stabilizers each qubit is a support of, for X and Z.
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
    
    def stab_to_qubit_d(self):
        """ Given a parity check matrix, returns a dictionary of stabilizers and the qubits in each stabilizer, for X and Z.
            Returns: (dict) stab to qubit mapping
        """
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

            if plaq not in d_z:
                d_z[plaq] = [cols_z[i]]
            else:
                d_z[plaq] += [cols_z[i]]
        sorted_d_z = dict(sorted(zip(d_z.keys(),d_z.values())))
        return sorted_d_x, sorted_d_z



    ############################################
    # 
    # Functions to make the circuits
    # 
    ############################################  

    
    def add_meas_round(self, curr_circuit, stab_d_x,  order_d_x, order_d_z, qubit_d_x, num_stabs, num_qubits_x, num_qubits_z, CD_data, p_gate, p_i_round, CD_type, p_i = 0, fully_biased=False):
        """
        Helper to add a round of measurements to the circuit. Equivalent to the syndrome extraction circuit for one round. Full idling noise is possible with parameter
        p_i, but the idling noise used in the results is only between rounds with p_i_round, so set p_i to 0 to match the results. The circuit is deep, so full
        idling noise is computationally expensive.

        :param curr_circuit: the circuit to add the round to
        :param stab_d_x: the dictionary of X stabilizers and the qubits in each stabilizer
        :param order_d_x: the dictionary of X stabilizers and the qubits in each stabilizer in the order they should be measured
        :param order_d_z: the dictionary of Z stabilizers and the qubits in each stabilizer in the order they should be measured
        :param qubit_d_x: the dictionary of qubits and the X stabilizers they are a part of
        :param num_stabs: the number of stabilizers in the circuit
        :param num_qubits_x: the number of X data qubits in the circuit
        :param num_qubits_z: the number of Z data qubits in the circuit
        :param CD_data: the dictionary of which qubits have a clifford deformation applied to them, and which transformation they have applied
        :param p_i: the probability of an idling error on qubits during the round
        :param p_gate: the probability of an error on a gate
        :param p_i_round: the probability of an idling error on qubits between rounds
        :param CD_type: the type of clifford deformation in the circuit, either "SC", "ZXXZonSqu", "XZZXonSq" to match compass_codes CD_data function
        :param fully_biased: whether to use fully biased gate errors - otherwise, use HBD noise model 

        :returns: the circuit with the round of measurements added
        """

        circuit = curr_circuit

        # idling error between rounds components 
        px = 0.5*p_i_round/(1+self.eta)
        pz = p_i_round*(self.eta/(1+self.eta))

        # gate error components, using HBD noise model 
        p_x_gate = p_gate/(12*(1+self.eta))
        p_z_gate = self.eta*p_gate/(3*(1+self.eta))
        
        # add idling errors on all qubits between rounds
        circuit.append("PAULI_CHANNEL_1",range(num_stabs + num_qubits_x), [px,px,pz])
        circuit.append("H", range(num_stabs))
        
        # gate errors on H
        if fully_biased:
            circuit.append("PAULI_CHANNEL_1", range(num_stabs), [p_x_gate, p_x_gate, p_z_gate])
        else:
            circuit.append("DEPOLARIZE1", range(num_stabs), p_gate) # depolarizing error on the stabilizers after H

        # full idling noise addition 
        if p_i > 0: circuit.append("Z_ERROR", [num_stabs + q for q in list(qubit_d_x.keys())], p_i) # idling error on the data qubits during round

        #
        # measure X stabilizers
        #

        for order in order_d_x:
            q_x_list = order_d_x[order] # (qubit, ancilla) in each stabilizer, not offset for x stabilizers for the z stabilizers, or stabilizers for the data qubits

            # full idling noise addition
            if p_i > 0:
                active_qubits = {q for q,_ in q_x_list} # the dummy list for qubits that are idling
                active_ancilla = order

                # keep track of the idling qubits outside the stabilizer
                inactive_stabilizers = [stab for stab in range(num_stabs) if stab != active_ancilla]
                inactive_qubits = [q + num_stabs for q in range(num_qubits_z) if q not in active_qubits]
                full_inactive_list = inactive_stabilizers + inactive_qubits
            
            # apply a CX to each qubit in the stabilizer in the correct order
            for q,stab in q_x_list:
                ctrl = anc
                target = q + num_stabs

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

                # fully idling noise addition
                if p_i > 0:
                    # apply idling errors to the qubits in the stabilizer without CX
                    for other_q in active_qubits - {q}:
                        circuit.append("Z_ERROR", [other_q + num_stabs], p_i) # Idling error on the X qubits
                    circuit.append("Z_ERROR", full_inactive_list, p_i) # Idling error on the stabilizers and qubits outside the stabilizer

        # fully idling noise addition
        if p_i > 0: circuit.append("Z_ERROR", [num_stabs + q for q in list(qubit_d_x.keys())], p_i)# idling error on the data qubits during round
        circuit.append("TICK")

        
        #
        # measure Z stabilizers
        #

        for order in order_d_z: 
            q_z_list = order_d_z[order] # (qubit, ancilla) in each stabilizer, not offset for x stabilizers for the z stabilizers, or stabilizers for the data qubits

            # full idling noise addition
            if p_i > 0:
                active_qubits = {q for q,_ in q_z_list} # the dummy list for qubits that are idling
                active_ancilla = order + len(stab_d_x) # the ancilla for this stabilizer, shifted to account for X stabs

                # keep track of the idling qubits outside the stabilizer
                inactive_stabilizers = [stab for stab in range(num_stabs) if stab != active_ancilla]
                inactive_qubits = [q + num_stabs for q in range(num_qubits_z) if q not in active_qubits]
                full_inactive_list = inactive_stabilizers + inactive_qubits

            # apply a CX to each qubit in the stabilizer in the correct order
            for q,stab in q_z_list:
                ctrl = stab + len(stab_d_x) # stabilizers are shifted to account for X stabs
                target = q + num_stabs

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

                # full idling noise addition
                if p_i > 0:
                    # apply idling errors to the qubits in the stabilizer without CX
                    for other_q in active_qubits - {q}:
                        circuit.append("Z_ERROR", [other_q + num_stabs], p_i) # Idling error on the X qubits
                    circuit.append("Z_ERROR", full_inactive_list, p_i) # Idling error on the stabilizers and qubits outside the stabilizer

        circuit.append("H", range(num_stabs))
        
        if fully_biased:
            circuit.append("PAULI_CHANNEL_1", range(num_stabs), [p_x_gate, p_x_gate, p_z_gate]) # try fully biased gates after H
        else:
            circuit.append("DEPOLARIZE1", range(num_stabs), p_gate) # depolarizing error on the stabilizers after H
        circuit.append("TICK")
        
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

        # assume that we are completing a full memory experiment with d rounds if num_rounds is not specified
        if num_rounds == None: 
            num_rounds = self.d

        # if we execute a full memory
        memory = self.memory

        #
        # noise components
        #

        p_gate = after_clifford_depolarization # gate error on two-qubit and single qubit cliffords (H, CNOT, CZ)
        p_meas = before_measure_flip # measurement bit/phase flip error before measurements unweighted
        p_data_meas = before_measure_pauli_channel # apply biased depolarizing error on DATA qubits before measurement
        p_data_dep = before_round_data_pauli_channel # apply biased depolarizing error on data qubits before each round
        p_i_round = between_round_idling_pauli_channel # idling error on all qubits between the measurement rounds
        p_i = idling_dephasing # idling error on all qubits during rounds

        # data and measurement error probabilities for the biased noise model
        px_data = 0.5*p_data_dep/(1+self.eta) # biased depolarizing error on data qubits before round
        pz_data = p_data_dep*(self.eta/(1+self.eta)) # biased depolarizing error on data qubits before round
        py_data = px_data

        px_meas = 0.5*p_data_meas/(1+self.eta) # biased depolarizing error on data qubits before measurement
        pz_meas = p_data_meas*(self.eta/(1+self.eta)) # biased depolarizing error on data qubits before measurement
        py_meas = px_meas

        # for phenomonlogical measurement error model, the measurement error scales with the size of the stabilizer
        p_phenom_meas = (0.5*p_meas/(1+self.eta) + p_meas*(self.eta/(1+self.eta)))/4 # reweight by stabilizer weight preparation with phenom circuit
        

        
        
        #
        # make the circuit
        #

        circuit = stim.Circuit()

        # get the qubit stabilizer mapping
        stab_d_x,stab_d_z = self.stab_to_qubit_d()
        
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
        num_stabs = len(stab_d_x) + len(stab_d_z) # total number of stabilizer to initialize
        num_qubits_x = len(qubit_d_x)
        num_qubits_z = len(qubit_d_z)
        
        data_q_x_list = [num_stabs + q for q in list(qubit_d_x.keys())] # all the x data qubits
        data_q_z_list = [num_stabs + q for q in list(qubit_d_z.keys())] # all the z data qubits
        data_q_list = [num_stabs + q for q in range(self.d**2)] # change this later when wanna do X and Z seperately


        # convention - X stabs first, then Z stabs starting with 0
        full_stab_L = range(num_stabs)
        
        #
        # Initialization
        #

        # stabilizer initialization
        circuit.append("R", full_stab_L)

        # reset the data qubits
        if self.type == "X":
        
            if CD_type != "SC":
                circuit.append("R", [q + num_stabs for q in CD_data_transform if CD_data_transform[q] == 2]) # put code into 0L of the CD code
                circuit.append("RX", [q + num_stabs for q in CD_data_transform if CD_data_transform[q] == 0])
            else: 
                circuit.append("RX", data_q_list)

            # before round biased pauli channel, idling if present
            circuit.append("PAULI_CHANNEL_1", data_q_list, [px_data, py_data, pz_data]) # biased pauli channel on data qubits before the round
            circuit.append("Z_ERROR", [stab for stab in range(num_stabs)], p_i) # idling error on the stabilizers
        
        elif self.type == "Z":
            
            if CD_type != "SC":
                circuit.append("RX", [q + num_stabs for q in CD_data_transform if CD_data_transform[q] == 2])
                circuit.append("R", [q + num_stabs for q in CD_data_transform if CD_data_transform[q] == 0]) # put the code into the 1L of the CD code
            else:
                circuit.append("R", data_q_list)
                
            # before round biased pauli channel, idling if present
            circuit.append("PAULI_CHANNEL_1", data_q_list, [px_data, py_data, pz_data])
            circuit.append("Z_ERROR", [stab for stab in range(num_stabs)], p_i) # idling error on the stabilizers
            
        #
        # start the for loop to repeat for d rounds - memory experiment round 1
        #

        # Round 0 - t=0 measurements
        circuit.append("TICK")
        circuit = self.add_meas_round(circuit, stab_d_x, stab_d_z, order_d_x, order_d_z, qubit_d_x, qubit_d_z, num_stabs, num_qubits_x, num_qubits_z, CD_data=CD_data_transform,p_i=p_i, p_gate=p_gate, p_i_round=0, CD_type=CD_type, fully_biased=fully_biased) # set the idling error between rounds to 0 on first round

        # idling errors on the data qubits during round 
        circuit.append("Z_ERROR", data_q_z_list, p_i)
        
        # add the measurement error to the stabilizers before the measurements, phenom model scale with the size of the stabilizer
        for stab in range(len(stab_d_x)):
            if phenom_meas:
                circuit.append("X_ERROR", stab, min(p_phenom_meas*len(stab_d_x[stab]),1))
            else:
                circuit.append("X_ERROR", stab, p_meas)
        for stab in range(len(stab_d_z)):
            if phenom_meas:
                circuit.append("X_ERROR", stab + len(stab_d_x), min(1,p_phenom_meas*len(stab_d_z[stab]))) 
            else:
                circuit.append("X_ERROR", stab + len(stab_d_x), p_meas) 
        
        #
        # first round measurements
        #

        circuit.append("MR", full_stab_L) # measure the stabilizers at t=0


        # initialize the t=0 detectors for the X or Z stabilizers
        if self.type == "X": # the Z stabilizers will be indeterministic at t=0
            for i in range(len(stab_d_x)):
                circuit.append("DETECTOR", stim.target_rec(-num_stabs + i))
        elif self.type == "Z": # the X stabilizers will be indeterministic at t=0
            for i in range(len(stab_d_z)):
                circuit.append("DETECTOR", stim.target_rec(-num_stabs + len(stab_d_x) + i ))
        
        circuit.append("TICK") # add a tick to the circuit to mark the end of the t=0 measurements

        #
        # start the for loop to repeat for d rounds - memory experiment rounds 2-d
        #
        if num_rounds > 1:
            loop_circuit = stim.Circuit() # create a loop circuit to repeat the following for d-1 rounds
            # All other d rounds - t>0 measurements

            # add error to the data qubits
            loop_circuit.append("PAULI_CHANNEL_1", data_q_list, [px_data, py_data, pz_data])
        
            loop_circuit = self.add_meas_round(loop_circuit, stab_d_x, stab_d_z, order_d_x, order_d_z, qubit_d_x, qubit_d_z, num_stabs, num_qubits_x, num_qubits_z, CD_data=CD_data_transform, p_i=p_i, p_gate=p_gate, p_i_round=p_i_round, CD_type=CD_type, fully_biased=fully_biased)


            # idling errors on the data qubits, measure the stabilizers, bit flip errors on measurements
            loop_circuit.append("Z_ERROR", data_q_z_list, p_i)
            
            # add the error to the stabilizers before the ancilla measurement, phenom model
            for stab in range(len(stab_d_x)):
                if phenom_meas:
                    loop_circuit.append("X_ERROR", stab,  min(p_phenom_meas*len(stab_d_x[stab]),1))
                else:
                    loop_circuit.append("X_ERROR", stab,  p_meas)
            for stab in range(len(stab_d_z)):
                if phenom_meas:
                    loop_circuit.append("X_ERROR", stab + len(stab_d_x), min(p_phenom_meas*len(stab_d_z[stab]),1)) 
                else:
                    loop_circuit.append("X_ERROR", stab + len(stab_d_x), p_meas) 

            loop_circuit.append("MR", full_stab_L) # measure the stabilizers at t>0

            # timelike detectors for the X or Z stabilizers
            for i in range(num_stabs):
                loop_circuit.append("DETECTOR", [stim.target_rec(-num_stabs + i), stim.target_rec(-2*num_stabs+ i)]) # stab round d tied to stab round d=0

            loop_circuit.append("TICK") # add a tick to the circuit to mark the end of the t>0 iteration
            
            if memory:
                # repeat the loop circuit d-1 times - circuit level only
                circuit.append(stim.CircuitRepeatBlock(repeat_count=(num_rounds-1), body=loop_circuit))# end the repeat block


        #
        # Stabilizer measurement reconstruction - perfect round
        #


        # reconstruct the stabilizers and measure the data qubits
        # for X mem measure X stabs
        if self.type == "X":
            # measure all the data qubits in the X stabilizers
            circuit.append("PAULI_CHANNEL_1", data_q_list, [px_meas, py_meas, pz_meas]) # apply biased depolarizing error on data qubits before measurement

            if CD_type != "SC":
                for q in CD_data_transform:
                    if CD_data_transform[q] == 2:
                        circuit.append("M", q + num_stabs)
                    elif CD_data_transform[q] == 0:
                        circuit.append("MX", q + num_stabs) # apply H to the qubits that have a transformation applied
            else:
                circuit.append("MX", data_q_list)

            # reconstruct each X stabilizer with a detector
            for stab in stab_d_x:
                q_x_list = stab_d_x[stab] # get the qubits in the stab
                detector_list =  [-num_qubits_x + q for q in q_x_list] + [-num_stabs + stab - num_qubits_x]
                circuit.append("DETECTOR", [stim.target_rec(d) for d in detector_list])
            
            
            # construct the logical observable to include - pick the column of qubits for X_L meas, protecting against Z errors
            circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-num_qubits_x + self.d*q) for q in range(self.d)], 0)
        
        # Z mem measure Z stabs
        if self.type == "Z":
            # measure all the data qubits in the Z stabilizers
            circuit.append("PAULI_CHANNEL_1", data_q_list, [px_meas, py_meas, pz_meas]) # apply biased depolarizing error on data qubits before measurement

            if CD_type != "SC":
                for q in CD_data_transform:
                    if CD_data_transform[q] == 2:
                        circuit.append("MX", q+num_stabs)
                    elif CD_data_transform[q] == 0:
                        circuit.append("M", q + num_stabs) # apply H to the qubits that have a transformation applied
            else:
                circuit.append("M", data_q_list)

            # reconstruct each stabilizer with a detector
            for stab in stab_d_z: 
                
                q_z_list = stab_d_z[stab] # get the qubits in the stab
                detector_list =  [-num_qubits_z + q for q in q_z_list] + [-num_stabs +len(stab_d_x)+ stab - num_qubits_z]
                circuit.append("DETECTOR", [stim.target_rec(d) for d in detector_list])
        
            # construct the logical observable to include - pick the top line of qubits since this is an Z_L meas, protecting against X errors
            circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-num_qubits_z + q) for q in range(self.d)], 0)
        return circuit
