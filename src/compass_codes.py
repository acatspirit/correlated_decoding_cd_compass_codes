'''
This file will contain all functions created to calculate logical error rates for Surface Codes and Compass Codes.
There are also functions that can apply Clifford Deformations.
Author: Julie Campos
Date: March 17, 2023
'''

# from typing import Callable, List
import random
from operator import itemgetter
import numpy as np
from pymatching import Matching
from scipy.sparse import csr_matrix


# General Functions
def parity_matrix(stabs, data_qbits):
    '''
    The purpose of this function is to create parity check matrices given appropriate set of stabilizer and
    data qubit dictionaries.
    Input:
        stabs: Stabilizer dictionary with keys indicating a single stabilizer location and values are
        list of data qubits the checks act on.
        data_qbits: dictionary whose keys are the position of the data qubit and the value is its label
    Outputs:
        H: Parity check matrices with rows corresponding to stabilizers and columns correspond to qubits
    '''
    ones_list = [] # Number of non-zero entries
    ind = [] # position of non-zero entry
    row_vals = [0] # distance between non-zero entries
    for val in list(stabs.values()):
        count = 0
        for inds in val:
            ones_list.append(1)
            ind.append(inds)
            count += 1
        row_vals.append(row_vals[-1]+count)

    sh = (len(stabs),len(data_qbits))
    H = csr_matrix((ones_list, ind,row_vals), shape = sh)
    return H


def prob_comb(probs):
    '''
    This function calculates the total probability of error for some list of probabilities, probs. This is
    useful in combining probabilites for merged edges.
    Inputs:
        probs: List of probabilities where each element corresponds to some edge in the matching graph
    Outputs:
        totalp: List of total probabilities for each edge in the matching graph
    '''
    totalp = 0  # Total probability

    for p in probs:
        tot = p
        xprobs = probs.copy()
        xprobs.remove(p)
        for xp in xprobs:
            tot *= 1 - xp
        totalp += tot

    return totalp

# Surface Code Class

# Compass Code Class
# Define blue and red checkerboard of size LxL
def CompassStabs(L, **kwargs):
    '''
    Inputs:
        L: size of lattice (LxL)
        kwargs:
            px: percentage of lattice to be red (Z stabilizers)
            pz: percentage of lattice to be blue (X stabilizers)
            l: Defines distance between blue cells (l = 2 corresponds to surface code)
    Outputs:
        stabs: Dictionary of X and Z stabilizers
    '''

    L_c = L - 1  # Length of colored blocks

    qcols, qrows = np.meshgrid(range(L), range(L))
    ccols, crows = qcols[0:-1, 0:-1], qrows[0:-1, 0:-1]

    qbit_dict = {}
    for (i, j, n) in zip(np.ndarray.flatten(qrows), np.ndarray.flatten(qcols), range(L ** 2)):
        qbit_dict[(i, j)] = n

    # Need L-1 x L-1 colored blocks
    colors = np.zeros((L_c, L_c))
    if 'l' in kwargs.keys():
        l = kwargs['l']
        for i in range(L_c):
            for j in range(L_c):
                if (i - j) % l == 0:
                    colors[i, j] = 1
    else:
        px = kwargs['px']
        pz = kwargs['pz']
        for i in range(L_c):
            for j in range(L_c):
                np.random.seed()
                colors[i, j] = np.random.choice([0, 1], p=[px, pz])  # red = 0, blue = 1

    # Combine blues going vertically
    x_stabs = []
    b = 2  # Reset

    for j in range(L_c):
        if b == 0:
            qs = set(itemgetter((i + 1, j - 1), (i + 1, j))(qbit_dict))
            x_stabs.append(list(qs))
        b = 2
        for i in range(L_c):
            if colors[i, j] == 1:  # If blue block
                if b == 0 or b == 2:  # Wasn't blue before
                    qs = itemgetter((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))(qbit_dict)
                    qs = set(list(qs))
                else:
                    qs = qs | set(itemgetter((i + 1, j), (i + 1, j + 1))(qbit_dict))
                b = 1
                if i == L_c - 1:
                    x_stabs.append(list(qs))
            else:  # Then red block
                if b == 1:
                    x_stabs.append(list(qs))  # Dump blue data, don't add more stabilizers
                else:
                    qs = set(itemgetter((i, j), (i, j + 1))(qbit_dict))
                    x_stabs.append(list(qs))
                b = 0
    if b == 0:
        qs = set(itemgetter((i + 1, j), (i + 1, j + 1))(qbit_dict))
        x_stabs.append(list(qs))

    # Combine reds going horizontally
    z_stabs = []
    r = 2  # Reset

    for i in range(L_c):
        if r == 0:
            qs = set(itemgetter((i - 1, j + 1), (i, j + 1))(qbit_dict))
            z_stabs.append(list(qs))
        r = 2
        for j in range(L_c):
            if colors[i, j] == 0:  # If red block
                if r == 0 or r == 2:  # Wasn't red before
                    qs = itemgetter((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))(qbit_dict)
                    qs = set(list(qs))
                else:
                    qs = qs | set(itemgetter((i, j + 1), (i + 1, j + 1))(qbit_dict))
                r = 1
                if j == L_c - 1:
                    z_stabs.append(list(qs))
            else:  # Then blue block
                if r == 1:
                    z_stabs.append(list(qs))  # Dump red data, don't add more stabilizers
                else:
                    qs = set(itemgetter((i, j), (i + 1, j))(qbit_dict))
                    z_stabs.append(list(qs))
                r = 0

    if r == 0:
        qs = set(itemgetter((i, j + 1), (i + 1, j + 1))(qbit_dict))
        z_stabs.append(list(qs))

    return x_stabs, z_stabs, qbit_dict, colors
    # Combine red going horizontally


class CompassCode:
    def __init__(self, d, **kwargs):
        '''
        Construct a class object containing information for a compass code of distance d. User can specify
        how to fix gauges by specifying l, or px & pz which state the probability of having X check blocks or
        Z check blocks.
        '''

        random.seed()
        x_stabs, z_stabs, qbit_dict, colors = CompassStabs(d, **kwargs)
        # Resulting colors: 1 = blue, 0 = red
        self.stabs = {'X': {}, 'Z': {}}
        for i in range(len(x_stabs)):
            self.stabs['X'][i] = x_stabs[i]
        for i in range(len(z_stabs)):
            self.stabs['Z'][i] = z_stabs[i]

        # Make Generator matrices
        Hx = parity_matrix(self.stabs['X'], qbit_dict)
        Hz = parity_matrix(self.stabs['Z'], qbit_dict)

        # Labels for matching graph in addition to boundary nodes needed
        xs = self.stabs['X']
        zs = self.stabs['Z']

        edges_e2v = {'X': {}, 'Z': {}}
        for q in qbit_dict.values():
            edges_e2v['X'].setdefault(q, [])
            edges_e2v['Z'].setdefault(q, [])
            for x in xs.keys():
                if q in xs[x]:
                    edges_e2v['X'][q].append(x)
            for z in zs.keys():
                if q in zs[z]:
                    edges_e2v['Z'][q].append(z)
        self.edge2vertices = edges_e2v
        boundary_nodes = {}
        for stabtype in edges_e2v.keys():
            for k in edges_e2v[stabtype].keys():
                if len(edges_e2v[stabtype][k]) == 1:  # If only one vertex
                    boundary_nodes.setdefault(stabtype, max(self.stabs[stabtype].keys()) + 1)
                    edges_e2v[stabtype][k].append(boundary_nodes[stabtype])

        # Final edges including boundary nodes
        edges = {'X': {}, 'Z': {}}
        for stabtype in ['X', 'Z']:
            for k1 in edges_e2v[stabtype].keys():
                e = tuple(edges_e2v[stabtype][k1])
                edges[stabtype].setdefault(e, [[]])
                edges[stabtype][e][0].append(k1)

        logicals = {'X': np.zeros(len(qbit_dict), dtype=int), 'Z': np.zeros(len(qbit_dict), dtype=int)}
        for j in [d * i for i in range(0, d)]:
            logicals['X'][j] = 1
        for i in range(0, d):
            logicals['Z'][i] = 1

            # Class Attributes
        self.qbit_dict = qbit_dict
        self.boundary_nodes = boundary_nodes
        self.edges = edges
        self.H = {'X': Hx, 'Z': Hz}
        self.logicals = logicals
        self.colors = colors

    def add_weight(self, **kwargs):
        '''
        This method updates edges and edge2vertices attributes of this class object to include error
        probabilities. These error probabilities may be p (same for X and Z errors) or px (pz) probability
        of X (Z) errors.
        Input:
            p: array with probability of error on each qubit, same for X and Z errors
            px: array with probability of X error on each qubit
            pz: array with probability of Z error on each qubit
        Outputs:
            self.edges: Updated edges now containing appropriate weights.
        '''
        if 'p' in kwargs.keys():
            probs = p.copy()

        edge_weights = {'X': {}, 'Z': {}}
        for stabtype in ['X', 'Z']:
            # Check if probabilities for Z errors are different than those for X errors
            if 'px' in kwargs.keys() and stabtype == 'Z':  # probabilities of X errors increase weights of Z checks
                px = kwargs['px']
                probs = px.copy()
            if 'pz' in kwargs.keys() and stabtype == 'X':
                pz = kwargs['pz']
                probs = pz.copy()

            for k1 in self.edge2vertices[stabtype].keys():  # Look through data qubits
                e = tuple(self.edge2vertices[stabtype][k1])  # corresponding stabilizer vertices
                edge_weights[stabtype].setdefault(e, [])  # Initialize list of weights
                edge_weights[stabtype][e].append(probs[k1])  # For each vertex, add weight of qubit if its in edge
                
            for edge in edge_weights[stabtype].keys():  # Go through vertex pairs
                if len(self.edges[stabtype][edge]) == 1:  # if no prior weights, we append list
                    self.edges[stabtype][edge].append([prob_comb(edge_weights[stabtype][edge])])
                elif len(self.edges[stabtype][edge]) == 2:  # if prior weight, we write over them
                    self.edges[stabtype][edge][1] = [prob_comb(edge_weights[stabtype][edge])]

        return self.edges

    def make_graph_withweights(self, checktype, ps = [1]):
        # kwargs: 'p', 'px','pz'. Should just have the probabilities that would affect the checktype. 
        # Thus, use 'px' if checktype = Z and 'pz' if checktype = X
        # Don't need to use add_weights here.
        m = Matching()
        for e, info in self.edges[checktype].items(): # Go through the stabilizers (pairs of vertices)
            for q in set(info[0]): # Go through qubits  to get minimum weight
                if len(ps) == 1:
                    w = 1
                else:
                    p = ps[q]
                    w = np.log((1 - p) / p)
                m.add_edge(e[0], e[1], fault_ids=q, weight=w, merge_strategy = "smallest-weight")  # fault_ids changed from set(info[0])
        m.set_boundary_nodes(set([self.boundary_nodes[checktype]]))
        return m
    
    def make_graph(self, checktype):
        # First make labels for stabilizers
        easy_vlabels = {}
        for stabname in self.edges[checktype].keys():
            for i in stabname:
                easy_vlabels.setdefault(i, )

        n = -1
        for key in np.sort(list(easy_vlabels.keys())):
            n += 1
            easy_vlabels[key] = n

        m = Matching()
        for e, info in self.edges[checktype].items():
            print("vertices: ",easy_vlabels[e[0]], easy_vlabels[e[1]])
            print("fault_ids: ", set(info[0]))
            if len(info) == 2:
                p = info[1][0]
                w = np.log((1 - p) / p)
            else:
                w = 1
            m.add_edge(easy_vlabels[e[0]], easy_vlabels[e[1]], fault_ids=max(set(info[0])), weight=w,
                       error_probability=p, merge_strategy = "smallest-weight")  # fault_ids changed from set(info[0])
            # Note here that we get the max of the fault_ids because of the case when error is on weight-2
            # stabilizer, we only have to apply a weight one correction.
        m.set_boundary_nodes(set([easy_vlabels[self.boundary_nodes[checktype]]]))
        return m
# Application of Clifford Deformations
# Note that all of the following functions will need "CD_data" which is a dictionary containing information on
# Clifford Deformations that will be applied.
#    CD_data keys: qubit number
#    CD_data values: 0 (identity), 1 (H_YZ) or 2 (Hadamard)
def Clifford_vectorTrans(xvec_i, zvec_i, CD_data):
    '''
     Inputs:
         xvec_i: Input binary vector with X information (could be a noise vector, state vector, operator)
         zvec_i: Input binary vector with Z information (could be a noise vector, state vector, operator)
         CD_data: dictionary with keys being the qubit position in the vector and the value is the type of
             Clifford applied represented by 0 (Identity), 1 (H_YZ), and 2 (Hadamard H_XZ).
     Outputs:
         xvec: Transformed xvec_i
         zvec: Transformed zvec_i
    '''

    # Transform bits
    xvec = xvec_i.copy()
    zvec = zvec_i.copy()

    for i, CD in CD_data.items():
        if CD == 0:
            None
        elif CD == 1:
            if zvec_i[i] == 1 and xvec_i[i] == 1:  # This is a Y, must go to Z
                xvec[i] = 0
                zvec[i] = 1
            elif zvec_i[i] == 1:  # This is a Z, must go to Y
                xvec[i] = 1
                zvec[i] = 1
        elif CD == 2:
            if zvec_i[i] == 1 and xvec_i[i] == 1:
                None
            elif zvec_i[i] == 1:
                xvec[i] = 1
                zvec[i] = 0
            elif xvec_i[i] == 1:
                xvec[i] = 0
                zvec[i] = 1

    return xvec, zvec


def Correction_Clifford(xnoise_i, znoise_i, g, mX, mZ, Hx, Hz, CD_data):
    '''
    Corrects input noise vectors according to the Clifford deformation specified in dictionary CD_data.
    Input:
        xnoise_i: Input X noise that Clifford Deformed code will see
        znoise_i: Input Z noise that Clifford Deformed code will see
        g: Pre-deformed code class object that will be used to correct errors. Contains stabilizer information
        that is not deformed yet.
        CD_data: Clifford Deformation information for each qubit
    Output:
        xerror: Overall X error on deformed code after correction
        zerror: Overall Z error on deformed code after correction
    '''
    # We assume that the incoming noise is detected by Clifford deformed code, so we undo it here to
    # use CSS-type decoding
    # logX, logZ = g.logicals['Z'], g.logicals['X']  # Logical operators
    # L = len(g.qbit_dict)  # Number of data qubits

    # Transform bits
    xnoise, znoise = Clifford_vectorTrans(xnoise_i, znoise_i, CD_data)
    # Assume input are already in CSS frame, so inputs are xnoise, znoise instead of 
    # xnoise_i and znoise_i

    # Obtain syndrome and correction
    zsynd = Hx @ znoise % 2
    zcorr = mX.decode(zsynd)

    xsynd = Hz @ xnoise % 2
    xcorr = mZ.decode(xsynd)

    # Overall error before final transformation, THIS IS IN CSS FRAME
    xerror, zerror = (xnoise + xcorr) % 2, (znoise + zcorr) % 2

    # # Transform resulting bits back to Clifford deformed frame
    # zcorr0 = zcorr.copy()
    # xcorr0 = xcorr.copy()

    # Correction in Clifford deformed code
    # xcorr, zcorr = Clifford_vectorTrans(xcorr0, zcorr0, CD_data)
    # # Calculate overall error
    # xerror, zerror = (xnoise_i + xcorr) % 2, (znoise_i + zcorr) % 2

    return xerror, zerror

# Error Correction functions counting number of logical errors
def num_decoding_failures_CD(g, mX, mZ, px, pz, num_trials, CD_data):
    '''
    This function counts the logical errors on a Clifford deformed code.
    Inputs:
        g: Code Class object (not deformed)
        px: Vector whose components "i" are probability of X error on qubit "i"
        pz: Vector whose components "i" are probability of Z error on qubit "i"
        num_trials: Number of times to do error correction
        CD_data: Dictionary with Clifford Deformation data
    Outputs:
        num_errors: Total number of logical errors
        num_xerrors: Total number of X logical errors
        num_zerrors: Total number of Z logical errors
    '''
    num_errors, num_zerrors, num_xerrors = 0, 0, 0
    # Transforming logical operators
    logX, logZ = g.logicals['X'], g.logicals['Z']
    num_qbits = len(g.qbit_dict)
    logX_xpart, logX_zpart = Clifford_vectorTrans(logX, np.zeros(num_qbits), CD_data)
    logZ_xpart, logZ_zpart = Clifford_vectorTrans(np.zeros(num_qbits), logZ, CD_data)
    # Make matching graphs with appropriately weighted edges
    # mX, mZ = g.make_graph('X'), g.make_graph('Z')  # graph objects
    # Parity Check Matrices
    Hx, Hz = g.H['X'].todense(), g.H['Z'].todense()  # Parity Check matrices
    
    for i in range(num_trials):
        # Initial noise
        # Noise vector: Each qubit can have no error (0), X error (1), Z error (2), or Y error (3)
        noise = np.random.choice(np.arange(4), p = [1-(2*px+pz), px, pz, px], size = num_qbits)

        # Creating noise vectors in the CSS frame
        # noise = []
        # for q in g.qbit_dict.values():
        #     err = np.random.choice(np.arange(4),p = [1-2*pxs[q]-pzs[q], pxs[q], pzs[q], pxs[q]])
        #     noise.append(err)
        
        # Split into X and Z
        xnoise_i = np.zeros(num_qbits)
        znoise_i = np.zeros(num_qbits)
        for q in range(num_qbits):
            if noise[q] == 1:
                xnoise_i[q] = 1
            elif noise[q] == 2:
                znoise_i[q] = 1
            elif noise[q] == 3:
                xnoise_i[q] = 1
                znoise_i[q] = 1

        # Corrected noise
        xerror, zerror = Correction_Clifford(xnoise_i, znoise_i, g, mX, mZ, Hx, Hz, CD_data)

        # Check for logical errors in CSS frame
        if np.any((zerror @ logX.T ) % 2) or np.any((xerror @ logZ.T) % 2):
            num_errors += 1
            if np.any((zerror @ logX.T ) % 2):
                num_zerrors += 1
            if np.any((xerror @ logZ.T) % 2):
                num_xerrors += 1
        
        # if np.any((zerror @ logX_xpart.T + xerror @ logX_zpart.T) % 2) or np.any(
        #         (xerror @ logZ_zpart.T + zerror @ logZ_xpart.T) % 2):
        #     num_errors += 1
        #     if np.any((zerror @ logX_xpart.T + xerror @ logX_zpart.T) % 2):
        #         num_zerrors += 1
        #     if np.any((xerror @ logZ_zpart.T + zerror @ logZ_xpart.T) % 2):
        #         num_xerrors += 1

    return num_errors, num_xerrors, num_zerrors

# Only total decoding failures
def num_decoding_failures_CD_totalerrors(g, px, pz, num_trials, CD_data):
    '''
    This function counts the logical errors on a Clifford deformed code.
    Inputs:
        g: Code Class object (not deformed)
        px: Vector whose components "i" are probability of X error on qubit "i"
        pz: Vector whose components "i" are probability of Z error on qubit "i"
        num_trials: Number of times to do error correction
        CD_data: Dictionary with Clifford Deformation data
    Outputs:
        num_errors: Total number of logical errors
        num_xerrors: Total number of X logical errors
        num_zerrors: Total number of Z logical errors
    '''
    num_errors = 0
    # Transforming logical operators
    logX, logZ = g.logicals['X'], g.logicals['Z']
    num_qbits = len(g.qbit_dict)
    logX_xpart, logX_zpart = Clifford_vectorTrans(logX, np.zeros(num_qbits), CD_data)
    logZ_xpart, logZ_zpart = Clifford_vectorTrans(np.zeros(num_qbits), logZ, CD_data)
    # Make matching graphs with appropriately weighted edges
    mX, mZ = g.make_graph('X'), g.make_graph('Z')  # graph objects
    # Parity Check Matrices
    Hx, Hz = g.H['X'].todense(), g.H['Z'].todense()  # Parity Check matrices
    
    for i in range(num_trials):
        # Initial noise
        print("MC {}/{}".format(i, num_trials))
        # Noise vector: Each qubit can have no error (0), X error (1), Z error (2), or Y error (3)
        noise = np.random.choice(np.arange(4), p = [1-(2*px+pz), px, pz, px], size = num_qbits)
        xnoise_i = np.zeros(num_qbits)
        znoise_i = np.zeros(num_qbits)

        for q in range(num_qbits):
            if noise[q] == 1:
                xnoise_i[q] = 1
            elif noise[q] == 2:
                znoise_i[q] = 1
            elif noise[q] == 3:
                xnoise_i[q] = 1
                znoise_i[q] = 1

        # Corrected noise
        xerror, zerror = Correction_Clifford(xnoise_i, znoise_i, g, mX, mZ, Hx, Hz, CD_data)

        # Check for logical errors
        if np.any((zerror @ logX_xpart.T + xerror @ logX_zpart.T) % 2) or np.any(
                (xerror @ logZ_zpart.T + zerror @ logZ_xpart.T) % 2):
            num_errors += 1
    print("------------Done------------")
    return num_errors

# As L changes, the Clifford deformation dictionary has to change.
def CD_data_func(qbits, **kwargs):
    '''
    Created Clifford Deformation Dictionary
    Inputs:
        qbits: list or array of qubit labels
        kwargs:
            'type': XZZX, XY
    Outputs:
        CD_data: Dictionary whose keys are qubit labels and values are 0,1,2 indication Clifford Deformation
    '''
    CD_data = {}  # Initialize dictionary
    for q in qbits:
        CD_data[q] = 0
        
    if 'type' in kwargs.keys():
        if kwargs['type'] == 'XZZX':
            for q in qbits:
                if (q % 2) == 0:
                    CD = 0
                elif (q % 2) == 1:
                    CD = 2
                CD_data[q] = CD
        elif kwargs['type'] == 'XY':
            for q in qbits:
                CD_data[q] = 2
        elif kwargs['type'] == 'I':
            return CD_data
        else:
            ValueError("Invalid 'type' argument. Enter either XZZX, XY, or I.")
    elif 'P_ZX' in kwargs.keys():
        P_ZX = kwargs['P_ZX']
        P_ZY = kwargs['P_ZY']
        for q in qbits:
            CD_data[q] = np.random.choice([0, 1, 2], p=[1 - P_ZX - P_ZY, P_ZY, P_ZX])
    elif 'ell' and 'special' in kwargs.keys():
        # Applies XZZX on squares 
        ell = kwargs['ell']
        size = kwargs['size']
        for q in qbits:
            CD_data[q] = 0
        if kwargs['special'] == 'XZZXonSqu':
            for i in np.arange(0,size-1):
                for j in np.arange(0,size-1):
                    if (i-j)%ell == 0:
                        # (i+1,j)
                        p2 = (i+1)*(size) + j
                        # (i,j+1)
                        p3 = i*(size) + j + 1
                        CD_data[p2], CD_data[p3]= 2,2 # Applying a Hadamard                
        elif kwargs['special'] == 'ZXXZonSqu':
            for i in np.arange(0,size-1):
                for j in np.arange(0,size-1):
                    if (i-j)%ell == 0:
                    # (i,j)
                        p1 = i*(size) + j
                        # (i+1,j+1)
                        p4 = (i+1)*(size) + j + 1 
                        CD_data[p1], CD_data[p4]= 2,2 # Applying a Hadamard  
            if (size-1)%ell == 0:
                CD_data[size-1] = 2
                CD_data[size**2-size] = 2
    return CD_data


def CDonCompassCode_Simulate(Ls, bias, num_trials, ps, **kwargs):
    '''
    This function calculates logical error rate of compass code.
    Input:
        Ls: distances of compass code
        bias: Z bias, depolarizing is 0.5
        num_trials: Number of shots
        ps: list of physical error rates
        **kwargs: Must include either l or p_blue & p_red for compass code creation. Must also include the Clifford
        deformation type ('XZZX' or 'XY') or P_ZX and P_ZY which give probabilities of applying H_ZX and H_ZY respectively.

    Output:
        log_errors_all_L: A list with logical error rates. First index corresponds to L, second index corresponds to p.
    '''
    # np.random.seed(2)

    log_errors_all_L = []
    log_xerrors_all_L = []
    log_zerrors_all_L = []
    for L in Ls:
        print("Simulating L={}...".format(L))
        g = CompassCode(L, **kwargs)
        kwargs['size'] = L
        num_qbits = len(g.qbit_dict)
        log_errors = []
        xlog_errors = []
        zlog_errors = []
        CD_data = CD_data_func(range(num_qbits), **kwargs)
        print(CD_data)
        for p in ps:
            pz = bias * p / (1 + bias)
            px = (p - pz) / 2
            pxs = np.ones(num_qbits) * px
            pzs = np.ones(num_qbits) * pz
            pxs_copy = pxs.copy()
            pzs_copy = pzs.copy()

            # Clifford Deformations change the weights
            for i in range(num_qbits):
                if CD_data[i] == 0:
                    pass
                elif CD_data[i] == 1:
                    pxs[i] = pxs_copy[i]
                    pzs[i] = pxs_copy[i]
                elif CD_data[i] == 2:
                    pxs[i] = pzs_copy[i]
                    pzs[i] = pxs_copy[i]

            # g.add_weight(px=pxs, pz=pzs) # Insert this if you want to use the make_graph method, else we use make_graph_withweights
            mX, mZ = g.make_graph_withweights('X',ps = pzs), g.make_graph_withweights('Z',ps = pxs)
            # num_errors = num_decoding_failures_CD_totalerrors(g, px, pz, num_trials, CD_data)
            num_errors, num_xerrors, num_zerrors = num_decoding_failures_CD(g,mX,mZ,px,pz,num_trials,CD_data)
            log_errors.append(num_errors / num_trials)
            xlog_errors.append(num_xerrors / num_trials)
            zlog_errors.append(num_zerrors / num_trials)

        log_errors_all_L.append(np.array(log_errors))
        log_xerrors_all_L.append(np.array(xlog_errors))
        log_zerrors_all_L.append(np.array(zlog_errors))

    return log_errors_all_L, log_xerrors_all_L, log_zerrors_all_L

def Thresh_function(ell, bias, delta, p0, num_trials, **kwargs):
    # Function for finding the threshold using binary search of intersection
    # CD: Clifford Deformation vector
    # ell: elongation parameter
    # delta: accuracy of threshold
    # num_trials: number of Monte Carlo runs
    # p0: Initial guess of threshold
    
    Ls = [13, 19]
    # Check as difference changes sign until the difference between 
    # ps is small enough
    shift = 0.01
    k = -1
    kwargs['l'] = ell
    print("Elongation: ", ell)
    data = []
    log_errors = CDonCompassCode_Simulate(Ls, bias, num_trials, ps = [p0], **kwargs)
    data.append([p0, log_errors])
    diff = log_errors[1]- log_errors[0]
    diff1 = diff
    while np.abs(diff) > delta: 
        k += 1
        while diff*diff1 > 0: 
            print("Difference: ", diff)
            if diff > 0: # We are above threshold
                print("Above threshold at: ", p0)
                p0 = p0 - shift/2**k
            if diff < 0: # We are below threshold
                print("Below threshold at: ", p0)
                p0 = p0 + shift/2**k
            # Find new difference between logical error rates
            log_errors = CDonCompassCode_Simulate(Ls, bias, num_trials, ps = [p0], **kwargs)
            data.append([p0, log_errors])
            diff1 = log_errors[1]- log_errors[0]
        diff = diff1
        print("Difference: ", diff)
    return p0, data
