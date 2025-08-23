# Core Logic File

"""
Core logic for simulating Clifford+S+T quantum circuits.

Provides functions for constructing polynomial representations of circuits,
evaluating polynomials, generating truth tables, and computing statevectors.
"""

import numpy as np, sys, psutil, time

def create_poly(qc, n: int):
    """
    Create the polynomial array representation of a quantum circuit.

    Args:
        qc: QuantumCircuit object.
        n (int): Number of qubits.

    Returns:
        tuple: (terms, wire_array, t) where terms is the polynomial, wire_array tracks variable names, and t is the total number of variables.
    """
    instructions = [(instruction.operation.name,
                    [qc.find_bit(q).index for q in instruction.qubits]) 
                    for index, instruction in enumerate(qc.data)]

    wire_array = [[i] for i in range(n)]
    max_new_var = n 
    terms = [] 

    for operation in instructions:
        gate = operation[0]
        qubits = operation[1]
        if gate == 'h':
            wire_array[qubits[0]].append(max_new_var)
            max_new_var += 1
            terms.append([4,[wire_array[qubits[0]][-2],wire_array[qubits[0]][-1]]])
        elif gate in ['z','cz','ccz']:
            terms.append([4,[wire_array[j][-1] for j in qubits]])
        elif gate == 's':
            terms.append([2,[wire_array[j][-1] for j in qubits]])
        elif gate == 't':
            terms.append([1,[wire_array[j][-1] for j in qubits]])
        elif gate == 'sdg':
            terms.append([6,[wire_array[j][-1] for j in qubits]])
        elif gate == 'tdg':
            terms.append([7,[wire_array[j][-1] for j in qubits]])
    t = max_new_var
    return terms, wire_array, t

def eval_f_no_ivs(terms, x, n):
    """
    Evaluate the polynomial equation without input variables.

    Args:
        terms (list): Polynomial terms.
        x (np.ndarray): Variable values.
        n (int): Number of qubits.

    Returns:
        np.int8: Evaluation result modulo 8.
    """
    val_out: np.int8 = 0
    for term in terms:
        weight = term[0]
        indices = term[1]
        v = bool(1)
        for j in indices:
            v &= x[j-n] 
        val_out = np.int8(val_out + weight*int(v))%8
    return val_out

def eval_f(terms, x, n):
    """
    Evaluate the polynomial equation.

    Args:
        terms (list): Polynomial terms.
        x (np.ndarray): Variable values.
        n (int): Number of qubits.

    Returns:
        np.int8: Evaluation result modulo 8.
    """
    val_out: np.int8 = 0
    for term in terms:
        weight = term[0]
        indices = term[1]
        v = bool(1)
        for j in indices:
            v &= x[j] 
        val_out = np.int8(val_out + weight*int(v))%8
    return val_out

def get_truthtable(terms, n, t, initial_state):
    """
    Generate the truth table for the polynomial given an initial state.

    Args:
        terms (list): Polynomial terms.
        n (int): Number of qubits.
        t (int): Total number of variables.
        initial_state (list): Initial state of the qubits.

    Returns:
        np.ndarray: Truth table values.
    """
    if n == t:
        state = "".join([str(int(i)) for i in initial_state])
        return
    x_range = 2**(t-n) 
    ttb = np.empty(x_range, dtype=np.uint8)
    x = np.empty(t, dtype='bool')
    x[0:n] = initial_state
    for i in range(x_range):
        y_bin = bin(i)[2:].zfill(t-n)
        for ind, val in enumerate(y_bin):
            x[n+ind] = bool(int(val))
        ttb[i] = eval_f(terms, x, n)
    return ttb

def get_truthtable_no_ivs(terms, n, t, initial_state):
    """
    Generate the truth table, removing input variables from the polynomial.

    Args:
        terms (list): Polynomial terms.
        n (int): Number of qubits.
        t (int): Total number of variables.
        initial_state (list): Initial state of the qubits.

    Returns:
        np.ndarray: Truth table values.
    """
    if n == t:
        state = "".join([str(int(i)) for i in initial_state])
        return
    new_terms = []
    for term in terms:
        indices = term[1]
        remove_term = False
        for q in indices:
            if q < n:
                remove_term = True
                break
        if not remove_term:
            new_terms.append(term)
    x_range = 2**(t-n) 
    ttb = np.empty(x_range, dtype=np.uint8)
    x_no_ivs = np.empty(t-n, dtype='bool')
    for i in range(x_range):
        y_bin = bin(i)[2:].zfill(t-n)
        for ind, val in enumerate(y_bin):
            x_no_ivs[ind] = bool(int(val))
        ttb[i] = eval_f_no_ivs(new_terms, x_no_ivs, n)
    return ttb

def get_statevector(ttb, n, t, ovs, starting_index=0):
    """
    Compute the statevector from the truth table.

    Args:
        ttb (np.ndarray): Truth table values.
        n (int): Number of qubits.
        t (int): Total number of variables.
        ovs (list): Output variable indices.
        starting_index (int, optional): Starting index. Defaults to 0.

    Returns:
        np.ndarray: Statevector as a complex array.
    """
    s = np.zeros(2**len(ovs), dtype=complex)
    s_ldic = dict()
    for k in range(0, len(ttb)):
        t_val = ttb[k]
        chosenbits = "".join([ ( bin(k)[2:].zfill(t) )[j] for j in ovs ])
        chosen_int = int(chosenbits,2)
        if chosen_int not in s_ldic:
            s_ldic[chosen_int] = np.array([0,0,0,0,0,0,0,0]) 
        s_ldic[chosen_int][t_val] += 1
    for k in s_ldic:
        tmp0 = (s_ldic[k][1] - s_ldic[k][5])/np.sqrt(2) 
        tmp1 = (s_ldic[k][3] - s_ldic[k][7])/np.sqrt(2)
        s_ldic[k] = (s_ldic[k][0] - s_ldic[k][4]) + tmp0 - tmp1 + (1j)*( (s_ldic[k][2] - s_ldic[k][6]) + tmp0 + tmp1 ) 
        s[k] = s_ldic[k] 
    stvector = s / (2**0.5)**(t-n)
    return stvector

def get_statevector_file(ttb, n, t, ovs, starting_index=0):
    """
    Compute the statevector and write amplitudes to a file.

    Args:
        ttb (np.ndarray): Truth table values.
        n (int): Number of qubits.
        t (int): Total number of variables.
        ovs (list): Output variable indices.
        starting_index (int, optional): Starting index. Defaults to 0.

    Returns:
        str: Filename where the statevector is written.
    """
    s_ldic = dict()
    start = time.time()
    for k in range(0, len(ttb)):
        t_val = ttb[k]
        chosenbits = "".join([ ( bin(k)[2:].zfill(t) )[j] for j in ovs ])
        chosen_int = int(chosenbits,2)
        if chosen_int not in s_ldic:
            s_ldic[chosen_int] = np.array([0,0,0,0,0,0,0,0], dtype=np.int8) 
        s_ldic[chosen_int][t_val] += 1
    stvec_filename = "Results/demo/stvec_tmp.txt"
    with open(stvec_filename, 'w') as f:
        for k in s_ldic:
            tmp0: float = (s_ldic[k][1] - s_ldic[k][5])/np.sqrt(2) 
            tmp1: float = (s_ldic[k][3] - s_ldic[k][7])/np.sqrt(2)
            amp = (s_ldic[k][0] - s_ldic[k][4]) + tmp0 - tmp1 + (1j)*( (s_ldic[k][2] - s_ldic[k][6]) + tmp0 + tmp1 ) 
            amp = amp / (2**0.5)**(t-n)
            binary_k = format(k, f'0{len(ovs)}b')
            f.write(f"k: {binary_k}, amp: {amp}\n")
    return

