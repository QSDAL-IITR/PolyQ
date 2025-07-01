from .engine import *
from qiskit import QuantumCircuit

def simulate(qc: QuantumCircuit, initial_state: int | str | list[int] = 0):
    """
    Simulates a quantum circuit from a given initial state.

    Args:
        qc (QuantumCircuit): Quantum circuit to be simulated.
        initial_state (int | str | list[int]): Initial state of the qubits.

    Returns:
        list[complex]: The resulting statevector.
    """
    n = qc.width() 
    h = list(instrct.operation.name for _index, instrct in enumerate(qc.data)).count('h') 
    t = n + h 
    terms, wire_array, max_new_var = create_poly(qc, n)
    assert t == max_new_var, "Value of 't' != 'max_new_var' from the create_poly function."
    ovs = [j[-1] for j in wire_array]
    if isinstance(initial_state, int):
        assert initial_state < 2**n, "Initial state must be between 0 and 2^(n-1)."
        bin_str = bin(initial_state)[2:].zfill(n)
        initial_state = [int(b) for b in bin_str]
    elif isinstance(initial_state, str):
        assert len(initial_state) == n, "Initial state must be of size n."
        initial_state = [int(b) for b in initial_state]
    print(f"Initial state is: {initial_state}")
    ttb = get_truthtable(terms, n, t, initial_state)
    stvec = get_statevector(ttb, n, t, ovs)
    # del ttb, terms, wire_array, max_new_var
    return stvec