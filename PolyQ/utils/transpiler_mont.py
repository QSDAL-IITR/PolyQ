"""
The function transpile_montennaro takes a circuit with 
{H, X, Z, CZ, CCZ, CX, CCX} gates and converts it to montennaro 
gate set {H, Z, CZ, CCZ}. It goes through all the gates in sequence
of application and creates a new transpiled circuit.

"""

### Transpilation
from qiskit import QuantumCircuit

def transpile_montennaro(circuit):

    tracker_array = [[] for k in range(circuit.width())]

    new_circuit = QuantumCircuit(circuit.width())

    def addH(qubits):

        # '''
        print(tracker_array[qubits[0]])
        print(tracker_array[qubits[0]][-1])
        print(tracker_array[qubits[0]][-1][0:-1])
        # '''

        try:
            ind = tracker_array[qubits[0]][-1][-1]
            if tracker_array[qubits[0]][-1][0:-1] == ['h',qubits]:
                del(new_circuit.data[ind])
                a = tracker_array[qubits[0]].pop()
                if a == [[0,0,0]]: tracker_array[qubits[0]].append(a)
            else:
                new_circuit.h(qubits)
                tracker_array[qubits[0]].append(['h',qubits,len(new_circuit.data)-1])
        except IndexError:
            #print('IndexError Exception')
            new_circuit.h(qubits)
            tracker_array[qubits[0]].append(['h',qubits,len(new_circuit.data)-1])

    def addZ(qubits): #Cautious hadamard add - will check if previous element is H and will delete instead
        try:
            ind = tracker_array[qubits[0]][-1][-1]
            if tracker_array[qubits[0]][-1][0:2] == ['z',qubits]:
                del(new_circuit.data[ind])
                #tracker_array[qubits[0]].pop()
                a = tracker_array[qubits[0]].pop()
                if a == [[0,0,0]]: tracker_array[qubits[0]].append(a)
            else:
                new_circuit.z(qubits)
                tracker_array[qubits[0]].append(['z',qubits,len(new_circuit.data)-1])
        except IndexError:
            #print('IndexError Exception')
            new_circuit.z(qubits)
            tracker_array[qubits[0]].append(['z',qubits,len(new_circuit.data)-1])

    def addX(qubits):
        addH(qubits)
        addZ(qubits)
        addH(qubits)

    def addCZ(qubits): #Cautious hadamard add - will check if previous element is H and will delete instead
        try:
            ind = tracker_array[qubits[0]][-1][-1]


            if (tracker_array[qubits[0]][-1][0:-1] == ['cz',qubits]) and (tracker_array[qubits[1]][-1][0:-1] == ['cz',qubits]):

                del(new_circuit.data[ind])
                #tracker_array[qubits[0]].pop()
                a = tracker_array[qubits[0]].pop()
                if a == [[0,0,0]]: tracker_array[qubits[0]].append(a)
                #tracker_array[qubits[1]].pop()
                a = tracker_array[qubits[1]].pop()
                if a == [[0,0,0]]: tracker_array[qubits[1]].append(a)
            else:

                control = qubits[0]
                target = qubits[1]
                new_circuit.cz(control,target)
                tracker_array[control].append(['cz',qubits,len(new_circuit.data)-1])
                tracker_array[target].append(['cz',qubits,len(new_circuit.data)-1])

        except IndexError:
            #print('IndexError Exception')
            control = qubits[0]
            target = qubits[1]
            new_circuit.cz(control,target)
            tracker_array[control].append(['cz',qubits,len(new_circuit.data)-1])
            tracker_array[target].append(['cz',qubits,len(new_circuit.data)-1])

    def addCCZ(qubits): #Cautious hadamard add - will check if previous element is H and will delete instead
        try:
            ind = tracker_array[qubits[0]][-1][-1]

            if (tracker_array[qubits[0]][-1][0:-1] == ['ccz',qubits]) and (tracker_array[qubits[1]][-1][0:-1] == ['ccz',qubits]) and (tracker_array[qubits[2]][-1][0:-1] == ['ccz',qubits]):
                del(new_circuit.data[ind])
                #tracker_array[qubits[0]].pop()
                a = tracker_array[qubits[0]].pop()
                if a == [[0,0,0]]: tracker_array[qubits[0]].append(a)
                #tracker_array[qubits[1]].pop()
                a = tracker_array[qubits[1]].pop()
                if a == [[0,0,0]]: tracker_array[qubits[1]].append(a)
                #tracker_array[qubits[2]].pop()
                a = tracker_array[qubits[2]].pop()
                if a == [[0,0,0]]: tracker_array[qubits[2]].append(a)
            else:
                control1 = qubits[0]
                control2 = qubits[1]
                target = qubits[2]
                new_circuit.ccz(control1,control2,target)
                tracker_array[control1].append(['ccz',qubits,len(new_circuit.data)-1])
                tracker_array[control2].append(['ccz',qubits,len(new_circuit.data)-1])
                tracker_array[target].append(['ccz',qubits,len(new_circuit.data)-1])
        except IndexError:
            #print('IndexError Exception')
            control1 = qubits[0]
            control2 = qubits[1]
            target = qubits[2]
            new_circuit.ccz(control1,control2,target)
            tracker_array[control1].append(['h',qubits,len(new_circuit.data)-1])
            tracker_array[control2].append(['h',qubits,len(new_circuit.data)-1])
            tracker_array[target].append(['h',qubits,len(new_circuit.data)-1])

    def addCX(qubits):
        target = [qubits[-1]]
        addH(target)
        addCZ(qubits)
        addH(target)

    def addCCX(qubits):
        target = [qubits[-1]]
        addH(target)
        addCCZ(qubits)
        addH(target)


    for index, instruction in enumerate(circuit.data):

        qubits = [circuit.find_bit(q).index for q in instruction.qubits]

        match instruction.operation.name:
            case 'h' : addH(qubits)
            case 'x' : addX(qubits)
            case 'z' : addZ(qubits)
            case 'cz': addCZ(qubits)
            case 'ccz': addCCZ(qubits)
            case 'cx': addCX(qubits)
            case 'ccx': addCCX(qubits)
        print(new_circuit)
    return new_circuit

