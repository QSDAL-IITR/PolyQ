### Range of n, h, g, d for random circuits

====== NEW ======

random_circuit_d_const ->
n -> (1, 30)
d -> (1, 30)
g -> arbitrary
h -> arbitrary

random_circuit_g_const ->
n -> (1, 30)
g -> (1, 500)
d -> arbitrary
h -> arbitrary

random_circuit_h_const ->
n -> (3, 100)
h -> (1, 100)
g -> arbitrary
d -> arbitrary

For PolySim -> limit `h` to 25
For DDSIM and Aer -> limit `n` to 25


====== h_prob ======

The probability of encountering an H gate in a random quantum circuit could range from 10% to 30% depending on the algorithm and its complexity.
    Lower bound (10%): In algorithms focused more on error correction or where other gates are more frequent.
    Upper bound (30%): In algorithms like Grover’s and Shor’s where the H gate plays a central role.

We will run the benchmark with range 5% to 40% with step size of 2.5%. That is 15 steps.



### Benchmark assumptions
1. initial condition is assumed to be state zero. If any other state is given then we can simply add X gates on corresponding qubits. But new_poly won't work in this case.
2. 
