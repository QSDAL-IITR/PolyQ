## Range of n, h, g, d etc. for random circuits and run2 results

| Function Name             | n Range     | d Range     | g Value     | h Value     |
|--------------------------|-------------|-------------|-------------|-------------|
| `random_circuit_d_const` | (1, 30)     | (1, 30)     | arbitrary   | arbitrary   |
| `random_circuit_g_const` | (1, 30)     | arbitrary   | (1, 500)    | arbitrary   |
| `random_circuit_h_const` | (3, 99)    | arbitrary   | arbitrary   | (1, 29)    |

All ranges are inclusive. 

For PolySim -> limit `h` to 25.

For DDSIM and Aer -> limit `n` to 25.


`h_prob`

The probability of encountering an H gate in a random quantum circuit could range from 10% to 30% depending on the algorithm and its complexity.

- Lower bound (10%): In algorithms focused more on error correction or where other gates are more frequent.
- Upper bound (30%): In algorithms like Grover’s and Shor’s where the H gate plays a central role.

We planned to run the benchmarks for range 5% to 40% with step size of 2.5%. That is 15 steps. Though the results came independant of `h_prob` because the gate count dependence is linear, we only ran the benchmark till 20%. 



### Benchmark assumptions
1. Initial condition is assumed to be state zero. If any other state is given then we can simply add X gates on corresponding qubits or modify the polynomial equation which is linear in gate count.
2. 
