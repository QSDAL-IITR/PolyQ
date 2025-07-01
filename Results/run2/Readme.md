This directory contains all the results we used for benchmarking our simulator against Qiskit's Aer simulator and MQT's DDSIM simulator. 

Keep in mind that Qiskit's Aer utilizes multiple cores but MQT's DDSIM and PolyQ runs on single core as of the time of benchmark.

> We used AMD Ryzen™ Threadripper™ PRO 9995WX processor which has 96 cores and 192 threads paired with 512 GB memory.

`arbitrary_h`: contains results for the random circuits with fixed number of H gates in range 1 to 29.

We did not run benchmark for any other type of random circuits, i.e. arbitrary depth or arbitrary gates, as the simulation time for all the 3 simulators eponentially depends either on `n` or `h`. 

Simulation time depends linearly on `g` (number of total gates) for all 3 simulators. 

> The program used for this benchmarks is same as in `demo.ipynb` file with slight modifications.




