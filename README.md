# 🧮 PolyQ
### A novel approach to simulate Quantum Circuits using Boolean Polynomials.

The approach of this simulator is based on the paper first published by Ashley Montanaro in 2017. He, along with previous researchers, proved the connection between Boolean Polynomial and the following Clifford gate set: `{H, Z, CZ, CCZ}`. We extended this approach to include T and S gates, thus making it universal gate set. So the currently supported gate set is `{H, Z, CZ, CCZ, T, S, T†, S†}`
 


## 🛠️ Get Started 

For a demo, look at the [demo](./demo.ipynb) file. It shows how to simulate a random circuit with supported gate set using Qiskit's Aer, MQT's DDSIM and PolyQ. In the end, it shows how to simulate a circuit in one line and get its state vector.

PolyQ for Qiskit `QuantumCircuit` object is available via [PyPI](https://pypi.org/project/PolyQ/0.1.0/) for Linux, macOS, and Windows and supports Python 3.11 and higher.

``` bash
(venv) $ pip install polyq
```

The following code gives an example on the usage:
```python
from qiskit import QuantumCircuit
import PolyQ

# GHZ state: 
# |GHZ⟩ = (|000⟩ + |111⟩) / √2
# Using the property: X = HZH
circ = QuantumCircuit(3)
circ.h(0)
circ.h(1)
circ.cz(0, 1)
circ.h(1)
circ.h(2)
circ.cz(1, 2)
circ.h(2)

print(circ.draw(fold=-1))

st_vec = PolyQ.simulate(circ)

print(st_vec)
```

## 📖 References

The full list of references used in the development of PolyQ is available in [`references.bib`](./references.bib).  
This file contains BibTeX entries for academic papers, libraries, and resources cited in the project and the upcoming paper.


## 📚 Citation

If you want to cite the **PolyQ**, please use the following format or BibTeX entry:

PolyQ: A novel approach to simulate Quantum Circuits using Boolean Polynomials.  
Author(s): C. A. Jothishwaran, Aarav Ratra, Satyam Sonaniya   
Affiliation: IIT Roorkee  
Repository: https://github.com/QSDAL-IITR/PolyQ  
Version: v0.1.0  
DOI: *To be added*  
Preprint / Paper: *To be added*  
License: Proprietary

### BibTeX :
```bibtex
@misc{polyq2025,
  title        = {PolyQ: A novel approach to simulate Quantum Circuits using Boolean Polynomials.},
  author       = {C. A. Jothishwaran and Aarav Ratra and Satyam Sonaniya},
  year         = {2025},
  note         = {Version 0.1.0. Available at \url{https://github.com/QSDAL-IITR/PolyQ}},
  howpublished = {\url{https://github.com/QSDAL-IITR/PolyQ}},
  license      = {Proprietary},
}
```