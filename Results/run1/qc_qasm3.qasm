

// Circuit: n=2, h=1, d=2, g=2
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
z q[1];
h q[1];
barrier q[0], q[1];


// Circuit: n=2, h=2, d=4, g=5
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
z q[1];
h q[0];
barrier q[0], q[1];
cz q[0], q[1];
z q[0];
h q[0];
barrier q[0], q[1];


// Circuit: n=2, h=3, d=6, g=7
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
cz q[1], q[0];
cz q[0], q[1];
h q[0];
barrier q[0], q[1];
h q[0];
barrier q[0], q[1];
z q[1];
z q[1];
h q[0];
barrier q[0], q[1];


// Circuit: n=2, h=4, d=16, g=19
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
z q[0];
h q[1];
barrier q[0], q[1];
cz q[0], q[1];
cz q[0], q[1];
cz q[0], q[1];
z q[0];
z q[1];
z q[1];
z q[0];
cz q[0], q[1];
h q[1];
barrier q[0], q[1];
z q[1];
cz q[1], q[0];
h q[0];
barrier q[0], q[1];
cz q[0], q[1];
z q[0];
cz q[0], q[1];
cz q[1], q[0];
h q[1];
barrier q[0], q[1];


// Circuit: n=2, h=5, d=8, g=10
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
barrier q[0], q[1];
z q[1];
z q[1];
h q[0];
barrier q[0], q[1];
z q[0];
h q[0];
barrier q[0], q[1];
z q[1];
h q[0];
barrier q[0], q[1];
z q[1];
h q[1];
barrier q[0], q[1];


// Circuit: n=2, h=6, d=18, g=19
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
z q[1];
cz q[1], q[0];
h q[0];
barrier q[0], q[1];
z q[0];
cz q[0], q[1];
cz q[1], q[0];
h q[0];
barrier q[0], q[1];
cz q[0], q[1];
cz q[0], q[1];
h q[0];
barrier q[0], q[1];
z q[1];
cz q[0], q[1];
cz q[0], q[1];
z q[1];
h q[0];
barrier q[0], q[1];
z q[0];
cz q[0], q[1];
h q[1];
barrier q[0], q[1];
h q[1];
barrier q[0], q[1];


// Circuit: n=2, h=7, d=11, g=12
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
barrier q[0], q[1];
cz q[1], q[0];
cz q[0], q[1];
h q[1];
barrier q[0], q[1];
h q[1];
barrier q[0], q[1];
z q[1];
h q[0];
barrier q[0], q[1];
h q[1];
barrier q[0], q[1];
cz q[1], q[0];
z q[0];
h q[0];
barrier q[0], q[1];
h q[1];
barrier q[0], q[1];


// Circuit: n=2, h=8, d=19, g=20
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[1];
barrier q[0], q[1];
z q[0];
z q[0];
cz q[0], q[1];
z q[1];
h q[0];
barrier q[0], q[1];
h q[1];
barrier q[0], q[1];
h q[0];
barrier q[0], q[1];
h q[0];
barrier q[0], q[1];
z q[1];
cz q[1], q[0];
cz q[1], q[0];
cz q[1], q[0];
cz q[1], q[0];
h q[0];
barrier q[0], q[1];
cz q[0], q[1];
cz q[1], q[0];
h q[1];
barrier q[0], q[1];
cz q[1], q[0];
h q[0];
barrier q[0], q[1];


// Circuit: n=2, h=9, d=31, g=36
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[1];
barrier q[0], q[1];
cz q[1], q[0];
z q[1];
cz q[1], q[0];
z q[1];
z q[0];
cz q[1], q[0];
z q[0];
cz q[0], q[1];
z q[1];
h q[0];
barrier q[0], q[1];
z q[1];
h q[1];
barrier q[0], q[1];
cz q[1], q[0];
cz q[0], q[1];
h q[1];
barrier q[0], q[1];
h q[1];
barrier q[0], q[1];
h q[0];
barrier q[0], q[1];
cz q[0], q[1];
cz q[0], q[1];
cz q[1], q[0];
z q[1];
z q[0];
cz q[0], q[1];
cz q[0], q[1];
h q[0];
barrier q[0], q[1];
cz q[1], q[0];
cz q[1], q[0];
cz q[1], q[0];
z q[0];
cz q[1], q[0];
z q[0];
z q[1];
z q[0];
h q[1];
barrier q[0], q[1];
h q[0];
barrier q[0], q[1];


// Circuit: n=2, h=10, d=28, g=30
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
cz q[0], q[1];
cz q[0], q[1];
h q[1];
barrier q[0], q[1];
h q[0];
barrier q[0], q[1];
h q[0];
barrier q[0], q[1];
h q[0];
barrier q[0], q[1];
z q[1];
z q[0];
h q[1];
barrier q[0], q[1];
z q[0];
cz q[1], q[0];
h q[1];
barrier q[0], q[1];
z q[0];
z q[1];
h q[0];
barrier q[0], q[1];
z q[0];
cz q[1], q[0];
cz q[1], q[0];
cz q[1], q[0];
h q[1];
barrier q[0], q[1];
cz q[1], q[0];
cz q[1], q[0];
cz q[0], q[1];
z q[0];
cz q[0], q[1];
h q[0];
barrier q[0], q[1];
cz q[1], q[0];
cz q[1], q[0];
z q[0];
h q[0];
barrier q[0], q[1];


// Circuit: n=2, h=11, d=29, g=31
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
z q[0];
z q[1];
cz q[1], q[0];
h q[1];
barrier q[0], q[1];
cz q[0], q[1];
h q[0];
barrier q[0], q[1];
cz q[1], q[0];
cz q[0], q[1];
cz q[0], q[1];
h q[1];
barrier q[0], q[1];
cz q[0], q[1];
cz q[0], q[1];
cz q[1], q[0];
h q[0];
barrier q[0], q[1];
z q[0];
cz q[1], q[0];
h q[1];
barrier q[0], q[1];
cz q[1], q[0];
h q[0];
barrier q[0], q[1];
cz q[0], q[1];
h q[0];
barrier q[0], q[1];
h q[0];
barrier q[0], q[1];
z q[1];
cz q[0], q[1];
h q[1];
barrier q[0], q[1];
cz q[1], q[0];
z q[1];
h q[0];
barrier q[0], q[1];
z q[0];
cz q[0], q[1];
h q[0];
barrier q[0], q[1];


// Circuit: n=2, h=12, d=41, g=44
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
cz q[1], q[0];
z q[1];
cz q[1], q[0];
h q[1];
barrier q[0], q[1];
h q[0];
barrier q[0], q[1];
z q[0];
h q[0];
barrier q[0], q[1];
z q[1];
z q[0];
cz q[1], q[0];
cz q[0], q[1];
z q[0];
cz q[1], q[0];
cz q[0], q[1];
cz q[0], q[1];
z q[0];
cz q[1], q[0];
cz q[1], q[0];
cz q[1], q[0];
h q[0];
barrier q[0], q[1];
z q[1];
cz q[1], q[0];
cz q[1], q[0];
cz q[1], q[0];
h q[0];
barrier q[0], q[1];
z q[1];
cz q[1], q[0];
z q[1];
z q[0];
h q[1];
barrier q[0], q[1];
z q[0];
cz q[0], q[1];
h q[1];
barrier q[0], q[1];
z q[1];
h q[0];
barrier q[0], q[1];
cz q[1], q[0];
h q[0];
barrier q[0], q[1];
cz q[1], q[0];
h q[1];
barrier q[0], q[1];
h q[1];
barrier q[0], q[1];
z q[1];
z q[1];
cz q[1], q[0];
h q[0];
barrier q[0], q[1];


// Circuit: n=2, h=13, d=43, g=45
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
cz q[0], q[1];
cz q[1], q[0];
z q[1];
h q[0];
barrier q[0], q[1];
h q[1];
barrier q[0], q[1];
h q[1];
barrier q[0], q[1];
h q[1];
barrier q[0], q[1];
cz q[1], q[0];
cz q[1], q[0];
h q[1];
barrier q[0], q[1];
h q[0];
barrier q[0], q[1];
cz q[0], q[1];
cz q[0], q[1];
h q[1];
barrier q[0], q[1];
cz q[1], q[0];
cz q[1], q[0];
cz q[0], q[1];
z q[1];
cz q[0], q[1];
h q[0];
barrier q[0], q[1];
h q[1];
barrier q[0], q[1];
z q[1];
cz q[1], q[0];
cz q[0], q[1];
cz q[0], q[1];
z q[1];
cz q[1], q[0];
h q[0];
barrier q[0], q[1];
cz q[0], q[1];
cz q[1], q[0];
h q[1];
barrier q[0], q[1];
z q[0];
z q[0];
z q[1];
cz q[1], q[0];
cz q[0], q[1];
z q[1];
cz q[0], q[1];
cz q[1], q[0];
cz q[1], q[0];
cz q[1], q[0];
h q[1];
barrier q[0], q[1];
cz q[0], q[1];
cz q[0], q[1];
h q[0];
barrier q[0], q[1];


// Circuit: n=2, h=14, d=39, g=41
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
cz q[0], q[1];
cz q[0], q[1];
h q[1];
barrier q[0], q[1];
z q[1];
cz q[0], q[1];
z q[0];
z q[1];
z q[1];
cz q[1], q[0];
h q[0];
barrier q[0], q[1];
z q[0];
z q[1];
z q[1];
cz q[1], q[0];
cz q[1], q[0];
h q[1];
barrier q[0], q[1];
z q[1];
cz q[1], q[0];
h q[1];
barrier q[0], q[1];
h q[0];
barrier q[0], q[1];
cz q[0], q[1];
cz q[1], q[0];
cz q[0], q[1];
h q[0];
barrier q[0], q[1];
z q[1];
h q[1];
barrier q[0], q[1];
h q[0];
barrier q[0], q[1];
cz q[1], q[0];
z q[0];
h q[0];
barrier q[0], q[1];
h q[0];
barrier q[0], q[1];
z q[1];
cz q[1], q[0];
cz q[0], q[1];
h q[1];
barrier q[0], q[1];
h q[1];
barrier q[0], q[1];
z q[0];
cz q[0], q[1];
h q[1];
barrier q[0], q[1];
cz q[1], q[0];
h q[1];
barrier q[0], q[1];


// Circuit: n=3, h=1, d=5, g=5
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[3] q;
ccz q[2], q[0], q[1];
ccz q[1], q[2], q[0];
cz q[2], q[0];
z q[0];
h q[0];
barrier q[0], q[1], q[2];


// Circuit: n=3, h=2, d=5, g=5
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[3] q;
ccz q[1], q[0], q[2];
ccz q[1], q[0], q[2];
h q[2];
barrier q[0], q[1], q[2];
cz q[1], q[0];
h q[1];
barrier q[0], q[1], q[2];


// Circuit: n=3, h=3, d=11, g=13
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[3] q;
cz q[2], q[1];
cz q[2], q[1];
h q[2];
barrier q[0], q[1], q[2];
z q[1];
cz q[1], q[2];
z q[2];
ccz q[1], q[2], q[0];
z q[2];
h q[1];
barrier q[0], q[1], q[2];
z q[0];
z q[1];
cz q[2], q[0];
h q[0];
barrier q[0], q[1], q[2];


// Circuit: n=3, h=4, d=8, g=11
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[2];
barrier q[0], q[1], q[2];
z q[2];
cz q[0], q[1];
h q[1];
barrier q[0], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2];
z q[2];
z q[0];
cz q[2], q[0];
cz q[2], q[1];
z q[0];
h q[1];
barrier q[0], q[1], q[2];


// Circuit: n=3, h=5, d=5, g=6
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[1];
barrier q[0], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2];
z q[1];
h q[0];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];


// Circuit: n=3, h=6, d=28, g=32
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[3] q;
h q[2];
barrier q[0], q[1], q[2];
ccz q[0], q[2], q[1];
ccz q[0], q[2], q[1];
z q[1];
cz q[2], q[0];
h q[0];
barrier q[0], q[1], q[2];
z q[0];
z q[2];
ccz q[2], q[0], q[1];
h q[2];
barrier q[0], q[1], q[2];
cz q[0], q[1];
z q[0];
z q[0];
z q[0];
z q[1];
z q[0];
z q[1];
ccz q[0], q[2], q[1];
ccz q[2], q[0], q[1];
cz q[0], q[1];
ccz q[0], q[2], q[1];
ccz q[2], q[0], q[1];
ccz q[2], q[1], q[0];
ccz q[2], q[1], q[0];
h q[1];
barrier q[0], q[1], q[2];
z q[2];
ccz q[2], q[1], q[0];
cz q[2], q[0];
cz q[0], q[2];
h q[2];
barrier q[0], q[1], q[2];
ccz q[0], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2];


// Circuit: n=3, h=7, d=21, g=22
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[3] q;
z q[0];
cz q[0], q[2];
h q[0];
barrier q[0], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2];
cz q[2], q[1];
ccz q[2], q[0], q[1];
h q[1];
barrier q[0], q[1], q[2];
z q[0];
ccz q[2], q[0], q[1];
ccz q[1], q[2], q[0];
ccz q[0], q[2], q[1];
h q[0];
barrier q[0], q[1], q[2];
z q[0];
cz q[2], q[1];
z q[2];
cz q[0], q[2];
z q[2];
ccz q[1], q[2], q[0];
h q[1];
barrier q[0], q[1], q[2];
z q[0];
h q[0];
barrier q[0], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2];


// Circuit: n=3, h=8, d=22, g=22
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[3] q;
h q[1];
barrier q[0], q[1], q[2];
cz q[2], q[1];
z q[1];
ccz q[0], q[2], q[1];
z q[0];
ccz q[2], q[0], q[1];
h q[0];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];
ccz q[1], q[0], q[2];
ccz q[2], q[1], q[0];
cz q[0], q[2];
h q[2];
barrier q[0], q[1], q[2];
z q[0];
cz q[0], q[1];
h q[1];
barrier q[0], q[1], q[2];
z q[2];
z q[2];
h q[2];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];
ccz q[0], q[2], q[1];
ccz q[2], q[1], q[0];
h q[2];
barrier q[0], q[1], q[2];


// Circuit: n=3, h=9, d=27, g=32
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[3] q;
cz q[0], q[1];
z q[0];
ccz q[2], q[1], q[0];
z q[0];
cz q[0], q[1];
z q[1];
cz q[2], q[0];
h q[0];
barrier q[0], q[1], q[2];
ccz q[2], q[0], q[1];
cz q[1], q[2];
h q[0];
barrier q[0], q[1], q[2];
ccz q[0], q[1], q[2];
cz q[2], q[0];
h q[2];
barrier q[0], q[1], q[2];
z q[2];
z q[0];
z q[1];
cz q[2], q[0];
ccz q[2], q[1], q[0];
ccz q[2], q[0], q[1];
z q[2];
cz q[2], q[1];
h q[2];
barrier q[0], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];
ccz q[0], q[1], q[2];
cz q[1], q[2];
z q[1];
h q[2];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];
z q[2];
h q[2];
barrier q[0], q[1], q[2];


// Circuit: n=3, h=10, d=30, g=36
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[3] q;
z q[2];
z q[0];
z q[0];
cz q[1], q[2];
z q[1];
ccz q[0], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2];
z q[2];
z q[0];
cz q[0], q[2];
ccz q[2], q[1], q[0];
ccz q[0], q[1], q[2];
z q[0];
ccz q[2], q[1], q[0];
z q[2];
cz q[2], q[1];
cz q[1], q[2];
h q[2];
barrier q[0], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2];
ccz q[2], q[0], q[1];
h q[2];
barrier q[0], q[1], q[2];
cz q[0], q[1];
cz q[2], q[0];
ccz q[2], q[1], q[0];
z q[1];
h q[2];
barrier q[0], q[1], q[2];
ccz q[2], q[1], q[0];
h q[1];
barrier q[0], q[1], q[2];
z q[2];
z q[0];
z q[2];
h q[1];
barrier q[0], q[1], q[2];
cz q[1], q[0];
h q[0];
barrier q[0], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2];


// Circuit: n=3, h=11, d=26, g=29
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[3] q;
cz q[1], q[2];
z q[0];
cz q[0], q[1];
ccz q[0], q[2], q[1];
cz q[1], q[0];
h q[0];
barrier q[0], q[1], q[2];
h q[2];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];
z q[2];
h q[1];
barrier q[0], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2];
cz q[1], q[0];
ccz q[1], q[2], q[0];
cz q[0], q[1];
cz q[1], q[2];
ccz q[2], q[1], q[0];
h q[2];
barrier q[0], q[1], q[2];
ccz q[0], q[2], q[1];
cz q[2], q[0];
ccz q[0], q[2], q[1];
h q[2];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];
ccz q[0], q[1], q[2];
cz q[2], q[1];
h q[0];
barrier q[0], q[1], q[2];
cz q[1], q[0];
cz q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];


// Circuit: n=3, h=12, d=30, g=36
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[3] q;
z q[1];
h q[2];
barrier q[0], q[1], q[2];
cz q[1], q[2];
cz q[1], q[0];
cz q[0], q[2];
h q[2];
barrier q[0], q[1], q[2];
h q[2];
barrier q[0], q[1], q[2];
z q[2];
h q[0];
barrier q[0], q[1], q[2];
cz q[2], q[1];
ccz q[1], q[0], q[2];
h q[2];
barrier q[0], q[1], q[2];
z q[0];
ccz q[0], q[2], q[1];
cz q[0], q[1];
z q[2];
ccz q[2], q[1], q[0];
ccz q[2], q[1], q[0];
h q[0];
barrier q[0], q[1], q[2];
ccz q[0], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];
cz q[2], q[1];
z q[2];
cz q[2], q[1];
h q[0];
barrier q[0], q[1], q[2];
ccz q[1], q[0], q[2];
z q[1];
h q[1];
barrier q[0], q[1], q[2];
z q[1];
z q[1];
z q[2];
cz q[0], q[2];
h q[2];
barrier q[0], q[1], q[2];
cz q[0], q[1];
h q[1];
barrier q[0], q[1], q[2];


// Circuit: n=3, h=13, d=32, g=40
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[3] q;
z q[2];
h q[1];
barrier q[0], q[1], q[2];
cz q[0], q[2];
z q[2];
z q[0];
h q[2];
barrier q[0], q[1], q[2];
cz q[0], q[2];
z q[1];
h q[0];
barrier q[0], q[1], q[2];
z q[1];
ccz q[1], q[0], q[2];
z q[0];
h q[0];
barrier q[0], q[1], q[2];
cz q[1], q[0];
ccz q[1], q[0], q[2];
z q[0];
ccz q[1], q[0], q[2];
z q[0];
z q[2];
h q[1];
barrier q[0], q[1], q[2];
z q[0];
h q[2];
barrier q[0], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2];
cz q[0], q[1];
z q[1];
cz q[0], q[2];
h q[0];
barrier q[0], q[1], q[2];
cz q[2], q[0];
z q[2];
h q[2];
barrier q[0], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2];
z q[1];
ccz q[2], q[1], q[0];
z q[1];
h q[0];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];
ccz q[1], q[0], q[2];
ccz q[2], q[1], q[0];
cz q[1], q[0];
h q[1];
barrier q[0], q[1], q[2];


// Circuit: n=3, h=14, d=34, g=42
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[3] q;
z q[0];
z q[2];
z q[2];
cz q[1], q[2];
z q[2];
ccz q[0], q[2], q[1];
ccz q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];
ccz q[2], q[1], q[0];
z q[1];
ccz q[0], q[1], q[2];
z q[1];
z q[0];
ccz q[1], q[2], q[0];
h q[1];
barrier q[0], q[1], q[2];
cz q[1], q[0];
z q[0];
h q[2];
barrier q[0], q[1], q[2];
z q[1];
h q[0];
barrier q[0], q[1], q[2];
z q[1];
z q[0];
z q[2];
ccz q[1], q[2], q[0];
ccz q[2], q[1], q[0];
ccz q[0], q[2], q[1];
h q[1];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];
ccz q[2], q[0], q[1];
cz q[2], q[0];
h q[2];
barrier q[0], q[1], q[2];
z q[2];
z q[0];
ccz q[0], q[2], q[1];
h q[2];
barrier q[0], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2];
z q[0];
h q[1];
barrier q[0], q[1], q[2];


// Circuit: n=4, h=1, d=6, g=7
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[4] q;
ccz q[0], q[1], q[2];
z q[2];
ccz q[2], q[3], q[1];
cz q[1], q[0];
cz q[1], q[3];
z q[1];
h q[0];
barrier q[0], q[1], q[2], q[3];


// Circuit: n=4, h=2, d=7, g=8
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[4] q;
h q[0];
barrier q[0], q[1], q[2], q[3];
cz q[2], q[0];
cz q[2], q[0];
z q[0];
cz q[0], q[3];
cz q[2], q[3];
ccz q[1], q[2], q[3];
h q[0];
barrier q[0], q[1], q[2], q[3];


// Circuit: n=4, h=3, d=3, g=3
OPENQASM 3.0;
include "stdgates.inc";
qubit[4] q;
h q[0];
barrier q[0], q[1], q[2], q[3];
h q[3];
barrier q[0], q[1], q[2], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3];


// Circuit: n=4, h=4, d=17, g=27
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[4] q;
z q[3];
z q[0];
z q[2];
z q[1];
z q[0];
z q[1];
h q[3];
barrier q[0], q[1], q[2], q[3];
z q[1];
cz q[3], q[0];
z q[2];
cz q[0], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3];
z q[2];
cz q[0], q[1];
cz q[1], q[0];
ccz q[1], q[2], q[0];
cz q[3], q[2];
cz q[2], q[1];
ccz q[0], q[3], q[1];
ccz q[3], q[1], q[0];
z q[0];
cz q[2], q[0];
z q[2];
cz q[0], q[1];
h q[0];
barrier q[0], q[1], q[2], q[3];
ccz q[0], q[1], q[3];
h q[0];
barrier q[0], q[1], q[2], q[3];


// Circuit: n=4, h=5, d=10, g=10
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[4] q;
h q[3];
barrier q[0], q[1], q[2], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3];
ccz q[2], q[0], q[3];
cz q[2], q[3];
ccz q[0], q[3], q[1];
z q[3];
cz q[3], q[2];
h q[3];
barrier q[0], q[1], q[2], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3];


// Circuit: n=4, h=6, d=11, g=14
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[4] q;
ccz q[2], q[3], q[1];
h q[3];
barrier q[0], q[1], q[2], q[3];
cz q[2], q[3];
z q[0];
ccz q[2], q[3], q[0];
ccz q[2], q[1], q[3];
z q[0];
h q[0];
barrier q[0], q[1], q[2], q[3];
z q[3];
h q[0];
barrier q[0], q[1], q[2], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3];
h q[3];
barrier q[0], q[1], q[2], q[3];
cz q[1], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3];


// Circuit: n=4, h=7, d=21, g=33
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[4] q;
cz q[3], q[2];
cz q[1], q[2];
h q[1];
barrier q[0], q[1], q[2], q[3];
z q[3];
z q[0];
ccz q[0], q[1], q[3];
h q[3];
barrier q[0], q[1], q[2], q[3];
z q[1];
cz q[1], q[2];
z q[3];
ccz q[2], q[1], q[3];
cz q[0], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3];
z q[0];
z q[2];
z q[3];
z q[1];
ccz q[1], q[0], q[2];
z q[1];
cz q[2], q[0];
z q[1];
z q[3];
z q[2];
z q[0];
cz q[3], q[0];
ccz q[0], q[2], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3];
ccz q[2], q[0], q[1];
z q[3];
cz q[1], q[2];
h q[3];
barrier q[0], q[1], q[2], q[3];
h q[3];
barrier q[0], q[1], q[2], q[3];


// Circuit: n=4, h=8, d=17, g=27
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[4] q;
z q[2];
h q[0];
barrier q[0], q[1], q[2], q[3];
cz q[0], q[1];
cz q[2], q[1];
h q[3];
barrier q[0], q[1], q[2], q[3];
z q[2];
z q[1];
ccz q[0], q[3], q[1];
cz q[2], q[3];
h q[0];
barrier q[0], q[1], q[2], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3];
ccz q[0], q[1], q[2];
z q[1];
z q[0];
z q[2];
h q[0];
barrier q[0], q[1], q[2], q[3];
ccz q[2], q[3], q[0];
z q[0];
z q[3];
h q[3];
barrier q[0], q[1], q[2], q[3];
ccz q[0], q[3], q[2];
cz q[1], q[2];
ccz q[2], q[1], q[0];
z q[3];
h q[3];
barrier q[0], q[1], q[2], q[3];
z q[2];
h q[0];
barrier q[0], q[1], q[2], q[3];


// Circuit: n=4, h=9, d=26, g=31
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[4] q;
cz q[0], q[1];
cz q[3], q[2];
ccz q[2], q[3], q[0];
cz q[2], q[1];
h q[1];
barrier q[0], q[1], q[2], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3];
cz q[3], q[1];
z q[3];
cz q[1], q[3];
ccz q[1], q[3], q[0];
h q[0];
barrier q[0], q[1], q[2], q[3];
z q[1];
h q[1];
barrier q[0], q[1], q[2], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3];
z q[1];
ccz q[3], q[0], q[1];
z q[3];
h q[3];
barrier q[0], q[1], q[2], q[3];
z q[1];
z q[2];
z q[3];
ccz q[2], q[0], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3];
ccz q[0], q[3], q[1];
z q[2];
ccz q[1], q[2], q[0];
h q[2];
barrier q[0], q[1], q[2], q[3];
ccz q[3], q[1], q[2];
cz q[2], q[0];
cz q[1], q[0];
h q[2];
barrier q[0], q[1], q[2], q[3];


// Circuit: n=4, h=10, d=29, g=38
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[4] q;
z q[2];
h q[3];
barrier q[0], q[1], q[2], q[3];
z q[3];
h q[3];
barrier q[0], q[1], q[2], q[3];
z q[2];
h q[0];
barrier q[0], q[1], q[2], q[3];
cz q[0], q[2];
ccz q[1], q[3], q[0];
cz q[0], q[2];
cz q[1], q[0];
ccz q[0], q[2], q[1];
h q[2];
barrier q[0], q[1], q[2], q[3];
cz q[2], q[1];
cz q[0], q[2];
z q[2];
cz q[0], q[3];
z q[1];
z q[0];
cz q[1], q[2];
ccz q[3], q[2], q[0];
ccz q[2], q[3], q[0];
h q[3];
barrier q[0], q[1], q[2], q[3];
ccz q[3], q[1], q[2];
z q[2];
z q[0];
cz q[3], q[0];
z q[2];
h q[2];
barrier q[0], q[1], q[2], q[3];
cz q[0], q[1];
ccz q[2], q[3], q[0];
h q[1];
barrier q[0], q[1], q[2], q[3];
cz q[1], q[3];
cz q[1], q[3];
h q[3];
barrier q[0], q[1], q[2], q[3];
z q[3];
h q[2];
barrier q[0], q[1], q[2], q[3];
z q[1];
h q[1];
barrier q[0], q[1], q[2], q[3];


// Circuit: n=4, h=11, d=24, g=26
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[4] q;
h q[0];
barrier q[0], q[1], q[2], q[3];
z q[3];
ccz q[0], q[2], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3];
h q[0];
barrier q[0], q[1], q[2], q[3];
z q[1];
h q[1];
barrier q[0], q[1], q[2], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3];
ccz q[0], q[2], q[3];
h q[3];
barrier q[0], q[1], q[2], q[3];
z q[1];
ccz q[1], q[2], q[3];
z q[1];
ccz q[1], q[0], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3];
ccz q[0], q[1], q[3];
cz q[3], q[2];
z q[1];
ccz q[2], q[0], q[3];
z q[0];
ccz q[2], q[0], q[3];
h q[3];
barrier q[0], q[1], q[2], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3];
z q[3];
h q[3];
barrier q[0], q[1], q[2], q[3];


// Circuit: n=4, h=12, d=22, g=28
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[4] q;
cz q[0], q[3];
cz q[3], q[0];
z q[0];
ccz q[1], q[0], q[2];
cz q[1], q[0];
h q[3];
barrier q[0], q[1], q[2], q[3];
ccz q[2], q[1], q[3];
ccz q[1], q[0], q[2];
h q[3];
barrier q[0], q[1], q[2], q[3];
h q[3];
barrier q[0], q[1], q[2], q[3];
h q[0];
barrier q[0], q[1], q[2], q[3];
h q[3];
barrier q[0], q[1], q[2], q[3];
z q[0];
h q[0];
barrier q[0], q[1], q[2], q[3];
z q[1];
h q[2];
barrier q[0], q[1], q[2], q[3];
z q[1];
h q[1];
barrier q[0], q[1], q[2], q[3];
h q[0];
barrier q[0], q[1], q[2], q[3];
cz q[2], q[0];
h q[2];
barrier q[0], q[1], q[2], q[3];
z q[1];
z q[2];
ccz q[0], q[3], q[2];
z q[3];
h q[1];
barrier q[0], q[1], q[2], q[3];
z q[1];
h q[3];
barrier q[0], q[1], q[2], q[3];


// Circuit: n=4, h=13, d=35, g=48
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[4] q;
ccz q[1], q[3], q[2];
h q[2];
barrier q[0], q[1], q[2], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3];
z q[0];
ccz q[0], q[1], q[2];
h q[2];
barrier q[0], q[1], q[2], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3];
cz q[0], q[2];
cz q[3], q[2];
z q[1];
z q[0];
ccz q[3], q[0], q[1];
z q[2];
z q[1];
z q[1];
ccz q[0], q[2], q[3];
cz q[1], q[0];
ccz q[0], q[2], q[1];
h q[0];
barrier q[0], q[1], q[2], q[3];
z q[2];
h q[3];
barrier q[0], q[1], q[2], q[3];
h q[0];
barrier q[0], q[1], q[2], q[3];
z q[2];
ccz q[0], q[1], q[3];
cz q[1], q[0];
h q[3];
barrier q[0], q[1], q[2], q[3];
cz q[0], q[1];
h q[2];
barrier q[0], q[1], q[2], q[3];
z q[1];
z q[0];
cz q[3], q[0];
z q[3];
h q[1];
barrier q[0], q[1], q[2], q[3];
z q[0];
z q[0];
ccz q[0], q[1], q[2];
ccz q[3], q[2], q[0];
h q[3];
barrier q[0], q[1], q[2], q[3];
ccz q[0], q[2], q[1];
z q[3];
z q[2];
ccz q[1], q[2], q[3];
ccz q[2], q[0], q[3];
ccz q[0], q[1], q[2];
z q[1];
h q[2];
barrier q[0], q[1], q[2], q[3];
ccz q[3], q[1], q[0];
h q[2];
barrier q[0], q[1], q[2], q[3];


// Circuit: n=4, h=14, d=39, g=54
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[4] q;
cz q[0], q[2];
h q[1];
barrier q[0], q[1], q[2], q[3];
ccz q[0], q[1], q[2];
ccz q[3], q[0], q[2];
h q[3];
barrier q[0], q[1], q[2], q[3];
ccz q[3], q[1], q[0];
z q[0];
h q[3];
barrier q[0], q[1], q[2], q[3];
h q[0];
barrier q[0], q[1], q[2], q[3];
z q[1];
h q[0];
barrier q[0], q[1], q[2], q[3];
h q[3];
barrier q[0], q[1], q[2], q[3];
z q[3];
z q[1];
ccz q[0], q[1], q[2];
z q[2];
cz q[2], q[0];
h q[3];
barrier q[0], q[1], q[2], q[3];
ccz q[2], q[1], q[0];
ccz q[0], q[1], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3];
ccz q[1], q[2], q[0];
cz q[3], q[0];
cz q[2], q[1];
cz q[1], q[0];
ccz q[1], q[2], q[0];
z q[2];
z q[2];
z q[1];
h q[0];
barrier q[0], q[1], q[2], q[3];
ccz q[3], q[0], q[2];
z q[0];
ccz q[3], q[0], q[2];
z q[3];
z q[3];
ccz q[0], q[1], q[2];
ccz q[2], q[3], q[0];
z q[3];
cz q[3], q[1];
h q[0];
barrier q[0], q[1], q[2], q[3];
cz q[0], q[2];
z q[2];
h q[2];
barrier q[0], q[1], q[2], q[3];
z q[3];
ccz q[1], q[0], q[2];
z q[3];
z q[2];
h q[0];
barrier q[0], q[1], q[2], q[3];
z q[1];
ccz q[3], q[1], q[0];
z q[1];
cz q[0], q[1];
h q[2];
barrier q[0], q[1], q[2], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3];


// Circuit: n=5, h=1, d=2, g=2
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[5] q;
ccz q[4], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];


// Circuit: n=5, h=2, d=3, g=3
OPENQASM 3.0;
include "stdgates.inc";
qubit[5] q;
cz q[2], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];


// Circuit: n=5, h=3, d=7, g=9
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[5] q;
z q[0];
z q[3];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[3], q[0], q[4];
z q[0];
ccz q[1], q[3], q[4];
cz q[4], q[3];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];


// Circuit: n=5, h=4, d=7, g=10
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[5] q;
z q[2];
ccz q[0], q[4], q[1];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
cz q[1], q[2];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
z q[1];
z q[3];
z q[2];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];


// Circuit: n=5, h=5, d=8, g=9
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[5] q;
z q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[3], q[1], q[2];
ccz q[0], q[2], q[1];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[0], q[2], q[3];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];


// Circuit: n=5, h=6, d=17, g=22
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[5] q;
z q[0];
ccz q[3], q[4], q[0];
ccz q[1], q[4], q[3];
cz q[0], q[3];
z q[1];
z q[2];
cz q[1], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
z q[3];
z q[2];
z q[2];
cz q[0], q[4];
ccz q[1], q[2], q[3];
ccz q[4], q[3], q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
z q[4];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4];
z q[0];
cz q[3], q[1];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];


// Circuit: n=5, h=7, d=10, g=15
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[5] q;
ccz q[2], q[0], q[1];
z q[1];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4];
cz q[2], q[1];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4];
z q[1];
cz q[1], q[2];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
z q[1];
z q[1];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];
z q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];


// Circuit: n=5, h=8, d=19, g=26
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[5] q;
ccz q[0], q[3], q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[4], q[2], q[0];
cz q[4], q[0];
z q[4];
z q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
z q[1];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4];
z q[2];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[2], q[0], q[3];
ccz q[0], q[3], q[1];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
cz q[4], q[0];
cz q[4], q[1];
cz q[2], q[1];
z q[3];
z q[4];
cz q[1], q[0];
cz q[4], q[1];
cz q[1], q[3];
ccz q[3], q[1], q[0];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];


// Circuit: n=5, h=9, d=22, g=31
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[5] q;
h q[0];
barrier q[0], q[1], q[2], q[3], q[4];
z q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
cz q[4], q[3];
ccz q[4], q[3], q[1];
ccz q[3], q[4], q[2];
cz q[4], q[0];
z q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
cz q[3], q[0];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
cz q[4], q[0];
z q[0];
cz q[3], q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[4], q[3], q[1];
ccz q[3], q[0], q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[2], q[0], q[3];
ccz q[2], q[4], q[0];
z q[1];
cz q[3], q[0];
ccz q[1], q[2], q[0];
z q[1];
ccz q[1], q[4], q[2];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4];
z q[3];
cz q[1], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];


// Circuit: n=5, h=10, d=15, g=22
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[5] q;
z q[1];
ccz q[2], q[0], q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
z q[3];
z q[2];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[3], q[1], q[2];
z q[1];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
z q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
cz q[1], q[4];
cz q[3], q[1];
z q[4];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[2], q[4], q[1];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];
cz q[4], q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];


// Circuit: n=5, h=11, d=26, g=40
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[5] q;
z q[4];
cz q[3], q[0];
z q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];
z q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
cz q[1], q[3];
ccz q[1], q[2], q[0];
cz q[4], q[2];
cz q[3], q[2];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4];
z q[4];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4];
z q[2];
z q[0];
cz q[3], q[4];
cz q[2], q[0];
cz q[3], q[1];
z q[3];
ccz q[2], q[0], q[3];
z q[2];
cz q[2], q[0];
cz q[4], q[1];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
cz q[4], q[1];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[3], q[4], q[1];
cz q[2], q[0];
cz q[3], q[1];
z q[4];
z q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[4], q[1], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
cz q[2], q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[1], q[4], q[2];
z q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4];


// Circuit: n=5, h=12, d=25, g=36
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[5] q;
ccz q[4], q[2], q[3];
ccz q[0], q[1], q[2];
cz q[2], q[1];
cz q[0], q[2];
cz q[3], q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[2], q[3], q[1];
z q[3];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];
cz q[1], q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[0], q[4], q[2];
z q[1];
z q[1];
z q[3];
cz q[4], q[1];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[3], q[4], q[0];
z q[1];
z q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
z q[4];
ccz q[3], q[2], q[4];
cz q[3], q[1];
cz q[2], q[3];
ccz q[1], q[2], q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4];
cz q[4], q[1];
cz q[2], q[3];
z q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];


// Circuit: n=5, h=13, d=35, g=44
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[5] q;
z q[0];
ccz q[4], q[0], q[3];
cz q[1], q[4];
z q[3];
z q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
z q[3];
ccz q[3], q[0], q[4];
cz q[0], q[4];
ccz q[2], q[4], q[0];
ccz q[2], q[4], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[0], q[4], q[3];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[3], q[4], q[0];
ccz q[2], q[4], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
z q[0];
ccz q[2], q[4], q[0];
cz q[4], q[2];
cz q[1], q[4];
z q[4];
ccz q[2], q[1], q[0];
z q[4];
z q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];
cz q[1], q[4];
ccz q[0], q[1], q[3];
cz q[2], q[1];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
z q[4];
cz q[4], q[2];
cz q[0], q[1];
cz q[3], q[4];
cz q[2], q[4];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4];
cz q[3], q[4];
z q[1];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];


// Circuit: n=5, h=14, d=27, g=39
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[5] q;
z q[0];
z q[2];
cz q[3], q[0];
z q[4];
z q[3];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4];
z q[0];
z q[3];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4];
z q[4];
ccz q[4], q[1], q[0];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4];
cz q[1], q[0];
z q[3];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[4], q[0], q[3];
z q[4];
ccz q[1], q[4], q[3];
ccz q[0], q[2], q[3];
cz q[2], q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];
z q[3];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4];
cz q[0], q[4];
cz q[3], q[4];
z q[2];
z q[4];
ccz q[1], q[3], q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4];
z q[3];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4];
cz q[4], q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4];
ccz q[2], q[4], q[1];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4];


// Circuit: n=6, h=1, d=3, g=4
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
cz q[1], q[0];
ccz q[1], q[4], q[2];
ccz q[0], q[1], q[2];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=2, d=2, g=2
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=3, d=8, g=16
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
ccz q[1], q[4], q[0];
cz q[2], q[5];
cz q[0], q[5];
cz q[4], q[0];
z q[1];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
ccz q[3], q[4], q[2];
z q[1];
ccz q[4], q[3], q[2];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[1];
z q[4];
z q[0];
cz q[3], q[1];
cz q[0], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=4, d=8, g=14
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
z q[1];
ccz q[5], q[0], q[4];
z q[3];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[2], q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[0], q[4];
z q[1];
cz q[4], q[2];
z q[3];
ccz q[5], q[4], q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[0];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=5, d=8, g=12
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
ccz q[5], q[2], q[0];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[4], q[5];
z q[3];
cz q[5], q[3];
z q[0];
z q[0];
cz q[0], q[1];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=6, d=10, g=17
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
z q[0];
z q[3];
z q[2];
cz q[0], q[1];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[3];
ccz q[1], q[0], q[5];
z q[0];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[2];
z q[4];
ccz q[4], q[3], q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[3];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=7, d=15, g=21
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
z q[5];
cz q[5], q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[1];
ccz q[1], q[4], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[3];
z q[3];
ccz q[2], q[4], q[3];
ccz q[3], q[4], q[5];
ccz q[4], q[5], q[3];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[4];
z q[2];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[3];
ccz q[0], q[2], q[5];
z q[5];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=8, d=15, g=20
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
cz q[0], q[5];
cz q[5], q[3];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[3], q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[1], q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[4], q[1];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
ccz q[1], q[2], q[5];
ccz q[2], q[0], q[1];
ccz q[1], q[5], q[0];
z q[5];
ccz q[0], q[5], q[2];
cz q[0], q[1];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[0];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=9, d=16, g=26
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
z q[5];
cz q[1], q[2];
z q[5];
z q[0];
z q[2];
z q[3];
cz q[1], q[2];
z q[3];
z q[3];
ccz q[5], q[3], q[1];
cz q[3], q[1];
ccz q[0], q[5], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
ccz q[3], q[4], q[1];
cz q[4], q[0];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[3];
z q[1];
cz q[0], q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=10, d=23, g=40
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
ccz q[5], q[2], q[0];
z q[5];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[0];
z q[2];
cz q[4], q[3];
cz q[5], q[3];
cz q[4], q[1];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[4];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[1], q[4];
z q[4];
cz q[1], q[5];
ccz q[4], q[0], q[1];
z q[5];
ccz q[5], q[0], q[4];
z q[3];
cz q[4], q[5];
cz q[3], q[1];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[2];
z q[4];
z q[0];
cz q[5], q[1];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
ccz q[2], q[4], q[5];
ccz q[2], q[0], q[1];
z q[3];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[1], q[2];
ccz q[0], q[1], q[5];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[3], q[0];
ccz q[0], q[4], q[5];
ccz q[2], q[5], q[3];
cz q[2], q[3];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=11, d=18, g=28
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
z q[5];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[3], q[5];
ccz q[0], q[5], q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[0], q[1];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[2], q[5];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[3];
cz q[3], q[5];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[0], q[1];
cz q[2], q[4];
ccz q[1], q[3], q[0];
z q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[0];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[0];
cz q[3], q[2];
z q[0];
cz q[3], q[2];
cz q[2], q[3];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=12, d=20, g=27
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
cz q[4], q[1];
ccz q[3], q[0], q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[1];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[5];
cz q[3], q[0];
ccz q[5], q[2], q[1];
cz q[4], q[0];
z q[4];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[3], q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[0], q[5];
ccz q[1], q[3], q[4];
z q[3];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[2], q[0];
cz q[3], q[0];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[2];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=13, d=31, g=45
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
ccz q[4], q[5], q[1];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
ccz q[1], q[2], q[3];
cz q[0], q[4];
cz q[4], q[0];
cz q[4], q[2];
ccz q[4], q[1], q[5];
cz q[3], q[2];
z q[2];
ccz q[5], q[1], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[5], q[4];
cz q[4], q[1];
cz q[0], q[1];
z q[2];
cz q[0], q[2];
cz q[3], q[0];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[2];
cz q[4], q[2];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[5], q[3];
cz q[2], q[0];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[5], q[2];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
ccz q[2], q[5], q[0];
z q[3];
cz q[4], q[1];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
ccz q[1], q[0], q[2];
z q[4];
cz q[1], q[4];
ccz q[4], q[2], q[0];
z q[2];
ccz q[3], q[5], q[4];
ccz q[1], q[5], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[1];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=14, d=30, g=44
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
ccz q[1], q[2], q[3];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[4];
z q[0];
z q[2];
z q[3];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[1];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[1], q[0];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[1];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
ccz q[1], q[0], q[2];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
ccz q[3], q[5], q[1];
cz q[3], q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[1], q[5];
cz q[3], q[1];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
ccz q[1], q[0], q[2];
cz q[3], q[0];
z q[0];
z q[2];
cz q[3], q[2];
z q[1];
z q[2];
cz q[4], q[2];
ccz q[5], q[1], q[0];
cz q[0], q[2];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[2], q[4];
z q[1];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
ccz q[0], q[4], q[1];
cz q[5], q[0];
cz q[4], q[0];
z q[0];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[3], q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=7, h=1, d=2, g=4
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[7] q;
cz q[5], q[4];
z q[4];
ccz q[1], q[2], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];


// Circuit: n=7, h=2, d=2, g=2
OPENQASM 3.0;
include "stdgates.inc";
qubit[7] q;
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];


// Circuit: n=7, h=3, d=5, g=12
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[7] q;
z q[3];
z q[1];
cz q[3], q[2];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[5], q[0], q[1];
z q[6];
z q[2];
z q[6];
cz q[0], q[1];
cz q[2], q[5];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];


// Circuit: n=7, h=4, d=6, g=9
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[7] q;
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[6];
cz q[5], q[2];
z q[4];
z q[1];
ccz q[0], q[6], q[2];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];


// Circuit: n=7, h=5, d=7, g=12
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[7] q;
z q[6];
ccz q[0], q[4], q[3];
z q[4];
z q[1];
z q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[2], q[1], q[0];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[6];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];


// Circuit: n=7, h=6, d=8, g=9
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[7] q;
z q[2];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[4], q[1], q[6];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[2], q[5], q[3];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];


// Circuit: n=7, h=7, d=12, g=17
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[7] q;
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[1];
z q[6];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[4], q[3], q[1];
z q[4];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[3];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[2], q[5], q[0];
cz q[3], q[1];
z q[3];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[6];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];


// Circuit: n=7, h=8, d=15, g=19
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[7] q;
z q[5];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[5], q[4], q[3];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[4], q[6], q[3];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[2], q[6], q[5];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[6];
z q[2];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
cz q[4], q[0];
ccz q[5], q[6], q[0];
cz q[5], q[4];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[0];
ccz q[2], q[3], q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];


// Circuit: n=7, h=9, d=19, g=27
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[7] q;
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[5];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[1], q[4], q[2];
ccz q[4], q[2], q[5];
ccz q[6], q[0], q[4];
z q[0];
z q[2];
ccz q[1], q[6], q[2];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[1], q[3], q[6];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
cz q[4], q[6];
z q[1];
ccz q[1], q[5], q[4];
z q[3];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
cz q[4], q[6];
z q[2];
cz q[2], q[4];
ccz q[1], q[4], q[0];
z q[4];
ccz q[6], q[1], q[3];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];


// Circuit: n=7, h=10, d=20, g=37
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[7] q;
z q[1];
z q[6];
z q[2];
cz q[2], q[1];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[4];
cz q[3], q[0];
z q[6];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[1];
z q[3];
z q[1];
z q[0];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[1], q[5], q[3];
z q[5];
ccz q[1], q[6], q[5];
ccz q[6], q[5], q[0];
z q[1];
z q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[0];
z q[3];
ccz q[0], q[5], q[2];
ccz q[4], q[0], q[5];
z q[0];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
cz q[0], q[2];
z q[0];
z q[2];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
cz q[4], q[6];
ccz q[4], q[6], q[1];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];


// Circuit: n=7, h=11, d=27, g=44
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[7] q;
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
cz q[4], q[1];
cz q[5], q[2];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[6], q[4], q[3];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[1];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[0], q[5], q[1];
ccz q[2], q[5], q[1];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
cz q[0], q[6];
cz q[0], q[1];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
cz q[0], q[4];
cz q[3], q[5];
z q[4];
z q[6];
cz q[0], q[6];
z q[0];
z q[6];
z q[1];
ccz q[5], q[2], q[3];
ccz q[4], q[0], q[5];
ccz q[3], q[6], q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[5];
cz q[3], q[4];
z q[4];
cz q[0], q[5];
z q[5];
z q[3];
ccz q[5], q[1], q[2];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[5];
ccz q[2], q[6], q[5];
z q[0];
z q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[5], q[0], q[3];
ccz q[0], q[5], q[4];
cz q[5], q[4];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];


// Circuit: n=7, h=12, d=28, g=44
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[7] q;
ccz q[0], q[6], q[5];
ccz q[6], q[2], q[4];
z q[2];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[4];
z q[3];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[3], q[4], q[0];
z q[1];
cz q[3], q[2];
ccz q[2], q[5], q[1];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[0];
cz q[5], q[3];
ccz q[5], q[2], q[0];
z q[2];
cz q[3], q[6];
z q[3];
ccz q[6], q[3], q[2];
z q[3];
ccz q[2], q[4], q[1];
z q[4];
cz q[3], q[2];
cz q[1], q[0];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[6];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[2], q[1], q[0];
cz q[1], q[4];
cz q[6], q[3];
ccz q[6], q[2], q[5];
z q[4];
cz q[2], q[6];
z q[2];
z q[4];
z q[4];
ccz q[3], q[2], q[6];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];


// Circuit: n=7, h=13, d=25, g=37
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[7] q;
cz q[0], q[6];
z q[3];
cz q[2], q[0];
ccz q[3], q[6], q[4];
z q[1];
cz q[2], q[0];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[2], q[1], q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
cz q[2], q[3];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
cz q[0], q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
cz q[5], q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
cz q[4], q[5];
cz q[1], q[2];
ccz q[6], q[3], q[4];
z q[3];
cz q[4], q[3];
z q[6];
ccz q[2], q[4], q[5];
z q[0];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
cz q[2], q[4];
ccz q[2], q[1], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[2];
cz q[0], q[4];
z q[4];
ccz q[2], q[0], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];


// Circuit: n=7, h=14, d=24, g=39
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[7] q;
z q[5];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[5], q[3], q[6];
z q[0];
ccz q[2], q[6], q[0];
ccz q[2], q[3], q[5];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
cz q[6], q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[4], q[3], q[2];
ccz q[5], q[2], q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[6];
z q[0];
cz q[3], q[5];
z q[4];
ccz q[4], q[5], q[6];
ccz q[1], q[2], q[5];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[4];
cz q[0], q[4];
z q[6];
ccz q[4], q[2], q[5];
cz q[1], q[6];
z q[0];
z q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
cz q[6], q[2];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[6], q[1], q[5];
cz q[1], q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
cz q[3], q[1];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];


// Circuit: n=8, h=1, d=3, g=4
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[8] q;
ccz q[0], q[2], q[5];
ccz q[0], q[2], q[7];
ccz q[2], q[5], q[0];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];


// Circuit: n=8, h=2, d=3, g=4
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[8] q;
ccz q[1], q[3], q[6];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[4];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];


// Circuit: n=8, h=3, d=5, g=8
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[8] q;
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[4], q[7];
z q[2];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[7];
ccz q[4], q[6], q[5];
z q[4];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];


// Circuit: n=8, h=4, d=6, g=12
OPENQASM 3.0;
include "stdgates.inc";
qubit[8] q;
cz q[1], q[7];
cz q[2], q[4];
z q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[2];
z q[2];
z q[3];
z q[1];
cz q[0], q[6];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];


// Circuit: n=8, h=5, d=8, g=14
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[8] q;
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[4], q[6];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[2];
z q[4];
z q[1];
z q[0];
z q[5];
ccz q[2], q[6], q[3];
ccz q[1], q[7], q[3];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[0], q[3];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];


// Circuit: n=8, h=6, d=13, g=19
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[8] q;
cz q[4], q[3];
z q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[3], q[2];
z q[3];
ccz q[2], q[5], q[7];
z q[0];
ccz q[1], q[0], q[3];
cz q[6], q[2];
ccz q[0], q[1], q[5];
z q[6];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[7];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[4];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[3];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];


// Circuit: n=8, h=7, d=10, g=15
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[8] q;
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
ccz q[6], q[7], q[2];
ccz q[3], q[7], q[0];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[3];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[5];
ccz q[2], q[4], q[0];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
ccz q[5], q[7], q[4];
cz q[4], q[1];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[7];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];


// Circuit: n=8, h=8, d=27, g=42
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[8] q;
ccz q[3], q[0], q[4];
cz q[0], q[2];
ccz q[6], q[0], q[4];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[4];
cz q[0], q[6];
ccz q[5], q[6], q[7];
ccz q[6], q[5], q[3];
cz q[0], q[7];
z q[6];
cz q[4], q[7];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[3];
cz q[5], q[1];
cz q[5], q[7];
cz q[3], q[1];
cz q[6], q[2];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
ccz q[7], q[1], q[2];
z q[1];
ccz q[1], q[4], q[0];
z q[2];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
ccz q[4], q[0], q[3];
z q[4];
ccz q[4], q[5], q[6];
cz q[2], q[3];
ccz q[4], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[6], q[4];
ccz q[7], q[0], q[4];
z q[4];
z q[4];
z q[5];
z q[4];
cz q[6], q[4];
cz q[6], q[4];
ccz q[5], q[2], q[1];
z q[3];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];


// Circuit: n=8, h=9, d=20, g=34
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[8] q;
cz q[4], q[3];
cz q[7], q[5];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
ccz q[7], q[1], q[2];
z q[0];
z q[1];
z q[0];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
ccz q[1], q[0], q[7];
cz q[0], q[5];
ccz q[7], q[6], q[1];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[4], q[1];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
ccz q[0], q[1], q[6];
ccz q[6], q[7], q[1];
cz q[5], q[3];
ccz q[6], q[3], q[1];
z q[3];
z q[4];
cz q[1], q[7];
cz q[4], q[6];
cz q[1], q[2];
ccz q[6], q[2], q[4];
z q[2];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[0];
ccz q[3], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[5], q[2];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];


// Circuit: n=8, h=10, d=20, g=34
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[8] q;
z q[2];
z q[4];
z q[5];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[1], q[6];
ccz q[0], q[4], q[6];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[5], q[7];
z q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
ccz q[0], q[7], q[4];
ccz q[4], q[5], q[1];
cz q[5], q[7];
ccz q[7], q[5], q[1];
cz q[3], q[0];
z q[5];
cz q[1], q[0];
ccz q[5], q[1], q[0];
z q[1];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[6];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[1];
cz q[1], q[7];
z q[3];
z q[2];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[1], q[4];
cz q[3], q[0];
cz q[2], q[4];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];


// Circuit: n=8, h=11, d=20, g=30
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[8] q;
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[4], q[0];
ccz q[4], q[3], q[6];
z q[1];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
ccz q[2], q[5], q[0];
ccz q[5], q[6], q[3];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[4];
z q[1];
ccz q[6], q[1], q[0];
ccz q[4], q[2], q[1];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[1];
cz q[5], q[1];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[6];
z q[4];
z q[6];
cz q[1], q[2];
z q[1];
cz q[0], q[2];
ccz q[3], q[4], q[5];
ccz q[3], q[7], q[4];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];


// Circuit: n=8, h=12, d=24, g=50
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[8] q;
ccz q[2], q[1], q[7];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[6], q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
ccz q[1], q[6], q[3];
z q[4];
z q[7];
z q[6];
z q[0];
cz q[6], q[2];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[0], q[3];
z q[3];
z q[0];
z q[1];
cz q[3], q[6];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[2];
ccz q[5], q[7], q[0];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[6], q[0];
ccz q[5], q[4], q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[3];
z q[2];
z q[6];
cz q[5], q[1];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[1], q[0];
z q[7];
cz q[7], q[2];
cz q[4], q[6];
cz q[4], q[6];
ccz q[4], q[7], q[2];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[5], q[4];
z q[3];
cz q[1], q[3];
cz q[7], q[6];
z q[0];
cz q[6], q[4];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[3];
z q[5];
ccz q[1], q[4], q[5];
z q[0];
ccz q[7], q[0], q[5];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];


// Circuit: n=8, h=13, d=17, g=35
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[8] q;
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[4];
z q[0];
z q[7];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[4];
cz q[6], q[0];
cz q[5], q[1];
z q[6];
z q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[5];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[5], q[6];
z q[3];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
ccz q[4], q[6], q[7];
z q[6];
z q[2];
cz q[5], q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[0];
z q[4];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[1], q[7];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[0];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[7], q[1];
ccz q[4], q[7], q[5];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];


// Circuit: n=8, h=14, d=34, g=56
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[8] q;
ccz q[6], q[4], q[2];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[5], q[0];
ccz q[3], q[4], q[1];
ccz q[2], q[1], q[3];
z q[0];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
ccz q[4], q[3], q[5];
ccz q[7], q[2], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
ccz q[4], q[0], q[2];
cz q[3], q[1];
cz q[0], q[3];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[2], q[6];
cz q[0], q[4];
ccz q[1], q[3], q[4];
cz q[0], q[3];
z q[4];
cz q[3], q[5];
z q[2];
cz q[6], q[1];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[0];
ccz q[7], q[0], q[2];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[7];
z q[4];
ccz q[7], q[3], q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
ccz q[1], q[2], q[6];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[1];
z q[6];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
cz q[3], q[0];
z q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
ccz q[4], q[5], q[0];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[6];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];
z q[6];
ccz q[3], q[5], q[0];
ccz q[7], q[4], q[0];
ccz q[1], q[0], q[4];
cz q[5], q[3];
z q[5];
z q[1];
ccz q[2], q[1], q[7];
ccz q[6], q[0], q[1];
z q[0];
z q[7];
z q[2];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7];


// Circuit: n=9, h=1, d=2, g=4
OPENQASM 3.0;
include "stdgates.inc";
qubit[9] q;
z q[7];
z q[1];
z q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];


// Circuit: n=9, h=2, d=6, g=8
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[9] q;
ccz q[6], q[3], q[4];
ccz q[7], q[6], q[2];
cz q[1], q[8];
ccz q[3], q[2], q[8];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[0];
ccz q[8], q[2], q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];


// Circuit: n=9, h=3, d=4, g=6
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[9] q;
ccz q[6], q[7], q[2];
ccz q[3], q[1], q[0];
z q[5];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];


// Circuit: n=9, h=4, d=9, g=20
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[9] q;
z q[1];
cz q[6], q[7];
z q[6];
z q[7];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[7], q[3], q[8];
z q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
cz q[6], q[2];
cz q[3], q[4];
ccz q[5], q[0], q[8];
cz q[0], q[4];
z q[3];
ccz q[6], q[5], q[4];
ccz q[8], q[7], q[2];
ccz q[5], q[3], q[4];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
cz q[3], q[7];
cz q[8], q[1];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];


// Circuit: n=9, h=5, d=9, g=18
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[9] q;
z q[7];
z q[4];
ccz q[8], q[7], q[2];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[3], q[7], q[8];
cz q[0], q[2];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[1], q[0], q[3];
ccz q[4], q[5], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[2];
ccz q[1], q[5], q[0];
z q[7];
z q[0];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[8];
z q[2];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];


// Circuit: n=9, h=6, d=12, g=19
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[9] q;
ccz q[0], q[8], q[1];
cz q[8], q[5];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
cz q[7], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
cz q[2], q[7];
z q[1];
cz q[1], q[0];
cz q[6], q[0];
z q[7];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
cz q[1], q[6];
z q[5];
ccz q[4], q[7], q[2];
z q[3];
z q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];


// Circuit: n=9, h=7, d=14, g=22
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[9] q;
z q[1];
cz q[0], q[3];
ccz q[7], q[1], q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[6];
ccz q[3], q[2], q[4];
cz q[6], q[8];
z q[6];
z q[6];
z q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
cz q[2], q[4];
ccz q[4], q[3], q[8];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
cz q[5], q[1];
ccz q[2], q[0], q[6];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[0], q[6], q[8];
ccz q[8], q[6], q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];


// Circuit: n=9, h=8, d=21, g=42
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[9] q;
z q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
cz q[8], q[0];
ccz q[2], q[1], q[6];
cz q[7], q[3];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[5];
cz q[2], q[4];
cz q[4], q[3];
z q[4];
ccz q[8], q[4], q[5];
z q[7];
ccz q[5], q[6], q[2];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[0], q[5], q[4];
z q[0];
ccz q[1], q[2], q[7];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[7];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
cz q[6], q[1];
ccz q[8], q[3], q[5];
ccz q[6], q[0], q[3];
cz q[1], q[0];
z q[6];
ccz q[6], q[4], q[2];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[3];
z q[4];
z q[5];
cz q[5], q[3];
cz q[8], q[2];
z q[5];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
cz q[6], q[1];
ccz q[3], q[2], q[0];
z q[4];
z q[4];
cz q[2], q[1];
ccz q[4], q[3], q[8];
cz q[1], q[7];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];


// Circuit: n=9, h=9, d=14, g=33
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[9] q;
cz q[3], q[2];
cz q[0], q[5];
z q[7];
z q[8];
ccz q[7], q[3], q[5];
z q[6];
ccz q[2], q[1], q[8];
z q[3];
ccz q[0], q[5], q[8];
cz q[3], q[5];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[5], q[6], q[3];
z q[1];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[0], q[7], q[1];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[6], q[1], q[2];
cz q[1], q[5];
z q[3];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[7];
z q[8];
z q[2];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
cz q[7], q[5];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[1];
cz q[2], q[3];
z q[7];
z q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];


// Circuit: n=9, h=10, d=26, g=46
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[9] q;
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[8];
cz q[7], q[8];
cz q[6], q[8];
z q[8];
z q[1];
z q[5];
cz q[8], q[6];
z q[6];
cz q[0], q[5];
z q[1];
ccz q[2], q[1], q[5];
z q[3];
z q[3];
cz q[5], q[6];
z q[0];
z q[8];
z q[1];
z q[7];
ccz q[5], q[1], q[2];
cz q[1], q[7];
cz q[2], q[1];
ccz q[0], q[4], q[2];
ccz q[2], q[1], q[4];
ccz q[5], q[8], q[6];
ccz q[2], q[0], q[3];
ccz q[6], q[2], q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[3], q[8], q[4];
z q[7];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[8], q[0], q[7];
z q[5];
z q[6];
ccz q[8], q[2], q[6];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[2], q[6], q[5];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
cz q[6], q[8];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[0];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[0], q[6], q[7];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];


// Circuit: n=9, h=11, d=28, g=43
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[9] q;
z q[0];
z q[8];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[8], q[0], q[7];
cz q[1], q[8];
z q[2];
cz q[0], q[6];
ccz q[3], q[6], q[1];
z q[1];
z q[3];
cz q[8], q[7];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[1], q[5], q[8];
ccz q[2], q[7], q[5];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[7], q[2], q[1];
cz q[5], q[2];
z q[0];
ccz q[8], q[3], q[7];
ccz q[2], q[8], q[3];
cz q[3], q[0];
ccz q[0], q[6], q[4];
ccz q[4], q[0], q[3];
ccz q[0], q[5], q[2];
z q[7];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[6];
ccz q[2], q[1], q[4];
cz q[6], q[7];
z q[6];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[5], q[1], q[4];
ccz q[5], q[1], q[2];
ccz q[3], q[5], q[1];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[2], q[6], q[3];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
cz q[2], q[0];
z q[1];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];


// Circuit: n=9, h=12, d=19, g=33
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[9] q;
cz q[3], q[4];
ccz q[0], q[3], q[7];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[3];
cz q[7], q[2];
z q[6];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[0];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[1];
cz q[0], q[5];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[1];
z q[0];
ccz q[5], q[6], q[4];
ccz q[2], q[7], q[5];
z q[2];
cz q[5], q[7];
z q[1];
z q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[2];
z q[8];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[3];
ccz q[0], q[7], q[4];
cz q[2], q[5];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];


// Circuit: n=9, h=13, d=32, g=60
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[9] q;
ccz q[4], q[7], q[3];
cz q[1], q[6];
z q[7];
z q[5];
ccz q[7], q[1], q[2];
cz q[1], q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[8], q[1], q[6];
cz q[4], q[0];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[4];
cz q[7], q[3];
z q[5];
cz q[3], q[2];
z q[6];
cz q[3], q[2];
cz q[7], q[5];
z q[0];
z q[1];
cz q[5], q[4];
cz q[8], q[0];
ccz q[5], q[7], q[3];
z q[7];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[0];
z q[0];
z q[1];
z q[5];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[6];
cz q[3], q[0];
ccz q[0], q[1], q[6];
ccz q[4], q[8], q[0];
z q[1];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[6], q[3], q[0];
cz q[8], q[7];
z q[0];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[6], q[0], q[2];
z q[0];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
cz q[5], q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[4];
z q[7];
ccz q[2], q[5], q[7];
ccz q[4], q[0], q[3];
z q[0];
z q[0];
cz q[5], q[1];
ccz q[5], q[6], q[8];
ccz q[3], q[7], q[0];
ccz q[6], q[3], q[7];
z q[8];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];


// Circuit: n=9, h=14, d=19, g=31
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[9] q;
ccz q[0], q[8], q[4];
z q[2];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[3];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[8], q[5], q[7];
cz q[2], q[3];
z q[7];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[7], q[4], q[2];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[4];
ccz q[6], q[0], q[7];
cz q[7], q[1];
cz q[8], q[2];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
ccz q[4], q[8], q[3];
cz q[6], q[3];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[8];
z q[3];
z q[6];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
z q[6];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8];


// Circuit: n=10, h=1, d=1, g=3
OPENQASM 3.0;
include "stdgates.inc";
qubit[10] q;
cz q[2], q[9];
z q[3];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];


// Circuit: n=10, h=2, d=5, g=6
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[10] q;
cz q[1], q[0];
ccz q[1], q[2], q[4];
ccz q[0], q[7], q[4];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[8], q[2];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];


// Circuit: n=10, h=3, d=6, g=14
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[10] q;
z q[1];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[7];
z q[3];
z q[2];
z q[4];
cz q[3], q[0];
ccz q[8], q[5], q[3];
ccz q[2], q[6], q[1];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[7];
z q[6];
ccz q[3], q[9], q[8];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];


// Circuit: n=10, h=4, d=6, g=15
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[10] q;
ccz q[4], q[8], q[2];
z q[3];
z q[8];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[4];
cz q[0], q[1];
ccz q[2], q[3], q[5];
z q[8];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[8], q[4];
cz q[2], q[9];
ccz q[3], q[6], q[1];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
ccz q[3], q[1], q[4];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];


// Circuit: n=10, h=5, d=6, g=10
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[10] q;
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
ccz q[1], q[4], q[9];
z q[6];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
ccz q[6], q[8], q[4];
z q[6];
ccz q[9], q[0], q[8];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];


// Circuit: n=10, h=6, d=7, g=10
OPENQASM 3.0;
include "stdgates.inc";
qubit[10] q;
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[2], q[8];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[7];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[1];
z q[4];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];


// Circuit: n=10, h=7, d=17, g=27
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[10] q;
cz q[3], q[7];
z q[7];
z q[3];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[5];
z q[6];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
ccz q[4], q[9], q[8];
cz q[6], q[4];
ccz q[9], q[8], q[4];
ccz q[7], q[0], q[8];
cz q[2], q[7];
z q[8];
cz q[4], q[0];
ccz q[0], q[4], q[9];
z q[9];
cz q[5], q[1];
z q[9];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[4], q[7];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[2];
z q[1];
z q[2];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];


// Circuit: n=10, h=8, d=24, g=36
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[10] q;
z q[2];
ccz q[1], q[2], q[3];
z q[6];
ccz q[7], q[9], q[6];
ccz q[8], q[2], q[1];
z q[6];
z q[7];
cz q[7], q[3];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[2], q[1];
ccz q[1], q[7], q[4];
z q[4];
z q[3];
cz q[9], q[8];
ccz q[5], q[2], q[4];
cz q[5], q[2];
cz q[0], q[4];
ccz q[9], q[5], q[1];
z q[8];
cz q[4], q[9];
ccz q[8], q[4], q[3];
ccz q[6], q[8], q[3];
ccz q[0], q[9], q[8];
ccz q[7], q[9], q[6];
ccz q[4], q[5], q[9];
ccz q[8], q[3], q[2];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
ccz q[3], q[1], q[6];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[3], q[7];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[2], q[9];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];


// Circuit: n=10, h=9, d=15, g=26
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[10] q;
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[4];
z q[0];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
ccz q[3], q[7], q[6];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[6];
z q[0];
z q[5];
z q[2];
z q[0];
ccz q[2], q[8], q[7];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[8];
ccz q[0], q[5], q[7];
ccz q[8], q[3], q[2];
z q[8];
cz q[2], q[3];
z q[0];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
ccz q[5], q[3], q[1];
cz q[5], q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];


// Circuit: n=10, h=10, d=22, g=42
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[10] q;
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[2], q[8];
cz q[6], q[4];
z q[5];
z q[7];
z q[0];
cz q[7], q[1];
cz q[1], q[9];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
ccz q[1], q[8], q[4];
cz q[4], q[5];
cz q[6], q[7];
ccz q[7], q[3], q[2];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[8];
ccz q[7], q[6], q[8];
ccz q[3], q[1], q[4];
ccz q[2], q[7], q[4];
z q[0];
cz q[2], q[5];
cz q[2], q[3];
cz q[9], q[6];
cz q[0], q[8];
z q[6];
z q[6];
ccz q[0], q[8], q[7];
cz q[6], q[2];
cz q[0], q[1];
cz q[0], q[4];
z q[0];
ccz q[2], q[4], q[7];
z q[3];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
ccz q[5], q[6], q[1];
ccz q[8], q[0], q[9];
cz q[6], q[4];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];


// Circuit: n=10, h=11, d=16, g=27
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[10] q;
z q[8];
cz q[6], q[9];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
ccz q[5], q[6], q[7];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[9], q[4];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[4], q[8];
ccz q[0], q[3], q[8];
z q[3];
ccz q[5], q[1], q[6];
ccz q[1], q[0], q[3];
cz q[9], q[2];
z q[0];
ccz q[8], q[5], q[4];
ccz q[3], q[6], q[9];
z q[9];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
ccz q[4], q[3], q[9];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
ccz q[7], q[8], q[1];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];


// Circuit: n=10, h=12, d=18, g=25
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[10] q;
cz q[8], q[9];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[3], q[4];
cz q[0], q[9];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[0], q[5];
z q[7];
cz q[2], q[6];
ccz q[8], q[6], q[9];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[9], q[0];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
ccz q[9], q[1], q[3];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[8], q[3];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
ccz q[2], q[1], q[9];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[7];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
ccz q[7], q[9], q[2];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];


// Circuit: n=10, h=13, d=24, g=41
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[10] q;
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[5], q[8];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
ccz q[8], q[3], q[2];
ccz q[5], q[2], q[0];
z q[0];
z q[4];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[9];
z q[6];
ccz q[9], q[6], q[1];
cz q[6], q[9];
z q[1];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[4];
z q[6];
ccz q[4], q[3], q[9];
cz q[7], q[4];
z q[0];
cz q[7], q[4];
z q[7];
ccz q[8], q[5], q[4];
z q[3];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
ccz q[5], q[9], q[1];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[7];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[0], q[1];
z q[5];
cz q[1], q[7];
ccz q[9], q[8], q[5];
ccz q[0], q[7], q[6];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[5];
ccz q[1], q[9], q[6];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];


// Circuit: n=10, h=14, d=33, g=67
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[10] q;
ccz q[4], q[5], q[7];
z q[6];
z q[6];
z q[6];
ccz q[2], q[9], q[6];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[6], q[2];
z q[0];
z q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
ccz q[5], q[6], q[4];
z q[7];
z q[9];
z q[5];
cz q[4], q[2];
ccz q[5], q[7], q[2];
z q[8];
z q[1];
cz q[8], q[1];
cz q[6], q[5];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[4], q[3];
z q[0];
z q[6];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[8];
z q[9];
z q[6];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
ccz q[6], q[7], q[9];
ccz q[6], q[9], q[7];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[2];
cz q[5], q[2];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[2];
z q[1];
z q[7];
z q[3];
z q[4];
ccz q[9], q[4], q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[4], q[7];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
z q[5];
cz q[0], q[1];
ccz q[3], q[2], q[0];
z q[4];
ccz q[3], q[4], q[9];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[0], q[1];
cz q[8], q[6];
cz q[5], q[7];
ccz q[6], q[8], q[1];
ccz q[1], q[3], q[7];
z q[0];
z q[6];
z q[9];
z q[4];
ccz q[9], q[5], q[3];
ccz q[1], q[6], q[3];
cz q[5], q[1];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];
cz q[6], q[0];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9];


// Circuit: n=11, h=1, d=1, g=2
OPENQASM 3.0;
include "stdgates.inc";
qubit[11] q;
z q[7];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];


// Circuit: n=11, h=2, d=4, g=6
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[11] q;
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[6];
ccz q[0], q[8], q[7];
cz q[0], q[1];
z q[8];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];


// Circuit: n=11, h=3, d=4, g=6
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[11] q;
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
ccz q[8], q[2], q[1];
z q[8];
ccz q[9], q[7], q[10];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];


// Circuit: n=11, h=4, d=8, g=12
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[11] q;
ccz q[5], q[4], q[8];
cz q[7], q[8];
z q[5];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
ccz q[4], q[7], q[8];
ccz q[10], q[4], q[7];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
ccz q[5], q[8], q[4];
ccz q[3], q[9], q[2];
ccz q[10], q[5], q[9];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];


// Circuit: n=11, h=5, d=7, g=9
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[11] q;
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[4];
ccz q[5], q[10], q[2];
cz q[4], q[3];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[7];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];


// Circuit: n=11, h=6, d=8, g=12
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[11] q;
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
ccz q[9], q[0], q[6];
cz q[4], q[7];
z q[3];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[5];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
ccz q[7], q[9], q[5];
z q[8];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];


// Circuit: n=11, h=7, d=19, g=44
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[11] q;
ccz q[5], q[2], q[1];
z q[4];
ccz q[6], q[0], q[7];
z q[8];
cz q[9], q[6];
z q[3];
cz q[0], q[5];
ccz q[6], q[2], q[5];
cz q[9], q[4];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
cz q[5], q[7];
ccz q[8], q[10], q[3];
ccz q[1], q[4], q[7];
z q[7];
z q[4];
z q[7];
ccz q[5], q[2], q[7];
z q[3];
z q[1];
z q[2];
ccz q[4], q[10], q[6];
z q[0];
z q[3];
z q[5];
z q[2];
cz q[5], q[1];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[10];
z q[3];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[9];
z q[10];
cz q[3], q[5];
cz q[6], q[8];
z q[3];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[2];
cz q[0], q[5];
cz q[4], q[1];
z q[6];
ccz q[8], q[5], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];


// Circuit: n=11, h=8, d=17, g=29
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[11] q;
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
ccz q[1], q[7], q[2];
z q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
ccz q[2], q[8], q[9];
cz q[2], q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
cz q[8], q[1];
ccz q[9], q[3], q[1];
z q[8];
ccz q[9], q[7], q[1];
ccz q[7], q[0], q[8];
z q[7];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
cz q[0], q[10];
ccz q[1], q[10], q[5];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[3];
z q[6];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[7];
cz q[9], q[3];
cz q[9], q[6];
z q[5];
cz q[1], q[6];
cz q[8], q[7];
cz q[4], q[0];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];


// Circuit: n=11, h=9, d=15, g=26
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[11] q;
ccz q[2], q[0], q[8];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
ccz q[1], q[9], q[4];
ccz q[4], q[7], q[2];
ccz q[7], q[2], q[3];
cz q[7], q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
cz q[9], q[4];
cz q[0], q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[7];
z q[8];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
ccz q[0], q[1], q[10];
cz q[8], q[2];
ccz q[7], q[9], q[1];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
cz q[6], q[7];
z q[8];
ccz q[3], q[1], q[10];
cz q[10], q[9];
cz q[10], q[0];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];


// Circuit: n=11, h=10, d=13, g=25
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[11] q;
z q[6];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
ccz q[2], q[6], q[0];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
ccz q[10], q[1], q[5];
z q[9];
z q[1];
cz q[3], q[2];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
cz q[3], q[2];
cz q[5], q[8];
cz q[7], q[6];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[6];
cz q[5], q[10];
z q[8];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[1];
ccz q[0], q[4], q[5];
cz q[10], q[2];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];


// Circuit: n=11, h=11, d=15, g=29
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[11] q;
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[6];
ccz q[1], q[5], q[8];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[0];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
ccz q[4], q[7], q[6];
z q[7];
z q[6];
ccz q[9], q[1], q[10];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[10];
z q[1];
cz q[7], q[9];
z q[4];
cz q[0], q[6];
ccz q[0], q[5], q[2];
ccz q[1], q[7], q[4];
ccz q[1], q[2], q[10];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
ccz q[9], q[3], q[7];
ccz q[2], q[9], q[10];
ccz q[4], q[0], q[7];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];


// Circuit: n=11, h=12, d=24, g=45
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[11] q;
cz q[10], q[5];
ccz q[7], q[10], q[8];
z q[5];
cz q[5], q[1];
z q[3];
ccz q[3], q[2], q[10];
cz q[5], q[1];
ccz q[7], q[4], q[10];
z q[5];
z q[6];
z q[9];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[4];
ccz q[9], q[2], q[3];
ccz q[4], q[5], q[10];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[8];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[8];
z q[6];
z q[6];
ccz q[1], q[4], q[6];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[9];
ccz q[6], q[4], q[8];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[1];
ccz q[1], q[8], q[9];
cz q[1], q[7];
cz q[4], q[2];
z q[1];
cz q[7], q[5];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
cz q[5], q[6];
z q[1];
cz q[5], q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
cz q[6], q[8];
ccz q[5], q[9], q[10];
ccz q[4], q[9], q[1];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];


// Circuit: n=11, h=13, d=20, g=38
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[11] q;
z q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
cz q[9], q[1];
z q[8];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
cz q[6], q[3];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[7];
z q[3];
cz q[7], q[0];
z q[9];
z q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[0];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
cz q[6], q[8];
ccz q[6], q[7], q[8];
ccz q[3], q[9], q[7];
z q[2];
z q[4];
z q[1];
cz q[8], q[6];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[2];
z q[6];
z q[6];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
ccz q[2], q[6], q[0];
ccz q[2], q[4], q[3];
z q[9];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
ccz q[8], q[2], q[4];
z q[5];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];


// Circuit: n=11, h=14, d=26, g=61
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[11] q;
ccz q[9], q[7], q[8];
z q[1];
z q[6];
z q[5];
z q[5];
z q[8];
z q[6];
z q[4];
z q[5];
cz q[1], q[8];
z q[10];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[7];
z q[2];
z q[3];
z q[3];
z q[8];
z q[6];
z q[0];
cz q[5], q[10];
ccz q[10], q[0], q[3];
ccz q[0], q[7], q[5];
z q[8];
cz q[0], q[9];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
cz q[10], q[3];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
cz q[4], q[10];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
z q[0];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
ccz q[8], q[2], q[4];
z q[6];
cz q[9], q[4];
z q[6];
z q[9];
z q[0];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
cz q[2], q[4];
z q[8];
ccz q[1], q[5], q[10];
z q[1];
z q[9];
ccz q[3], q[9], q[7];
cz q[2], q[5];
cz q[2], q[3];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
ccz q[7], q[10], q[9];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
ccz q[8], q[7], q[2];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
ccz q[9], q[7], q[4];
z q[1];
ccz q[4], q[0], q[6];
ccz q[8], q[1], q[7];
z q[10];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10];


// Circuit: n=12, h=1, d=1, g=1
OPENQASM 3.0;
include "stdgates.inc";
qubit[12] q;
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];


// Circuit: n=12, h=2, d=7, g=14
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[12] q;
cz q[3], q[5];
ccz q[5], q[1], q[2];
ccz q[5], q[4], q[9];
z q[9];
z q[8];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[8], q[10];
z q[6];
ccz q[11], q[2], q[8];
cz q[2], q[8];
z q[7];
z q[7];
z q[11];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];


// Circuit: n=12, h=3, d=3, g=5
OPENQASM 3.0;
include "stdgates.inc";
qubit[12] q;
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[3];
z q[0];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];


// Circuit: n=12, h=4, d=6, g=14
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[12] q;
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[11];
z q[1];
ccz q[4], q[9], q[6];
ccz q[1], q[8], q[0];
z q[9];
z q[7];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[0];
ccz q[7], q[3], q[2];
ccz q[2], q[4], q[6];
ccz q[9], q[8], q[3];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];


// Circuit: n=12, h=5, d=6, g=11
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[12] q;
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[3];
z q[11];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[4], q[11];
z q[7];
ccz q[4], q[5], q[8];
z q[1];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];


// Circuit: n=12, h=6, d=14, g=36
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[12] q;
cz q[0], q[8];
z q[9];
cz q[5], q[0];
z q[1];
cz q[9], q[7];
z q[11];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[9];
cz q[10], q[5];
ccz q[4], q[8], q[2];
z q[11];
z q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[0];
cz q[11], q[3];
z q[11];
z q[1];
z q[11];
ccz q[1], q[10], q[0];
ccz q[8], q[2], q[7];
cz q[1], q[2];
z q[1];
cz q[3], q[11];
z q[5];
ccz q[5], q[8], q[1];
ccz q[3], q[10], q[1];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[0];
ccz q[7], q[1], q[6];
cz q[3], q[11];
z q[2];
ccz q[3], q[4], q[5];
z q[2];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];


// Circuit: n=12, h=7, d=10, g=16
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[12] q;
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[5], q[0];
cz q[8], q[9];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
ccz q[7], q[9], q[10];
z q[8];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[5], q[8];
cz q[10], q[8];
z q[1];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[2];
ccz q[5], q[2], q[4];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];


// Circuit: n=12, h=8, d=20, g=45
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[12] q;
z q[2];
z q[6];
z q[2];
cz q[9], q[5];
z q[10];
cz q[1], q[6];
z q[2];
cz q[10], q[9];
z q[7];
z q[2];
ccz q[3], q[6], q[2];
z q[5];
cz q[0], q[4];
z q[2];
ccz q[8], q[11], q[2];
cz q[5], q[8];
cz q[9], q[1];
z q[2];
z q[0];
ccz q[5], q[4], q[6];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[8];
ccz q[11], q[9], q[2];
ccz q[3], q[5], q[2];
z q[2];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
ccz q[1], q[9], q[4];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
ccz q[5], q[3], q[7];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[2], q[8];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
ccz q[2], q[10], q[11];
z q[6];
z q[8];
cz q[3], q[2];
z q[6];
ccz q[5], q[4], q[2];
z q[0];
z q[1];
cz q[8], q[9];
cz q[0], q[6];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];


// Circuit: n=12, h=9, d=11, g=25
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[12] q;
ccz q[3], q[11], q[0];
cz q[5], q[10];
z q[6];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
ccz q[4], q[3], q[8];
z q[7];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[8], q[1];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[4], q[0];
ccz q[0], q[7], q[10];
z q[6];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[1], q[7];
z q[8];
cz q[9], q[10];
ccz q[4], q[9], q[7];
z q[5];
z q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[6];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];


// Circuit: n=12, h=10, d=21, g=41
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[12] q;
z q[11];
z q[0];
z q[10];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[7], q[0];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[2], q[11];
ccz q[9], q[5], q[3];
cz q[7], q[6];
z q[5];
ccz q[10], q[2], q[9];
cz q[8], q[2];
z q[8];
ccz q[0], q[2], q[5];
z q[8];
cz q[0], q[11];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
ccz q[6], q[0], q[3];
z q[7];
cz q[6], q[10];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[6];
ccz q[8], q[7], q[3];
z q[5];
cz q[2], q[0];
z q[3];
ccz q[2], q[8], q[1];
cz q[5], q[0];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[3];
z q[11];
ccz q[4], q[9], q[2];
cz q[0], q[10];
ccz q[6], q[10], q[11];
cz q[0], q[9];
cz q[11], q[8];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];


// Circuit: n=12, h=11, d=20, g=31
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[12] q;
ccz q[6], q[9], q[10];
cz q[3], q[9];
ccz q[4], q[5], q[10];
ccz q[4], q[11], q[9];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[5], q[9];
z q[6];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[6];
cz q[7], q[0];
ccz q[5], q[6], q[11];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
ccz q[0], q[3], q[9];
cz q[0], q[3];
cz q[10], q[3];
cz q[4], q[10];
ccz q[2], q[1], q[3];
z q[9];
z q[1];
z q[4];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
ccz q[2], q[0], q[11];
z q[1];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
ccz q[6], q[8], q[11];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];


// Circuit: n=12, h=12, d=21, g=44
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[12] q;
z q[5];
ccz q[1], q[3], q[10];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[11];
ccz q[2], q[0], q[11];
z q[3];
ccz q[0], q[6], q[1];
cz q[10], q[11];
cz q[7], q[10];
cz q[9], q[0];
cz q[9], q[6];
ccz q[1], q[9], q[5];
cz q[0], q[3];
ccz q[1], q[2], q[3];
z q[6];
cz q[11], q[6];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[0];
cz q[4], q[9];
z q[10];
ccz q[1], q[8], q[11];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
ccz q[6], q[7], q[1];
cz q[3], q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
ccz q[5], q[6], q[10];
ccz q[3], q[5], q[11];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[4];
z q[6];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[7];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[10];
ccz q[7], q[1], q[9];
z q[11];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[10];
ccz q[10], q[8], q[6];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[11];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];


// Circuit: n=12, h=13, d=23, g=33
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[12] q;
ccz q[9], q[11], q[1];
cz q[9], q[5];
cz q[8], q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[0];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[7];
ccz q[5], q[11], q[7];
z q[2];
ccz q[1], q[9], q[11];
z q[10];
z q[8];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[6], q[4];
ccz q[9], q[1], q[2];
z q[9];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[10];
ccz q[7], q[1], q[6];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
ccz q[1], q[6], q[8];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[3];
z q[11];
ccz q[3], q[5], q[4];
cz q[9], q[3];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];


// Circuit: n=12, h=14, d=28, g=50
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[12] q;
ccz q[11], q[10], q[7];
z q[11];
cz q[9], q[5];
cz q[11], q[0];
z q[5];
z q[8];
z q[7];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[8], q[11];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
ccz q[4], q[10], q[9];
ccz q[9], q[6], q[5];
cz q[2], q[5];
ccz q[4], q[5], q[2];
ccz q[0], q[7], q[9];
cz q[5], q[3];
cz q[9], q[7];
ccz q[3], q[0], q[4];
z q[6];
ccz q[2], q[4], q[7];
z q[5];
cz q[5], q[11];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[8];
ccz q[0], q[6], q[8];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[3], q[4];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[9];
cz q[9], q[7];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
ccz q[8], q[6], q[9];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[10];
ccz q[7], q[1], q[4];
cz q[1], q[0];
z q[10];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[1];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[9], q[1];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[1], q[5];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[11];
cz q[7], q[9];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[6], q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];


// Circuit: n=13, h=1, d=4, g=9
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[13] q;
cz q[11], q[0];
z q[9];
z q[6];
z q[8];
ccz q[5], q[1], q[7];
ccz q[10], q[3], q[5];
cz q[8], q[5];
cz q[9], q[2];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];


// Circuit: n=13, h=2, d=3, g=6
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[13] q;
z q[5];
ccz q[2], q[4], q[6];
ccz q[6], q[11], q[8];
z q[9];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];


// Circuit: n=13, h=3, d=5, g=14
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[13] q;
cz q[9], q[10];
z q[2];
z q[2];
cz q[9], q[10];
cz q[3], q[11];
z q[8];
ccz q[8], q[5], q[7];
z q[4];
ccz q[11], q[1], q[3];
cz q[8], q[5];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[1];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];


// Circuit: n=13, h=4, d=4, g=6
OPENQASM 3.0;
include "stdgates.inc";
qubit[13] q;
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
cz q[9], q[0];
cz q[2], q[5];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];


// Circuit: n=13, h=5, d=10, g=18
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[13] q;
z q[5];
cz q[9], q[0];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
ccz q[10], q[8], q[11];
z q[12];
cz q[10], q[7];
ccz q[6], q[12], q[4];
cz q[12], q[2];
ccz q[6], q[12], q[5];
cz q[6], q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[12];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
ccz q[3], q[2], q[11];
ccz q[2], q[10], q[0];
cz q[9], q[4];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];


// Circuit: n=13, h=6, d=14, g=26
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[13] q;
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[4];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
ccz q[6], q[1], q[11];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[7];
z q[8];
z q[8];
cz q[6], q[7];
ccz q[0], q[6], q[5];
ccz q[2], q[11], q[10];
cz q[4], q[8];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
ccz q[3], q[11], q[10];
ccz q[9], q[1], q[3];
z q[12];
z q[11];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[10];
cz q[7], q[9];
cz q[12], q[6];
ccz q[5], q[7], q[1];
cz q[0], q[5];
ccz q[8], q[11], q[1];
z q[6];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];


// Circuit: n=13, h=7, d=16, g=35
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[13] q;
cz q[7], q[10];
h q[12];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[12];
cz q[10], q[7];
z q[4];
cz q[9], q[11];
z q[4];
cz q[9], q[0];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
cz q[7], q[11];
ccz q[10], q[3], q[11];
ccz q[3], q[2], q[0];
ccz q[11], q[10], q[9];
z q[6];
cz q[0], q[2];
h q[12];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
ccz q[4], q[0], q[10];
z q[3];
z q[3];
ccz q[10], q[6], q[11];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[10];
ccz q[10], q[12], q[4];
cz q[11], q[8];
z q[8];
cz q[7], q[0];
cz q[3], q[2];
z q[2];
cz q[0], q[10];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
ccz q[2], q[0], q[10];
z q[4];
cz q[1], q[11];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];


// Circuit: n=13, h=8, d=10, g=20
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[13] q;
ccz q[5], q[0], q[1];
z q[3];
cz q[12], q[2];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[1];
z q[8];
z q[11];
ccz q[10], q[3], q[7];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[4];
cz q[0], q[11];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
ccz q[10], q[5], q[8];
ccz q[5], q[7], q[1];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
ccz q[10], q[12], q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];


// Circuit: n=13, h=9, d=18, g=33
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[13] q;
cz q[2], q[7];
cz q[12], q[8];
ccz q[12], q[9], q[3];
z q[12];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
cz q[5], q[0];
z q[1];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
cz q[5], q[2];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
cz q[2], q[4];
z q[12];
z q[2];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[12];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
ccz q[8], q[5], q[0];
z q[2];
z q[7];
z q[0];
ccz q[0], q[11], q[3];
z q[2];
z q[2];
ccz q[9], q[2], q[8];
z q[10];
ccz q[11], q[1], q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[1];
cz q[11], q[2];
ccz q[8], q[0], q[5];
z q[3];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];


// Circuit: n=13, h=10, d=19, g=39
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[13] q;
cz q[5], q[11];
ccz q[11], q[9], q[1];
z q[11];
z q[6];
ccz q[6], q[4], q[7];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[5];
z q[4];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[8];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[0];
cz q[10], q[6];
z q[7];
cz q[1], q[11];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
ccz q[11], q[7], q[3];
z q[5];
cz q[12], q[8];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
ccz q[5], q[7], q[3];
z q[3];
cz q[1], q[0];
z q[7];
ccz q[7], q[2], q[11];
cz q[3], q[1];
z q[2];
ccz q[1], q[3], q[0];
z q[1];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
ccz q[5], q[8], q[3];
ccz q[11], q[6], q[8];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
cz q[11], q[5];
z q[3];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[0];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];


// Circuit: n=13, h=11, d=17, g=30
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[13] q;
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
ccz q[11], q[9], q[0];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[6];
ccz q[8], q[6], q[2];
ccz q[10], q[5], q[1];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[10];
z q[3];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[10];
z q[5];
z q[10];
z q[12];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
cz q[0], q[10];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
cz q[0], q[6];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[2];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
cz q[2], q[9];
z q[1];
z q[6];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[5];
z q[4];
ccz q[3], q[7], q[1];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];


// Circuit: n=13, h=12, d=17, g=26
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[13] q;
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[11];
z q[3];
z q[7];
ccz q[4], q[3], q[10];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
ccz q[3], q[6], q[12];
z q[9];
z q[11];
ccz q[3], q[9], q[7];
h q[12];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
cz q[3], q[12];
z q[6];
cz q[9], q[6];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
ccz q[8], q[6], q[4];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[11];
z q[1];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];


// Circuit: n=13, h=13, d=14, g=25
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[13] q;
cz q[1], q[5];
ccz q[11], q[3], q[6];
z q[9];
z q[3];
cz q[12], q[0];
z q[8];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[5];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
cz q[5], q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[12];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
cz q[6], q[4];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[1];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
ccz q[11], q[4], q[6];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];


// Circuit: n=13, h=14, d=25, g=51
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[13] q;
z q[12];
ccz q[11], q[8], q[3];
cz q[11], q[4];
cz q[11], q[2];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
cz q[9], q[5];
cz q[4], q[11];
ccz q[12], q[2], q[9];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[4];
cz q[10], q[5];
z q[7];
z q[7];
cz q[1], q[12];
cz q[9], q[4];
z q[2];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
ccz q[11], q[10], q[5];
z q[4];
z q[7];
cz q[7], q[9];
z q[8];
z q[10];
ccz q[1], q[5], q[7];
z q[10];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[11];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
ccz q[0], q[6], q[8];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[7];
cz q[10], q[11];
ccz q[5], q[6], q[11];
cz q[12], q[4];
cz q[4], q[7];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[5];
ccz q[11], q[7], q[1];
z q[8];
z q[3];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[12];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[11];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[3];
z q[8];
h q[12];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];
z q[11];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12];


// Circuit: n=14, h=1, d=3, g=6
OPENQASM 3.0;
include "stdgates.inc";
qubit[14] q;
cz q[5], q[11];
cz q[2], q[5];
z q[0];
cz q[8], q[11];
cz q[10], q[2];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];


// Circuit: n=14, h=2, d=8, g=19
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[14] q;
z q[13];
ccz q[12], q[10], q[13];
z q[11];
cz q[10], q[11];
cz q[10], q[11];
z q[9];
ccz q[1], q[6], q[13];
z q[1];
ccz q[0], q[1], q[3];
cz q[0], q[1];
ccz q[13], q[11], q[5];
z q[9];
z q[2];
z q[6];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[6];
ccz q[3], q[4], q[1];
cz q[6], q[1];
h q[13];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];


// Circuit: n=14, h=3, d=3, g=8
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[14] q;
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[3];
z q[12];
ccz q[5], q[1], q[8];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[3];
cz q[1], q[10];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];


// Circuit: n=14, h=4, d=6, g=12
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[14] q;
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[5];
z q[8];
ccz q[4], q[3], q[10];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
ccz q[13], q[10], q[6];
z q[6];
ccz q[1], q[4], q[3];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[2];
ccz q[13], q[3], q[1];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];


// Circuit: n=14, h=5, d=9, g=21
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[14] q;
h q[13];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
cz q[0], q[12];
ccz q[3], q[2], q[6];
z q[3];
ccz q[13], q[10], q[1];
cz q[9], q[1];
z q[11];
z q[8];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
ccz q[5], q[9], q[8];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[8];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
ccz q[2], q[4], q[10];
cz q[11], q[0];
cz q[1], q[6];
cz q[12], q[9];
ccz q[7], q[1], q[11];
z q[6];
z q[1];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];


// Circuit: n=14, h=6, d=9, g=27
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[14] q;
z q[2];
cz q[8], q[11];
z q[13];
cz q[5], q[1];
cz q[10], q[11];
z q[13];
z q[12];
z q[5];
z q[10];
z q[3];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[1];
z q[12];
ccz q[8], q[5], q[11];
z q[8];
z q[12];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[3];
z q[5];
z q[8];
ccz q[2], q[1], q[12];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
ccz q[11], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
ccz q[7], q[6], q[5];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];


// Circuit: n=14, h=7, d=9, g=15
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[14] q;
ccz q[0], q[5], q[2];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[6];
ccz q[8], q[11], q[10];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
ccz q[2], q[0], q[5];
z q[8];
ccz q[11], q[3], q[1];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[13];
ccz q[7], q[13], q[3];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];


// Circuit: n=14, h=8, d=10, g=23
OPENQASM 3.0;
include "stdgates.inc";
qubit[14] q;
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
cz q[7], q[1];
z q[6];
cz q[6], q[7];
z q[1];
cz q[2], q[12];
z q[0];
cz q[10], q[8];
z q[5];
cz q[2], q[4];
z q[1];
z q[4];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[0];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[4];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
cz q[0], q[7];
z q[3];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];


// Circuit: n=14, h=9, d=14, g=27
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[14] q;
z q[9];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
cz q[1], q[4];
z q[6];
z q[10];
cz q[2], q[4];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[11];
ccz q[5], q[4], q[12];
z q[11];
cz q[11], q[1];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[6];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
cz q[8], q[4];
ccz q[13], q[9], q[4];
z q[12];
z q[4];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
cz q[12], q[13];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[1];
z q[0];
h q[12];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[12];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];


// Circuit: n=14, h=10, d=24, g=58
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[14] q;
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[12];
z q[6];
z q[9];
z q[7];
z q[13];
cz q[11], q[7];
cz q[7], q[0];
ccz q[1], q[6], q[13];
cz q[1], q[6];
z q[6];
z q[10];
z q[8];
ccz q[6], q[1], q[11];
cz q[11], q[0];
z q[9];
ccz q[8], q[4], q[7];
cz q[9], q[2];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
ccz q[2], q[12], q[0];
ccz q[6], q[11], q[13];
z q[10];
cz q[10], q[7];
ccz q[1], q[3], q[7];
ccz q[0], q[9], q[11];
cz q[5], q[6];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
ccz q[2], q[13], q[9];
z q[1];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
ccz q[6], q[12], q[2];
ccz q[1], q[2], q[0];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[0];
ccz q[13], q[4], q[8];
z q[1];
z q[11];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[7];
z q[1];
ccz q[0], q[9], q[3];
z q[11];
z q[3];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
cz q[5], q[2];
z q[10];
z q[8];
z q[12];
z q[1];
ccz q[5], q[12], q[6];
cz q[12], q[1];
z q[6];
cz q[6], q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[2];
h q[13];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
ccz q[0], q[2], q[12];
h q[12];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];


// Circuit: n=14, h=11, d=13, g=27
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[14] q;
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
cz q[8], q[1];
ccz q[7], q[13], q[4];
cz q[5], q[1];
ccz q[11], q[7], q[0];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
cz q[12], q[7];
z q[13];
z q[8];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
cz q[9], q[4];
h q[13];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
ccz q[5], q[1], q[12];
cz q[3], q[11];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
ccz q[2], q[5], q[12];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
cz q[11], q[0];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
h q[13];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[12];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[10];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
ccz q[1], q[0], q[8];
cz q[1], q[0];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];


// Circuit: n=14, h=12, d=20, g=37
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[14] q;
cz q[8], q[5];
z q[5];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
ccz q[1], q[11], q[7];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[2];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[10];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
ccz q[3], q[11], q[10];
z q[9];
z q[10];
ccz q[13], q[3], q[7];
z q[11];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[8];
ccz q[13], q[4], q[9];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
ccz q[1], q[2], q[0];
z q[0];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[7];
cz q[5], q[10];
cz q[7], q[11];
z q[7];
ccz q[8], q[12], q[3];
ccz q[10], q[4], q[12];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[8];
cz q[11], q[0];
ccz q[10], q[4], q[12];
ccz q[12], q[3], q[2];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[13];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];


// Circuit: n=14, h=13, d=21, g=46
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[14] q;
ccz q[1], q[6], q[9];
cz q[8], q[4];
cz q[6], q[1];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[7];
cz q[1], q[11];
z q[9];
cz q[11], q[12];
cz q[11], q[4];
z q[10];
cz q[3], q[8];
z q[10];
cz q[6], q[0];
z q[12];
z q[12];
z q[11];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
cz q[2], q[8];
cz q[8], q[10];
z q[6];
cz q[13], q[9];
cz q[6], q[5];
z q[0];
cz q[4], q[13];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[6];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[9];
cz q[8], q[4];
z q[7];
z q[7];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
cz q[10], q[1];
z q[11];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[12];
ccz q[1], q[10], q[11];
ccz q[1], q[9], q[12];
cz q[9], q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];


// Circuit: n=14, h=14, d=24, g=48
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[14] q;
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
ccz q[10], q[5], q[2];
z q[2];
z q[3];
cz q[7], q[1];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[13];
ccz q[13], q[7], q[5];
cz q[6], q[5];
z q[4];
cz q[11], q[8];
ccz q[3], q[5], q[13];
z q[13];
z q[5];
z q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
cz q[11], q[9];
cz q[6], q[0];
z q[4];
z q[13];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
ccz q[11], q[3], q[5];
h q[13];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[13];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
cz q[7], q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
cz q[9], q[13];
cz q[6], q[0];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
z q[8];
z q[5];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
ccz q[10], q[4], q[12];
z q[12];
cz q[13], q[6];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
cz q[0], q[11];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
cz q[3], q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];
cz q[8], q[0];
cz q[5], q[12];
z q[1];
ccz q[2], q[8], q[13];
z q[7];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13];

// Circuit: n=15, h=15, d=24, g=49
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[15] q;
z q[12];
z q[4];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14];
cz q[9], q[13];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14];
z q[9];
z q[10];
z q[3];
z q[7];
cz q[2], q[7];
z q[5];
ccz q[13], q[0], q[2];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14];
cz q[9], q[4];
ccz q[4], q[13], q[3];
ccz q[9], q[14], q[3];
h q[12];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14];
cz q[14], q[0];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14];
ccz q[3], q[10], q[5];
h q[12];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14];
cz q[7], q[8];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14];
z q[14];
ccz q[1], q[13], q[12];
cz q[8], q[4];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14];
z q[5];
cz q[7], q[12];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14];
ccz q[1], q[0], q[12];
z q[11];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14];
ccz q[7], q[13], q[1];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14];
cz q[6], q[0];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14];
z q[7];
ccz q[11], q[6], q[14];
cz q[12], q[7];
cz q[7], q[1];
z q[12];
z q[11];
cz q[7], q[9];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14];
z q[11];
cz q[8], q[13];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14];


// Circuit: n=20, h=15, d=23, g=51
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[20] q;
h q[13];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19];
ccz q[12], q[6], q[19];
cz q[14], q[10];
ccz q[10], q[5], q[9];
z q[0];
h q[14];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19];
ccz q[8], q[3], q[12];
z q[6];
z q[6];
z q[5];
cz q[19], q[17];
h q[15];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19];
cz q[11], q[10];
z q[3];
cz q[7], q[1];
ccz q[14], q[2], q[11];
ccz q[9], q[12], q[17];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19];
z q[9];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19];
ccz q[7], q[9], q[5];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19];
ccz q[7], q[17], q[5];
z q[2];
cz q[11], q[19];
ccz q[13], q[1], q[5];
h q[17];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19];
z q[0];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19];
ccz q[17], q[11], q[19];
ccz q[11], q[5], q[14];
ccz q[1], q[17], q[11];
ccz q[19], q[0], q[12];
z q[9];
z q[15];
ccz q[0], q[18], q[2];
z q[16];
ccz q[15], q[19], q[0];
z q[7];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19];
ccz q[15], q[11], q[16];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19];
h q[17];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19];
h q[15];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19];
ccz q[12], q[0], q[14];
cz q[9], q[4];
cz q[6], q[8];
cz q[7], q[15];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19];


// Circuit: n=22, h=15, d=26, g=67
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[22] q;
h q[20];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21];
ccz q[17], q[12], q[8];
cz q[14], q[12];
h q[18];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21];
ccz q[13], q[8], q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21];
h q[20];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21];
cz q[20], q[14];
z q[12];
cz q[1], q[5];
z q[19];
z q[16];
cz q[8], q[17];
h q[16];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21];
z q[21];
z q[4];
z q[5];
cz q[15], q[4];
h q[17];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21];
ccz q[14], q[19], q[7];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21];
z q[0];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21];
z q[14];
cz q[19], q[13];
z q[20];
ccz q[1], q[3], q[6];
z q[21];
cz q[3], q[21];
z q[1];
z q[6];
cz q[18], q[1];
ccz q[6], q[8], q[12];
ccz q[6], q[19], q[17];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21];
ccz q[13], q[5], q[8];
cz q[21], q[10];
z q[20];
z q[20];
z q[5];
z q[10];
ccz q[8], q[14], q[13];
z q[9];
h q[16];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21];
cz q[11], q[15];
cz q[8], q[15];
ccz q[14], q[21], q[5];
z q[10];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21];
z q[1];
z q[6];
z q[3];
ccz q[15], q[2], q[21];
cz q[8], q[13];
z q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21];
z q[16];
cz q[7], q[16];
ccz q[15], q[11], q[14];
ccz q[4], q[14], q[8];
h q[15];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21];
ccz q[20], q[19], q[18];
ccz q[16], q[15], q[13];
z q[15];
ccz q[10], q[8], q[17];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21];


// Circuit: n=23, h=15, d=20, g=48
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[23] q;
ccz q[6], q[5], q[1];
ccz q[3], q[2], q[1];
h q[15];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22];
cz q[15], q[17];
ccz q[14], q[9], q[19];
z q[10];
ccz q[17], q[14], q[5];
z q[22];
ccz q[12], q[7], q[3];
z q[7];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22];
z q[18];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22];
z q[22];
ccz q[11], q[21], q[17];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22];
z q[21];
h q[19];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22];
h q[12];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22];
h q[13];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22];
ccz q[21], q[11], q[12];
ccz q[19], q[1], q[0];
h q[18];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22];
h q[12];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22];
z q[14];
z q[18];
z q[8];
ccz q[21], q[8], q[22];
ccz q[10], q[0], q[3];
ccz q[5], q[0], q[19];
h q[13];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22];
z q[8];
z q[12];
cz q[18], q[17];
h q[14];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22];
cz q[22], q[8];
ccz q[6], q[8], q[15];
z q[21];
cz q[9], q[5];
cz q[17], q[20];
ccz q[15], q[9], q[22];
ccz q[5], q[21], q[1];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22];
z q[8];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22];
ccz q[8], q[20], q[13];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22];


// Circuit: n=24, h=15, d=18, g=42
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[24] q;
h q[20];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
cz q[16], q[7];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
h q[15];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
ccz q[23], q[10], q[4];
ccz q[17], q[12], q[1];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
z q[13];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
z q[18];
z q[12];
h q[14];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
cz q[18], q[11];
z q[2];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
z q[6];
ccz q[23], q[8], q[21];
ccz q[18], q[6], q[11];
h q[12];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
z q[2];
cz q[18], q[23];
z q[17];
h q[14];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
ccz q[2], q[15], q[13];
z q[7];
ccz q[3], q[18], q[19];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
ccz q[15], q[14], q[8];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
z q[15];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
z q[23];
z q[22];
ccz q[19], q[10], q[14];
cz q[0], q[17];
z q[0];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
cz q[15], q[7];
z q[10];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
ccz q[8], q[9], q[17];
h q[23];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];


// Circuit: n=25, h=15, d=21, g=37
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[25] q;
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
h q[19];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
cz q[22], q[24];
cz q[9], q[8];
cz q[13], q[23];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
cz q[7], q[23];
h q[24];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
ccz q[17], q[6], q[8];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
ccz q[15], q[0], q[16];
h q[21];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
z q[19];
ccz q[7], q[14], q[8];
z q[14];
cz q[6], q[5];
z q[0];
z q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
cz q[15], q[0];
z q[23];
z q[23];
cz q[1], q[10];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
z q[11];
ccz q[15], q[18], q[22];
ccz q[8], q[13], q[2];
ccz q[20], q[18], q[3];
h q[18];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
h q[19];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
ccz q[13], q[7], q[3];
cz q[2], q[18];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];


// Circuit: n=24, h=15, d=18, g=40
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[24] q;
ccz q[14], q[18], q[4];
h q[15];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
cz q[20], q[5];
ccz q[2], q[22], q[16];
ccz q[8], q[4], q[1];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
h q[18];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
h q[21];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
z q[19];
cz q[3], q[10];
z q[3];
z q[0];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
ccz q[6], q[22], q[3];
h q[21];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
h q[17];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
z q[1];
z q[0];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
z q[19];
h q[17];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
z q[3];
h q[23];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
ccz q[20], q[23], q[0];
cz q[14], q[7];
ccz q[19], q[4], q[3];
z q[19];
z q[3];
z q[1];
h q[21];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
ccz q[7], q[13], q[22];
h q[19];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
z q[22];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
ccz q[10], q[20], q[8];
ccz q[7], q[18], q[9];
z q[2];
ccz q[22], q[1], q[20];
h q[15];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23];


// Circuit: n=25, h=15, d=21, g=54
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[25] q;
z q[24];
z q[0];
z q[3];
z q[11];
z q[19];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
z q[16];
ccz q[3], q[2], q[20];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
ccz q[9], q[17], q[24];
z q[1];
h q[22];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
h q[23];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
z q[3];
ccz q[20], q[13], q[4];
cz q[8], q[18];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
h q[24];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
cz q[5], q[0];
ccz q[17], q[5], q[12];
z q[11];
ccz q[11], q[24], q[5];
ccz q[23], q[19], q[7];
z q[19];
z q[2];
ccz q[20], q[21], q[9];
cz q[24], q[20];
cz q[9], q[12];
h q[12];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
z q[3];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
ccz q[14], q[16], q[6];
z q[6];
z q[17];
cz q[22], q[14];
cz q[17], q[24];
ccz q[23], q[15], q[21];
z q[5];
z q[1];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
cz q[9], q[17];
z q[5];
cz q[13], q[22];
z q[5];
ccz q[15], q[14], q[11];
h q[13];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
ccz q[1], q[2], q[17];
z q[2];
ccz q[13], q[10], q[11];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
h q[23];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
h q[14];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24];


// Circuit: n=26, h=15, d=19, g=38
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[26] q;
z q[24];
cz q[9], q[22];
z q[4];
cz q[15], q[1];
z q[4];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25];
h q[19];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25];
z q[12];
z q[22];
ccz q[22], q[6], q[1];
h q[21];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25];
cz q[16], q[0];
cz q[9], q[7];
h q[23];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25];
h q[15];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25];
ccz q[11], q[25], q[12];
ccz q[20], q[12], q[4];
h q[17];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25];
h q[19];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25];
h q[18];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25];
cz q[3], q[2];
h q[22];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25];
z q[4];
h q[7];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25];
ccz q[24], q[5], q[20];
ccz q[1], q[4], q[11];
cz q[17], q[2];
cz q[18], q[0];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25];
z q[24];
ccz q[19], q[22], q[15];
z q[14];
ccz q[25], q[4], q[12];
cz q[8], q[2];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25];


// Circuit: n=27, h=15, d=24, g=67
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[27] q;
z q[5];
h q[25];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26];
z q[10];
cz q[25], q[19];
ccz q[4], q[3], q[15];
z q[15];
h q[24];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26];
cz q[22], q[20];
ccz q[2], q[19], q[20];
z q[16];
cz q[25], q[21];
ccz q[13], q[0], q[26];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26];
ccz q[14], q[9], q[1];
z q[7];
z q[18];
z q[20];
h q[19];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26];
z q[20];
z q[4];
cz q[9], q[4];
z q[4];
h q[22];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26];
z q[11];
cz q[2], q[13];
h q[16];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26];
z q[7];
cz q[22], q[3];
z q[1];
cz q[16], q[4];
z q[11];
h q[12];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26];
ccz q[13], q[3], q[5];
z q[24];
ccz q[2], q[12], q[17];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26];
z q[9];
cz q[26], q[10];
cz q[10], q[8];
z q[22];
cz q[12], q[26];
z q[19];
z q[2];
h q[21];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26];
cz q[18], q[8];
ccz q[17], q[6], q[23];
z q[15];
cz q[8], q[26];
cz q[21], q[6];
z q[9];
cz q[8], q[26];
cz q[26], q[7];
z q[26];
cz q[5], q[20];
cz q[7], q[18];
ccz q[20], q[0], q[23];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26];
z q[2];
z q[13];
z q[3];
h q[20];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26];
z q[1];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26];
z q[5];
h q[15];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26];


// Circuit: n=28, h=15, d=18, g=37
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[28] q;
cz q[9], q[15];
z q[27];
h q[20];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26], q[27];
z q[0];
ccz q[4], q[12], q[9];
ccz q[26], q[4], q[18];
cz q[11], q[12];
z q[23];
ccz q[18], q[25], q[27];
ccz q[22], q[8], q[21];
z q[18];
z q[9];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26], q[27];
ccz q[17], q[16], q[1];
z q[15];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26], q[27];
ccz q[12], q[15], q[18];
h q[27];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26], q[27];
h q[16];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26], q[27];
z q[17];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26], q[27];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26], q[27];
z q[27];
h q[9];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26], q[27];
z q[13];
h q[11];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26], q[27];
z q[10];
h q[24];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26], q[27];
ccz q[18], q[11], q[27];
ccz q[14], q[1], q[2];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26], q[27];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26], q[27];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26], q[27];
z q[26];
z q[23];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26], q[27];
h q[12];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[24], q[25], q[26], q[27];



// Circuit: n=3, h=3, d=3, g=6
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[1];
barrier q[0], q[1], q[2];
h q[0];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];


// Circuit: n=3, h=3, d=3, g=6
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[2];
barrier q[0], q[1], q[2];
h q[2];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];


// Circuit: n=3, h=3, d=3, g=6
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[2];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];


// Circuit: n=3, h=3, d=3, g=6
OPENQASM 3.0;
include "stdgates.inc";
qubit[3] q;
h q[2];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];
h q[1];
barrier q[0], q[1], q[2];


// Circuit: n=6, h=3, d=3, g=8
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
ccz q[2], q[0], q[3];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[2], q[5];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=3, d=3, g=7
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[0];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=3, d=3, g=7
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;
cz q[0], q[2];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=3, d=3, g=8
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
ccz q[1], q[3], q[4];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[5], q[2];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=3, d=4, g=9
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
ccz q[2], q[1], q[5];
ccz q[4], q[3], q[1];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[4], q[3];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=0, d=0, g=0
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;


// Circuit: n=6, h=0, d=0, g=0
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;


// Circuit: n=6, h=0, d=0, g=0
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;


// Circuit: n=6, h=0, d=0, g=0
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;


// Circuit: n=6, h=0, d=0, g=0
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;


// Circuit: n=6, h=0, d=0, g=0
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;


// Circuit: n=6, h=0, d=0, g=0
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;


// Circuit: n=6, h=0, d=0, g=0
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;


// Circuit: n=6, h=0, d=0, g=0
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;


// Circuit: n=6, h=6, d=6, g=13
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[1];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=6, d=7, g=14
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[0], q[1];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[5];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=6, d=6, g=13
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[4];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=6, d=8, g=17
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[3], q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
ccz q[1], q[5], q[4];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[0];
z q[3];
z q[4];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=6, d=6, g=14
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[0], q[3];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[2];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=6, d=6, g=12
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=6, d=7, g=16
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[2];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[0], q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[1];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=2, d=2, g=4
OPENQASM 3.0;
include "stdgates.inc";
qubit[6] q;
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=6, d=7, g=16
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[4];
z q[2];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[3], q[0];
ccz q[3], q[1], q[5];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=6, d=8, g=15
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
cz q[2], q[3];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
ccz q[0], q[4], q[5];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[2];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=6, h=6, d=50, g=81
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[6] q;
ccz q[3], q[4], q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[1], q[2];
cz q[2], q[5];
z q[5];
cz q[3], q[5];
ccz q[0], q[1], q[3];
cz q[4], q[2];
ccz q[3], q[2], q[1];
ccz q[0], q[1], q[3];
cz q[5], q[0];
z q[5];
ccz q[0], q[5], q[4];
z q[4];
cz q[2], q[0];
z q[4];
z q[2];
cz q[2], q[0];
z q[0];
z q[3];
z q[4];
cz q[5], q[4];
cz q[3], q[4];
ccz q[1], q[2], q[0];
z q[0];
z q[4];
ccz q[5], q[3], q[2];
z q[4];
z q[1];
ccz q[0], q[3], q[1];
ccz q[5], q[1], q[3];
z q[1];
ccz q[1], q[5], q[3];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
ccz q[2], q[0], q[5];
z q[3];
ccz q[5], q[4], q[2];
ccz q[0], q[2], q[4];
ccz q[5], q[2], q[4];
ccz q[0], q[1], q[5];
cz q[1], q[3];
z q[0];
z q[5];
ccz q[5], q[3], q[0];
cz q[2], q[0];
cz q[2], q[0];
z q[0];
ccz q[1], q[2], q[5];
cz q[2], q[0];
ccz q[3], q[2], q[4];
cz q[5], q[3];
z q[1];
cz q[3], q[5];
cz q[3], q[4];
ccz q[3], q[4], q[2];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
cz q[4], q[1];
z q[4];
ccz q[1], q[5], q[2];
ccz q[2], q[3], q[4];
cz q[0], q[4];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[3];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5];
z q[0];
z q[4];
ccz q[1], q[2], q[0];
ccz q[4], q[5], q[2];
z q[1];
cz q[3], q[4];
cz q[3], q[1];
z q[0];
cz q[3], q[1];
ccz q[5], q[4], q[2];
h q[3];
barrier q[0], q[1], q[2], q[3], q[4], q[5];


// Circuit: n=12, h=6, d=16, g=47
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[12] q;
cz q[3], q[7];
ccz q[11], q[6], q[10];
ccz q[5], q[8], q[10];
z q[1];
z q[1];
cz q[0], q[10];
h q[0];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[4];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[0], q[3];
z q[5];
h q[10];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[9], q[7];
z q[8];
z q[3];
z q[3];
ccz q[6], q[0], q[2];
ccz q[7], q[10], q[0];
z q[6];
cz q[10], q[11];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
cz q[5], q[11];
z q[3];
z q[2];
cz q[4], q[1];
ccz q[1], q[2], q[11];
z q[8];
cz q[5], q[0];
z q[1];
cz q[1], q[9];
z q[0];
z q[10];
z q[7];
ccz q[3], q[5], q[9];
z q[7];
cz q[7], q[4];
ccz q[10], q[11], q[9];
h q[4];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];
z q[7];
z q[2];
h q[8];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8], q[9], q[10], q[11];


// Circuit: n=7, h=6, d=54, g=105
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[7] q;
z q[1];
ccz q[3], q[2], q[0];
ccz q[1], q[2], q[0];
cz q[1], q[6];
cz q[4], q[3];
z q[4];
ccz q[1], q[3], q[2];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[1], q[2], q[0];
cz q[2], q[3];
cz q[2], q[0];
ccz q[1], q[6], q[0];
ccz q[0], q[6], q[1];
z q[1];
ccz q[3], q[5], q[0];
ccz q[0], q[3], q[4];
ccz q[3], q[6], q[5];
ccz q[0], q[5], q[6];
z q[6];
z q[2];
cz q[1], q[3];
z q[6];
z q[5];
cz q[0], q[6];
z q[5];
z q[5];
z q[6];
ccz q[0], q[3], q[5];
cz q[1], q[0];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[3], q[5], q[0];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[4], q[1], q[6];
z q[0];
z q[2];
z q[5];
z q[0];
z q[2];
z q[1];
cz q[4], q[5];
z q[2];
cz q[5], q[3];
z q[4];
ccz q[6], q[1], q[3];
z q[2];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[0], q[5], q[1];
ccz q[5], q[1], q[3];
cz q[2], q[6];
cz q[4], q[3];
z q[3];
cz q[0], q[1];
ccz q[1], q[4], q[3];
cz q[2], q[5];
z q[3];
ccz q[5], q[4], q[0];
ccz q[2], q[6], q[5];
cz q[1], q[2];
z q[0];
z q[5];
cz q[0], q[4];
z q[5];
z q[4];
cz q[3], q[4];
z q[0];
cz q[3], q[4];
cz q[0], q[1];
ccz q[5], q[0], q[1];
ccz q[6], q[1], q[0];
cz q[4], q[2];
z q[5];
z q[4];
ccz q[3], q[2], q[4];
z q[1];
ccz q[0], q[1], q[3];
cz q[0], q[5];
cz q[6], q[0];
ccz q[0], q[6], q[5];
z q[1];
ccz q[5], q[0], q[4];
ccz q[4], q[5], q[2];
z q[5];
ccz q[3], q[4], q[0];
z q[2];
cz q[2], q[3];
z q[2];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[3];
z q[6];
cz q[4], q[3];
z q[0];
z q[4];
cz q[5], q[2];
z q[0];
z q[1];
cz q[3], q[2];
ccz q[2], q[1], q[0];
z q[6];
h q[1];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];


// Circuit: n=7, h=6, d=52, g=90
OPENQASM 3.0;
include "stdgates.inc";
gate ccz _gate_q_0, _gate_q_1, _gate_q_2 {
  h _gate_q_2;
  ccx _gate_q_0, _gate_q_1, _gate_q_2;
  h _gate_q_2;
}
qubit[7] q;
ccz q[1], q[6], q[0];
cz q[2], q[1];
z q[2];
z q[6];
z q[4];
ccz q[1], q[4], q[2];
ccz q[3], q[4], q[1];
z q[4];
z q[6];
z q[3];
ccz q[1], q[4], q[0];
z q[0];
z q[0];
ccz q[2], q[4], q[5];
ccz q[4], q[2], q[3];
cz q[0], q[5];
cz q[0], q[5];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
cz q[5], q[1];
z q[5];
cz q[6], q[4];
z q[1];
ccz q[6], q[4], q[0];
cz q[0], q[6];
z q[0];
z q[3];
cz q[3], q[4];
z q[5];
z q[0];
ccz q[1], q[4], q[6];
z q[5];
ccz q[0], q[6], q[4];
ccz q[5], q[3], q[1];
cz q[3], q[0];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[2], q[1], q[6];
ccz q[6], q[5], q[3];
ccz q[6], q[2], q[5];
cz q[0], q[2];
z q[5];
cz q[5], q[3];
ccz q[4], q[0], q[6];
cz q[0], q[6];
ccz q[5], q[6], q[4];
cz q[3], q[1];
h q[6];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[5], q[6], q[1];
cz q[4], q[1];
ccz q[5], q[1], q[3];
z q[0];
cz q[5], q[6];
z q[6];
cz q[5], q[6];
ccz q[5], q[3], q[6];
z q[4];
ccz q[1], q[0], q[3];
z q[4];
cz q[3], q[1];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
z q[2];
ccz q[2], q[4], q[0];
cz q[1], q[4];
ccz q[5], q[6], q[3];
ccz q[6], q[1], q[2];
ccz q[3], q[6], q[1];
ccz q[4], q[1], q[6];
z q[5];
cz q[1], q[5];
z q[0];
z q[3];
z q[3];
z q[3];
cz q[5], q[3];
cz q[0], q[1];
z q[5];
h q[2];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];
ccz q[1], q[2], q[3];
ccz q[0], q[2], q[3];
cz q[0], q[5];
cz q[0], q[2];
ccz q[0], q[6], q[1];
z q[6];
ccz q[1], q[3], q[5];
h q[5];
barrier q[0], q[1], q[2], q[3], q[4], q[5], q[6];


// Circuit: n=2, h=3, d=3, g=4
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
h q[1];
cz q[0], q[1];
h q[1];


// Circuit: n=2, h=3, d=3, g=4
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
h q[1];
cz q[0], q[1];
h q[1];


// Circuit: n=2, h=3, d=3, g=4
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
h q[1];
cz q[0], q[1];
h q[1];


// Circuit: n=2, h=3, d=3, g=4
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
h q[1];
cz q[0], q[1];
h q[1];


// Circuit: n=2, h=3, d=3, g=4
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
h q[0];
h q[1];
cz q[0], q[1];
h q[1];
