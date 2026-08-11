import numpy as np

# This file is used to deposit matrix representations

levi_civitta = np.array([[[int((i - j) * (j - k) * (k - i) // 2) for k in range(3)] for j in range(3)] for i in range(3)])

g0 = np.eye(3)
g1 = np.zeros((3,3)); g1[0,1] = 1; g1[1,0] = 1
g2 = np.zeros((3,3), dtype=np.complex128); g2[0,1] = -1j; g2[1,0] = 1j
g3 = np.diag([1,-1,0])
g4 = np.zeros((3,3)); g4[0,2] = 1; g4[2,0] = 1
g5 = np.zeros((3,3), dtype=np.complex128); g5[0,2] = -1j; g5[2,0] = 1j
g6 = np.zeros((3,3)); g6[1,2] = 1; g6[2,1] = 1
g7 = np.zeros((3,3), dtype=np.complex128); g7[1,2] = -1j; g7[2,1] = 1j
g8 = np.diag([1,1,-2])/np.sqrt(3)
gell_mann = [g0,g1,g2,g3,g4,g5,g6,g7,g8]

gx = np.diag([1,0,0])
gy = np.diag([0,1,0])
gz = np.diag([0,0,1])
gdiag = [gx,gy,gz]

l1 = -g7
l2 = g5
l3 = -g2
angular = (l1,l2,l3)


s0 = np.eye(2)
s1 = np.array([[0,1],[1,0]])
s2 = np.array([[0,-1j],[1j,0]])
s3 = np.diag([1,-1])
pauli = (s1,s2,s3)

sx = np.kron(g0, s1)
sy = np.kron(g0, s2)
sz = np.kron(g0, s3)
pauli_cross = np.array([sx,sy,sz])

lx = np.kron(l1,s0)
ly = np.kron(l2,s0)
lz = np.kron(l3,s0)

ax = np.kron(gx, s0)
ay = np.kron(gy, s0)
az = np.kron(gz, s0)

Lx = np.kron(l1,s1)
Ly = np.kron(l2,s2)
Lz = np.kron(l3,s3)
soc = (Lx,Ly,Lz)

V = -np.kron(g7,s1) + np.kron(g5,s2) - np.kron(g2,s3)
I = np.kron(g0,s0)