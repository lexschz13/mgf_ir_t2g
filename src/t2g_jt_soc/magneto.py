import numpy as np
import sparse_ir as ir
from .ohmatrix import ohevaluate, ohfit
from .reciprocal_space import k_convolution
from .phys_prop import conductivity_from_ir, conductivity_real_ir



#######
# Matrices

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

##################




def tau_triple_conv(smat, stau, a,b,c, sum_idx, **kwargs):
    # Arguments on freq space
    d = np.einsum(sum_idx, a, b, c, optimize=True)
    return stau.evaluate(smat.fit(d, **kwargs).real, **kwargs)


def GD2h(Gkl, angle, ejt, smat, mode='t'):
    Gkiw = ohevaluate(smat, Gkl, axis=0)
    
    if mode=='t':
        a = np.sin(angle)
        b = np.cos(angle)
    elif mode=='o':
        a = np.cos(angle)
        b = -np.sin(angle)
    
    detGkiw = Gkiw.a**2 - 2*Gkiw.b**2 + Gkiw.a*Gkiw.b
    alphakiw = (Gkiw.a + Gkiw.b)/detGkiw
    gammakiw = -Gkiw.b/detGkiw
    
    pkiw = alphakiw - ejt*(a + b/np.sqrt(3))
    qkiw = alphakiw + ejt*(a - b/np.sqrt(3))
    rkiw = alphakiw + ejt*2*b/np.sqrt(3)
    
    deltakiw = pkiw*qkiw*rkiw + 2*gammakiw**3 - (pkiw+qkiw+rkiw)*gammakiw**2
    
    akl = smat.fit((qkiw*rkiw-gammakiw**2)/deltakiw, axis=0).real
    bkl = smat.fit((pkiw*rkiw-gammakiw**2)/deltakiw, axis=0).real
    ckl = smat.fit((pkiw*qkiw-gammakiw**2)/deltakiw, axis=0).real
    xkl = smat.fit(gammakiw*(gammakiw-pkiw)/deltakiw, axis=0).real
    ykl = smat.fit(gammakiw*(gammakiw-qkiw)/deltakiw, axis=0).real
    zkl = smat.fit(gammakiw*(gammakiw-rkiw)/deltakiw, axis=0).real
    
    return (+ akl[None,None,...] * ax[:,:,None,None,None,None]
            + bkl[None,None,...] * ay[:,:,None,None,None,None]
            + ckl[None,None,...] * az[:,:,None,None,None,None]
            - xkl[None,None,...] * Lx[:,:,None,None,None,None]
            - ykl[None,None,...] * Ly[:,:,None,None,None,None]
            - zkl[None,None,...] * Lz[:,:,None,None,None,None]
            )


def Gspin_proj(Gkl, beta, t, angle, ejt, irbf, jt_mode):
    stauf = ir.TauSampling(irbf)
    smatf = ir.MatsubaraSampling(irbf)
    # staub = ir.TauSampling(irbb)
    # smatb = ir.MatsubaraSampling(irbb)
    
    print("Breaking G symmetry")
    Gkl_D2h_iso = GD2h(Gkl, angle, ejt, smatf, jt_mode)
    k_sz = Gkl.shape[1]
    ky,kx,kz = np.meshgrid(*(np.arange(0,2*np.pi,2*np.pi/k_sz),)*3)
    # sin = np.sin(np.array([kx,ky,kz]))
    # cos = np.cos(np.array([kx,ky,kz]))
    # upsx = 6*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[0,0]) / k_sz**3
    # upsy = 6*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[2,2]) / k_sz**3
    # upsz = 6*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[4,4]) / k_sz**3
    A = np.array([[1,-1,0],
                  [np.sqrt(1/3), np.sqrt(1/3), -np.sqrt(4/3)],
                  [1,1,1]])
    qx = -2*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[0,0]) / k_sz**3
    qy = -2*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[2,2]) / k_sz**3
    qz = -2*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[4,4]) / k_sz**3
    
    Q3 = qx - qy
    Q8 = (qx + qy - 2*qz) / np.sqrt(3)
    
    upsx, upsy, upsz = np.linalg.inv(A) @ np.array([Q3, Q8, 0])
    # ups = np.array([upsx, upsy, upsz])
    
    print("Adding hopping anisotropy")
    Gkiw = ohevaluate(smatf, Gkl, axis=0)
    Gkiw = (Gkiw**-1 + 2*t * (upsx*np.cos(kx) + upsy*np.cos(ky) + upsz*np.cos(kz)))**-1
    Gkl = ohfit(smatf, Gkiw, axis=0).real
    Gkl_jt = GD2h(Gkl, angle, ejt, smatf, jt_mode)
    Gktau_jt = stauf.evaluate(Gkl_jt, axis=2)
    Gkiw_jt = smatf.evaluate(Gkl_jt, axis=2)
    
    # Projection
    print("Computing dynamical projections")
    _tmp1 = np.einsum("aij,jk...->aik...", pauli_cross, Gktau_jt, optimize=True)
    _tmp2 = np.einsum("aij...,aji...->a...", _tmp1, -_tmp1[:,:,:,::-1], optimize=True)
    psiktau = np.einsum("a...,aij->aij...", _tmp2, pauli_cross, optimize=True)
    del _tmp1, _tmp2
    # psiktau = np.einsum("aij,jk...,akl,li...,axy->axy...", pauli_cross, Gktau_jt, pauli_cross, -Gktau_jt[:,:,::-1], pauli_cross, optimize=True)
    # psikl = stauf.fit(psiktau, axis=3)
    
    # Self-energy
    print("Computing self-energy")
    sesktau = k_convolution(Gktau_jt, psiktau, einidxs="ij...,ajk...->ik...")
    seskl = stauf.fit(sesktau, axis=2)
    del sesktau
    seskiw = smatf.evaluate(seskl, axis=2)
    del seskl
    
    # New Green
    # print("Computing spin projected Green's function")
    # Gkiw_s = (Gkiw_jt**-1 - seskiw)**-1
    # Gkl_s = stauf.evaluate(Gkiw_s, axis=2)
    
    # 1st order Green expansion
    print("Computing 1st order exmapnsion of Green's funciton")
    G1kiw = np.einsum("ij...,jk...,kl...->il...", Gkiw_jt, seskiw, Gkiw_jt, optimize=True)
    del seskiw
    G1kl = smatf.fit(G1kiw, axis=2)
    G1ktau = stauf.evaluate(G1kl, axis=2)
    
    return Gktau_jt, G1ktau


def susc_mo(Gkl, beta, t, angle, ejt, irbf, irbb, jt_mode):
    stauf = ir.TauSampling(irbf)
    smatf = ir.MatsubaraSampling(irbf)
    staub = ir.TauSampling(irbb)
    smatb = ir.MatsubaraSampling(irbb)
    
    print("Breaking G symmetry")
    Gkl_D2h_iso = GD2h(Gkl, angle, ejt, smatf, jt_mode)
    k_sz = Gkl.shape[1]
    ky,kx,kz = np.meshgrid(*(np.arange(0,2*np.pi,2*np.pi/k_sz),)*3)
    sin = np.sin(np.array([kx,ky,kz]))
    # cos = np.cos(np.array([kx,ky,kz]))
    # upsx = 6*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[0,0]) / k_sz**3
    # upsy = 6*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[2,2]) / k_sz**3
    # upsz = 6*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[4,4]) / k_sz**3
    A = np.array([[1,-1,0],
                  [np.sqrt(1/3), np.sqrt(1/3), -np.sqrt(4/3)],
                  [1,1,1]])
    qx = -2*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[0,0]) / k_sz**3
    qy = -2*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[2,2]) / k_sz**3
    qz = -2*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[4,4]) / k_sz**3
    
    Q3 = qx - qy
    Q8 = (qx + qy - 2*qz) / np.sqrt(3)
    
    upsx, upsy, upsz = np.linalg.inv(A) @ np.array([Q3, Q8, 0])
    ups = np.array([upsx, upsy, upsz])
    
    print("Adding hopping anisotropy")
    Gkiw = ohevaluate(smatf, Gkl, axis=0)
    Gkiw = (Gkiw**-1 + 2*t * (upsx*np.cos(kx) + upsy*np.cos(ky) + upsz*np.cos(kz)))**-1
    Gkl = ohfit(smatf, Gkiw, axis=0).real
    Gkl_jt = GD2h(Gkl, angle, ejt, smatf, jt_mode)
    Gktau_jt = stauf.evaluate(Gkl_jt, axis=2)
    Gkiw_jt = smatf.evaluate(Gkl_jt, axis=2)
    
    # Projection
    print("Computing dynamical projections")
    _tmp1 = np.einsum("aij,jk...->aik...", pauli_cross, Gktau_jt, optimize=True)
    _tmp2 = np.einsum("aij...,aji...->a...", _tmp1, -_tmp1[:,:,:,::-1], optimize=True)
    psiktau = np.einsum("a...,aij->aij...", _tmp2, pauli_cross, optimize=True)
    del _tmp1, _tmp2
    # psiktau = np.einsum("aij,jk...,akl,li...,axy->axy...", pauli_cross, Gktau_jt, pauli_cross, -Gktau_jt[:,:,::-1], pauli_cross, optimize=True)
    # psikl = stauf.fit(psiktau, axis=3)
    
    # Self-energy
    print("Computing self-energy")
    sesktau = k_convolution(Gktau_jt, psiktau, einidxs="ij...,ajk...->ik...")
    seskl = stauf.fit(sesktau, axis=2)
    del sesktau
    seskiw = smatf.evaluate(seskl, axis=2)
    del seskl
    
    # 1st order Green expansion
    print("Computing 1st order exmapnsion of Green's funciton")
    G1kiw = np.einsum("ij...,jk...,kl...->il...", Gkiw_jt, seskiw, Gkiw_jt, optimize=True)
    del seskiw
    G1kl = smatf.fit(G1kiw, axis=2)
    G1ktau = stauf.evaluate(G1kl, axis=2)
    
    # Current
    print("Computing current")
    jk = sin * ups[:,None,None,None]
    GGkiw_jt = np.einsum("ij...,jk...->ik...", Gkiw_jt, Gkiw_jt, optimize=True)
    dkx_Gkiw_jt = -np.einsum("ijw...,...->ijw...", GGkiw_jt, jk[0], optimize=True)
    dky_Gkiw_jt = -np.einsum("ijw...,...->ijw...", GGkiw_jt, jk[1], optimize=True)
    dkz_Gkiw_jt = -np.einsum("ijw...,...->ijw...", GGkiw_jt, jk[2], optimize=True)
    del GGkiw_jt
    # dk_Gkiw_jt = -np.einsum("ijw...,a,a...,jkw...->aikw...", Gkiw_jt, ups, sin, Gkiw_jt, optimize=True)
    dkx_Gkl_jt = smatf.fit(dkx_Gkiw_jt, axis=2)
    dky_Gkl_jt = smatf.fit(dky_Gkiw_jt, axis=2)
    dkz_Gkl_jt = smatf.fit(dkz_Gkiw_jt, axis=2)
    del dkx_Gkiw_jt,dky_Gkiw_jt,dkz_Gkiw_jt
    #
    dkx_Gktau_jt = stauf.evaluate(dkx_Gkl_jt, axis=2)
    dky_Gktau_jt = stauf.evaluate(dky_Gkl_jt, axis=2)
    dkz_Gktau_jt = stauf.evaluate(dkz_Gkl_jt, axis=2)
    del dkx_Gkl_jt,dky_Gkl_jt,dkz_Gkl_jt
    #
    djxktau = k_convolution(dkx_Gktau_jt, psiktau, einidxs="ij...,ajk...->ik...")
    djyktau = k_convolution(dky_Gktau_jt, psiktau, einidxs="ij...,ajk...->ik...")
    djzktau = k_convolution(dkz_Gktau_jt, psiktau, einidxs="ij...,ajk...->ik...")
    del dkx_Gktau_jt,dky_Gktau_jt,dkz_Gktau_jt
    djxkl = stauf.fit(djxktau, axis=2)
    djykl = stauf.fit(djyktau, axis=2)
    djzkl = stauf.fit(djzktau, axis=2)
    del djxktau, djyktau, djzktau
    djxkiw = smatf.evaluate(djxkl, axis=2)
    djykiw = smatf.evaluate(djykl, axis=2)
    djzkiw = smatf.evaluate(djzkl, axis=2)
    # djkbeta = np.einsum("l,rijl...->rij...", irbf.u(beta), djkl, optimize=True)
    del djxkl, djykl, djzkl
    
    # Current-Green conv
    print("Computing current-green convolution")
    Fxkiw = np.einsum("ij...,jk...->ik...", djxkiw, Gkiw_jt, optimize=True)
    Fykiw = np.einsum("ij...,jk...->ik...", djykiw, Gkiw_jt, optimize=True)
    Fzkiw = np.einsum("ij...,jk...->ik...", djzkiw, Gkiw_jt, optimize=True)
    Fxkl = smatf.fit(Fxkiw, axis=2)
    Fykl = smatf.fit(Fykiw, axis=2)
    Fzkl = smatf.fit(Fzkiw, axis=2)
    del Fxkiw,Fykiw,Fzkiw
    Fxktau = stauf.evaluate(Fxkl, axis=2)
    Fyktau = stauf.evaluate(Fykl, axis=2)
    Fzktau = stauf.evaluate(Fzkl, axis=2)
    del Fxkl,Fykl,Fzkl
    Hxkiw = np.einsum("ij...,jk...->ik...", djxkiw, G1kiw, optimize=True)
    Hykiw = np.einsum("ij...,jk...->ik...", djykiw, G1kiw, optimize=True)
    Hzkiw = np.einsum("ij...,jk...->ik...", djzkiw, G1kiw, optimize=True)
    del djxkiw, djykiw, djzkiw
    Hxkl = smatf.fit(Hxkiw, axis=2)
    Hykl = smatf.fit(Hykiw, axis=2)
    Hzkl = smatf.fit(Hzkiw, axis=2)
    del Hxkiw,Hykiw,Hzkiw
    Hxktau = stauf.evaluate(Hxkl, axis=2)
    Hyktau = stauf.evaluate(Hykl, axis=2)
    Hzktau = stauf.evaluate(Hzkl, axis=2)
    del Hxkl,Hykl,Hzkl
    
    # Susceptibility
    print("Computing susceptibility")
    chixktau = np.einsum("...,ijt...,jit...->t...", jk[1], G1ktau, Fzktau[:,:,::-1], optimize=True) - np.einsum("...,ijt...,jit...->t...", jk[2], G1ktau, Fyktau[:,:,::-1], optimize=True)
    chiyktau = np.einsum("...,ijt...,jit...->t...", jk[2], G1ktau, Fxktau[:,:,::-1], optimize=True) - np.einsum("...,ijt...,jit...->t...", jk[0], G1ktau, Fzktau[:,:,::-1], optimize=True)
    chizktau = np.einsum("...,ijt...,jit...->t...", jk[0], G1ktau, Fyktau[:,:,::-1], optimize=True) - np.einsum("...,ijt...,jit...->t...", jk[1], G1ktau, Fxktau[:,:,::-1], optimize=True)
    #
    chixktau += np.einsum("...,ijt...,jit...->t...", jk[1], Gktau_jt, Hzktau[:,:,::-1], optimize=True) - np.einsum("...,ijt...,jit...->t...", jk[2], Gktau_jt, Hyktau[:,:,::-1], optimize=True)
    chiyktau += np.einsum("...,ijt...,jit...->t...", jk[2], Gktau_jt, Hxktau[:,:,::-1], optimize=True) - np.einsum("...,ijt...,jit...->t...", jk[0], Gktau_jt, Hzktau[:,:,::-1], optimize=True)
    chizktau += np.einsum("...,ijt...,jit...->t...", jk[0], Gktau_jt, Hyktau[:,:,::-1], optimize=True) - np.einsum("...,ijt...,jit...->t...", jk[1], Gktau_jt, Hxktau[:,:,::-1], optimize=True)
    # _temp = np.einsum("abc,b...->ac...", levi_civitta, jk, optimize=True)
    # chiktau = np.einsum("ac...,ijt...,cjit...->at...", _temp, G1ktau, Fktau[:,:,:,::-1,...], optimize=True)
    # chiktau += np.einsum("ac...,ijt...,cjit...->at...", _temp, Gktau_jt, Hktau[:,:,:,::-1,...], optimize=True)
    # chiktau = np.einsum("abc,b...,ijt...,cjit...->at...", levi_civitta, jk, G1ktau, Fktau[:,:,:,::-1,...], optimize=True)
    # chiktau += np.einsum("abc,b...,ijt...,cjit...->at...", levi_civitta, jk, Gktau_jt, Hktau[:,:,:,::-1,...], optimize=True)
    chiktau = np.array([chixktau,chiyktau,chizktau])
    
    return -conductivity_real_ir(irbb, staub, smatb, chiktau + chiktau[:,::-1], axis=1)