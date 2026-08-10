from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from ..sym_matrix import matrixevaluate, matrixfit
from ..k_space import k_convolution
from ..analytical_continuation import boson_continuation, hilbert_ir
from .__matrices import ax,ay,az,Lx,Ly,Lz,pauli_cross


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..dyson import DysonSolver



def GD2h(dy_solver: DysonSolver, angle: float, mode: str = 't') -> NDArray:
    """
    Takes a Green's function solved from a DysonSolver and applies symmetry breakings from a fixed distortion.
    These distortions correspond to JT eg modes repredented by Gell-Mann matrices 3 and 8.

    Parameters
    ----------
    dy_solver : DysonSolver
        Dyson solver class containing Green's functions and bases. It must be solved
    angle : float
        Angle that defines JT distortion.
    mode : str, optional
        Distortion mode.
        For tetragonal mode 't' or "tetragonal". The distortion is defined by sin(angle)*gm3+cos(angle)*gm8.
        Fot orthorhombic mode 'o' or "orthorhombic". The distortion is defined by cos(angle)*gm3-sin(angle)*gm8.
        The default is 't'.

    Returns
    -------
    np.ndarray
        Matrix representation of distorted Green's function.
        Matrix axes are the first two.

    """
    
    if not dy_solver.is_solved:
        raise TypeError("DysonSolver must be solved.")
    
    Gkl = dy_solver.gkl
    ejt = dy_solver.eeph
    smat = dy_solver.smat
    Gkiw = matrixevaluate(smat, Gkl, axis=0)
    
    
    if mode in ['t', "tetragonal"]:
        a = np.sin(angle)
        b = np.cos(angle)
    elif mode in ['o', "orthorhombic"]:
        a = np.cos(angle)
        b = -np.sin(angle)
    else:
        raise ValueError("Only two modes, t or tetragonal and o or orthorhombic")
    
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


def Gspin_proj(dy_solver, angle, jt_mode):
    """
    Takes a Green's function solved from a DysonSolver and, from a fixed distortion, computes the expansion of spin projection.
    These distortions correspond to JT eg modes repredented by Gell-Mann matrices 3 and 8.

    Parameters
    ----------
    dy_solver : DysonSolver
        Dyson solver class containing Green's functions and bases. It must be solved
    angle : float
        Angle that defines JT distortion.
    mode : str, optional
        Distortion mode.
        For tetragonal mode 't' or "tetragonal". The distortion is defined by sin(angle)*gm3+cos(angle)*gm8.
        Fot orthorhombic mode 'o' or "orthorhombic". The distortion is defined by cos(angle)*gm3-sin(angle)*gm8.
        The default is 't'.

    Returns
    -------
    Gkl_jt : np.ndarray
        Matrix representation of distorted Green's function after hopping assymetry is applied.
        This is the zeroth order spin projection expansion of the Green's function.
    G1kl : np.ndarray
        First order spin projection expansion of the Green's function.
    seskl : np.ndarray
        Self-energy of spin-projection.

    """
    stauf = dy_solver.stauf
    smatf = dy_solver.smatf
    # staub = ir.TauSampling(irbb)
    # smatb = ir.MatsubaraSampling(irbb)
    
    Gkl = dy_solver.gkl
    irbf = dy_solver.irbf
    beta = dy_solver.beta
    t = dy_solver.t
    ejt = dy_solver.eeph
    
    print("Breaking G symmetry")
    Gkl_D2h_iso = GD2h(dy_solver, angle, jt_mode)
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
    Gkiw = matrixevaluate(smatf, Gkl, axis=0)
    Gkiw = (Gkiw**-1 + 2*t * (upsx*np.cos(kx) + upsy*np.cos(ky) + upsz*np.cos(kz)))**-1
    Gkl = matrixfit(smatf, Gkiw, axis=0).real
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
    # if get_se:
    #     return seskl
    del sesktau
    seskiw = smatf.evaluate(seskl, axis=2)
    # del seskl
    
    # New Green
    # print("Computing spin projected Green's function")
    # Gkiw_s = (Gkiw_jt**-1 - seskiw)**-1
    # Gkl_s = stauf.evaluate(Gkiw_s, axis=2)
    
    # 1st order Green expansion
    print("Computing 1st order exmapnsion of Green's funciton")
    G1kiw = np.einsum("ij...,jk...,kl...->il...", Gkiw_jt, seskiw, Gkiw_jt, optimize=True)
    del seskiw
    G1kl = smatf.fit(G1kiw, axis=2)
    
    return Gkl_jt, G1kl, seskl


def conductivity_mo(dy_solver, angle, jt_mode, alpha=10**-1.1, guess=None, solver="lsql2"):
    """
    Takes a Green's function solved from a DysonSolver and, from a fixed distortion, computes the antisymmetric component of associeted conductivity tensor on ir-basis.
    These distortions correspond to JT eg modes repredented by Gell-Mann matrices 3 and 8.

    Parameters
    ----------
    dy_solver : DysonSolver
        Dyson solver class containing Green's functions and bases. It must be solved
    angle : float
        Angle that defines JT distortion.
    mode : str, optional
        Distortion mode.
        For tetragonal mode 't' or "tetragonal". The distortion is defined by sin(angle)*gm3+cos(angle)*gm8.
        Fot orthorhombic mode 'o' or "orthorhombic". The distortion is defined by cos(angle)*gm3-sin(angle)*gm8.
        The default is 't'.
    alpha : (int,float), optional
        See :func:"boson_continuation" documantetion.
        The default is 10**-1.1.
    guess : np.ndarray, optional
        See :func:"boson_continuation" documantetion.
        The default is None.
    solver : str, optional
        See :func:"boson_continuation" documantetion.
        The default is "lsql2".

    Returns
    -------
    np.ndarray
        Antisymmetric conductivity tensor on ir-basis.

    """
    Gkl = dy_solver.gkl
    irbf = dy_solver.irbf
    beta = dy_solver.beta
    stauf = dy_solver.stauf
    smatf = dy_solver.smatf
    staub = dy_solver.staub
    
    k_sz = Gkl.shape[1]
    ky,kx,kz = np.meshgrid(*(np.arange(0,2*np.pi,2*np.pi/k_sz),)*3)
    sin = np.sin(np.array([kx,ky,kz]))
    
    Gkl_D2h_iso = GD2h(dy_solver, angle, jt_mode)
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
    
    Gkl_jt, G1kl, seskl = Gspin_proj(dy_solver, angle, jt_mode)
    del seskl
    Gktau_jt = stauf.evaluate(Gkl_jt, axis=2)
    Gkiw_jt = smatf.evaluate(Gkl_jt, axis=2)
    G1ktau = stauf.evaluate(G1kl, axis=2)
    G1kiw = smatf.evaluate(G1kl, axis=2)
    
    # Projection
    print("Computing dynamical projections")
    _tmp1 = np.einsum("aij,jk...->aik...", pauli_cross, Gktau_jt, optimize=True)
    _tmp2 = np.einsum("aij...,aji...->a...", _tmp1, -_tmp1[:,:,:,::-1], optimize=True)
    psiktau = np.einsum("a...,aij->aij...", _tmp2, pauli_cross, optimize=True)
    del _tmp1, _tmp2
    
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
    #
    chixktau += np.einsum("ijt...,jit...->t...", Fyktau, Fzktau[:,:,::-1])
    chiyktau += np.einsum("ijt...,jit...->t...", Fzktau, Fxktau[:,:,::-1])
    chizktau += np.einsum("ijt...,jit...->t...", Fxktau, Fyktau[:,:,::-1])
    # _temp = np.einsum("abc,b...->ac...", levi_civitta, jk, optimize=True)
    # chiktau = np.einsum("ac...,ijt...,cjit...->at...", _temp, G1ktau, Fktau[:,:,:,::-1,...], optimize=True)
    # chiktau += np.einsum("ac...,ijt...,cjit...->at...", _temp, Gktau_jt, Hktau[:,:,:,::-1,...], optimize=True)
    # chiktau = np.einsum("abc,b...,ijt...,cjit...->at...", levi_civitta, jk, G1ktau, Fktau[:,:,:,::-1,...], optimize=True)
    # chiktau += np.einsum("abc,b...,ijt...,cjit...->at...", levi_civitta, jk, Gktau_jt, Hktau[:,:,:,::-1,...], optimize=True)
    chiktau = np.array([chixktau,chiyktau,chizktau])
    chikl = staub.fit(chiktau)
    
    recondl = boson_continuation(chikl - chikl[:,::-1], alpha, axis=1, guess=guess, solver=solver)
    imcondl = hilbert_ir(recondl, dy_solver.irbb)
    return recondl + 1j*imcondl
