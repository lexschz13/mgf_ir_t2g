from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from ..sym_matrix import matrixevaluate, matrixfit
from ..k_space import k_convolution
from ..analytical_continuation import boson_continuation, hilbert_ir
from .__matrices import ax,ay,az,Lx,Ly,Lz,pauli_cross
from ..__utils.__new_types import RealScalar


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..dyson import DysonSolver



def GD2h(dy_solver: DysonSolver, angle: RealScalar, mode: str = 't') -> NDArray[float]:
    """
    Takes a Green's function solved from a DysonSolver and applies symmetry breakings from a fixed distortion.
    These distortions correspond to JT eg modes repredented by Gell-Mann matrices 3 and 8.

    Parameters
    ----------
    dy_solver : DysonSolver
        Dyson solver class containing Green's functions and bases. It must be solved
    angle : RealScalar
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
        Matrix axes are the last two (compatible with :func:"np.matmul").

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
    
    return (+ akl[...,None,None] * ax.reshape(akl.ndim*(1,) + ax.shape)
            + bkl[...,None,None] * ay.reshape(bkl.ndim*(1,) + ay.shape)
            + ckl[...,None,None] * az.reshape(ckl.ndim*(1,) + az.shape)
            - xkl[...,None,None] * Lx.reshape(xkl.ndim*(1,) + Lx.shape)
            - ykl[...,None,None] * Ly.reshape(ykl.ndim*(1,) + Ly.shape)
            - zkl[...,None,None] * Lz.reshape(zkl.ndim*(1,) + Lz.shape)
            )


def Gspin_proj(dy_solver: DysonSolver, angle: RealScalar, jt_mode: str = 't') -> tuple[NDArray[float]]:
    """
    Takes a Green's function solved from a DysonSolver and, from a fixed distortion, computes the expansion of spin projection.
    These distortions correspond to JT eg modes repredented by Gell-Mann matrices 3 and 8.

    Parameters
    ----------
    dy_solver : DysonSolver
        Dyson solver class containing Green's functions and bases. It must be solved
    angle : RealScalar
        Angle that defines JT distortion.
    jt_mode : str, optional
        Distortion mode.
        For tetragonal mode 't' or "tetragonal". The distortion is defined by sin(angle)*gm3+cos(angle)*gm8.
        Fot orthorhombic mode 'o' or "orthorhombic". The distortion is defined by cos(angle)*gm3-sin(angle)*gm8.
        The default is 't'.

    Returns
    -------
    Gkl_jt : NDArray[float]
        Matrix representation of distorted Green's function after hopping assymetry is applied.
        This is the zeroth order spin projection expansion of the Green's function.
    G1kl : NDArray[float]
        First order spin projection expansion of the Green's function.
    seskl : NDArray[float]
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
    latt_shape = dy_solver.latt_shape
    latt_size = dy_solver.latt_size
    kx,ky,kz = tuple(np.arange(0,2*np.pi,2*np.pi/latt_shape[i]) for i in range(len(latt_shape)))
    ky,kx,kz = np.meshgrid(ky,kx,kz)
    A = np.array([[1,-1,0],
                  [np.sqrt(1/3), np.sqrt(1/3), -np.sqrt(4/3)],
                  [1,1,1]])
    qx = -2*np.sum(irbf.u(beta).reshape((irbf.size,) + (1,)*len(latt_shape)) * Gkl_D2h_iso[...,0,0]) / latt_size
    qy = -2*np.sum(irbf.u(beta).reshape((irbf.size,) + (1,)*len(latt_shape)) * Gkl_D2h_iso[...,2,2]) / latt_size
    qz = -2*np.sum(irbf.u(beta).reshape((irbf.size,) + (1,)*len(latt_shape)) * Gkl_D2h_iso[...,4,4]) / latt_size
    
    Q3 = qx - qy
    Q8 = (qx + qy - 2*qz) / np.sqrt(3)
    
    upsx, upsy, upsz = np.linalg.inv(A) @ np.array([Q3, Q8, 0])
    
    print("Adding hopping anisotropy")
    Gkiw = matrixevaluate(smatf, Gkl, axis=0)
    Gkiw = (Gkiw**-1 + 2*t * (upsx*np.cos(kx) + upsy*np.cos(ky) + upsz*np.cos(kz)))**-1
    Gkl = matrixfit(smatf, Gkiw, axis=0).real
    Gkl_jt = GD2h(Gkl, angle, ejt, smatf, jt_mode)
    Gktau_jt = stauf.evaluate(Gkl_jt, axis=0)
    Gkiw_jt = smatf.evaluate(Gkl_jt, axis=0)
    
    # Projection
    print("Computing dynamical projections")
    _tmp1 = pauli_cross @ Gktau_jt[...,None,:,:] # (Gstruct,spin_axes,matrix,matrix)
    _tmp2 = np.trace(_tmp1 - _tmp1[::-1]) # (Gstruct,spin_axes)
    psiktau = _tmp2 * pauli_cross # (Gstruct,spin_axes,matrix,matrix)
    del _tmp1, _tmp2
    
    # Self-energy
    print("Computing self-energy")
    sesktau = k_convolution(Gktau_jt, psiktau, axes=(1,2,3), einidxs="tij...,tajk...->tik...") # (Gstruct,matrix,matrix)
    seskl = stauf.fit(sesktau, axis=0)
    # if get_se:
    #     return seskl
    del sesktau
    seskiw = smatf.evaluate(seskl, axis=0)
    # del seskl
    
    # 1st order Green expansion
    print("Computing 1st order exmapnsion of Green's funciton")
    G1kiw = Gkiw_jt @ seskiw @ Gkiw_jt # (Gstruct,matrix,matrix)
    del seskiw
    G1kl = smatf.fit(G1kiw, axis=0) # (Gstruct,matrix,matrix)
    
    return Gkl_jt, G1kl, seskl


def conductivity_mo(dy_solver: DysonSolver, angle: RealScalar, jt_mode: str = 't', alpha: RealScalar = 10**-1.1,
                    guess: None | NDArray[RealScalar] = None, solver: str = "lsql2") -> NDArray[complex]:
    """
    Takes a Green's function solved from a DysonSolver and, from a fixed distortion, computes the antisymmetric component of associeted conductivity tensor on ir-basis.
    These distortions correspond to JT eg modes repredented by Gell-Mann matrices 3 and 8.

    Parameters
    ----------
    dy_solver : DysonSolver
        Dyson solver class containing Green's functions and bases. It must be solved
    angle : RealScalar
        Angle that defines JT distortion.
    jt_mode : str, optional
        Distortion mode.
        For tetragonal mode 't' or "tetragonal". The distortion is defined by sin(angle)*gm3+cos(angle)*gm8.
        Fot orthorhombic mode 'o' or "orthorhombic". The distortion is defined by cos(angle)*gm3-sin(angle)*gm8.
        The default is 't'.
    alpha : RealScalar, optional
        See :func:"boson_continuation" documantetion.
        The default is 10**-1.1.
    guess : None | NDArray[RealScalar], optional
        See :func:"boson_continuation" documantetion.
        The default is None.
    solver : str, optional
        See :func:"boson_continuation" documantetion.
        The default is "lsql2".

    Returns
    -------
    NDArray[float]
        Antisymmetric conductivity tensor on ir-basis.

    """
    irbf = dy_solver.irbf
    beta = dy_solver.beta
    stauf = dy_solver.stauf
    smatf = dy_solver.smatf
    staub = dy_solver.staub
    
    latt_shape = dy_solver.latt_shape
    latt_size = dy_solver.latt_size
    kx,ky,kz = tuple(np.arange(0,2*np.pi,2*np.pi/latt_shape[i]) for i in range(len(latt_shape)))
    ky,kx,kz = np.meshgrid(ky,kx,kz)
    sin = np.sin(np.array([kx,ky,kz]))
    
    Gkl_D2h_iso = GD2h(dy_solver, angle, jt_mode)
    A = np.array([[1,-1,0],
                  [np.sqrt(1/3), np.sqrt(1/3), -np.sqrt(4/3)],
                  [1,1,1]])
    qx = -2*np.sum(irbf.u(beta).reshape((irbf.size,) + (1,)*len(latt_shape)) * Gkl_D2h_iso[...,0,0]) / latt_size
    qy = -2*np.sum(irbf.u(beta).reshape((irbf.size,) + (1,)*len(latt_shape)) * Gkl_D2h_iso[...,2,2]) / latt_size
    qz = -2*np.sum(irbf.u(beta).reshape((irbf.size,) + (1,)*len(latt_shape)) * Gkl_D2h_iso[...,4,4]) / latt_size
    
    Q3 = qx - qy
    Q8 = (qx + qy - 2*qz) / np.sqrt(3)
    
    upsx, upsy, upsz = np.linalg.inv(A) @ np.array([Q3, Q8, 0])
    ups = np.array([upsx, upsy, upsz]) # (hoppings,)
    
    Gkl_jt, G1kl, seskl = Gspin_proj(dy_solver, angle, jt_mode) # (Gstruct,matrix,matrix)
    del seskl
    Gktau_jt = stauf.evaluate(Gkl_jt, axis=0)
    Gkiw_jt = smatf.evaluate(Gkl_jt, axis=0)
    G1ktau = stauf.evaluate(G1kl, axis=0)
    G1kiw = smatf.evaluate(G1kl, axis=0)
    
    # Projection
    print("Computing dynamical projections")
    _tmp1 = pauli_cross @ Gktau_jt[...,None,:,:] # (Gstruct,spin_axes,matrix,matrix)
    _tmp2 = np.trace(_tmp1 - _tmp1[::-1]) # (Gstruct,spin_axes)
    psiktau = _tmp2 * pauli_cross # (Gstruct,spin_axes,matrix,matrix)
    del _tmp1, _tmp2
    
    # Current
    print("Computing current")
    jk = sin * ups.reshape((3,)+(1,)*len(latt_shape)) # (hoppings,k-dims)
    GGkiw_jt = Gkiw_jt @ Gkiw_jt # (Gstruct,matrix,matrix)
    dkx_Gkiw_jt = -GGkiw_jt * jk[0,None,...,None,None] # (Gstruct,matrix,matrix)
    dky_Gkiw_jt = -GGkiw_jt * jk[1,None,...,None,None] # (Gstruct,matrix,matrix)
    dkz_Gkiw_jt = -GGkiw_jt * jk[2,None,...,None,None] # (Gstruct,matrix,matrix)
    del GGkiw_jt
    dkx_Gkl_jt = smatf.fit(dkx_Gkiw_jt, axis=0)
    dky_Gkl_jt = smatf.fit(dky_Gkiw_jt, axis=0)
    dkz_Gkl_jt = smatf.fit(dkz_Gkiw_jt, axis=0)
    del dkx_Gkiw_jt,dky_Gkiw_jt,dkz_Gkiw_jt
    #
    dkx_Gktau_jt = stauf.evaluate(dkx_Gkl_jt, axis=0)
    dky_Gktau_jt = stauf.evaluate(dky_Gkl_jt, axis=0)
    dkz_Gktau_jt = stauf.evaluate(dkz_Gkl_jt, axis=0)
    del dkx_Gkl_jt,dky_Gkl_jt,dkz_Gkl_jt
    #
    djxktau = k_convolution(dkx_Gktau_jt, psiktau, axes=(1,2,3), einidxs="tij...,tajk...->tik...") # (Gstruct,matrix,matrix)
    djyktau = k_convolution(dky_Gktau_jt, psiktau, axes=(1,2,3), einidxs="tij...,tajk...->tik...") # (Gstruct,matrix,matrix)
    djzktau = k_convolution(dkz_Gktau_jt, psiktau, axes=(1,2,3), einidxs="tij...,tajk...->tik...") # (Gstruct,matrix,matrix)
    del dkx_Gktau_jt,dky_Gktau_jt,dkz_Gktau_jt
    djxkl = stauf.fit(djxktau, axis=0)
    djykl = stauf.fit(djyktau, axis=0)
    djzkl = stauf.fit(djzktau, axis=0)
    del djxktau, djyktau, djzktau
    djxkiw = smatf.evaluate(djxkl, axis=0)
    djykiw = smatf.evaluate(djykl, axis=0)
    djzkiw = smatf.evaluate(djzkl, axis=0)
    del djxkl, djykl, djzkl
    
    # Current-Green conv
    print("Computing current-green convolution")
    Fxkiw = djxkiw @ Gkiw_jt # (Gstruct,matrix,matrix)
    Fykiw = djykiw @ Gkiw_jt # (Gstruct,matrix,matrix)
    Fzkiw = djzkiw @ Gkiw_jt # (Gstruct,matrix,matrix)
    Fxkl = smatf.fit(Fxkiw, axis=0)
    Fykl = smatf.fit(Fykiw, axis=0)
    Fzkl = smatf.fit(Fzkiw, axis=0)
    del Fxkiw,Fykiw,Fzkiw
    Fxktau = stauf.evaluate(Fxkl, axis=0)
    Fyktau = stauf.evaluate(Fykl, axis=0)
    Fzktau = stauf.evaluate(Fzkl, axis=0)
    del Fxkl,Fykl,Fzkl
    Hxkiw = djxkiw @ G1kiw # (Gstruct,matrix,matrix)
    Hykiw = djykiw @ G1kiw # (Gstruct,matrix,matrix)
    Hzkiw = djzkiw @ G1kiw # (Gstruct,matrix,matrix)
    del djxkiw, djykiw, djzkiw
    Hxkl = smatf.fit(Hxkiw, axis=0)
    Hykl = smatf.fit(Hykiw, axis=0)
    Hzkl = smatf.fit(Hzkiw, axis=0)
    del Hxkiw,Hykiw,Hzkiw
    Hxktau = stauf.evaluate(Hxkl, axis=0)
    Hyktau = stauf.evaluate(Hykl, axis=0)
    Hzktau = stauf.evaluate(Hzkl, axis=0)
    del Hxkl,Hykl,Hzkl
    
    # Susceptibility
    print("Computing susceptibility")
    chixktau = (
        jk[1,None,...]*np.trace(G1ktau @ Fzktau[::-1] + Gktau_jt @ Hzktau[::-1]) -
        jk[2,None,...]*np.trace(G1ktau @ Fyktau[::-1] + Gktau_jt @ Hyktau[::-1])
        ) + np.trace(Fyktau @ Fzktau[::-1])
    chiyktau = (
        jk[2,None,...]*np.trace(G1ktau @ Fxktau[::-1] + Gktau_jt @ Hxktau[::-1]) -
        jk[0,None,...]*np.trace(G1ktau @ Fzktau[::-1] + Gktau_jt @ Hzktau[::-1])
        ) + np.trace(Fzktau @ Fxktau[::-1])
    chizktau = (
        jk[0,None,...]*np.trace(G1ktau @ Fyktau[::-1] + Gktau_jt @ Hyktau[::-1]) -
        jk[1,None,...]*np.trace(G1ktau @ Fxktau[::-1] + Gktau_jt @ Hxktau[::-1])
        ) + np.trace(Fxktau @ Fyktau[::-1])

    chiktau = np.array([chixktau,chiyktau,chizktau])
    chikl = staub.fit(chiktau, axis=1)
    
    recondl = boson_continuation(chikl - chikl[:,::-1], alpha, axis=1, guess=guess, solver=solver)
    imcondl = hilbert_ir(recondl, dy_solver.irbb)
    return recondl + 1j*imcondl
