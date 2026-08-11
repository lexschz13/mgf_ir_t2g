import numpy as np
from typing import TYPE_CHECKING
from ..sym_matrix import ohmatrix, matrixfit
from ..k_space import k_convolution


if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sparse_ir import TauSampling
    from ..sym_matrix import OhMatrix
    from ..__utils.__new_types import RealScalar


def sehf_ee(gloc_beta: RealScalar, U: RealScalar, J: RealScalar) -> OhMatrix:
    # Hartree-Fock diagrams of electronic correlations (Oh sym)
    sehf_a = -2*gloc_beta.a * (3*U - 5*J)
    sehf_b = gloc_beta.b * (U - 2*J)
    return ohmatrix(sehf_a, sehf_b)

def sehf_phm(gkbeta: NDArray[float], J: RealScalar) -> OhMatrix:
    # Hartree-Fock diagrams of nearest-neighbour interaction to model Cooperative JTE (Oh sym)
    lattice_shape = gkbeta.shape
    lattice_size = gkbeta.size
    ky,kx,kz = np.meshgrid(*tuple(np.arange(0,2*np.pi,2*np.pi/lattice_shape[i]) for i in [1,0,2]))
    gammak = 2*J*(np.cos(kx) + np.cos(ky) + np.cos(kz))
    return ohmatrix(k_convolution(4*gkbeta.a, gammak), k_convolution(-2*gkbeta.b, gammak)) / 3 / lattice_size
    
    # Direct convolution implementation, unused
    # qidxs = np.transpose(np.indices(lattice_shape), (1,2,3,0)).reshape((lattice_size,3))
    # sehf = 0
    # for qidx in qidxs:
    #     gqbeta = gkbeta[tuple(qidx)]
    #     qx,qy,qz = qidx / np.array(lattice_shape) * 2*np.pi
    #     gammakq = 2*J*(np.cos(kx-qx) + np.cos(ky-qy) + np.cos(kz-qz))
    #     sehf += ohmatrix(4*gqbeta.a, -2*gqbeta.b)/3 * gammakq / lattice_size
    
    # return sehf

def se2b_ee(gloctau: OhMatrix, U: RealScalar, J: RealScalar, stau: TauSampling) -> OhMatrix:
    # Second-Bohr diagrams of electroonic correlations (Oh sym)
    
    if stau.basis.statistics != 'F':
        raise TypeError("Sampling must be fermionic")
    
    se2btau_a = (
        (5*U**2 - 20*U*J + 28*J**2)*gloctau.a**2*gloctau.a[::-1]
        +8*(U**2 - 4*U*J + 3*J**2)*gloctau.a*gloctau.b*gloctau.b[::-1]
        -2*(U**2 - 4*U*J + 5*J**2)*gloctau.b**2*(gloctau.a[::-1] + gloctau.b[::-1])
                 )
    se2btau_b = (
        +(U**2 - 4*U*J + 5*J**2)*gloctau.a**2*gloctau.b[::-1]
        -2*(U**2 - 2*U*J + 3*J**2)*gloctau.a*gloctau.b*(2*gloctau.a[::-1] - gloctau.b[::-1])
        +(U**2 - 4*U*J + 3*J**2)*gloctau.b**2*gloctau.a[::-1]
        -(9*U**2 - 36*U*J + 38*J**2)*gloctau.b**2*gloctau.b[::-1]
                 )
    return matrixfit(stau, ohmatrix(se2btau_a, se2btau_b))

def seep(gloctau: OhMatrix, dtau: NDArray[float], g: RealScalar, stau: TauSampling) -> OhMatrix:
    # Migdal second order electron-phonon diagram
    seepftau = -g**2/3 * dtau * ohmatrix(4/3*gloctau.a, -2/3*gloctau.b)
    return matrixfit(stau, seepftau)

def seb(gloctau: OhMatrix, g: RealScalar, stau: TauSampling) -> NDArray[float]:
    # Phonon self-energy from Eliashberg Theory
    ptau = -4*(gloctau.a*gloctau.a[::-1] - gloctau.b*gloctau.b[::-1]) * g**2 / 3
    return stau.fit(ptau)
