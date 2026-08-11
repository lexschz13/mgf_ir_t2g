from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sparse_ir import MatsubaraSampling
    from ..sym_matrix.sym_matrix import _AbstractMatrix
    from ..__utils.__new_types import RealScalar



def phonon_propagator(iw: NDArray[complex], w0: RealScalar, seiw: NDArray[complex],
                      smat: MatsubaraSampling) -> NDArray[float]:
    # Phonon (boson) Green's function from Dyson equation, scalar
    if smat.basis.statistics != 'B':
        raise TypeError("Sampling must be bosonic")
    d0iw =  (2*w0/(iw**2 - w0**2)).real
    diw = (d0iw**(-1) - seiw)**(-1)
    return smat.fit(diw).real


def electron_propagator_single_site(iw: NDArray[complex], mu: RealScalar, H: _AbstractMatrix,
                                    seiw: NDArray[complex]) -> NDArray[complex]:
    # Electron (fermion) single site Green's function from Dyson equation, sym_matrix
    return (iw - H + mu - seiw)**-1


def electron_propagator_lattice(iw: NDArray[complex], mu: RealScalar, H: _AbstractMatrix,
                                    seiw: NDArray[complex]) -> NDArray[complex]:
    # Electron (fermion) lattice Green's function from Dyson equation, sym_matrix
    return (iw[:,None,None,None] - H[None,:,:,:] + mu - seiw)**-1