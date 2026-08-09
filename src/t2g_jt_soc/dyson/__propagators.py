def phonon_propagator(iw, w0, seiw, smat):
    # Phonon (boson) Green's function from Dyson equation, scalar
    d0iw =  (2*w0/(iw**2 - w0**2)).real
    diw = (d0iw**(-1) - seiw)**(-1)
    return smat.fit(diw).real


def electron_propagator_single_site(iw, mu, H, seiw):
    # Electron (fermion) single site Green's function from Dyson equation, sym_matrix
    return (iw - H + mu - seiw)**-1


def electron_propagator_lattice(iw, mu, H, seiw):
    # Electron (fermion) lattice Green's function from Dyson equation, sym_matrix
    return (iw[:,None,None,None] - H[None,:,:,:] + mu - seiw)**-1