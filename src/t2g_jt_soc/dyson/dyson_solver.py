import numpy as np
import sparse_ir as ir
import h5py
from ..sym_matrix import (ohmatrix,
                          ohzeros,
                          matrixfit,
                          matrixevaluate,
                          matrixsum,
                          matrixcopy)
# from ..sym_matrix.sym_matrix import ohmatrix, ohzeros
# from ..sym_matrix.ir_utils import matrixfit, matrixevaluate
# from ..sym_matrix.array_utils import matrixcopy, matrixsum
from .__utils import (check_discrete_parameter,
                      check_physical_param,
                      check_shape,
                      fprint,
                      frobenius_inner)
from .__propagators import (phonon_propagator,
                            electron_propagator_lattice,
                            electron_propagator_single_site)
from .__self_energy import (se2b_ee,
                            sehf_ee,
                            sehf_phm,
                            seb)
from ..convergence import (implemented_conv,
                           diis)



class DysonSolver:
    def __init__(self, *args, **kwargs):
        """
        From a set of parameters solves self-consistent Dyson equation for Oh symmetric system with SOC and dynamic JT.
        This can be initialized with parameters setting at zero all Green's functions and self energies, prepared to solved and saved.
        Also it can be initialized from a file getting the solved functions.
        The initialization mode depends on which parameters are introduces.

        Parameters (New init mode)
        ----------
        irbf : FiniteTempBasis
            Fermionic basis from package sparse_ir.
        irbb : FiniteTempBasis
            Bosonic basis from package sparse_ir.
        N : (int, float)
            Desired particle density, from 0 to 6.
            This solver works with secodn Bohr approximations, for stability interval 1 to 5 is recomended.
        t : (int, float)
            Hopping amplitude.
            Greater-equal than zero.
            If t=0 only local Green's function is computed.
        U : (int, float)
            Direct correlation integral.
            Greater-equal than zero.
        J : (int, float)
            Exchange integral.
            Greater-equal than U.
        Jphm : (int, flaot)
            Cooperative JTE exchange integral.
            Greater-equal than zero.
        w0 : (int, float)
            Phonon natural frequency.
            Greater-equal than zero.
        g : (int, float)
            Phonon-electron coupling.
            Greater-equal than zero.
        lbd : (int, float)
            SOC amplitude.
            Greater-equal than zero.
        latt_shape : (int, iterable(int))
            Lattice shape.
        fl_out : str, optional
            Output file.
            The default is "t2g_soc_jtpol.out".
        
        Parameters (File load mode)
        ----------
        file : str
            File name.
        irbf : (None,FiniteTempBasis), optional
            Fermionic basis from package sparse_ir.
            If None a new basis is initialized from file medtadata.
            Basis metadata must coincide with fiel metadata.
            The default is None.
        irbb : (None,FiniteTempBasis), optional
            Fermionic basis from package sparse_ir.
            If None a new basis is initialized from file medtadata.
            Basis metadata must coincide with fiel metadata.
            The default is None.

        """
        num_args = len(args)+len(kwargs)
        if num_args in [1,2,3]:
            self.__solver_load(*args, **kwargs)
        elif num_args in [11,12,13]:
            self.__solver_init(*args, **kwargs)
        else:
            raise TypeError("Invalid number of arguments.")
    
    
    ######################################################################
    #Initializers
    def __solver_init(self, irbf, irbb, N, t, U, J, Jphm, w0, g, lbd, latt_shape, fl_out="t2g_soc_jtpol.out"):
        """
        Unsolved initialization mode.

        Parameters
        ----------
        irbf : FiniteTempBasis
            Fermionic basis from package sparse_ir.
        irbb : FiniteTempBasis
            Bosonic basis from package sparse_ir.
        N : (int, float)
            Desired particle density, from 0 to 6.
            This solver works with secodn Bohr approximations, for stability interval 1 to 5 is recomended.
        t : (int, float)
            Hopping amplitude.
            Greater-equal than zero.
            If t=0 only local Green's function is computed.
        U : (int, float)
            Direct correlation integral.
            Greater-equal than zero.
        J : (int, float)
            Exchange integral.
            Greater-equal than U.
        Jphm : (int, flaot)
            Cooperative JTE exchange integral.
            Greater-equal than zero.
        w0 : (int, float)
            Phonon natural frequency.
            Greater-equal than zero.
        g : (int, float)
            Phonon-electron coupling.
            Greater-equal than zero.
        lbd : (int, float)
            SOC amplitude.
            Greater-equal than zero.
        latt_shape : (int, iterable(int))
            Lattice shape.
        fl_out : str, optional
            Output file.
            The default is "t2g_soc_jtpol.out".

        """
        # self.__T = check_physical_param(T, #Temperaturre in K
        #                                 0,
        #                                 text_type_error="Temperature must be a number",
        #                                 text_value_error="Temperature must be a positive number") 
        # self.__wM = check_physical_param(wM, #Cutoff frequency
        #                                  0,
        #                                  text_type_error="Cut frequency must be a number",
        #                                  text_value_error="Cut frequency must be a positive number")  
        
        self.__N = check_physical_param(N, #Particle density
                                        0,
                                        6,
                                        text_type_error="Particle density must be a number.",
                                        text_value_error=r"Particle density of a $t_{2g}$ shell goes from 0 to 6.")  
        self.__t = check_physical_param(t, #Kinetic integral
                                        0,
                                        text_type_error="Kinetic integral must be a number.",
                                        text_value_error="Kinetic integral must be a positive number.") 
        self.__U = check_physical_param(U, #Electron-electron direct integral
                                        0,
                                        text_type_error="Electron-electron direct integral must be a number.",
                                        text_value_error="Electron-electron direct integral must be a positive number.") 
        self.__J = check_physical_param(J, #Hund's coupling
                                        U,
                                        text_type_error="Hund's coupling must be a number greater than direct correlation integral.",
                                        text_value_error="Hund's coupling must be a positive number.")
        self.__Jphm = check_physical_param(Jphm, #Orbital exchange
                                           0,
                                           text_type_error="Orbital exchange must be a number.",
                                           text_value_error="Orbital exchange must be a positive number.")
        self.__w0 = check_physical_param(w0, #Phonon natural frequency
                                         0,
                                         text_type_error="Phonon natural frequency must be a number.",
                                         text_value_error="Phonon natural frequency must be a positive number.") 
        self.__g = check_physical_param(g, #Phonon-electron coupling constant
                                        0,
                                        text_type_error="Phonon-electron coupling constant must be a number.",
                                        text_value_error="Phonon-electron coupling constant must be a positive number.")
        self.__lbd = check_physical_param(lbd, #Spin-orbit coupling constant
                                          0,
                                          text_type_error="Spin-orbit coupling constant must be a number.",
                                          text_value_error="Spin-orbit coupling constant must be a positive number.")
        self.__latt_shape = check_shape(latt_shape,
                                        3,
                                        "Non value number of lattice dimensions.",
                                        "Introduce three element iterable or an integer to define shape.")
        # self.__diis_mem = check_discrete_parameter(diis_mem, #Memory for diis extrapolation
        #                                            2,
        #                                            text_value_error="Memory for DIIS extrapolation must be a number",
        #                                            text_type_error="Memory for DIIS extrapolation must be an integer bigger than 1") 
        
        self.__fl_out = fl_out
        
        # Kinetic term
        ky,kx,kz = np.meshgrid(*tuple(np.arange(0,2*np.pi,2*np.pi/latt_shape[i]) for i in [1,0,2]))
        self.__Hlatt = -2*t*(np.cos(kx) + np.cos(ky) + np.cos(kz))
        
        
        # Sparse basis
        if type(irbf) != ir.FiniteTempBasis:
            raise TypeError("irbf must be a fermionic basis")
        if type(irbb) != ir.FiniteTempBasis:
            raise TypeError("irbb must be a fermionic basis")
        if irbf.statistics != 'F':
            raise TypeError("irbf must be a fermionic basis")
        if irbb.statistics != 'B':
            raise TypeError("irbb must be a fermionic basis")
        if irbf.beta != irbb.beta or irbf.wmax != irbb.wmax:
            raise ValueError("Fermionic and bosonic basis must coincide")
        self.__irbf = self.__irbf
        self.__stauf = ir.TauSampling(self.__irbf)
        self.__smatf = ir.MatsubaraSampling(self.__irbf)
        self.__freqf = 1j*self.__smatf.wn*np.pi/self.beta
        self.__irbb = self.__irbb
        self.__staub = ir.TauSampling(self.__irbb)
        self.__smatb = ir.MatsubaraSampling(self.__irbb)
        self.__freqb = 1j*self.__smatb.wn*np.pi/self.beta
        
        
        # Green's func
        self.__mu = 0
        self.__sehf = ohmatrix(0,0)
        self.__sephm = ohzeros(latt_shape)
        self.__seepl = ohzeros(self.__irbf.size)
        self.__se2bl = ohzeros(self.__irbf.size)
        self.__sebl = np.zeros(self.__irbb.size)
        self.__glocl = ohzeros(self.__irbf.size)
        self.__gkl = ohzeros((self.__irbf.size,) + latt_shape)
        self.__dl = phonon_propagator(self.freqb, self.w0, self.sebiw, self.smatb)
        self.__nph0 = -2*np.sum(self.irbb.u(self.beta)*self.dl)
        self.__hybl = ohzeros(self.__irbf.size)
        
        
        # SC loop
        self.__conv_method = None
        self.__diis_mem = None
        self.__conv_ls = []
        # self.__diis_vals = ohzeros(1) if diis_mem==0 else ohzeros((diis_mem, self.irbf.size+1))
        # self.__diis_err = ohzeros(1) if diis_mem==0 else ohzeros((diis_mem, self.irbf.size+1))
        
        self.__solved = False
        
        return
    
    def __solver_load(self, file, irbf=None, irbb=None):
        """
        DysonSolver loader from file.

        Parameters
        ----------
        file : str
            File name.
        irbf : (None,FiniteTempBasis), optional
            Fermionic basis from package sparse_ir.
            If None a new basis is initialized from file medtadata.
            Basis metadata must coincide with fiel metadata.
            The default is None.
        irbb : (None,FiniteTempBasis), optional
            Fermionic basis from package sparse_ir.
            If None a new basis is initialized from file medtadata.
            Basis metadata must coincide with fiel metadata.
            The default is None.

        """
        
        if type(file) != str:
            raise TypeError("File name of data storage must be a string.")
        if file[-5:] != ".hdf5": file += ".hdf5"
        with h5py.File(file, "r") as fl:
            # self.__T = fl["T"][()]
            # self.__wM = fl["wmax"][()]
            self.__N = fl["N"][()]
            self.__t = fl["t"][()]
            self.__U = fl["U"][()]
            self.__J = fl["J"][()]
            self.__Jphm = fl["Jphm"][()]
            self.__w0 = fl["w0"][()] 
            self.__g = fl["g"][()]
            self.__lbd = fl["lbd"][()]
            self.__latt_shape = fl["latt_shape"][:]
            self.__diis_mem = fl["diis_mem"][()]
            self.__conv_method = fl["conv_method"][()]
            
            # Kinetic term
            ky,kx,kz = np.meshgrid(*tuple(np.arange(0,2*np.pi,2*np.pi/self.__latt_shape[i]) for i in [1,0,2]))
            self.__Hlatt = -2*self.__t*(np.cos(kx) + np.cos(ky) + np.cos(kz))
            
            
            # Sparse basis
            if irbf in None:
                self.__irbf = ir.FiniteTempBasis("F", fl["beta"][()], fl["wmax"][()])
            else:
                if type(irbf) != ir.FiniteTempBasis:
                    raise TypeError("irbf must be a fermionic basis")
                if irbf.statistics != 'F':
                    raise TypeError("irbf must be a fermionic basis")
                if irbf.beta != fl["beta"][()] or irbf.wmax != fl["wmax"][()]:
                    raise ValueError("Fermionic basis parameters must coincide with file")
                self.__irbf = irbf
            if irbf in None:
                self.__irbb = ir.FiniteTempBasis("B", fl["beta"][()], fl["wmax"][()])
            else:
                if type(irbb) != ir.FiniteTempBasis:
                    raise TypeError("irbb must be a bosonic basis")
                if irbb.statistics != 'B':
                    raise TypeError("irbb must be a bosonic basis")
                if irbb.beta != fl["beta"][()] or irbb.wmax != fl["wmax"][()]:
                    raise ValueError("Bosonic basis parameters must coincide with file")
                self.__irbb = irbb
            self.__stauf = ir.TauSampling(self.__irbf)
            self.__smatf = ir.MatsubaraSampling(self.__irbf)
            self.__freqf = 1j*self.__smatf.wn*np.pi/self.beta
            self.__staub = ir.TauSampling(self.__irbb)
            self.__smatb = ir.MatsubaraSampling(self.__irbb)
            self.__freqb = 1j*self.__smatb.wn*np.pi/self.beta
            
            # Green's func
            self.__mu = fl["mu"][()]
            self.__sehf = ohmatrix(fl["sehf_a"][()], fl["sehf_b"][()])
            self.__seepl = ohmatrix(fl["seepl_a"][:], fl["seepl_b"][:])
            self.__se2bl = ohmatrix(fl["se2bl_a"][:], fl["se2bl_b"][:])
            self.__sebl = fl["sebl"][:]
            self.__glocl = ohmatrix(fl["glocl_a"][:], fl["glocl_b"][:])
            self.__dl = fl["dl"][()]
            d0iw = (2*self.w0/(self.freqb**2 - self.w0**2)).real
            d0l = self.smatb.fit(d0iw).real
            self.__nph0 = -2*np.sum(self.irbb.u(self.beta)*d0l)
            self.__hybl = ohmatrix(fl["hybl_a"][:], fl["hybl_b"][:])
            
            if self.Jphm != 0:
                gkbeta = ohmatrix(fl["gkbeta_a"][:], fl["gkbeta_b"][:])
                self.__sephm = sehf_phm(gkbeta, self.Jphm)
            else:
                self.__sephm = ohzeros(self.__latt_shape)
            if self.__t != 0:
                gkiw = electron_propagator_lattice(self.freqf,
                                                   self.mu,
                                                   self.Hlatt + self.lbd*ohmatrix(0,1),
                                                   self.sehf +
                                                   self.se2biw[:,None,None,None] +
                                                   self.sephm[None,:,:,:] +
                                                   2*self.seeepiw)
                self.__gkl = matrixfit(self.smatf, gkiw)
            
            # SC loop
            self.__conv_ls = fl["conv"][:]
            # self.__diis_vals = ohzeros(1) if diis_mem==0 else ohzeros((diis_mem, self.irbf.size+1))
            # self.__diis_err = ohzeros(1) if diis_mem==0 else ohzeros((diis_mem, self.irbf.size+1))
            
            self.__solved = True
        
    
    ######################################################################
    #Properties
    
    @property
    def T(self): return 11604.522110519543/self.beta
    @property
    def beta(self): return self.irbf.beta
    @property
    def wM(self): return self.irbf.wmax
    @property
    def N(self): return self.__N
    @property
    def t(self): return self.__t
    @property
    def U(self): return self.__U
    @property
    def Up(self): return self.__U - 2*self.__J
    @property
    def J(self): return self.__J
    @property
    def Jphm(self): return self.__Jphm
    @property
    def w0(self): return self.__w0
    @property
    def g(self): return self.__g
    @property
    def lbd(self): return self.__lbd
    @property
    def latt_shape(self): return self.__latt_shape
    @property
    def latt_size(self): return np.prod(self.__latt_shape)
    @property
    def diis_mem(self): return self.__diis_mem
    @property
    def Hlatt(self): return self.__Hlatt
    @property
    def irbf(self): return self.__irbf
    @property
    def smatf(self): return self.__smatf
    @property
    def stauf(self): return self.__stauf
    @property
    def freqf(self): return self.__freqf
    @property
    def irbb(self): return self.__irbb
    @property
    def smatb(self): return self.__smatb
    @property
    def staub(self): return self.__staub
    @property
    def freqb(self): return self.__freqb
    @property
    def mu(self): return self.__mu
    @property
    def sehf(self): return self.__sehf
    @property
    def sephm(self): return self.__sephm
    @property
    def se2bl(self): return self.__se2bl
    @property
    def se2btau(self): return matrixevaluate(self.__stauf, self.se2bl)
    @property
    def se2biw(self): return matrixevaluate(self.__smatf, self.se2bl)
    @property
    def seepl(self): return self.__seepl
    @property
    def seeptau(self): return matrixevaluate(self.__stauf, self.seepl)
    @property
    def seepiw(self): return matrixevaluate(self.__smatf, self.seepl)
    @property
    def sebl(self): return self.__sebl
    @property
    def sebtau(self): return self.__staub.evaluate(self.sebl)
    @property
    def sebiw(self): return self.__smatb.evaluate(self.sebl)
    @property
    def glocl(self): return self.__glocl
    @property
    def gloctau(self): return matrixevaluate(self.__stauf, self.glocl)
    @property
    def glociw(self): return matrixevaluate(self.__smatf, self.glocl)
    @property
    def gkl(self): return self.__gkl
    @property
    def gktau(self): return matrixevaluate(self.__stauf, self.gkl, axis=0)
    @property
    def gkiw(self): return matrixevaluate(self.__smatf, self.gkl, axis=0)
    @property
    def hybl(self): return self.__hybl
    @property
    def hybtau(self): return matrixevaluate(self.stauf, self.hybl)
    @property
    def hybiw(self): return matrixevaluate(self.smatf, self.hybl)
    @property
    def dl(self): return self.__dl
    @property
    def dtau(self): return self.__staub.evaluate(self.dl)
    @property
    def diw(self): return self.__smatb.evaluate(self.dl)
    @property
    def conv_ls(self): return self.__conv_ls
    @property
    def is_solved(self):
        return self.__solved
    # Energies
    @property
    def ekin(self):
        gkbeta = matrixsum(self.irbf.u(self.beta)[:,None,None,None] * self.gkl, axis=0)
        return -matrixsum(gkbeta * self.Hlatt).trace / self.latt_size
    @property
    def esoc(self):
        return -0.5*self.lbd*(matrixsum(self.irbf.u(self.beta)*self.glocl) * ohmatrix(0,1)).trace
    @property
    def eeph(self):
        Fepl = matrixfit(self.smatf, self.glociw * self.seepiw).real
        Fepbeta = matrixsum(self.irbf.u(self.beta) * Fepl)
        return -Fepbeta.trace
    @property
    def eint(self):
        Feel = matrixfit(self.smatf, self.glociw * (self.sehf+self.se2biw)).real
        Feebeta = matrixsum(self.irbf.u(self.beta) * Feel)
        return -Feebeta.trace
    @property
    def ephm(self):
        Fphml = matrixfit(self.smatf, matrixsum(self.gkiw*self.sephm[None,:,:,:], axis=(1,2,3))/self.latt_size).real
        return -matrixsum(Fphml * self.irbf.u(self.beta)).trace
    @property
    def eche(self):
        return -matrixsum(self.irbf.u(self.beta)*self.glocl).trace * self.mu
    @property
    def ephn(self):
        return -2*np.sum(self.irbb.u(self.beta)*self.dl) * self.w0
    @property
    def etot(self):
        return self.ekin + self.esoc + self.eeph + self.eint + self.ephm + self.eche + self.ephn
    #Densities
    @property
    def nexp(self):
        return -6 * np.sum(self.irbf.u(self.beta) * self.glocl.a)
    @property
    def nph(self):
        return -2*np.sum(self.irbf.u(self.beta)*self.dl) - self.__nph0
    @property
    def varnel(self):
        gloc0 = matrixsum(self.irbf.u(0) * self.glocl)
        glocf = matrixsum(self.irbf.u(self.beta) * self.glocl)
        return np.sqrt(12*gloc0.a*glocf.a + 6*gloc0.b*glocf.b)
    #Correlators
    @property
    def correlations(self):
        gloc0 = matrixsum(self.irbf.u(0) * self.glocl)
        glocf = matrixsum(self.irbf.u(self.beta) * self.glocl)
        return np.array([[gloc0.a*glocf.a, gloc0.a*glocf.b], [gloc0.b*glocf.a, gloc0.b*glocf.b]])
    
    
    ######################################################################
    #Self-consistency updates
    def __update_green(self, out_fl, tol=1e-6, delta=0.1, max_iter=10000):
        # Electronic Green's function updating
        # Chemical potential is adjusted
        
        # Half-filling approximation for w->infty (only static components)
        self.__mu = (self.sehf + matrixsum(self.Hlatt+self.sephm, axis=(-1,-2,-3))/self.latt_size).trace.real/6
        
        iterations = 0
        while True:
            fprint("Starting with mu=%.8f" % self.mu, out_fl)
            if self.__t != 0:
                gkiw = electron_propagator_lattice(self.freqf,
                                                   self.mu,
                                                   self.Hlatt + self.lbd*ohmatrix(0,1),
                                                   self.sehf +
                                                   self.se2biw[:,None,None,None] +
                                                   self.sephm[None,:,:,:] +
                                                   2*self.seepiw[:,None,None,None])
                glociw = matrixsum(gkiw, axis=(1,2,3)) / self.latt_size
            else: # When t=0 it is assumed isolated site
                glociw = electron_propagator_single_site(self.freqf,
                                                         self.mu,
                                                         self.Hlatt + self.lbd*ohmatrix(0,1),
                                                         self.sehf +
                                                         self.se2biw +
                                                         2*self.seepiw)
            
            self.__glocl = matrixfit(self.smatf, glociw).real
            hybiw = glociw**-1 - self.sehf - self.se2biw - matrixsum(self.sephm, axis=(-1,-2,-3))/self.latt_size - 2*self.seepiw
            self.__hybl = matrixfit(self.smatf, hybiw).real # Hybridization function
                                                            # Unused, no DMFT algorithm, but stored
            
            fprint("Finished with Nexp=%.8f" % (self.nexp), out_fl)
            
            # Adjustment of mu by gradient descent
            DN = self.N-self.nexp
            if abs(DN) <= tol:
                return
            etaiw = np.sum((self.glociw*self.glociw).trace) if self.__t == 0 else np.sum((self.gkiw * self.gkiw).trace, axis=(-1,-2,-3)) / self.latt_size
            etal = self.smatf.fit(etaiw).real
            etabeta = np.sum(self.irbf.u(self.beta) * etal) # d<n>/dmu
            self.__mu += DN/etabeta
            
            # Variable step algorithm for mu adjust, deprecated
            # if DN > 0:
            #     if last_sign == -1:
            #         delta /= 2
            #     self.__mu += delta
            #     last_sign = +1
            # elif DN < 0:
            #     if last_sign == +1:
            #         delta /= 2
            #     self.__mu -= delta
            #     last_sign = -1
            iterations += 1
            if iterations >= max_iter:
                raise RuntimeError("No convergence for chemical potential")
                return
        self.__dl = phonon_propagator(self.freqb, self.w0, self.sebiw, self.smatb)
        return
    
    
    def __update_self_energy(self):
        # Self-energies update
        if self.g != 0: self.__sebl = seb(self.gloctau, self.g, self.staub)
        if self.U != 0: self.__sehf = sehf_ee(matrixsum(self.irbf.u(self.beta).reshape((self.irbf.size,) + (1,)*self.glocl.ndim-1) * self.glocl, axis=0),
                                              self.U, self.J)
        if self.U != 0: self.__se2bl = se2b_ee(self.gloctau, self.U, self.J, self.stauf)
        if self.Jphm != 0: self.__sephm = sehf_phm(matrixsum(self.irbf.u(self.beta).reshape((self.irbf.size,) + (1,)*self.gkl.ndim-1) * self.gkl, axis=0),
                                                   self.Jphm)
    
    
    ######################################################################
    #Self-consistency solve
    def solve(self, conv_method=None, tol=5e-6, mutol=1e-6, maxiter=10000,
              diis_mem=None):
        """
        

        Parameters
        ----------
        conv_method : (None, str), optional
            Convergence acceleration method.
            If None no convergence acceleration method is executed.
            If correlation integrals and coupling constants are 0 it is set to None.
            The default is None.
        tol : float, optional
            Convergence tolerance.
            The default is 5e-6.
            Greater-equal than 1e-15.
        mutol : float, optional
            Tolerance for chemical potential adjustment.
            Greater-equal than 1e-15.
            The default is 1e-6.
        maxiter : int, optional
            Maximum iterations of self-consistency.
            Greater than 0.
            The default is 10000.
        diis_mem : (None, int), optional
            DIIS memory truncation.
            Only considered if DIIS convergence method is chosen, then it must be an integer greater than 1.
            The default is None.

        Raises
        ------
        NotImplementedError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        tol = check_physical_param(tol,
                                   1e-15,
                                   text_type_error="Tolerance must be a number.",
                                   text_value_error="Minimum tolerance allowed is 1e-15 (machine precission).")
        mutol = check_physical_param(mutol,
                                     1e-15,
                                     text_type_error="Tolerance must be a number.",
                                     text_value_error="Minimum tolerance allowed is 1e-15 (machine precission).")
        maxiter = check_discrete_parameter(maxiter,
                                           1,
                                           text_type_error="Maximum iterations must be an integer.",
                                           text_value_error="No allowed 0 or negative maximum iterations.")
        
        if self.U == 0 and self.g==0 and self.Jphm==0:
            conv_method = None
        
        if not conv_method is None and not conv_method in implemented_conv:
            raise NotImplementedError("This convergence method is not implemented")
        
        if conv_method == "diis":
            self.__conv_method = conv_method
            self.__diis_vals = ohzeros((self.diis_mem, self.irbf.size+1))
            self.__diis_err = ohzeros((self.diis_mem, self.irbf.size+1))
            self.__diis_mem = check_discrete_parameter(diis_mem, #Memory for diis extrapolation
                                                       2,
                                                       text_value_error="Memory for DIIS extrapolation must be a number.",
                                                       text_type_er="Memory for DIIS extrapolation must be an integer bigger than 1.")
        
        out_fl = open(self.__fl_out, 'w')
        fprint("Starting execution with the following paramters", file=out_fl)
        fprint("T=%.3fK" % self.T, file=out_fl)
        fprint("beta=%.3f" % self.beta, file=out_fl)
        fprint("wM=%.3feV" % self.wM, file=out_fl)
        fprint("N=%.3f" % self.N, file=out_fl)
        fprint("t=%.3feV" % self.t, file=out_fl)
        fprint("U=%.3feV" % self.U, file=out_fl)
        fprint("J=%.3feV" % self.J, file=out_fl)
        fprint("Jphm=%.3feV" % self.Jphm, file=out_fl)
        fprint("w0=%.3feV" % self.w0, file=out_fl)
        fprint("g=%.3feV" % self.g, file=out_fl)
        fprint("lbd=%.3feV" % self.lbd, file=out_fl)
        fprint("latt_sz=%i" % self.latt_size, file=out_fl)
        fprint("Convergence method %s" % conv_method if conv_method is not None else "No convergence method")
        if conv_method == "diis":
            fprint("diis_mem=%i" % (self.diis_mem), file=out_fl)
        fprint("-"*15+"\n", file=out_fl)
        fprint("Computing non-interactive Green's function", file=out_fl)
        self.__update_green(out_fl)
        fprint('\n'*2, file=out_fl)
        iterations = 0
        while True:
            last_g = matrixcopy(self.glocl)
            fprint("Starting iteration %i" % (iterations+1), file=out_fl)
            fprint("Updating self-energies", file=out_fl)
            self.__update_self_energy()
            
            # DIIS
            if conv_method == "diis":
                self.__diis_vals[:-1] = matrixcopy(self.__diis_vals[1:])
                self.__diis_err[:-1] = matrixcopy(self.__diis_err[1:])
                self.__diis_vals[-1,0] = matrixcopy(self.sehf)
                self.__diis_vals[-1,1:] = matrixcopy(self.se2btau)
                self.__diis_err[-1] = self.__diis_vals[-1] - self.__diis_vals[-2]
                if iterations >= self.diis_mem:
                    self.__diis_vals = diis(self.__diis_vals,
                                            self.__diis_err,
                                            frobenius_inner,
                                            matrixsum)
                    self.__diis_err[-1] = self.__diis_vals[-1] - self.__diis_vals[-2]
                    seext = self.__diis_vals[-1]
                    self.__sehf = seext[0]
                    self.__se2bl = matrixfit(self.stauf, seext[1:])
            
            fprint("Computing electron and phonon propagators", file=out_fl)
            self.__update_green(out_fl)
            fprint("Expected phononic excitations is %.5f" % (-2*np.sum(self.irbf.u(self.beta)*self.dl) - self.__nph0),
                   file=out_fl)
            fprint('\n', file=out_fl)
            iterations += 1
            conv = np.sqrt(frobenius_inner(self.glocl-last_g, self.glocl-last_g))
            self.__conv_ls.append(conv)
            fprint("iteration  %i finished with convergence %.8e" % (iterations, conv), file=out_fl)
            fprint('-'*15, file=out_fl)
            fprint('\n'*2, file=out_fl)
            if conv <= tol:
                fprint("Finished", file=out_fl)
                fprint("\n"*3, file=out_fl)
                fprint("-"*15, file=out_fl)
                self.__solved = True
                out_fl.close()
                return
            if iterations >= maxiter:
                fprint("Reached max iterations", file=out_fl)
                out_fl.close()
                return
    
    ######################################################################
    #File save
    def save(self, sv_fl):
        """
        Save solver data in hdf5 file only if it is solved.

        Parameters
        ----------
        sv_fl : str
            File name.

        """
        if not self.__solved:
            print("Not solved yet, nothing to save")
            return
        if type(sv_fl) != str:
            raise TypeError("File name of data storage must be a string.")
        if sv_fl[-5:] != ".hdf5": sv_fl += ".hdf5"
        with h5py.File(sv_fl, "w") as fl:
            print("Saving data on file")
            # Password
            fl.create_dataset("password", data = "dyson_solver") # Added to recognize the DysonSolver files
            # Metadata
            fl.create_dataset("T", data = self.T)
            fl.create_dataset("beta", data = self.beta)
            fl.create_dataset("wmax", data = self.wM)
            fl.create_dataset("N", data = self.N)
            fl.create_dataset("t", data = self.t)
            fl.create_dataset("U", data = self.U)
            fl.create_dataset("J", data = self.J)
            fl.create_dataset("Jphm", data = self.Jphm)
            fl.create_dataset("Up", data = self.Up)
            fl.create_dataset("w0", data = self.w0)
            fl.create_dataset("g", data = self.g)
            fl.create_dataset("lbd", data = self.lbd)
            fl.create_dataset("latt_shape", data = self.latt_shape)
            fl.create_dataset("diis_mem", data = self.diis_mem)
            fl.create_dataset("mu", data = self.mu)
            fl.create_dataset("conv", data = np.array(self.__conv_ls))
            fl.create_dataset("conv_method", data=self.__conv_method)
            
            # Green's funcions objetcs
            fl.create_dataset("glocl_a", data = self.glocl.a)
            fl.create_dataset("glocl_b", data = self.glocl.b)
            fl.create_dataset("hybl_a", data = self.hybl.a)
            fl.create_dataset("hybl_b", data = self.hybl.b)
            fl.create_dataset("dl", data = self.dl)
            fl.create_dataset("sehf_a", data = self.sehf.a)
            fl.create_dataset("sehf_b", data = self.sehf.b)
            fl.create_dataset("se2bl_a", data = self.se2bl.a)
            fl.create_dataset("se2bl_b", data = self.se2bl.b)
            fl.create_dataset("seepl_a", data = self.seepl.a)
            fl.create_dataset("seepl_b", data = self.seepl.b)
            fl.create_dataset("sebl", data = self.sebl)
            
            # Energies and particle densities
            fl.create_dataset("esoc", data = self.esoc)
            fl.create_dataset("eeph", data = self.eeph)
            fl.create_dataset("eint", data = self.eint)
            fl.create_dataset("ephm", data = self.ephm)
            fl.create_dataset("ekin", data = self.ekin)
            fl.create_dataset("eche", data = self.eche)
            fl.create_dataset("ephn", data = self.ephn)
            fl.create_dataset("etot", data = self.etot)
            
            fl.create_dataset("nexp", data = self.nexp)
            fl.create_dataset("nph", data = self.nph)
            fl.create_dataset("varn", data=self.varnel)
            
            # Lattice beta
            gkbeta = matrixsum(self.irbf.u(self.beta)[:,None,None,None] * self.gkl, axis=0)
            fl.create_dataset("gkbeta_a", data = gkbeta.a)
            fl.create_dataset("gkbeta_b", data = gkbeta.b)
            
            # Local ocrrelators
            corrs = self.correlations
            fl.create_dataset("correldiag", data=corrs[0,0])
            fl.create_dataset("correloffd", data=corrs[1,1])
            fl.create_dataset("correlcrs1", data=corrs[0,1])
            fl.create_dataset("correlcrs2", data=corrs[1,0])
            
            
            # Correlators in irreducible reciprocal zone
            # print("Computing correlators")
            # kidxs = np.transpose(np.indices((self.latt_size,)*3), (1,2,3,0)).reshape((self.latt_size,3))
            # qidxs = kidxs[np.where((kidxs[:,0] <= self.latt_size//2) * (kidxs[:,1] <= kidxs[:,0]) * (kidxs[:,2] <= kidxs[:,0]) * (kidxs[:,2] <= kidxs[:,1]))]
            # corrdiag = np.zeros(qidxs.shape[0])
            # corroffd = np.zeros(qidxs.shape[0])
            # corrcrs1 = np.zeros(qidxs.shape[0])
            # corrcrs2 = np.zeros(qidxs.shape[0])
            # for i,q in enumerate(qidxs):
            #     kidxst = (kidxs - q) % self.latt_size
            #     kidxstf = self.latt_size**2*kidxst[:,0] + self.latt_size*kidxst[:,1] + kidxst[:,2]
            #     gklconv = self.gkl.reshape((self.irbf.size,self.latt_size))[:,kidxstf].reshape((self.irbf.size,)+(self.latt_size,)*3)
            #     gkzero = matrixsum(self.irbf.u(0)[:,None,None,None] * self.gkl, axis=0)
            #     gkconvbeta = matrixsum(self.irbf.u(self.beta)[:,None,None,None] * gklconv, axis=0)
            #     corrdiag[i] = np.sum(gkzero.a * gkconvbeta.a) / self.latt_size
            #     corroffd[i] = np.sum(gkzero.b * gkconvbeta.b) / self.latt_size
            #     corrcrs1[i] = -np.sum(gkzero.a * gkconvbeta.b) / self.latt_size
            #     corrcrs2[i] = -np.sum(gkzero.b * gkconvbeta.a) / self.latt_size
            # fl.create_dataset("correldiag", data=corrdiag)
            # fl.create_dataset("correloffd", data=corroffd)
            # fl.create_dataset("correlcrs1", data=corrcrs1)
            # fl.create_dataset("correlcrs2", data=corrcrs2)
            # fl.create_dataset("irrBZ", data=qidxs)


