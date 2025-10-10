import numpy as np
import sparse_ir as ir
import h5py
from .ohmatrix import ohmatrix, ohsum, ohcopy, ohzeros
from .magneto import commGtau, gyrocurrent


def ohfit(sampling, M, **kwargs):
    return ohmatrix(sampling.fit(M.a, **kwargs), sampling.fit(M.b, **kwargs))

def ohevaluate(sampling, M, **kwargs):
    return ohmatrix(sampling.evaluate(M.a, **kwargs), sampling.evaluate(M.b, **kwargs))

def fprint(string, file, **kwargs):
    print(string, **kwargs)
    print(string, file=file, **kwargs)



class DysonSolver:
    def __init__(self, T, wM, N, t, U, J, Jphm, w0, g, lbd, sz, diis_mem, fl="t2g_soc_jtpol.out"):
        self.__T = T #Temperaturre in K
        self.__wM = wM #Cutoff frequency
        self.__N = N #Particle density
        self.__t = t #Kinetic term
        self.__U = U #Direct interaction
        self.__J = J #Exchange interaction
        self.__Jphm = Jphm #Orbital exchange
        self.__w0 = w0 #Phonon natural frequency
        self.__g = g #Phonon-electron coupling constant
        self.__lbd = lbd #soc constant
        self.__diis_mem = diis_mem #Memory for diis extrapolation
        
        self.__fl = fl
        
        # Kinetic term
        ky,kx,kz = np.meshgrid(*(np.arange(0,2*np.pi,2*np.pi/sz),)*3)
        self.__Hlatt = -2*t*(np.cos(kx) + np.cos(ky) + np.cos(kz))
        
        
        # Sparse basis
        self.__irbf = ir.FiniteTempBasis('F', self.beta, wM)
        self.__stauf = ir.TauSampling(self.__irbf)
        self.__smatf = ir.MatsubaraSampling(self.__irbf)
        self.__freqf = 1j*self.__smatf.wn*np.pi/self.beta
        self.__irbb = ir.FiniteTempBasis('B', self.beta, wM)
        self.__staub = ir.TauSampling(self.__irbb)
        self.__smatb = ir.MatsubaraSampling(self.__irbb)
        self.__freqb = 1j*self.__smatb.wn*np.pi/self.beta
        
        
        # Green's func
        self.__mu = 0
        self.__sehf = ohmatrix(0,0)
        self.__sephm = ohzeros((sz, sz, sz))
        self.__seepl = ohzeros(self.__irbf.size)
        self.__se2bl = ohzeros(self.__irbf.size)
        self.__sebl = np.zeros(self.__irbb.size)
        self.__glocl = ohzeros(self.__irbf.size)
        self.__gkl = ohzeros((self.__irbf.size, sz, sz, sz))
        self.__flocl = ohzeros(self.__irbf.size) # Residual without electronic interactions
        self.__fkl = ohzeros((self.__irbf.size, sz, sz, sz))
        self.__dl = 0
        
        
        # SC loop
        self.__conv_ls = []
        self.__diis_vals = ohzeros(1) if diis_mem==0 else ohzeros((diis_mem, self.irbf.size+1))
        self.__diis_err = ohzeros(1) if diis_mem==0 else ohzeros((diis_mem, self.irbf.size+1))
        
        self.__solved = False
        
        self.__update_gb()
        self.__nph0 = -2*np.sum(self.irbb.u(self.beta)*self.dl)
    
    
    @property
    def T(self):
        return self.__T
    
    @property
    def beta(self):
        return 11604.522110519543/self.__T
    
    @property
    def wM(self):
        return self.__wM
    
    @property
    def N(self):
        return self.__N
    
    @property
    def t(self):
        return self.__t
    
    @property
    def U(self):
        return self.__U
    
    @property
    def Up(self):
        return self.__U - 2*self.__J
    
    @property
    def J(self):
        return self.__J
    
    @property
    def Jphm(self):
        return self.__Jphm
    
    @property
    def w0(self):
        return self.__w0
    
    @property
    def g(self):
        return self.__g
    
    @property
    def lbd(self):
        return self.__lbd
    
    @property
    def k_sz(self):
        return self.__Hlatt.shape[0]
    
    @property
    def diis_mem(self):
        return self.__diis_mem
    
    @property
    def Hlatt(self):
        return self.__Hlatt
    
    @property
    def irbf(self):
        return self.__irbf
    
    @property
    def smatf(self):
        return self.__smatf
    
    @property
    def stauf(self):
        return self.__stauf
    
    @property
    def freqf(self):
        return self.__freqf
    
    @property
    def irbb(self):
        return self.__irbb
    
    @property
    def smatb(self):
        return self.__smatb
    
    @property
    def staub(self):
        return self.__staub
    
    @property
    def freqb(self):
        return self.__freqb
    
    @property
    def mu(self):
        return self.__mu
    
    @property
    def sehf(self):
        return self.__sehf
    
    @property
    def sephm(self):
        return self.__sephm
    
    @property
    def se2bl(self):
        return self.__se2bl
    
    @property
    def se2btau(self):
        return ohevaluate(self.__stauf, self.se2bl)
    
    @property
    def se2biw(self):
        return ohevaluate(self.__smatf, self.se2bl)
    
    @property
    def seepl(self):
        return self.__seepl
    
    @property
    def seeptau(self):
        return ohevaluate(self.__stauf, self.seepl)
    
    @property
    def seepiw(self):
        return ohevaluate(self.__smatf, self.seepl)
    
    @property
    def sebl(self):
        return self.__sebl
    
    @property
    def sebtau(self):
        return self.__staub.evaluate(self.sebl)
    
    @property
    def sebiw(self):
        return self.__smatb.evaluate(self.sebl)
    
    @property
    def glocl(self):
        return self.__glocl
    
    @property
    def gloctau(self):
        return ohevaluate(self.__stauf, self.glocl)
    
    @property
    def glociw(self):
        return ohevaluate(self.__smatf, self.glocl)
    
    @property
    def gkl(self):
        return self.__gkl
    
    @property
    def gktau(self):
        return ohevaluate(self.__stauf, self.gkl, axis=0)
    
    @property
    def gkiw(self):
        return ohevaluate(self.__smatf, self.gkl, axis=0)
    
    @property
    def flocl(self):
        return self.__flocl
    
    @property
    def floctau(self):
        return ohevaluate(self.__stauf, self.flocl)
    
    @property
    def flociw(self):
        return ohevaluate(self.__smatf, self.flocl)
    
    @property
    def fkl(self):
        return self.__fkl
    
    @property
    def fktau(self):
        return ohevaluate(self.__stauf, self.fkl, axis=0)
    
    @property
    def fkiw(self):
        return ohevaluate(self.__smatf, self.fkl, axis=0)
    
    @property
    def dl(self):
        return self.__dl
    
    @property
    def dtau(self):
        return self.__staub.evaluate(self.dl)
    
    @property
    def diw(self):
        return self.__smatb.evaluate(self.dl)
    
    @property
    def conv_ls(self):
        return self.__conv_ls
    
    
    # Update functions

    def __update_green(self, out_fl, tol=1e-6, delta=0.1):
        self.__mu = np.sum((self.sehf + ohsum(self.irbf.u(self.beta)*(self.se2bl+2*self.seepl))).real.eigvals)/2 # Approximates near to half filling
        last_sign = 0
        while True:
            fprint("Starting with mu=%.8f" % self.mu, out_fl)
            if self.__t != 0:
                gkiw = (self.freqf[:,None,None,None] - self.Hlatt[None,:,:,:] - self.sehf - self.se2biw[:,None,None,None] - self.sephm[None,:,:,:] - 2*self.seepiw[:,None,None,None] + self.mu - 0.5*self.lbd*ohmatrix(0,1) )**-1
                fkiw = (self.freqf[:,None,None,None] - self.Hlatt[None,:,:,:] - self.sephm[None,:,:,:] - 2*self.seepiw[:,None,None,None] + self.mu - 0.5*self.lbd*ohmatrix(0,1) )**-1
                self.__gkl = ohfit(self.smatf, gkiw, axis=0).real
                self.__fkl = ohfit(self.smatf, fkiw, axis=0).real
                glociw = ohsum(gkiw, axis=(1,2,3)) / self.k_sz**3
                flociw = ohsum(fkiw, axis=(1,2,3)) / self.k_sz**3
            else:
                glociw = (self.freqf - self.sehf - self.se2biw - 2*self.seepiw + self.mu - 0.5*self.lbd*ohmatrix(0,1) )**-1
                flociw = (self.freqf - self.sehf - 2*self.seepiw + self.mu - 0.5*self.lbd*ohmatrix(0,1) )**-1
            self.__glocl = ohfit(self.smatf, glociw).real
            self.__flocl = ohfit(self.smatf, flociw).real
            Nexp = -6 * np.sum(self.irbf.u(self.beta) * self.glocl.a)
            fprint("Finished with Nexp=%.8f" % (Nexp.real), out_fl)
            DN = self.N-Nexp
            if abs(DN) <= tol:
                return
            if DN > 0:
                if last_sign == -1:
                    delta /= 2
                self.__mu += delta
                last_sign = +1
            elif DN < 0:
                if last_sign == +1:
                    delta /= 2
                self.__mu -= delta
                last_sign = -1
    
    def __update_gb(self):
        d0iw =  (2*self.w0/(self.freqb**2 - self.w0**2)).real
        diw = (d0iw**(-1) - self.sebiw)**(-1)
        self.__dl = self.smatb.fit(diw).real
        return
    
    def __update_sehf(self):
        gloc_beta = ohsum(self.irbf.u(self.beta)*self.glocl)
        self.__sehf = ohmatrix(-2*gloc_beta.a*(3*self.U-5*self.J), gloc_beta.b*(self.U-2*self.J))
        return
    
    def __update_sephm(self):
        ky,kx,kz = np.meshgrid(*(np.arange(0,2*np.pi,2*np.pi/self.k_sz),)*3)
        qidxs = np.transpose(np.indices((self.k_sz,)*3), (1,2,3,0)).reshape((self.k_sz**3,3))
        self.__sephm *= 0
        gkbeta = ohsum(self.gkl*self.irbf.u(self.beta)[:,None,None,None], axis=0)
        for qidx in qidxs:
            gqbeta = gkbeta[tuple(qidx)]
            qx,qy,qz = qidx / self.k_sz * 2*np.pi
            gammakq = 2*self.Jphm*(np.cos(kx-qx) + np.cos(ky-qy) + np.cos(kz-qz))
            self.__sephm += ohmatrix(4*gqbeta.a, -2*gqbeta.b)/3 * gammakq / self.k_sz**3
                    
        return
    
    def __update_se2b(self):
        se2btau_a = (
            (5*self.U**2 - 20*self.U*self.J + 28*self.J**2)*self.gloctau.a**2*self.gloctau.a[::-1]
            +8*(self.U**2 - 4*self.U*self.J + 3*self.J**2)*self.gloctau.a*self.gloctau.b*self.gloctau.b[::-1]
            -2*(self.U**2 - 4*self.U*self.J + 5*self.J**2)*self.gloctau.b**2*(self.gloctau.a[::-1] + self.gloctau.b[::-1])
                     )
        se2btau_b = (
            +(self.U**2 - 4*self.U*self.J + 5*self.J**2)*self.gloctau.a**2*self.gloctau.b[::-1]
            -2*(self.U**2 - 2*self.U*self.J + 3*self.J**2)*self.gloctau.a*self.gloctau.b*(2*self.gloctau.a[::-1] - self.gloctau.b[::-1])
            +(self.U**2 - 4*self.U*self.J + 3*self.J**2)*self.gloctau.b**2*self.gloctau.a[::-1]
            -(9*self.U**2 - 36*self.U*self.J + 38*self.J**2)*self.gloctau.b**2*self.gloctau.b[::-1]
                     )
        self.__se2bl = ohfit(self.stauf, ohmatrix(se2btau_a, se2btau_b))
        return


    def __update_seep(self):
        seepftau = -self.g**2/3 * self.dtau * ohmatrix(4/3*self.gloctau.a, -2/3*self.gloctau.b)
        self.__seepl = ohfit(self.stauf, seepftau)
        return


    def __update_seb(self):
        ptau = -4*(self.gloctau.a*self.gloctau.a[::-1] - self.gloctau.b*self.gloctau.b[::-1]) * self.g**2 / 3
        self.__sebl = self.staub.fit(ptau)
        return
    
    def __seel_res_tau(self):
        eiw = (self.flociw**-1 - self.glociw**-1) - self.sehf - self.se2biw
        el = ohfit(self.smatf, eiw).real
        return ohevaluate(self.stauf, el)
    
    
    
    def solve(self, diis_active = True, tol = 1e-6, max_iter = 10000):
        if self.diis_mem == 0:
            diis_active = False
        if self.U == 0:
            self.__J = 0
            self.__diis_mem = 0
            diis_active = False
        out_fl = open(self.__fl, 'w')
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
        fprint("k_sz=%i" % self.k_sz, file=out_fl)
        fprint("diis_mem=%i" % (diis_active*self.diis_mem), file=out_fl)
        fprint("-"*15+"\n", file=out_fl)
        fprint("Computing non-interactive Green's function", file=out_fl)
        self.__update_green(out_fl)
        self.__update_gb()
        fprint('\n'*2, file=out_fl)
        iterations = 0
        while True:
            last_g = ohcopy(self.glocl)
            fprint("Starting iteration %i" % (iterations+1), file=out_fl)
            fprint("Updating self-energies", file=out_fl)
            self.__update_seb()
            self.__update_sehf()
            self.__update_sephm()
            self.__update_se2b()
            self.__update_seep()
            # DIIS
            if diis_active:
                self.__diis_vals[:-1] = ohcopy(self.__diis_vals[1:])
                self.__diis_err[:-1] = ohcopy(self.__diis_err[1:])
                self.__diis_vals[-1,0] = ohcopy(self.sehf)
                self.__diis_vals[-1,1:] = ohcopy(self.se2btau)
                self.__diis_err[-1] = self.__diis_vals[-1] - self.__diis_vals[-2]
                if iterations >= self.diis_mem:
                    fprint("Starting diis extrapolation", file=out_fl)
                    B = np.zeros((self.diis_mem,)*2)
                    for i in range(self.diis_mem):
                        for j in range(i, self.diis_mem):
                            B[i,j] = np.sum((self.__diis_err[i] * self.__diis_err[j]).trace)
                            if i != j:
                                B[j,i] = np.copy(B[i,j])
                    B /= np.mean(B)
                    c_prime = np.linalg.inv(B) @ np.ones((self.diis_mem,))
                    c = c_prime / np.sum(c_prime)
                    fprint("DIIS coeficients", file=out_fl)
                    for k in range(self.diis_mem):
                        fprint("c%i = %.8f" % (k, c[k]), file=out_fl)
                    seext = ohsum(c[:,None] * self.__diis_vals, axis=0)
                    self.__sehf = seext[0]
                    self.__se2bl = ohfit(self.stauf, seext[1:])
                    self.__diis_vals[-1] = ohcopy(seext)
                    self.__diis_err[-1] = self.__diis_vals[-1] - self.__diis_vals[-2]
            fprint("Computing gloc", file=out_fl)
            self.__update_green(out_fl)
            fprint("Computing phonon propagator", file=out_fl)
            self.__update_gb()
            fprint("Expected phononic excitations is %.5f" % (-2*np.sum(self.irbb.u(self.beta)*self.dl)), file=out_fl)
            fprint('\n', file=out_fl)
            iterations += 1
            conv = np.sum(((self.glocl - last_g)**2).sqrt().trace)
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
            if iterations >= max_iter:
                fprint("Reached max iterations", file=out_fl)
                out_fl.close()
                return
            check_loop = True
            if iterations >= 5:
                for ii in range(1,5):
                    check_loop *= abs(conv-self.conv_ls[-1-ii]) <= tol/1000
                if check_loop:
                    if conv <= tol*100:
                        fprint("Finished in a loop with satisfactory convergence", file=out_fl)
                        fprint("\n"*3, file=out_fl)
                        fprint("-"*15, file=out_fl)
                        self.__solved = True
                        out_fl.close()
                        return
                    else:
                        fprint("Aborted in a loop", file=out_fl)
                        fprint("\n"*3, file=out_fl)
                        fprint("-"*15, file=out_fl)
                        self.__solved = False
                        out_fl.close()
                        return
    
    def save(self, sv_fl):
        if not self.__solved:
            print("Not solved yet, nothing to save")
            return
        with h5py.File(sv_fl+'.hdf5', "w") as fl:
            print("Saving data on file")
            fl.create_dataset("T", data = self.T)
            fl.create_dataset("beta", data = self.beta)
            fl.create_dataset("wmax", data = self.wM)
            fl.create_dataset("N", data = self.N)
            fl.create_dataset("nexp", data = -6 * np.sum(self.irbf.u(self.beta) * self.glocl.a))
            fl.create_dataset("t", data = self.t)
            fl.create_dataset("U", data = self.U)
            fl.create_dataset("J", data = self.J)
            fl.create_dataset("Jphm", data = self.Jphm)
            fl.create_dataset("Up", data = self.Up)
            fl.create_dataset("w0", data = self.w0)
            fl.create_dataset("g", data = self.g)
            fl.create_dataset("lbd", data = self.lbd)
            fl.create_dataset("k_sz", data = self.k_sz)
            fl.create_dataset("diis_mem", data = self.diis_mem)
            fl.create_dataset("mu", data = self.mu)
            fl.create_dataset("conv", data = np.array(self.__conv_ls))
            
            fl.create_dataset("glocl_a", data = self.glocl.a)
            fl.create_dataset("glocl_b", data = self.glocl.b)
            # fl.create_dataset("gkl_a", data = self.gkl.a)
            # fl.create_dataset("gkl_b", data = self.gkl.b)
            fl.create_dataset("dl", data = self.dl)
            fl.create_dataset("sehf_a", data = self.sehf.a)
            fl.create_dataset("sehf_b", data = self.sehf.b)
            fl.create_dataset("se2bl_a", data = self.se2bl.a)
            fl.create_dataset("se2bl_b", data = self.se2bl.b)
            fl.create_dataset("seepl_a", data = self.seepl.a)
            fl.create_dataset("seepl_b", data = self.seepl.b)
            fl.create_dataset("sebl", data = self.sebl)
            
            
            fl.create_dataset("epot", data = -0.5*self.lbd*(ohsum(self.irbf.u(self.beta)*self.glocl) * ohmatrix(0,1)).trace)

            Fepl = ohfit(self.smatf, self.glociw * self.seepiw).real
            Fepbeta = ohsum(self.irbf.u(self.beta) * Fepl)
            fl.create_dataset("eeph", data = -Fepbeta.trace)

            Feel = ohfit(self.smatf, self.glociw * (self.sehf+self.se2biw)).real
            Feebeta = ohsum(self.irbf.u(self.beta) * Feel)
            fl.create_dataset("eint", data = -Feebeta.trace)
            
            Fphml = ohfit(self.smatf, ohsum(self.gkiw*self.sephm[None,:,:,:], axis=(1,2,3))/self.k_sz**3).real
            fl.create_dataset("ephm", data = -ohsum(Fphml * self.irbf.u(self.beta)).trace)

            gkbeta = ohsum(self.irbf.u(self.beta)[:,None,None,None] * self.gkl, axis=0)
            fl.create_dataset("ekin", data = -ohsum(gkbeta * self.Hlatt).trace / self.k_sz**3)

            fl.create_dataset("eche", data = -ohsum(self.irbf.u(self.beta)*self.glocl).trace * self.mu)

            fl.create_dataset("ephn", data = -2*np.sum(self.irbb.u(self.beta)*self.dl) * self.w0)
            fl.create_dataset("nph", data = -2*np.sum(self.irbb.u(self.beta)*self.dl))
            fl.create_dataset("nphel", data = self.__nph0)

            fl.create_dataset("etot", data = fl["epot"][()] + fl["eeph"][()] + fl["ephm"][()] + fl["eint"][()] + fl["ekin"][()] + fl["eche"][()] + fl["ephn"][()])
            
            gloc0 = ohsum(self.irbf.u(0) * self.glocl)
            glocf = ohsum(self.irbf.u(self.beta) * self.glocl)
            var = np.sqrt(12*gloc0.a*glocf.a + 6*gloc0.b*glocf.b)
            fl.create_dataset("varn", data=var)
            
            gkbeta = ohsum(self.irbf.u(self.beta)[:,None,None,None] * self.gkl, axis=0)
            fl.create_dataset("gkbeta_a", data = gkbeta.a)
            fl.create_dataset("gkbeta_b", data = gkbeta.b)
            
            
            # Correlators in irreducible reciprocal zone
            print("Computing correlators")
            kidxs = np.transpose(np.indices((self.k_sz,)*3), (1,2,3,0)).reshape((self.k_sz**3,3))
            qidxs = kidxs[np.where((kidxs[:,0] <= self.k_sz//2) * (kidxs[:,1] <= kidxs[:,0]) * (kidxs[:,2] <= kidxs[:,0]) * (kidxs[:,2] <= kidxs[:,1]))]
            corrdiag = np.zeros(qidxs.shape[0])
            corroffd = np.zeros(qidxs.shape[0])
            corrcrs1 = np.zeros(qidxs.shape[0])
            corrcrs2 = np.zeros(qidxs.shape[0])
            for i,q in enumerate(qidxs):
                kidxst = (kidxs - q) % self.k_sz
                kidxstf = self.k_sz**2*kidxst[:,0] + self.k_sz*kidxst[:,1] + kidxst[:,2]
                gklconv = self.gkl.reshape((self.irbf.size,self.k_sz**3))[:,kidxstf].reshape((self.irbf.size,)+(self.k_sz,)*3)
                gkzero = ohsum(self.irbf.u(0)[:,None,None,None] * self.gkl, axis=0)
                gkconvbeta = ohsum(self.irbf.u(self.beta)[:,None,None,None] * gklconv, axis=0)
                corrdiag[i] = np.sum(gkzero.a * gkconvbeta.a) / self.k_sz**3
                corroffd[i] = np.sum(gkzero.b * gkconvbeta.b) / self.k_sz**3
                corrcrs1[i] = -np.sum(gkzero.a * gkconvbeta.b) / self.k_sz**3
                corrcrs2[i] = -np.sum(gkzero.b * gkconvbeta.a) / self.k_sz**3
            fl.create_dataset("correldiag", data=corrdiag)
            fl.create_dataset("correloffd", data=corroffd)
            fl.create_dataset("correlcrs1", data=corrcrs1)
            fl.create_dataset("correlcrs2", data=corrcrs2)
            fl.create_dataset("irrBZ", data=qidxs)
            
            print("Computing paramagnetic conductivity")
            k_vecs = kidxs.reshape((self.k_sz,)*3 + (3,)) * 2*np.pi / self.k_sz
            optcond = np.zeros((self.irbf.size,3,3))
            gkimagtime = self.gktau
            for i in range(3):
                for j in range(3):
                    optcond[:,i,j] = -self.t**2 * np.sum(np.sin(k_vecs[None,:,:,:,i]) * np.sin(k_vecs[None,:,:,:,j]) * (gkimagtime * gkimagtime[::-1]).trace, axis=(1,2,3)) / self.k_sz**3
            fl.create_dataset("optcondpl", data=self.stauf.fit(optcond, axis=0))
            
            
            print("Computing magneto-optical conductivity")
            magnetocond = np.zeros((self.stauf.tau.size,12,3))
            kx = k_vecs[...,0]
            ky = k_vecs[...,1]
            kz = k_vecs[...,2]
            for i in range(12):
                uk,vk,wk = commGtau(self.gkl, (i//2)*np.pi/3, -Fepbeta.trace, self.stauf, self.smatf, mode=['t','o'][i%2])
                jxk,jyk,jzk = gyrocurrent(self.gkl, self.t, self.Jphm, self.irbf, self.stauf, self.smatf)
                magnetocond[:,i,0] = -32*self.t**2*np.sum((jyk[None,...]*np.sin(kz)[None,...]*vk - jzk[None,...]*np.sin(ky)[None,...]*wk), axis=(1,2,3))
                magnetocond[:,i,1] = -32*self.t**2*np.sum((jzk[None,...]*np.sin(kx)[None,...]*wk - jxk[None,...]*np.sin(kz)[None,...]*uk), axis=(1,2,3))
                magnetocond[:,i,2] = -32*self.t**2*np.sum((jxk[None,...]*np.sin(ky)[None,...]*uk - jyk[None,...]*np.sin(kx)[None,...]*vk), axis=(1,2,3))
            fl.create_dataset("magnetocond", data = self.stauf.fit(magnetocond, axis=0))




def greenk_from_file(h5fl, irbf=None, irbb=None):
    beta = h5fl["beta"][()]
    wm = h5fl["wmax"][()]
    create_irbf = True
    if not irbf is None:
        try:
            if irbf.beta==beta and irbf.wmax==wm:
                create_irbf = False
        except:
            pass
    if create_irbf:
        irbf = ir.FiniteTempBasis('F', beta, wm)
    
    create_irbb = True
    if not irbb is None:
        try:
            if irbb.beta==beta and irbf.wmax==wm:
                create_irbb = False
        except:
            pass
    if create_irbb:
        irbb = ir.FiniteTempBasis('B', beta, wm)
    
    
    #Imag freq
    stauf = ir.TauSampling(irbf)
    smatf = ir.MatsubaraSampling(irbf)
    iwn = 1j * smatf.wn * np.pi/beta
    
    #Chemichal potential
    mu = h5fl["mu"][()]
    
    # Kinetic term
    k_sz = int(h5fl["k_sz"][()])
    t = h5fl["t"][()]
    ky,kx,kz = np.meshgrid(*(np.arange(0,2*np.pi,2*np.pi/k_sz),)*3)
    Hlatt = -2*t*(np.cos(kx) + np.cos(ky) + np.cos(kz))
    
    #Gloc
    glocl = ohmatrix(h5fl["glocl_a"][:], h5fl["glocl_b"][:])
    gloc_beta = ohsum(irbf.u(beta)*glocl)
    
    #hf
    U = h5fl["U"][()]
    J = h5fl["J"][()]
    sehf = sehf = ohmatrix(-2*gloc_beta.a*(3*U-5*J), gloc_beta.b*(U-2*J))
    
    #sephm
    Jphm = h5fl["Jphm"][()]
    gkbeta = ohmatrix(h5fl["gkbeta_a"][:], h5fl["gkbeta_b"][:])
    qidxs = np.transpose(np.indices((k_sz,)*3), (1,2,3,0)).reshape((k_sz**3,3))
    sephm = ohzeros((k_sz,)*3)
    for qidx in qidxs:
        gqbeta = gkbeta[tuple(qidx)]
        qx,qy,qz = qidx / k_sz * 2*np.pi
        gammakq = 2*Jphm*(np.cos(kx-qx) + np.cos(ky-qy) + np.cos(kz-qz))
        sephm += ohmatrix(4*gqbeta.a, -2*gqbeta.b)/3 * gammakq / k_sz**3
    
    #se2b
    gloctau = ohevaluate(stauf, glocl)
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
    se2bl = ohfit(stauf, ohmatrix(se2btau_a, se2btau_b))
    se2biw = ohevaluate(smatf, se2bl)
    
    #seeb
    staub = ir.TauSampling(irbb)
    g = h5fl["g"][()]
    dl = h5fl["dl"][:]
    dtau = staub.evaluate(dl)
    seepftau = -g**2/3 * dtau * ohmatrix(4/3*gloctau.a, -2/3*gloctau.b)
    seepl = ohfit(stauf, seepftau)
    seepiw = ohevaluate(smatf, seepl)
    
    lbd = h5fl["lbd"][()]
    gkiw = (iwn[:,None,None,None] - Hlatt[None,:,:,:] - sehf - se2biw[:,None,None,None] - sephm[None,:,:,:] - 2*seepiw[:,None,None,None] + mu - 0.5*lbd*ohmatrix(0,1))**-1
    return ohfit(smatf, gkiw, axis=0).real


def correct_magnetocurrent_from_file(h5fl, irbf=None, irbb=None):
    beta = h5fl["beta"][()]
    wm = h5fl["wmax"][()]
    create_irbf = True
    if not irbf is None:
        try:
            if irbf.beta==beta and irbf.wmax==wm:
                create_irbf = False
        except:
            pass
    if create_irbf:
        irbf = ir.FiniteTempBasis('F', beta, wm)
    
    create_irbb = True
    if not irbb is None:
        try:
            if irbb.beta==beta and irbf.wmax==wm:
                create_irbb = False
        except:
            pass
    if create_irbb:
        irbb = ir.FiniteTempBasis('B', beta, wm)
        
    stauf = ir.TauSampling(irbf)
    smatf = ir.MatsubaraSampling(irbf)
    print("Recovering Green's function")
    gkl = greenk_from_file(h5fl, irbf, irbb)
    t = h5fl["t"][()]
    Jphm = h5fl["Jphm"][()]
    ejt = h5fl["eeph"][()]
    k_sz = h5fl["k_sz"][()]
    kidxs = np.transpose(np.indices((k_sz,)*3), (1,2,3,0)).reshape((k_sz**3,3))
    k_vecs = kidxs.reshape((k_sz,)*3 + (3,)) * 2*np.pi / k_sz
    magnetocond = np.zeros((stauf.tau.size,12,3))
    kx = k_vecs[...,0]
    ky = k_vecs[...,1]
    kz = k_vecs[...,2]
    print("Computing magneto-optical conductivity")
    for i in range(12):
        print("Polarization %i of 12" % (i+1))
        uk,vk,wk = commGtau(gkl, (i//2)*np.pi/3, ejt, stauf, smatf, mode=['t','o'][i%2])
        jxk,jyk,jzk = gyrocurrent(gkl, t, Jphm, irbf, stauf, smatf)
        magnetocond[:,i,0] = -32*t**2*np.sum((jyk[None,...]*np.sin(kz)[None,...]*vk - jzk[None,...]*np.sin(ky)[None,...]*wk), axis=(1,2,3))
        magnetocond[:,i,1] = -32*t**2*np.sum((jzk[None,...]*np.sin(kx)[None,...]*wk - jxk[None,...]*np.sin(kz)[None,...]*uk), axis=(1,2,3))
        magnetocond[:,i,2] = -32*t**2*np.sum((jxk[None,...]*np.sin(ky)[None,...]*uk - jyk[None,...]*np.sin(kx)[None,...]*vk), axis=(1,2,3))
    # h5fl.create_dataset("magnetocond_corr", data = stauf.fit(magnetocond, axis=0))
    h5fl["magnetocond"][:] = stauf.fit(magnetocond, axis=0)