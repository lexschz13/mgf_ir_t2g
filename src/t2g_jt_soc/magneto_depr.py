import numpy as np
import sparse_ir as ir
from .ohmatrix import ohevaluate, ohfit
from .reciprocal_space import k_convolution
from .phys_prop import conductivity_from_ir, conductivity_real_ir


__levi_civitta = np.array([[[int((i - j) * (j - k) * (k - i) // 2) for k in range(3)] for j in range(3)] for i in range(3)])
__abs_levi = np.abs(__levi_civitta)





def commGtau(Gkl, angle, ejt, stau, smat, mode='t'):
    from .dyson import ohevaluate
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
    
    akiw = (qkiw*rkiw-gammakiw**2)/deltakiw
    bkiw = (pkiw*rkiw-gammakiw**2)/deltakiw
    ckiw = (pkiw*qkiw-gammakiw**2)/deltakiw
    xkiw = gammakiw*(gammakiw-pkiw)/deltakiw
    ykiw = gammakiw*(gammakiw-qkiw)/deltakiw
    zkiw = gammakiw*(gammakiw-rkiw)/deltakiw
    
    aktau = stau.evaluate(smat.fit(akiw, axis=0).real, axis=0)
    bktau = stau.evaluate(smat.fit(bkiw, axis=0).real, axis=0)
    cktau = stau.evaluate(smat.fit(ckiw, axis=0).real, axis=0)
    xktau = stau.evaluate(smat.fit(xkiw, axis=0).real, axis=0)
    yktau = stau.evaluate(smat.fit(ykiw, axis=0).real, axis=0)
    zktau = stau.evaluate(smat.fit(zkiw, axis=0).real, axis=0)
    
    return ((bktau-cktau)*xktau[::-1] + (cktau[::-1]-bktau[::-1])*xktau + zktau*yktau[::-1] - yktau*zktau[::-1],
            (cktau-aktau)*yktau[::-1] + (aktau[::-1]-cktau[::-1])*yktau + xktau*zktau[::-1] - zktau*xktau[::-1],
            (aktau-bktau)*zktau[::-1] + (bktau[::-1]-aktau[::-1])*zktau + yktau*xktau[::-1] - xktau*yktau[::-1])



def gyrocurrent(Gkl, t, Jphm, irb, stau, smat):
    from .dyson import ohevaluate
    from .ohmatrix import ohsum
    beta = irb.beta
    k_sz = Gkl.shape[-1]
    Gkbeta = ohsum(irb.u(beta)[:,None,None,None] * Gkl, axis=0)
    Gkiw = ohevaluate(smat, Gkl, axis=0)
    
    ky,kx,kz = np.meshgrid(*(np.arange(0,2*np.pi,2*np.pi/k_sz),)*3)
    uk = 2*t*np.sin(np.array([kx,ky,kz]))
    vk = np.zeros((3,)+(k_sz,)*3)
    qidxs = np.transpose(np.indices((k_sz,)*3), (1,2,3,0)).reshape((k_sz**3,3))
    planar_vector = np.array([np.cos(ky)+np.cos(kz),
                              np.cos(kz)+np.cos(kx),
                              np.cos(kx)+np.cos(ky)])
    for qidx in qidxs:
        qx,qy,qz = qidx / k_sz * 2*np.pi
        uk -= 4/3 * Jphm * 2*Gkbeta.a * np.sin(np.array([kx-qx,ky-qy,kz-qz])) / k_sz**3
        vk -= 4/3 * Jphm *   Gkbeta.b * np.sin(np.array([kx-qx,ky-qy,kz-qz])) / k_sz**3
    
    jkiw = 2*((Gkiw.a[:,None,:,:,:]*(uk-vk)[None,:,:,:,:] - Gkiw.b[:,None,:,:,:]*(3*vk-uk)[None,:,:,:,:])*Gkiw.b[:,None,:,:,:]) * planar_vector[None,:,:,:,:]
    return np.sum(irb.u(beta)[:,None,None,None,None] * smat.fit(jkiw, axis=0).real, axis=0)




def tau_triple_conv(smat, stau, a,b,c, sum_idx, **kwargs):
    # Arguments on freq space
    d = np.einsum(sum_idx, a, b, c, optimize=True)
    return stau.evaluate(smat.fit(d, **kwargs).real, **kwargs)


def GD2h(Gkl, angle, ejt, smat, mode='t'):
    from .dyson import ohevaluate
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
    
    return akl, bkl, ckl, -xkl, -ykl, -zkl



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
    # upsx = 6*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[0]) / k_sz**3
    # upsy = 6*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[1]) / k_sz**3
    # upsz = 6*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[2]) / k_sz**3
    A = np.array([[1,-1,0],
                  [np.sqrt(1/3), np.sqrt(1/3), -np.sqrt(4/3)],
                  [1,1,1]])
    qx = -2*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[0]) / k_sz**3
    qy = -2*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[1]) / k_sz**3
    qz = -2*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[2]) / k_sz**3
    
    Q3 = qx - qy
    Q8 = (qx + qy - 2*qz) / np.sqrt(3)
    
    upsx, upsy, upsz = np.linalg.inv(A) @ np.array([Q3, Q8, 0])
    
    print("Adding hopping anisotropy")
    Gkiw = ohevaluate(smatf, Gkl, axis=0)
    Gkiw = (Gkiw**-1 + 2*t * (upsx*np.cos(kx) + upsy*np.cos(ky) + upsz*np.cos(kz)))**-1
    Gkl = ohfit(smatf, Gkiw, axis=0).real
    Gkl_jt = GD2h(Gkl, angle, ejt, smatf, jt_mode)
    gkl = np.array(Gkl_jt[:3])
    pkl = np.array(Gkl_jt[3:])
    gktau = stauf.evaluate(gkl, axis=1)
    pktau = stauf.evaluate(pkl, axis=1)
    # gkbeta = np.sum(irb.u(beta)[None,:,None,None,None] * gkl, axis=1)
    # pkbeta = np.sum(irb.u(beta)[None,:,None,None,None] * pkl, axis=1)
    gkiw = smatf.evaluate(gkl, axis=1)
    pkiw = smatf.evaluate(pkl, axis=1)
    
    # Projection
    print("Computing dynamical projections")
    psi0ktau = -2*np.sum(gktau*gktau[:,::-1], axis=0) + 4*np.sum(pktau*pktau[:,::-1], axis=0)
    psiktau = psi0ktau[None,...] - 8*pktau*pktau[:,::-1]
    psikl = stauf.fit(psiktau, axis=1)
    psikbeta = np.sum(irbf.u(beta)[None,:,None,None,None] * psikl, axis=1)
    
    # Self-energy
    print("Computing self-energy")
    fktau = k_convolution(gktau[:,None,...], psiktau[None,:,...])
    qktau = k_convolution(pktau[:,None,...], psiktau[None,:,...])
    fkl = stauf.fit(fktau, axis=2)
    qkl = stauf.fit(qktau, axis=2)
    fkiw = smatf.evaluate(fkl, axis=2)
    qkiw = smatf.evaluate(qkl, axis=2)
    
    # Current
    # a,r,l,...
    print("Computing current density")
    dk_gkiw = sin[None,:,None,...] * np.sum(np.eye(3)[:,:,None,None,None,None] * gkiw[None,:,...]**2 +
                                            (np.ones((3,3))-np.eye(3))[:,:,None,None,None,None] * pkiw[None,:,...]**2, axis=1)[:,None,...]
    dk_pkiw = sin[None,:,None,...] * np.sum(0.5*np.abs(__levi_civitta)[...,None,None,None,None] * (pkiw[None,:,None,...]*pkiw[None,None,:,...]
                                                                                               -pkiw[:,None,None,...]*(gkiw[None,:,None,...]+gkiw[None,None,:,...])), axis=(1,2))[:,None,...]
    dk_gkl = smatf.fit(dk_gkiw, axis=2).real
    dk_pkl = smatf.fit(dk_pkiw, axis=2).real
    dk_gktau = stauf.evaluate(dk_gkl, axis=2)
    dk_gktau = stauf.evaluate(dk_gkl, axis=2)
    dk_pktau = stauf.evaluate(dk_pkl, axis=2)
    # a,b,r,l,...
    dk_fktau = k_convolution(dk_gktau[:,None,:,...], psiktau[None,:,None,...])
    dk_qktau = k_convolution(dk_pktau[:,None,:,...], psiktau[None,:,None,...])
    dk_fkl = stauf.fit(dk_fktau, axis=3).real
    dk_qkl = stauf.fit(dk_qktau, axis=3).real
    jk = np.sum(irbf.u(beta)[None,None,None,:,None,None,None] * dk_fkl, axis=3)
    ck = np.sum(irbf.u(beta)[None,None,None,:,None,None,None] * dk_qkl, axis=3)
    
    # Susceptibility terms
    # alpha,beta,gamma,rho,l,kx,ky,kz
    print("Computing susceptibility")
    ximoktau = (+np.einsum("abc,aar...,bl...,abcl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, gkiw, "ba...,a...,c...->abc...", axis=3), optimize=True)
                -np.einsum("abc,aar...,bl...,abl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "ba...,b...,b...->ab...", axis=2), optimize=True)
                -np.einsum("abc,aar...,bl...,acl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "ca...,a...,a...->ac...", axis=2), optimize=True)
                -np.einsum("abc,aar...,bl...,abcl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "ca...,b...,c...->abc...", axis=3), optimize=True)
                #
                +np.einsum("abc,abr...,bl...,abcl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "ab...,c...,c...->abc...", axis=3), optimize=True)
                +np.einsum("abc,abr...,bl...,abl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "ab...,a...,b...->ab...", axis=2), optimize=True)
                -np.einsum("abc,abr...,bl...,abcl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "cb...,a...,a...->abc...", axis=3), optimize=True)
                -np.einsum("abc,abr...,bl...,bcl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "cb...,b...,c...->bc...", axis=2), optimize=True)
                #
                -np.einsum("abc,abr...,cl...,abl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "ab...,b...,b...->ab...", axis=2), optimize=True)
                -np.einsum("abc,abr...,cl...,abcl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "ab...,a...,c...->abc...", axis=3), optimize=True)
                +np.einsum("abc,abr...,cl...,abcl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, gkiw, "cb...,a...,b...->abc...", axis=3), optimize=True)
                -np.einsum("abc,abr...,cl...,bcl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "cb...,c...,c...->bc...", axis=2), optimize=True)
                #
                -np.einsum("abc,aar...,bl...,abcl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "ba...,c...,c...->abc...", axis=3), optimize=True)
                -np.einsum("abc,aar...,bl...,abl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "ba...,a...,b...->ab...", axis=2), optimize=True)
                +np.einsum("abc,aar...,bl...,abcl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "ca...,b...,b...->abc...", axis=3), optimize=True)
                +np.einsum("abc,aar...,bl...,acl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "ca...,a...,c...->ac...", axis=2), optimize=True)
                #
                +np.einsum("abc,aar...,bl...,abl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "ba...,a...,a...->ab...", axis=2), optimize=True)
                +np.einsum("abc,aar...,bl...,abcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "ba...,b...,c...->abc...", axis=3), optimize=True)
                -np.einsum("abc,aar...,bl...,abcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, gkiw, "ca...,a...,b...->abc...", axis=3), optimize=True)
                +np.einsum("abc,aar...,bl...,acl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "ca...,c...,c...->ac...", axis=2), optimize=True)
                #
                +np.einsum("abc,abr...,bl...,abcl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, fkiw, pkiw, pkiw, "ab...,b...,c...->abc...", axis=3), optimize=True)
                +np.einsum("abc,abr...,bl...,abl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, pkiw, "bb...,b...,a...->ab...", axis=2), optimize=True)
                -np.einsum("abc,abr...,bl...,abcl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, pkiw, "cb...,c...,a...->abc...", axis=3), optimize=True)
                +np.einsum("abc,abr...,bl...,bcl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "bb...,c...,c...->bc...", axis=2), optimize=True)
                -np.einsum("abc,abr...,bl...,abl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "bb...,a...,b...->ab...", axis=2), optimize=True)
                #
                -np.einsum("abc,abr...,cl...,abcl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, fkiw, pkiw, pkiw, "ab...,b...,c...->abc...", axis=3), optimize=True)
                -np.einsum("abc,abr...,cl...,abl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, pkiw, "bb...,b...,a...->ab...", axis=2), optimize=True)
                +np.einsum("abc,abr...,cl...,abcl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, pkiw, "cb...,c...,a...->abc...", axis=3), optimize=True)
                -np.einsum("abc,abr...,cl...,bcl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "bb...,c...,c...->bc...", axis=2), optimize=True)
                +np.einsum("abc,abr...,cl...,abl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "bb...,a...,b...->ab...", axis=2), optimize=True)
                #
                +np.einsum("abc,abr...,al...,abcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, pkiw, pkiw, "ab...,c...,c...->abc...", axis=3), optimize=True)
                -np.einsum("abc,abr...,al...,abl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, pkiw, pkiw, "ab...,b...,b...->ab...", axis=2), optimize=True)
                +np.einsum("abc,abr...,al...,abl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, pkiw, pkiw, "bb...,a...,a...->ab...", axis=2), optimize=True)
                -np.einsum("abc,abr...,al...,bl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, gkiw, "bb...,b...,b...->b...", axis=1), optimize=True)
                +np.einsum("abc,abr...,al...,abcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, pkiw, pkiw, "cb...,a...,a...->abc...", axis=3), optimize=True)
                -np.einsum("abc,abr...,al...,bcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, gkiw, "cb...,c...,c...->bc...", axis=2), optimize=True)
                -np.einsum("abc,abr...,al...,bcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "bb...,c...,b...->bc...", axis=2), optimize=True)*2
                -np.einsum("abc,abr...,al...,abcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "bb...,a...,c...->abc...", axis=3), optimize=True)*2
                #
                -np.einsum("abc,abr...,bl...,abcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, pkiw, "ab...,a...,c...->abc...", axis=3), optimize=True)
                +np.einsum("abc,abr...,bl...,bcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, pkiw, "bb...,b...,c...->bc...", axis=2), optimize=True)
                +np.einsum("abc,abr...,bl...,abcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, pkiw, pkiw, "cb...,a...,b...->abc...", axis=3), optimize=True)
                +np.einsum("abc,abr...,bl...,abl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "bb...,a...,a...->ab...", axis=2), optimize=True)
                -np.einsum("abc,abr...,bl...,bcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "bb...,b...,c...->bc...", axis=2), optimize=True)
                #
                +np.einsum("abc,abr...,cl...,abl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, pkiw, "ab...,a...,b...->ab...", axis=2), optimize=True)
                +np.einsum("abc,abr...,cl...,abcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, pkiw, pkiw, "bb...,a...,c...->abc...", axis=3), optimize=True)
                +np.einsum("abc,abr...,cl...,bcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, pkiw, "cb...,c...,b...->bc...", axis=2), optimize=True)
                +np.einsum("abc,abr...,cl...,abcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, gkiw, "bb...,a...,c...->abc...", axis=3), optimize=True)
                +np.einsum("abc,abr...,cl...,bl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "bb...,b...,b...->b...", axis=1), optimize=True)
                )
    
    ximotau_xy = -np.sum(np.sin(kx)[None,...]*(ximoktau[1] - ximoktau[1,::-1]) + np.sin(ky)[None,...]*(ximoktau[0] - ximoktau[0,::-1]), axis=(-3,-2,-1)) * (1+upsx)*(1+upsy)
    ximotau_yz = -np.sum(np.sin(ky)[None,...]*(ximoktau[2] - ximoktau[2,::-1]) + np.sin(kz)[None,...]*(ximoktau[1] - ximoktau[1,::-1]), axis=(-3,-2,-1)) * (1+upsy)*(1+upsz)
    ximotau_zx = -np.sum(np.sin(kz)[None,...]*(ximoktau[0] - ximoktau[0,::-1]) + np.sin(kx)[None,...]*(ximoktau[2] - ximoktau[2,::-1]), axis=(-3,-2,-1)) * (1+upsz)*(1+upsx)
    
    # ximol_xy = conductivity_real_ir(irbb, staub, smatb, ximotau_xy, axis=-1)
    # ximol_yz = conductivity_real_ir(irbb, staub, smatb, ximotau_yz, axis=-1)
    # ximol_zx = conductivity_real_ir(irbb, staub, smatb, ximotau_zx, axis=-1)
    
    ximol_xy = staub.fit(ximotau_xy, axis=-1)
    ximol_yz = staub.fit(ximotau_yz, axis=-1)
    ximol_zx = staub.fit(ximotau_zx, axis=-1)
    
    return 4*t**2* np.array([ximol_yz, ximol_zx, ximol_xy])


def susc_mo_kslice(Gkl, beta, t, angle, ejt, irbf, irbb, jt_mode, plane="z"):
    stauf = ir.TauSampling(irbf)
    smatf = ir.MatsubaraSampling(irbf)
    staub = ir.TauSampling(irbb)
    smatb = ir.MatsubaraSampling(irbb)
    
    if not plane in "xyz":
        raise ValueError("Not allowed plane name, specify x, y or z")
    
    plane_num = "xyz".index(plane)
    
    print("Breaking G symmetry")
    Gkl_D2h_iso = GD2h(Gkl, angle, ejt, smatf, jt_mode)
    k_sz = Gkl.shape[1]
    ky,kx,kz = np.meshgrid(*(np.arange(0,2*np.pi,2*np.pi/k_sz),)*3)
    sin = np.sin(np.array([kx,ky,kz]))
    # cos = np.cos(np.array([kx,ky,kz]))
    # upsx = 6*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[0]) / k_sz**3
    # upsy = 6*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[1]) / k_sz**3
    # upsz = 6*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[2]) / k_sz**3
    A = np.array([[1,-1,0],
                  [np.sqrt(1/3), np.sqrt(1/3), -np.sqrt(4/3)],
                  [1,1,1]])
    qx = -2*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[0]) / k_sz**3
    qy = -2*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[1]) / k_sz**3
    qz = -2*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[2]) / k_sz**3
    
    Q3 = qx - qy
    Q8 = (qx + qy - 2*qz) / np.sqrt(3)
    
    upsx, upsy, upsz = np.linalg.inv(A) @ np.array([Q3, Q8, 0])
    
    print("Adding hopping anisotropy")
    Gkiw = ohevaluate(smatf, Gkl, axis=0)
    Gkiw = (Gkiw**-1 + 2*t * (upsx*np.cos(kx) + upsy*np.cos(ky) + upsz*np.cos(kz)))**-1
    Gkl = ohfit(smatf, Gkiw, axis=0).real
    Gkl_jt = GD2h(Gkl, angle, ejt, smatf, jt_mode)
    gkl = np.array(Gkl_jt[:3])
    pkl = np.array(Gkl_jt[3:])
    gktau = stauf.evaluate(gkl, axis=1)
    pktau = stauf.evaluate(pkl, axis=1)
    # gkbeta = np.sum(irb.u(beta)[None,:,None,None,None] * gkl, axis=1)
    # pkbeta = np.sum(irb.u(beta)[None,:,None,None,None] * pkl, axis=1)
    gkiw = smatf.evaluate(gkl, axis=1)
    pkiw = smatf.evaluate(pkl, axis=1)
    
    # Projection
    print("Computing dynamical projections")
    psi0ktau = -2*np.sum(gktau*gktau[:,::-1], axis=0) + 4*np.sum(pktau*pktau[:,::-1], axis=0)
    psiktau = psi0ktau[None,...] - 8*pktau*pktau[:,::-1]
    psikl = stauf.fit(psiktau, axis=1)
    psikbeta = np.sum(irbf.u(beta)[None,:,None,None,None] * psikl, axis=1)
    
    # Self-energy
    print("Computing self-energy")
    fktau = k_convolution(gktau[:,None,...], psiktau[None,:,...])
    qktau = k_convolution(pktau[:,None,...], psiktau[None,:,...])
    fkl = stauf.fit(fktau, axis=2)
    qkl = stauf.fit(qktau, axis=2)
    fkiw = smatf.evaluate(fkl, axis=2)
    qkiw = smatf.evaluate(qkl, axis=2)
    
    # Current
    # a,r,l,...
    print("Computing current density")
    dk_gkiw = sin[None,:,None,...] * np.sum(np.eye(3)[:,:,None,None,None,None] * gkiw[None,:,...]**2 +
                                            (np.ones((3,3))-np.eye(3))[:,:,None,None,None,None] * pkiw[None,:,...]**2, axis=1)[:,None,...]
    dk_pkiw = sin[None,:,None,...] * np.sum(0.5*np.abs(__levi_civitta)[...,None,None,None,None] * (pkiw[None,:,None,...]*pkiw[None,None,:,...]
                                                                                               -pkiw[:,None,None,...]*(gkiw[None,:,None,...]+gkiw[None,None,:,...])), axis=(1,2))[:,None,...]
    dk_gkl = smatf.fit(dk_gkiw, axis=2).real
    dk_pkl = smatf.fit(dk_pkiw, axis=2).real
    dk_gktau = stauf.evaluate(dk_gkl, axis=2)
    dk_gktau = stauf.evaluate(dk_gkl, axis=2)
    dk_pktau = stauf.evaluate(dk_pkl, axis=2)
    # a,b,r,l,...
    dk_fktau = k_convolution(dk_gktau[:,None,:,...], psiktau[None,:,None,...])
    dk_qktau = k_convolution(dk_pktau[:,None,:,...], psiktau[None,:,None,...])
    dk_fkl = stauf.fit(dk_fktau, axis=3).real
    dk_qkl = stauf.fit(dk_qktau, axis=3).real
    jk = np.sum(irbf.u(beta)[None,None,None,:,None,None,None] * dk_fkl, axis=3)
    ck = np.sum(irbf.u(beta)[None,None,None,:,None,None,None] * dk_qkl, axis=3)
    
    # Confinement
    def conf(funk):
        confk = np.swapaxes(funk, plane_num-3, -1)[...,0]
        if plane != "z":
            confk = np.swapaxes(confk, -1, -2)
        return confk
    jk = conf(jk)
    ck = conf(ck)
    gktau = conf(gktau)
    pktau = conf(pktau)
    gkiw = conf(gkiw)
    pkiw = conf(pkiw)
    fkiw = conf(fkiw)
    qkiw = conf(qkiw)
    
    # Susceptibility terms
    # alpha,beta,gamma,rho,l,kx,ky,kz
    print("Computing susceptibility")
    ximoktau = (+np.einsum("abc,aar...,bl...,abcl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, gkiw, "ba...,a...,c...->abc...", axis=3), optimize=True)
                -np.einsum("abc,aar...,bl...,abl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "ba...,b...,b...->ab...", axis=2), optimize=True)
                -np.einsum("abc,aar...,bl...,acl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "ca...,a...,a...->ac...", axis=2), optimize=True)
                -np.einsum("abc,aar...,bl...,abcl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "ca...,b...,c...->abc...", axis=3), optimize=True)
                #
                +np.einsum("abc,abr...,bl...,abcl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "ab...,c...,c...->abc...", axis=3), optimize=True)
                +np.einsum("abc,abr...,bl...,abl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "ab...,a...,b...->ab...", axis=2), optimize=True)
                -np.einsum("abc,abr...,bl...,abcl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "cb...,a...,a...->abc...", axis=3), optimize=True)
                -np.einsum("abc,abr...,bl...,bcl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "cb...,b...,c...->bc...", axis=2), optimize=True)
                #
                -np.einsum("abc,abr...,cl...,abl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "ab...,b...,b...->ab...", axis=2), optimize=True)
                -np.einsum("abc,abr...,cl...,abcl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "ab...,a...,c...->abc...", axis=3), optimize=True)
                +np.einsum("abc,abr...,cl...,abcl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, gkiw, "cb...,a...,b...->abc...", axis=3), optimize=True)
                -np.einsum("abc,abr...,cl...,bcl...->rl...", __abs_levi, jk, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "cb...,c...,c...->bc...", axis=2), optimize=True)
                #
                -np.einsum("abc,aar...,bl...,abcl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "ba...,c...,c...->abc...", axis=3), optimize=True)
                -np.einsum("abc,aar...,bl...,abl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "ba...,a...,b...->ab...", axis=2), optimize=True)
                +np.einsum("abc,aar...,bl...,abcl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "ca...,b...,b...->abc...", axis=3), optimize=True)
                +np.einsum("abc,aar...,bl...,acl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "ca...,a...,c...->ac...", axis=2), optimize=True)
                #
                +np.einsum("abc,aar...,bl...,abl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "ba...,a...,a...->ab...", axis=2), optimize=True)
                +np.einsum("abc,aar...,bl...,abcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "ba...,b...,c...->abc...", axis=3), optimize=True)
                -np.einsum("abc,aar...,bl...,abcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, gkiw, "ca...,a...,b...->abc...", axis=3), optimize=True)
                +np.einsum("abc,aar...,bl...,acl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "ca...,c...,c...->ac...", axis=2), optimize=True)
                #
                +np.einsum("abc,abr...,bl...,abcl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, fkiw, pkiw, pkiw, "ab...,b...,c...->abc...", axis=3), optimize=True)
                +np.einsum("abc,abr...,bl...,abl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, pkiw, "bb...,b...,a...->ab...", axis=2), optimize=True)
                -np.einsum("abc,abr...,bl...,abcl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, pkiw, "cb...,c...,a...->abc...", axis=3), optimize=True)
                +np.einsum("abc,abr...,bl...,bcl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "bb...,c...,c...->bc...", axis=2), optimize=True)
                -np.einsum("abc,abr...,bl...,abl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "bb...,a...,b...->ab...", axis=2), optimize=True)
                #
                -np.einsum("abc,abr...,cl...,abcl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, fkiw, pkiw, pkiw, "ab...,b...,c...->abc...", axis=3), optimize=True)
                -np.einsum("abc,abr...,cl...,abl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, pkiw, "bb...,b...,a...->ab...", axis=2), optimize=True)
                +np.einsum("abc,abr...,cl...,abcl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, pkiw, "cb...,c...,a...->abc...", axis=3), optimize=True)
                -np.einsum("abc,abr...,cl...,bcl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "bb...,c...,c...->bc...", axis=2), optimize=True)
                +np.einsum("abc,abr...,cl...,abl...->rl...", __abs_levi, ck, gktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "bb...,a...,b...->ab...", axis=2), optimize=True)
                #
                +np.einsum("abc,abr...,al...,abcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, pkiw, pkiw, "ab...,c...,c...->abc...", axis=3), optimize=True)
                -np.einsum("abc,abr...,al...,abl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, pkiw, pkiw, "ab...,b...,b...->ab...", axis=2), optimize=True)
                +np.einsum("abc,abr...,al...,abl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, pkiw, pkiw, "bb...,a...,a...->ab...", axis=2), optimize=True)
                -np.einsum("abc,abr...,al...,bl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, gkiw, "bb...,b...,b...->b...", axis=1), optimize=True)
                +np.einsum("abc,abr...,al...,abcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, pkiw, pkiw, "cb...,a...,a...->abc...", axis=3), optimize=True)
                -np.einsum("abc,abr...,al...,bcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, gkiw, "cb...,c...,c...->bc...", axis=2), optimize=True)
                -np.einsum("abc,abr...,al...,bcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "bb...,c...,b...->bc...", axis=2), optimize=True)*2
                -np.einsum("abc,abr...,al...,abcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "bb...,a...,c...->abc...", axis=3), optimize=True)*2
                #
                -np.einsum("abc,abr...,bl...,abcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, pkiw, "ab...,a...,c...->abc...", axis=3), optimize=True)
                +np.einsum("abc,abr...,bl...,bcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, pkiw, "bb...,b...,c...->bc...", axis=2), optimize=True)
                +np.einsum("abc,abr...,bl...,abcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, pkiw, pkiw, "cb...,a...,b...->abc...", axis=3), optimize=True)
                +np.einsum("abc,abr...,bl...,abl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, pkiw, "bb...,a...,a...->ab...", axis=2), optimize=True)
                -np.einsum("abc,abr...,bl...,bcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "bb...,b...,c...->bc...", axis=2), optimize=True)
                #
                +np.einsum("abc,abr...,cl...,abl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, pkiw, "ab...,a...,b...->ab...", axis=2), optimize=True)
                +np.einsum("abc,abr...,cl...,abcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, pkiw, pkiw, "bb...,a...,c...->abc...", axis=3), optimize=True)
                +np.einsum("abc,abr...,cl...,bcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, fkiw, gkiw, pkiw, "cb...,c...,b...->bc...", axis=2), optimize=True)
                +np.einsum("abc,abr...,cl...,abcl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, gkiw, gkiw, "bb...,a...,c...->abc...", axis=3), optimize=True)
                +np.einsum("abc,abr...,cl...,bl...->rl...", __abs_levi, ck, pktau, tau_triple_conv(smatf, stauf, qkiw, pkiw, pkiw, "bb...,b...,b...->b...", axis=1), optimize=True)
                )
    
    ximoktau_xy = -np.sin(conf(kx))[None,...]*(ximoktau[1] - ximoktau[1,::-1]) + np.sin(conf(ky))[None,...]*(ximoktau[0] - ximoktau[0,::-1]) * (1+upsx)*(1+upsy)
    ximoktau_yz = -np.sin(conf(ky))[None,...]*(ximoktau[2] - ximoktau[2,::-1]) + np.sin(conf(kz))[None,...]*(ximoktau[1] - ximoktau[1,::-1]) * (1+upsy)*(1+upsz)
    ximoktau_zx = -np.sin(conf(kz))[None,...]*(ximoktau[0] - ximoktau[0,::-1]) + np.sin(conf(kx))[None,...]*(ximoktau[2] - ximoktau[2,::-1]) * (1+upsz)*(1+upsx)
    
    # ximol_xy = conductivity_real_ir(irbb, staub, smatb, ximotau_xy, axis=-1)
    # ximol_yz = conductivity_real_ir(irbb, staub, smatb, ximotau_yz, axis=-1)
    # ximol_zx = conductivity_real_ir(irbb, staub, smatb, ximotau_zx, axis=-1)
    
    ximol_xy = staub.fit(ximoktau_xy, axis=-3)
    ximol_yz = staub.fit(ximoktau_yz, axis=-3)
    ximol_zx = staub.fit(ximoktau_zx, axis=-3)
    
    return 4*t**2* np.array([ximol_yz, ximol_zx, ximol_xy])


def print_dt(Gkl, beta, t, angle, ejt, irbf, irbb, jt_mode):
    # stauf = ir.TauSampling(irbf)
    smatf = ir.MatsubaraSampling(irbf)
    # staub = ir.TauSampling(irbb)
    # smatb = ir.MatsubaraSampling(irbb)
    
    A = np.array([[1,-1,0],
                  [np.sqrt(1/3), np.sqrt(1/3), -np.sqrt(4/3)],
                  [1,1,1]])
    
    print("Breaking G symmetry")
    Gkl_D2h_iso = GD2h(Gkl, angle, ejt, smatf, jt_mode)
    k_sz = Gkl.shape[1]
    ky,kx,kz = np.meshgrid(*(np.arange(0,2*np.pi,2*np.pi/k_sz),)*3)
    # sin = np.sin(np.array([kx,ky,kz]))
    # cos = np.cos(np.array([kx,ky,kz]))
    qx = -2*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[0]) / k_sz**3
    qy = -2*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[1]) / k_sz**3
    qz = -2*np.sum(irbf.u(beta)[:,None,None,None] * Gkl_D2h_iso[2]) / k_sz**3
    
    Q3 = qx - qy
    Q8 = (qx + qy - 2*qz) / np.sqrt(3)
    
    x, y, z = np.linalg.inv(A) @ np.array([Q3, Q8, 0])
    
    print(Q3, Q8, x, y, z)


# def greenk_from_h5(h5fl, irb_ls=None):
#     beta = h5fl["beta"][()]
#     wm = h5fl["wmax"][()]
    
#     create_irb = True
#     if not irb_ls is None:
#         for irb_el in irb_ls:
#             if irb_el.beta==beta and irb_el.wmax==wm:
#                 irb = irb_el
#                 create_irb = False
#                 break
#     if create_irb:
#         irb = ir.FiniteTempBasis('F', beta, wm)
    
#     t = h5fl["t"][()]
#     lbd = h5fl["lbd"][()]