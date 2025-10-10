import numpy as np
import sparse_ir as ir
# from .dyson import ohevaluate





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