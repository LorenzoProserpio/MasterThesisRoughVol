import numpy as np
from variance_curve import variance_curve
import ImpliedDrift
from scipy.special import gamma

#######################Simulation rHeston################################

# Psi for the QE Scheme of Lemma 7.
def psi_m(psi, ev, w):
    #psi minus
    
    beta2 = psi
    mask = psi > 0
    mask1 = psi <= 0
    if np.any(mask):
        beta2[mask] = 2./psi[mask]-1+np.sqrt(2./psi[mask]* \
                                             np.abs(2./psi[mask]-1))   
    if np.any(mask1):
        beta2[mask1] = 0.    
    return ev/(1+beta2)*(np.sqrt(np.abs(beta2))+w)**2 
    
def psi_p(psi, ev, u):
    #psi plus
    
    p = 2/(1+psi)
    res = (u<p)*(-ev)/2*(1+psi)
    mask = u > 0
    if np.any(mask):
        res[mask] = np.log(u[mask]/p[mask])
    return res

# functions for K_i, K_ii and K_01
def Gi(eta, alpha, dt, i):
    return np.sqrt(2*alpha-1)*eta/alpha * dt**alpha * ((i+1)**alpha - i**alpha)

def Gii(eta, H, dt, i):
    aux = 2*H
    return eta**2 * dt**aux * ((i+1)**aux - i**aux)

def G01(eta, alpha, dt):
    return Gi(eta,alpha,dt,0)*Gi(eta,alpha,dt,1)/dt

def HQE_sim(theta, H, rho, T, S0, paths, steps, eps0 = 1e-10):
    # HQE scheme

    # theta, H, rho: parameters of the rHeston model
    # T: final time of the simulations, in years
    # S0: spot price at time 0
    # paths: number of paths to simulate
    # steps: number of timesteps between 0 and T
    # eps0: lower bound for xihat
    
    dt = T/steps
    dt_sqrt = np.sqrt(dt)
    alpha = H + 0.5
    eta = theta/(gamma(alpha)*np.sqrt(2*H))
    rho2m1 = np.sqrt(1-rho*rho)
    
    W = np.random.normal(0.,1.,size = (steps,paths))
    Wperp = np.random.normal(0.,1.,size = (steps,paths))
    Z = np.random.normal(0.,1.,size = (steps,paths))
    U = np.random.uniform(0.,1.,size = (steps,paths))
    Uperp = np.random.uniform(0.,1.,size = (steps,paths))
    
    tj = np.arange(0,steps,1)*dt
    tj += dt
    
    xij = variance_curve(tj)
    G0del = Gi(eta,alpha,dt,0)
    G00del = Gii(eta,alpha,dt,0)
    G11del = Gii(eta,alpha,dt,1)
    G01del = G01(eta,alpha,dt)
    G00j = np.zeros(steps)
    
    for j in range(steps):
        G00j[j] = Gii(eta,H,dt,j)
    bstar = np.sqrt((G00j)/dt)
    
    rho_vchi = G0del/np.sqrt(G00del*dt)
    beta_vchi = G0del/dt
    
    u = np.zeros((steps,paths))
    chi = np.zeros((steps,paths))
    v = np.ones(paths)*variance_curve(0)
    hist_v = np.zeros((steps,paths))
    hist_v[0,:] = v
    xihat = np.ones(paths)*xij[0]
    x = np.zeros((steps,paths))
    y = np.zeros(paths)
    w = np.zeros(paths)
    
    for j in range(steps):
        xibar = (xihat + 2*H*v)/(1+2*H)
        
        psi_chi = 2*beta_vchi*xibar*dt/(xihat**2)
        psi_eps =  2/(xihat**2)*xibar*(G00del - G0del**2/dt)
        aux_ = xihat/2
        
        z_chi = np.zeros(paths)
        z_eps = np.zeros(paths)
        
        mask1 = psi_chi < 1.5
        mask2 = psi_chi >= 1.5
        mask3 = psi_eps < 1.5
        mask4 = psi_eps >= 1.5
        
        if np.any(mask1):            
            z_chi[mask1] = psi_m(psi_chi[mask1],aux_[mask1],W[j,mask1])
        if np.any(mask2):
            z_chi[mask2] = psi_m(psi_chi[mask2],aux_[mask2],U[j,mask2])
        if np.any(mask3):
            z_eps[mask3] = psi_m(psi_eps[mask3],aux_[mask3],Wperp[j,mask3])
        if np.any(mask4):
            z_eps[mask4] = psi_m(psi_eps[mask4],aux_[mask4],Uperp[j,mask4])
        
        chi[j,:] = (z_chi-aux_)/beta_vchi
        eps = z_eps - aux_
        u[j,:] = beta_vchi*chi[j,:]+eps
        vf = xihat + u[j,:]
        vf[vf < eps0] = eps0
        
        dw = (v+vf)/2*dt
        w += dw
        y += chi[j,:]
        x[j,:] = x[j-1,:] + ImpliedDrift.drift(T)*dt - dw/2 + np.sqrt(dw) \
                * (rho2m1*Z[j,:]) + rho*chi[j,:]
        
        btilde = np.flip(bstar[1:j+1])
        if j < steps-1:
            xihat = xij[j+1] + (np.matmul(btilde,chi[:j,:]))
        v = vf
        hist_v[j,:] = v
    return np.vstack([np.ones(paths)*S0,(np.exp(x)*S0)]),hist_v
