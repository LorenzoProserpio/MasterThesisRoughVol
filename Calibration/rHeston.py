import numpy as np
import ImpliedDrift
import Heston
import BlackScholes
import scipy.integrate

from variance_curve import variance_curve, Gompertz
from scipy.special import gamma

#######################Padè rHeston################################

def Pade33(u, t, H, rho, theta):
    alpha = H + 0.5
  
    aa = np.sqrt(u * (u + (0+1j)) - rho**2 * u**2)
    rm = -(0+1j) * rho * u - aa
    rp = -(0+1j) * rho * u + aa
  
    b1 = -u*(u+1j)/(2 * gamma(1+alpha))
    b2 = (1-u*1j) * u**2 * rho/(2* gamma(1+2*alpha))               
    b3 = gamma(1+2*alpha)/gamma(1+3*alpha) * \
        (u**2*(1j+u)**2/(8*gamma(1+alpha)**2)+(u+1j)*u**3*rho**2/(2*gamma(1+2*alpha)))

    g0 = rm
    g1 = -rm/(aa*gamma(1-alpha))
    g2 = rm/aa**2/gamma(1-2*alpha) * \
         (1 + rm/(2*aa)*gamma(1-2*alpha)/gamma(1-alpha)**2)

    den = g0**3 +2*b1*g0*g1-b2*g1**2+b1**2*g2+b2*g0*g2

    p1 = b1
    p2 = (b1**2*g0**2 + b2*g0**3 + b1**3*g1 + b1*b2*g0*g1 - \
          b2**2*g1**2 +b1*b3*g1**2 +b2**2*g0*g2 - b1*b3*g0*g2)/den
    q1 = (b1*g0**2 + b1**2*g1 - b2*g0*g1 + b3*g1**2 - b1*b2*g2 -b3*g0*g2)/den
    q2 = (b1**2*g0 + b2*g0**2 - b1*b2*g1 - b3*g0*g1 + b2**2*g2 - b1*b3*g2)/den
    q3 = (b1**3 + 2*b1*b2*g0 + b3*g0**2 -b2**2*g1 +b1*b3*g1 )/den
    p3 = g0*q3

    y = t**alpha
    
    return (p1*y + p2*y**2 + p3*y**3)/(1 + q1*y + q2*y**2 + q3*y**3)

# def DH_Pade33(u, t, H, rho, theta):
#     alpha = H + 0.5
    
#     aa = np.sqrt(u * (u + (0+1j)) - rho**2 * u**2)
#     rm = -(0+1j) * rho * u - aa
#     rp = -(0+1j) * rho * u + aa
  
#     b1 = -u*(u+1j)/(2 * gamma(1+alpha))
#     b2 = (1-u*1j) * u**2 * rho/(2* gamma(1+2*alpha))               
#     b3 = gamma(1+2*alpha)/gamma(1+3*alpha) * \
#         (u**2*(1j+u)**2/(8*gamma(1+alpha)**2)+(u+1j)*u**3*rho**2/(2*gamma(1+2*alpha)))

#     g0 = rm
#     g1 = -rm/(aa*gamma(1-alpha))
#     g2 = rm/aa**2/gamma(1-2*alpha) * (1 + rm/(2*aa)*gamma(1-2*alpha)/gamma(1-alpha)**2)

#     den = g0**3 +2*b1*g0*g1-b2*g1**2+b1**2*g2+b2*g0*g2

#     p1 = b1
#     p2 = (b1**2*g0**2 + b2*g0**3 + b1**3*g1 + b1*b2*g0*g1 - b2**2*g1**2 +b1*b3*g1**2 \
#           +b2**2*g0*g2 - b1*b3*g0*g2)/den
#     q1 = (b1*g0**2 + b1**2*g1 - b2*g0*g1 + b3*g1**2 - b1*b2*g2 -b3*g0*g2)/den
#     q2 = (b1**2*g0 + b2*g0**2 - b1*b2*g1 - b3*g0*g1 + b2**2*g2 - b1*b3*g2)/den
#     q3 = (b1**3 + 2*b1*b2*g0 + b3*g0**2 -b2**2*g1 +b1*b3*g1 )/den
#     p3 = g0*q3

#     y = t**alpha

#     h_pade = (p1*y + p2*y**2 + p3*y**3)/(1 + q1*y + q2*y**2 + q3*y**3)

#     res = 1/2*(h_pade-rm)*(h_pade-rp)

#     return res

def phi_rhest(u, t, H, rho, theta, N = 1000):
    if u == 0:
        return 1.
    
    term1 = 1j*u*t*((1j*u-1)*0.5*Gompertz(t)**2)
    alpha = H + 0.5
    dt = t/N
    tj = np.linspace(0,N,N+1,endpoint = True)*dt
    
    x = theta**(1./alpha)*tj
    xi = np.flip(variance_curve(tj))
    
    aux = Pade33(u, x, H, rho, theta)
    term2 = 1j*u*theta*rho*np.matmul(aux,xi)*dt
    term3 = theta**2*0.5*np.matmul(aux**2,xi)*dt
    return np.exp(term1 + term2 + term3)    

# def phi_rhest(u, t, H, rho, theta, N = 1000):
#     if u == 0:
#         return 1.
    
#     alpha = H + 0.5
#     dt = t/N
#     tj = np.linspace(0,N,N+1,endpoint = True)*dt
    
#     x = theta**(1./alpha)*tj
#     xi = np.flip(variance_curve(tj))
#     return np.exp(np.matmul(DH_Pade33(u, x, H, rho, theta),xi)*dt)    
    

def integral(x, t, H, rho, theta):
    
    # Pseudo-probabilities 

    integrand = (lambda u: np.real(np.exp((1j*u)*x) * \
                                   phi_rhest(u - 0.5j, t, H, rho, theta)) / \
                (u**2 + 0.25))
    
    i, err = scipy.integrate.quad_vec(integrand, 0, np.inf)
    
    return i

def analytic_rhest(S0, strikes, t, H, rho, theta, options_type):
    
    # Pricing of vanilla options under "analytic" rHeston
    
    a = np.log(S0/strikes) + ImpliedDrift.drift(t)*t 
    i = integral(a, t, H, rho, theta)
    r = ImpliedDrift.r(t)
    q = ImpliedDrift.q(t)
    out = S0 * np.exp(-q*t) - np.sqrt(S0*strikes) * np.exp(-(r+q)*t*0.5)/np.pi * i
    out = np.array([out]).flatten()

    for k in range(len(options_type)):
        if options_type[k] == 0:
            out[k] = Heston.call_put_parity(out[k], S0, strikes[k], r, q, t)
    
    return out

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
