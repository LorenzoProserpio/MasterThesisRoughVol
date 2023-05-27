import numpy as np
import ImpliedDrift
import Heston
import BlackScholes
import scipy.integrate

from variance_curve import variance_curve, Gompertz
from scipy.special import gamma

####################### Padè rHeston ################################

def Pade33(u, t, H, rho, theta):
    alpha = H + 0.5
    
    aa = np.sqrt(u * (u + (0+1j)) - rho**2 * u**2)
    rm = -(0+1j) * rho * u - aa
    rp = -(0+1j) * rho * u + aa

    gamma1 = gamma(1+alpha)
    gamma2 = gamma(1+2*alpha)
    gammam1 = gamma(1-alpha)
    gammam2 = gamma(1-2*alpha)

    b1 = -u*(u+1j)/(2 * gamma1)
    b2 = (1-u*1j) * u**2 * rho/(2* gamma2)               
    b3 = gamma2/gamma(1+3*alpha) * \
        (u**2*(1j+u)**2/(8*gamma1**2)+(u+1j)*u**3*rho**2/(2*gamma2))

    g0 = rm
    g1 = -rm/(aa*gammam1)
    g2 = rm/aa**2/gammam2 * \
         (1 + rm/(2*aa)*gammam2/gammam1**2)

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

####################### Analytic rHeston ################################

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

###################### COS METHOD Le Floch #########################
    
def chi_k(k, c, d, a, b):
    # Auxiliary function for U_k
    
    aux_1 = k*np.pi/(b-a)
    aux_2 = np.exp(d)
    aux_3 = np.exp(c)
    
    return  (np.cos(aux_1*(d-a))*aux_2 - \
            aux_3 + \
            aux_1*np.sin(aux_1*(d-a))*aux_2) / (1+aux_1**2)

def psi_k(k, c, d, a, b):    
    # Auxiliary function for U_k
    
    if k == 0:
        return d - c
    
    aux = k*np.pi/(b-a)
    return np.sin(aux*(d-a)) / aux
    
def U_k_put(k, a, b):
    # Auxiliary for cos_method
    
    return 2./(b-a) * (psi_k(k, a, 0, a, b) - chi_k(k, a, 0, a, b))

def optimal_ab(t, H, theta, rho, L = 12):
    # Compute the optimal interval for the truncation
    
    c1 =  (ImpliedDrift.drift(t)-Gompertz(t)**2*0.5) * t
    
    aux = theta*Gompertz(1000)**2
    aux2 = H+1.5
    aux3 = gamma(aux2)
    c2 = aux*t**aux2*rho/(aux2*aux3) - theta*aux*t**(2*H+2)*0.25/((2*H+2)*aux3**2)
    
    return c1 - 12*np.sqrt(np.abs(c2)), c1 + 12*np.sqrt(np.abs(c2))


def precomputed_terms(t, H, theta, rho, L, N):
    # Auxiliary term precomputed
    
    a,b = optimal_ab(t, H, theta, rho, L)
    
    aux = np.pi/(b-a)
    out = np.zeros(N-1)
    
    for k in range(1,N):
        out[k-1] = np.real(np.exp(-1j*k*a*aux)*\
                         phi_rhest(k*aux, t, H, rho, theta)*\
                         np.exp(1j*k*aux*ImpliedDrift.drift(t)*t))
    
    return out, a, b

def V_k_put(k, a, b, S0, K, z):
    # V_k coefficients for puts   
    
    return 2./(b-a)*(K*psi_k(k, a, z, a, b) - S0*chi_k(k, a, z, a, b))

def cos_method_Heston_LF(precomp_term, a, b, t,  H, theta, rho, S0,\
                         strikes, N, options_type, L=12):
    # Cosine Fourier Expansion for evaluating vanilla options under rHeston using LeFloch correction
    # Should be better for deep otm options.
    
    # precomp_term: precomputed terms from the function precomputed_terms
    # a,b: extremes of the interval to approximate
    # t: time to expiration (annualized) (must be a number)
    # H, theta, rho: rHeston parameters
    # S0: initial spot price
    # strikes: np.array of strikes
    # N: number of terms of the truncated expansion
    # options_type: binary np.array (1 for calls, 0 for puts)
    # L: truncation level

    z = np.log(strikes/S0)
    
    r = ImpliedDrift.r(t)
    q = ImpliedDrift.q(t)
    
    out = 0.5 * V_k_put(0, a, b, S0, strikes, z)
    
    for k in range(1,N):
        out = out + precomp_term[k-1]*V_k_put(k, a, b, S0, strikes, z)
    
    D = np.exp(-r*t)
    out = out*D
    
    for k in range(len(strikes)):
        if options_type[k] == 1:
            out[k] = Heston.put_call_parity(out[k], S0, strikes[k], r, q, t)

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
