import numpy as np
import scipy.integrate
from scipy.special import iv

#################### PUT-CALL PARITY #############################

def put_call_parity(put, S0, strike, r, q, tau):
    # Standard put_call_parity
    return put + S0*np.exp(-q*tau) - strike*np.exp(-r*tau)

def call_put_parity(call, S0, strike, r, q, tau):
    return call - S0*np.exp(-q*tau) + strike*np.exp(-r*tau)



##################### ANALYTIC HESTON ############################

def phi_hest(u, tau, sigma_0, kappa, eta, theta, rho):
    
    # Compute the characteristic function for Heston Model
    
    # u: argument of the function (where you want to evaluate)
    # tau: time to expiration
    # sigma_0, kappa, eta, theta, rho: Heston parameters 

    alpha_hat = -0.5 * u * (u + 1j)
    beta = kappa - 1j * u * theta * rho
    gamma = 0.5 * theta ** 2
    d = np.sqrt(beta**2 - 4 * alpha_hat * gamma)
    g = (beta - d) / (beta + d)
    h = np.exp(-d*tau)
    A_ = (beta - d)*tau - 2*np.log((g*h-1) / (g-1))
    A = kappa * eta / (theta**2) * A_
    B = (beta - d) / (theta**2) * (1 - h) / (1 - g*h)
    return np.exp(A + B * sigma_0)

def integral(x, tau, sigma_0, kappa, eta, theta, rho):
    
    # Pseudo-probabilities 
    
    # x: log-prices discounted
    
    integrand = (lambda u: np.real(np.exp((1j*u + 0.5)*x) * \
                                   phi_hest(u - 0.5j, tau, sigma_0, kappa, eta, theta, rho)) / \
                (u**2 + 0.25))
    
    i, err = scipy.integrate.quad_vec(integrand, 0, np.inf)
    
    return i

def analytic_hest(S0, strikes, tau, r, q,  kappa, theta, rho, eta, sigma_0, options_type):
    
    # Pricing of vanilla options under analytic Heston
    
    a = np.log(S0/strikes) + (r-q)*tau 
    i = integral(a, tau, sigma_0, kappa, eta, theta, rho)
    
    out = S0 * np.exp(-q*tau) - strikes * np.exp(-r*tau)/np.pi * i
    
    for k in range(len(out)):
        if options_type[k] == 0:
            out[k] = call_put_parity(out[k], S0, strikes[k], r, q, tau)
    
    return out

###################### COS METHOD Le Floch #########################

def phi_hest_0(u, tau, r, q, sigma_0, kappa, eta, theta, rho):
    
    # Compute the characteristic function for Heston Model with log_asset = 0
    
    # u: argument of the function (where you want to evaluate)
    # r: risk-free-rate
    # q: annual percentage yield
    # tau: time to expiration
    # sigma_0, kappa, eta, theta, rho: Heston parameters 
    
    beta = (kappa - 1j*rho*u*theta)
    d = np.sqrt(beta**2 + (theta**2)*(1j*u+u**2))    
    r_minus = (beta - d)
    g = r_minus/(beta + d)    
    aux = np.exp(-d*tau)
    
    term_1 = sigma_0/(theta**2) *  ((1-aux)/(1-g*aux)) * r_minus
    term_2 = kappa*eta/(theta**2) * (tau*r_minus - 2*np.log((1-g*aux) / (1-g)))                    
    term_3 = 1j*(r-q)*u*tau
                    
    return np.exp(term_1)*np.exp(term_2)*np.exp(term_3)
    
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

def optimal_ab(r, tau, sigma_0, kappa, eta, theta, rho, L = 12):
    # Compute the optimal interval for the truncation
    aux = np.exp(-kappa* tau)
    c1 =  (1 - aux) \
            * (eta - sigma_0)/(2*kappa) - eta * tau / 2

#     c2 = 1/(8 * kappa**3) \
#             * (theta * tau* kappa * np.exp(-kappa * tau) \
#             * (sigma_0 - eta) * (8 * kappa * rho - 4 * theta) \
#             + kappa * rho * theta * (1 - np.exp(-kappa * tau)) \
#             * (16 * eta - 8 * sigma_0) + 2 * eta * kappa * tau \
#             * (-4 * kappa * rho * theta + theta**2 + 4 * kappa**2) \
#             + theta**2 * ((eta - 2 * sigma_0) * np.exp(-2*kappa*tau) \
#             + eta * (6 * np.exp(-kappa*tau) - 7) + 2 * sigma_0) \
#             + 8 * kappa**2 * (sigma_0 - eta) * (1 - np.exp(-kappa*tau)))

    c2 = (sigma_0) / (4*kappa**3) * (4*kappa**2*(1+(rho*theta*tau-1)*aux) \
                                    + kappa*(4*rho*theta*(aux-1)\
                                             -2*theta**2*tau*aux) \
                                    +theta**2*(1-aux*aux)) \
        + eta/(8*kappa**3)*(8*kappa**3*tau - 8*kappa**2*(1+ rho*theta*tau +(rho*theta*tau-1)*aux)\
                            + 2*kappa*((1+2*aux)*theta**2*tau+8*(1-aux)*rho*theta)\
                            + theta**2*(aux*aux+4*aux-5))
    
    
    a = c1 - L * np.abs(c2)**.5
    b = c1 + L * np.abs(c2)**.5
    
    
    return c1 - 12*np.sqrt(np.abs(c2)), c1 + 12*np.sqrt(np.abs(c2))


def precomputed_terms(r, q, tau, sigma_0, kappa, eta, theta, rho, L, N):
    # Auxiliary term precomputed
    
    a,b = optimal_ab(r, tau, sigma_0, kappa, eta, theta, rho, L)
    aux = np.pi/(b-a)
    out = np.zeros(N-1)
    
    for k in range(1,N):
        out[k-1] = np.real(np.exp(-1j*k*a*aux)*\
                         phi_hest_0(k*aux, tau, r, q, sigma_0, kappa, eta, theta, rho))
    
    return out, a, b

def V_k_put(k, a, b, S0, K, z):
    # V_k coefficients for puts   
    
    return 2./(b-a)*(K*psi_k(k, a, z, a, b) - S0*chi_k(k, a, z, a, b))

def cos_method_Heston_LF(precomp_term, a, b, tau, r, q, sigma_0, kappa, eta, theta, rho, S0,\
                         strikes, N, options_type, L=12):
    # Cosine Fourier Expansion for evaluating vanilla options under Heston using LeFloch correction
    # Should be better for deep otm options.
    
    # precomp_term: precomputed terms from the function precomputed_terms
    # a,b: extremes of the interval to approximate
    # tau: time to expiration (annualized) (must be a number)
    # r: risk-free-rate 
    # q: yield
    # sigma_0, kappa, eta, theta, rho: Heston parameters
    # S0: initial spot price
    # strikes: np.array of strikes
    # N: number of terms of the truncated expansion
    # options_type: binary np.array (1 for calls, 0 for puts)
    # L: truncation level

    z = np.log(strikes/S0)
    
    out = 0.5 * np.real(phi_hest_0(0, tau, r, q, sigma_0, kappa, eta, theta, rho))*\
          V_k_put(0, a, b, S0, strikes, z)
    
    for k in range(1,N):
        out = out + precomp_term[k-1]*V_k_put(k, a, b, S0, strikes, z)
    
    D = np.exp(-r*tau)
    out = out*D
    
    for k in range(len(strikes)):
        if options_type[k] == 1:
            out[k] = put_call_parity(out[k], S0, strikes[k], r, q, tau)

    return out

#######################Gradient Heston################################
# Da sistemare
######################################################################

def grad_h(u, tau, sigma_0, kappa, eta, theta, rho, S0, r, q):
    # Gradient of the Heston characteristic function (Le Cui)
    
    h = np.zeros(5, dtype = complex)  
    
    eps = kappa - theta*rho*1j*u
    d = np.sqrt(eps**2 + theta**2*(u**2+1j*u))
    aux1 = np.cosh(d*tau/2)
    aux2 = np.sinh(d*tau/2)
    A1 = (u**2+1j*u)*aux2
    A2 = d*aux1 + eps*aux2
    B = d*np.exp(kappa*tau/2)/A2
    D = np.log(B)
    A = A1/A2
    
    # partial derivatives
    
    dp = -eps*theta*1j*u/d
    A2p = - theta*1j*u*(2+tau*eps)/(2*d)*(eps*aux1 + d*aux2)
    A1p = - (1j*u*(u**2+1j*u)*tau*eps*theta/(2*d))*aux1
    Ap = 1./A2*A1p - A/A2*A2p
    
    Bk = 1j*np.exp(kappa*tau/2)/(theta*u)*(1/A2*dp-d/(A2**2)*A2p) + tau*B/2
    
    dt = (rho/theta - 1/eps)*dp + theta*u**2/d
    A1t = (u**2+1j*u)*tau/2*dt*aux1
    A2t = rho/theta*A2p - (2+tau*eps)/(1j*u*tau*eps)*A1p + theta*tau*A1/2
    At = 1/A2 * A1t - A/A2 * A2t
    
    # h
    
    h[0] = -A
    h[1] = 2*kappa*D/theta**2 - tau*kappa*rho*1j*u/theta
    h[2] = -sigma_0**2 * Ap + 2*kappa*eta/(theta**2*d)*(dp-d/A2*A2p)-\
            tau*kappa*eta*1j*u/theta
    h[3] = sigma_0**2/(theta*1j*u)*Ap + 2*eta/(theta**2)*D \
           + 2*kappa*eta/(theta**2*B)*Bk-tau*rho*eta*1j*u/theta
    h[4] = -sigma_0**2 *At - 4*kappa*eta/(theta**3)*D + 2*kappa*eta/(theta**2*d)*\
           (dt - d/A2*A2t) + tau*rho*eta*kappa*1j*u/(theta**2)
    
    # phi
    phi = np.exp(1j*u*(np.log(S0) + (r-q)*tau))*np.exp(-tau*kappa*eta*rho*1j*u/theta)*np.exp(-\
                      sigma_0**2*A + 2*kappa*eta/(theta**2)*D)
    
    return phi*h

def grad_c(tau, strikes, sigma_0, kappa, eta, theta, rho, S0, r, q, num, a, b):
    # gradient of a European call (or put, it makes no difference) using Gauss-Legendre integration
    
    [u,w] = np.polynomial.legendre.leggauss(num)
    length = strikes.shape[0]
    out = np.zeros((length,5))
    
    for k in range(length):
        for i in range(num):
            out[k,:] += np.real(strikes[k]**(-1j*u[i])/(1j*u[i])*\
                                grad_h(u[i] - 1j, tau, sigma_0, kappa, eta, theta, rho, S0, r, q))*w[i] - \
                        strikes[k]*np.real(strikes[k]**(-1j*u[i])/(1j*u[i])*\
                                grad_h(u[i], tau, sigma_0, kappa, eta, theta, rho, S0, r, q))*w[i]   
    
    return np.exp(-(r-q)*tau)/np.pi*out
    
#######################Simulation Heston################################

def create_totems(base, start, end):
    totems = np.ones(end-start+1)
    index = 0
    for j in range(start, end+1):
        totems[index] = base**j
        index += 1
    return totems

def calc_nu_bar(kappa, eta, theta):
    return 4*kappa*eta/theta**2

def x2_exp_var(nu_bar, kappa, theta, dt):
    aux = kappa*dt/2.
    c1 = np.cosh(aux)/np.sinh(aux)
    c2 = (1./np.sinh(aux))**2
    exp_x2 = nu_bar*theta**2*((-2.+kappa*dt*c1)/(4*kappa**2))
    var_x2 = nu_bar*theta**4*((-8.+2*kappa*dt*c1+\
                              kappa**2*dt**2*c2)/(8*kappa**4))
    return exp_x2, var_x2

def Z_exp_var(nu_bar, exp_x2, var_x2):
    return 4*exp_x2/nu_bar, 4*var_x2/nu_bar

def xi_exp(nu_bar, kappa, theta, dt, totem):
    z = 2*kappa*np.sqrt(totem) / (theta**2*np.sinh(kappa*dt/2.))
    
    iv_pre = iv(nu_bar/2.-1., z)
    
    exp_xi = (z*iv(nu_bar/2.,z))/(2*iv_pre)
    exp_xi2 = exp_xi + (z**2*iv(nu_bar/2.+1,z))/(4.*iv_pre)
    return exp_xi, exp_xi2

def create_caches(base, start, end, kappa, eta, theta, dt):
    totems = create_totems(base, start, end)
    caches_exp = np.zeros(end-start+1)
    caches_var = np.zeros(end-start+1)
    nu_bar = calc_nu_bar(kappa, eta, theta)
    exp_x2, var_x2 = x2_exp_var(nu_bar, kappa, theta, dt)
    exp_Z, var_Z = Z_exp_var(nu_bar, exp_x2, var_x2)
    
    for j in range(end-start+1):
        exp_xi, exp_xi2 = xi_exp(nu_bar, kappa, theta, dt, totems[j])
        caches_exp[j] = exp_x2 + exp_xi*exp_Z
        caches_var[j] = var_x2 + exp_xi*var_Z + \
                        (exp_xi2-exp_xi**2)*exp_Z**2
        
    return totems, caches_exp, caches_var

def x1_exp_var(kappa, theta, dt, vt, vT):
    aux = kappa*dt/2.
    c1 = np.cosh(aux)/np.sinh(aux)
    c2 = (1./np.sinh(aux))**2
    
    exp_x1 = (vt + vT)*(c1/kappa - dt*c2/2)
    var_x1 = (vt + vT)*theta**2*(c1/kappa**3 + dt*c2/(2*kappa**2) \
                                - dt**2*c1*c2/(2*kappa))
    
    return exp_x1, var_x1

def lin_interp(vtvT, totems, caches_exp, caches_var):
    exp_int = np.interp(vtvT, totems, caches_exp)
    var_int = np.interp(vtvT, totems, caches_var)
    return exp_int, var_int

