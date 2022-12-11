import numpy as np
import scipy.integrate

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

###################### COS METHOD ################################

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
            np.cos(aux_1*(c-a))*aux_3 + \
            aux_1*np.sin(aux_1*(d-a))*aux_2 - \
            aux_1*np.sin(aux_1*(c-a))*aux_3) / (1+aux_1**2)

def psi_k(k, c, d, a, b):    
    # Auxiliary function for U_k
    
    if k == 0:
        return d - c
    
    aux = k*np.pi/(b-a)
    return (np.sin(aux*(d-a)) - np.sin(aux*(c-a))) / aux
    
def U_k_put(k, a, b):
    # Auxiliary for cos_method
    
    return 2./(b-a) * (psi_k(k, a, 0, a, b) - chi_k(k, a, 0, a, b))

def optimal_ab(r, tau, sigma_0, kappa, eta, theta, rho, L = 12):
    # Compute the optimal interval for the truncation
    
    c1 = r * tau \
            + (1 - np.exp(-kappa* tau)) \
            * (eta - sigma_0)/2/kappa - eta * tau / 2

    c2 = 1/(8 * kappa**3) \
            * (theta * tau* kappa * np.exp(-kappa * tau) \
            * (sigma_0 - eta) * (8 * kappa * rho - 4 * theta) \
            + kappa * rho * theta * (1 - np.exp(-kappa * tau)) \
            * (16 * eta - 8 * sigma_0) + 2 * eta * kappa * tau \
            * (-4 * kappa * rho * theta + theta**2 + 4 * kappa**2) \
            + theta**2 * ((eta - 2 * sigma_0) * np.exp(-2*kappa*tau) \
            + eta * (6 * np.exp(-kappa*tau) - 7) + 2 * sigma_0) \
            + 8 * kappa**2 * (sigma_0 - eta) * (1 - np.exp(-kappa*tau)))

    a = c1 - L * np.abs(c2)**.5
    b = c1 + L * np.abs(c2)**.5
    
    
    return c1 - 12*np.sqrt(np.abs(c2)), c1 + 12*np.sqrt(np.abs(c2))

def cos_method_Heston(tau, r, q, sigma_0, kappa, eta, theta, rho, S0, strikes, N, options_type, L=12):
    # Cosine Fourier Expansion for evaluating vanilla options under Heston
    
    # tau: time to expiration (annualized) (must be a number)
    # r: risk-free-rate 
    # q: yield
    # sigma_0, kappa, eta, theta, rho: Heston parameters
    # S0: initial spot price
    # strikes: np.array of strikes
    # a,b: extremes of the interval to approximate
    # N: number of terms of the truncated expansion
    # options_type: binary np.array (1 for calls, 0 for puts)
    # L: truncation level
    
    a, b = optimal_ab(r, tau, sigma_0, kappa, eta, theta, rho, L)
      
    x = np.log(S0/strikes) 
    aux = np.pi/(b-a)
    
    # first term
    out = 0.5 * phi_hest_0(0, tau, r, q, sigma_0, kappa, eta, theta, rho) \
        * U_k_put(0,a,b)
    
    # other terms
    for k in range(1,N):
        out = out + phi_hest_0(k*aux, tau, r, q, sigma_0, kappa, eta, theta, rho) \
        * U_k_put(k,a,b) * np.exp(1j*k*aux*(x - a))
        
    out = out.real
    D = np.exp(-r*tau)
    out = strikes*out*D
    
    for k in range(len(strikes)):
        if options_type[k] == 1:
            out[k] = put_call_parity(out[k], S0, strikes[k], r, q, tau)
        
    return out



###################### COS METHOD Le Floch #########################

def precomputed_terms(r, q, tau, sigma_0, kappa, eta, theta, rho, L, N):
    # Auxiliary term precomputed
    
    a,b = optimal_ab(r, tau, sigma_0, kappa, eta, theta, rho, L)
    aux = np.pi/(b-a)
    out = np.zeros(N-1)
    
    for k in range(1,N):
        out[k-1] = np.real(np.exp(-1j*k*a*aux)*\
                         phi_hest_0(k*aux, tau, r, q, sigma_0, kappa, eta, theta, rho))
    
    return out, a, b

def V_k_put(k, a, b, F, K, z):
    # V_k coefficients for puts   
    
    return 2./(b-a)*(K*psi_k(k, a, z, a, b) - F*chi_k(k, a, z, a, b))

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