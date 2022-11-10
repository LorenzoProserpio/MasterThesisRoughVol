import numpy as np
from scipy.fft import fft
from scipy.interpolate import CubicSpline

def Heston_characteristic(r, q, T, sigma_0, kappa, eta, theta, rho, u):
    
    # Compute the characteristic function for Heston Model
    
    # r: risk-free-rate
    # q: annual percentage yield
    # T: time to expiration
    # sigma_0, kappa, eta, theta, rho: Heston parameters
    
    
    aux = (rho*theta*u*1j - kappa)
    d = np.sqrt(aux**2. - theta**2.*(-1j*u - u**2.))
    
    g = (-aux - d)/(-aux + d)
    
    p1 = 1j*u*(r-q)*T
    aux_2 = np.exp(-d*T)
    p2 = eta*kappa*theta**(-2.)*((-aux -d)*T - 2.*np.log((1 - g*aux_2)/(1 - g)))
    p3 = sigma_0**2.*theta**(-2.)*(-aux - d)*(1 - aux_2)/(1 - g*aux_2)
    
    return np.exp(p1)*np.exp(p2)*np.exp(p3)

def Heston_FFT(K, r, q, T, sigma_0, kappa, eta, theta, rho, Option_type = 1, integral_rule = 0):
    
    # Compute call and put prices given strikes with Heston model, using FFT method proposed by Carr-Madan
    
    # K: np.array of strikes
    # r: risk-free-rate
    # q: annual percentage yield
    # T: time to expiration
    # sigma_0, kappa, eta, theta, rho: Heston parameters
    # Option_type: 1 for calls, 0 for puts
    # integral_rule: 1 for Cavalieri-Simpson, 0 for rectangular
    
    N = 4096
    alpha = 1.5
    eta = 0.25
    lambda_ = 2.*np.pi/(N*eta)
    b = lambda_*N/2.
    
    k = np.linspace(-b, b-lambda_, int(np.floor(2*b/lambda_)))
    
    v = np.linspace(0, (N-1)*eta, N)
    u = v - (alpha + 1)*1j
    rho = np.exp(-r*T)*Heston_characteristic(r, q, T, sigma_0, kappa, eta, theta, rho, u)
    rho = rho/(alpha**2. + alpha - v**2. + 1j*(2*alpha +1)*v)
    
    if integral_rule:
        simp = np.concatenate(([1./3.],(3+(-1)**np.arange(2,4096+1))/3))
        a = np.real(fft(np.exp(1j*v*b)*rho*eta*simp))
    else:
        a = np.real(fft(np.exp(1j*v*b)*rho*eta))
    
    call_price = (1./np.pi)*np.exp(-alpha*k)*a
    
    try:
        inter_K = CubicSpline(np.exp(k), call_price)
        out = inter_K(K)
    
    except:
        out = 1e8*np.ones(K.shape[0])
        
    if Option_type == 0:
        out = out + K*np.exp(-r*T) - np.exp(-q*T)
        
    return out