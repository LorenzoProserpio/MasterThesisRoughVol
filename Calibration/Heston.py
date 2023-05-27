import numpy as np
import QuantLib as ql
import scipy.integrate
from scipy.special import ive

#################### PUT-CALL PARITY #############################

def put_call_parity(put, S0, strike, r, q, tau):
    # Standard put_call_parity
    return put + S0*np.exp(-q*tau) - strike*np.exp(-r*tau)

def call_put_parity(call, S0, strike, r, q, tau):
    return call - S0*np.exp(-q*tau) + strike*np.exp(-r*tau)



##################### Analytic Heston ############################

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
    
    out = S0 * np.exp(-q*t) - np.sqrt(S0*strikes) * np.exp(-(r+q)*t*0.5)/np.pi * i
    out = np.array([out]).flatten()
    
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

def optimal_ab(r, q, tau, sigma_0, kappa, eta, theta, rho, L = 12):
    # Compute the optimal interval for the truncation
    aux = np.exp(-kappa* tau)
    c1 =  (r-q)*tau - sigma_0 * tau / 2

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
    
    a,b = optimal_ab(r, q, tau, sigma_0, kappa, eta, theta, rho, L)
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

#######################Calibration######################################

def setup_model(_yield_ts, _dividend_ts, _spot, 
                init_condition):
    # Setup Heston model object
    
    # _yield_ts: Term Structure for yield (QuantLib object)
    # _dividend_ts: Term Structure for dividend_ts (QuantLib object)
    # init_condition: eta, kappa, theta, rho, sigma_0
    
    eta, kappa, theta, rho, sigma_0 = init_condition
    process = ql.HestonProcess(_yield_ts, _dividend_ts, 
                           ql.QuoteHandle(ql.SimpleQuote(_spot)), 
                           sigma_0, kappa, eta, theta, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model) 
    return model, engine

def setup_helpers(engine, expiration_dates, strikes, 
                  data, ref_date, spot, yield_ts, 
                  dividend_ts, calendar):
    # Helpers for Heston Calibration
    
    # engine: Heston.setup_model output
    # expiration_dates: maturities
    # data: IV market data
    # ref_date: date for the calculation
    # yield_ts: Term Structure for yield (QuantLib object)
    # dividend_ts: Term Structure for dividend_ts (QuantLib object)
    # calendar: type of calendar for calculations
    
    heston_helpers = []
    grid_data = []
    for i, date in enumerate(expiration_dates):
        for j, s in enumerate(strikes):
            t = (date - ref_date )
            p = ql.Period(t, ql.Days)
            vols = data[i][j]
            helper = ql.HestonModelHelper(
                p, calendar, spot, s, 
                ql.QuoteHandle(ql.SimpleQuote(vols)),
                yield_ts, dividend_ts)
            helper.setPricingEngine(engine)
            heston_helpers.append(helper)
            grid_data.append((date, s))
    return heston_helpers, grid_data

def cost_function_generator(model, helpers, norm=False):
    # Define cost function for the calibration (usually Mean Square Error)
    
    def cost_function(params):
        params_ = ql.Array(list(params))
        model.setParams(params_)
        error = [h.calibrationError() for h in helpers]
        if norm:
            return np.sqrt(np.sum(np.abs(error)))
        else:
            return error
    return cost_function
    
#######################Simulation Heston################################

def create_totems(base, start, end):
    # create the grid 
    
    totems = np.ones(end-start+2)
    index = 1
    for j in range(start, end+1):
        totems[index] = base**j
        index += 1
        
    totems[0] = 0
    return totems

def calc_nu_bar(kappa, eta, theta):
    # compute v bar
    return 4*kappa*eta/theta**2

def x2_exp_var(nu_bar, kappa, theta, dt):
    # compute E[X_2] and Var[X_2]
    
    aux = kappa*dt/2.
    c1 = np.cosh(aux)/np.sinh(aux)
    c2 = (1./np.sinh(aux))**2
    exp_x2 = nu_bar*theta**2*((-2.+kappa*dt*c1)/(4*kappa**2))
    var_x2 = nu_bar*theta**4*((-8.+2*kappa*dt*c1+\
                              kappa**2*dt**2*c2)/(8*kappa**4))
    return exp_x2, var_x2

def Z_exp_var(nu_bar, exp_x2, var_x2):
    # compute E[Z] and Var[Z]
    
    return 4*exp_x2/nu_bar, 4*var_x2/nu_bar

def xi_exp(nu_bar, kappa, theta, dt, totem):
    # compute E[\Xi] and E[\Xi^2]
    
    z = 2*kappa*np.sqrt(totem) / (theta**2*np.sinh(kappa*dt/2.))
    iv_pre = ive(nu_bar/2.-1., z)    
    exp_xi = (z*ive(nu_bar/2.,z))/(2*iv_pre)
    exp_xi2 = exp_xi + (z**2*ive(nu_bar/2.+1,z))/(4.*iv_pre)    
    
    return exp_xi, exp_xi2

def create_caches(base, start, end, kappa, eta, theta, dt):
    # precompute the caches for IV*
    
    totems = create_totems(base, start, end)
    caches_exp = np.zeros(end-start+2)
    caches_var = np.zeros(end-start+2)
    nu_bar = calc_nu_bar(kappa, eta, theta)
    exp_x2, var_x2 = x2_exp_var(nu_bar, kappa, theta, dt)
    exp_Z, var_Z = Z_exp_var(nu_bar, exp_x2, var_x2)
    
    for j in range(1,end-start+2):
        exp_xi, exp_xi2 = xi_exp(nu_bar, kappa, theta, dt, totems[j])
        caches_exp[j] = exp_x2 + exp_xi*exp_Z
        caches_var[j] = var_x2 + exp_xi*var_Z + \
                        (exp_xi2-exp_xi**2)*exp_Z**2
        
    caches_exp[0] = exp_x2
    caches_var[0] = var_x2
    return totems, caches_exp, caches_var

def x1_exp_var(kappa, theta, dt, vt, vT):
    # compute E[X_1] and Var[X_1]
    
    aux = kappa*dt/2.
    c1 = np.cosh(aux)/np.sinh(aux)
    c2 = (1./np.sinh(aux))**2
    
    exp_x1 = (vt + vT)*(c1/kappa - dt*c2/2)
    var_x1 = (vt + vT)*theta**2*(c1/kappa**3 + dt*c2/(2*kappa**2) \
                                - dt**2*c1*c2/(2*kappa))
    
    return exp_x1, var_x1

def lin_interp(vtvT, totems, caches_exp, caches_var):
    # compute linear interpolation for value not in caches
    
    exp_int = np.interp(vtvT, totems, caches_exp)
    var_int = np.interp(vtvT, totems, caches_var)
    return exp_int, var_int

def sample_vT(vt, dt, kappa, theta, nu_bar):
    # sample vT from a noncentral chisquare (given vt)
    
    aux = (theta**2*(1-np.exp(-kappa*dt)))/(4*kappa)
    n = np.exp(-kappa*dt)/aux *vt
    return np.random.noncentral_chisquare(nu_bar, n)*aux

def generate_path(S0, T, dt, kappa, eta, theta, rho, r, q, sigma_0, totems, caches_exp, caches_var):
    # This function generate one path for the Heston model using the Gamma Approx algorithm
    
    # T: final time
    # r: risk-free-rate 
    # q: yield
    # sigma_0, kappa, eta, theta, rho: Heston parameters
    # S0: initial spot price
    # dt: temporal step
    # totems, caches_exp, caches_var: precomputed grid, caches for expectation and caches for var
    
    t = dt
    index = 0
    vt = sigma_0
    xt = np.log(S0)
    
    path = np.zeros(int(np.ceil(T/dt))+1)
    path[0] = S0
    
    variance = np.zeros(int(np.ceil(T/dt))+1)
    variance[0] = vt
    
    nu_bar = calc_nu_bar(kappa, eta, theta)
    
    while t < T:
        vT = sample_vT(vt, dt, kappa, theta, nu_bar)
        
        exp_int, var_int = lin_interp(vt*vT, totems, caches_exp, caches_var)
       
        exp_x1, var_x1 = x1_exp_var(kappa, theta, dt, vt, vT)
        exp_int += exp_x1
        var_int += var_x1
        
        gamma_t = var_int/exp_int
        gamma_k = exp_int**2/var_int  
        
        iv_t = np.random.gamma(gamma_k, gamma_t)
        z = np.random.normal()
        
        xt += (r-q)*dt + (- 0.5 + kappa*rho/theta)*iv_t + \
              rho/theta*(vT-vt-kappa*eta*dt) + \
              z*np.sqrt(1-rho**2)*np.sqrt(iv_t)
        
        index += 1
        path[index] = np.exp(xt)
        vt = vT
        variance[index] = vt        
        t += dt
        
    return path[:-1], variance[:-1]
