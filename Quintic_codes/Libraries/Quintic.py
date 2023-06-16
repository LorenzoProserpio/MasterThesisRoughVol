import numpy as np
import variance_curve as vc
import ImpliedDrift as iD
import scipy
import BlackScholes as bs

from scipy.integrate import quad

def horner_vector(poly, n, x):
    #Initialize result
    result = poly[0].reshape(-1,1)
    for i in range(1,n):
        result = result*x + poly[i].reshape(-1,1)
    return result
 

 
def gauss_dens(mu,sigma,x):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/(2*sigma**2))
    
    

def vix_futures(H, eps, T, a_k_part, k, r, q, n_steps, index = 0):

    a2,a4 = (0,0)
    a0,a1,a3,a5 = a_k_part
    a_k = np.array([a0, a1, a2, a3, a4, a5])
    
    kappa_tild = (0.5-H)/eps
    eta_tild = eps**(H-0.5)

    delt = 30/365
    T_delta = T + delt

    std_X = eta_tild*np.sqrt(1/(2*kappa_tild)*(1-np.exp(-2*kappa_tild*T)))
    dt = delt/(n_steps)
    tt = np.linspace(T, T_delta, n_steps+1)
    
    FV_curve_all_vix = vc.variance_curve(tt, index)
    
    exp_det = np.exp(-kappa_tild*(tt-T))
    cauchy_product = np.convolve(a_k,a_k)
    
    std_Gs_T = eta_tild*np.sqrt(1/(2*kappa_tild)*(1-np.exp(-2*kappa_tild*(tt-T))))
    std_X_t = eta_tild*np.sqrt(1/(2*kappa_tild)*(1-np.exp(-2*kappa_tild*tt)))
    std_X_T = std_X
    
    n = len(a_k)
    
    normal_var = np.sum(cauchy_product[np.arange(0,2*n,2)].reshape(-1,1)*std_X_t**(np.arange(0,2*n,2).reshape(-1,1))*\
    scipy.special.factorial2(np.arange(0,2*n,2).reshape(-1,1)-1),axis=0) #g(u)
    
    beta = []
    for i in range(0,2*n-1):
        k_array = np.arange(i,2*n-1)
        beta_temp = ((std_Gs_T**((k_array-i).reshape(-1,1))*((k_array-i-1)%2).reshape(-1,1)*\
            scipy.special.factorial2(k_array-i-1).reshape(-1,1)*\
            (scipy.special.comb(k_array,i)).reshape(-1,1))*\
            exp_det**(i))*cauchy_product[k_array].reshape(-1,1)
        beta.append(np.sum(beta_temp,axis=0))

    beta = np.array(beta)*FV_curve_all_vix/normal_var
    beta = (np.sum((beta[:,:-1]+beta[:,1:])/2,axis=1))*dt
    
    sigma = np.sqrt(eps**(2*H)/(1-2*H)*(1-np.exp((2*H-1)*T/eps)))
    
    f = lambda x: np.sqrt(horner_vector(beta[::-1], len(beta), x)/delt)*100 * gauss_dens(0, sigma, x)

    Ft, err = quad(f, -np.inf, np.inf)
    
    return Ft * np.exp((r-q)*T)



def vix_iv(H, eps, T, a_k_part, K, r, q, n_steps, index = 0):

    a2,a4 = (0,0)
    a0,a1,a3,a5 = a_k_part
    a_k = np.array([a0, a1, a2, a3, a4, a5])
    
    kappa_tild = (0.5-H)/eps
    eta_tild = eps**(H-0.5)

    delt = 30/365
    T_delta = T + delt

    std_X = eta_tild*np.sqrt(1/(2*kappa_tild)*(1-np.exp(-2*kappa_tild*T)))
    dt = delt/(n_steps)
    tt = np.linspace(T, T_delta, n_steps+1)
    
    FV_curve_all_vix = vc.variance_curve(tt, index)
    
    exp_det = np.exp(-kappa_tild*(tt-T))
    cauchy_product = np.convolve(a_k,a_k)
    
    std_Gs_T = eta_tild*np.sqrt(1/(2*kappa_tild)*(1-np.exp(-2*kappa_tild*(tt-T))))
    std_X_t = eta_tild*np.sqrt(1/(2*kappa_tild)*(1-np.exp(-2*kappa_tild*tt)))
    std_X_T = std_X
    
    n = len(a_k)
    
    normal_var = np.sum(cauchy_product[np.arange(0,2*n,2)].reshape(-1,1)*std_X_t**(np.arange(0,2*n,2).reshape(-1,1))*\
    scipy.special.factorial2(np.arange(0,2*n,2).reshape(-1,1)-1),axis=0) #g(u)
    
    beta = []
    for i in range(0,2*n-1):
        k_array = np.arange(i,2*n-1)
        beta_temp = ((std_Gs_T**((k_array-i).reshape(-1,1))*((k_array-i-1)%2).reshape(-1,1)*\
            scipy.special.factorial2(k_array-i-1).reshape(-1,1)*\
            (scipy.special.comb(k_array,i)).reshape(-1,1))*\
            exp_det**(i))*cauchy_product[k_array].reshape(-1,1)
        beta.append(np.sum(beta_temp,axis=0))

    beta = np.array(beta)*FV_curve_all_vix/normal_var
    beta = (np.sum((beta[:,:-1]+beta[:,1:])/2,axis=1))*dt
    
    sigma = np.sqrt(eps**(2*H)/(1-2*H)*(1-np.exp((2*H-1)*T/eps)))
    
    N = len(K); P = np.zeros(N);
    
    for i in range(N):
    
        f = lambda x: np.maximum(np.sqrt(horner_vector(beta[::-1], len(beta), x)/delt)*100 - K[i], 0) * gauss_dens(0, sigma, x)
        P[i], err = quad(f, -np.inf, np.inf)
    
    return P * np.exp((r-q)*T)



def dW(n_steps,N_sims):
    w = np.random.normal(0, 1, (n_steps, N_sims))
    #Antithetic variates
    w = np.concatenate((w, -w), axis = 1)
    return w



def local_reduction(rho,H,eps,T,a_k_part,S0,strike_array,n_steps,N_sims,w1,r,q, index = 0):
    
    eta_tild = eps**(H-0.5)
    kappa_tild = (0.5-H)/eps
    
    a_0,a_1,a_3,a_5 = a_k_part
    a_k = np.array([a_0,a_1,0,a_3,0,a_5])

    dt = T/n_steps
    tt = np.linspace(0., T, n_steps + 1)

    exp1 = np.exp(kappa_tild*tt)
    exp2 = np.exp(2*kappa_tild*tt)

    diff_exp2 = np.concatenate((np.array([0.]),np.diff(exp2)))
    std_vec = np.sqrt(diff_exp2/(2*kappa_tild))[:,np.newaxis] #to be broadcasted columnwise 
    exp1 = exp1[:,np.newaxis] 
    X = (1/exp1)*(eta_tild*np.cumsum(std_vec*w1, axis = 0)) 
    Xt = np.array(X[:-1])
    del X
    
    tt = tt[:-1]
    std_X_t = np.sqrt(eta_tild**2/(2*kappa_tild)*(1-np.exp(-2*kappa_tild*tt)))
    n = len(a_k)
    
    cauchy_product = np.convolve(a_k,a_k)
    normal_var = np.sum(cauchy_product[np.arange(0,2*n,2)].reshape(-1,1)*std_X_t**(np.arange(0,2*n,2).reshape(-1,1))*\
        scipy.special.factorial2(np.arange(0,2*n,2).reshape(-1,1)-1),axis=0)
    
    f_func = horner_vector(a_k[::-1], len(a_k), Xt)
        
    del Xt
    
    fv_curve = vc.variance_curve(tt, index).reshape(-1,1)

    volatility = f_func/np.sqrt(normal_var.reshape(-1,1))
    del f_func
    volatility = np.sqrt(fv_curve)*volatility
    
    logS1 = np.log(S0)
    for i in range(w1.shape[0]-1):
        logS1 = logS1 - 0.5*dt*(volatility[i]*rho)**2 + np.sqrt(dt)*rho*volatility[i]*w1[i+1] + rho**2*(r-q)*dt
    del w1
    ST1 = np.exp(logS1)
    del logS1 

    int_var = np.sum(volatility[:-1,]**2*dt,axis=0)
    Q = np.max(int_var)+1e-9
    del volatility
    X = (bs.BSCall(ST1, strike_array.reshape(-1,1), T, r, q, np.sqrt((1-rho**2)*int_var/T))).T
    Y = (bs.BSCall(ST1, strike_array.reshape(-1,1), T, r, q, np.sqrt(rho**2*(Q-int_var)/T))).T
    del int_var
    eY = (bs.BSCall(S0, strike_array.reshape(-1,1), T, r, q, np.sqrt(rho**2*Q/T))).T
    
    c = []
    for i in range(strike_array.shape[0]):
        cova = np.cov(X[:,i]+10,Y[:,i]+10)[0,1]
        varg = np.cov(X[:,i]+10,Y[:,i]+10)[1,1]
        if (cova or varg)<1e-8:
            temp = 1e-40
        else:
            temp = np.nan_to_num(cova/varg,1e-40)
        temp = np.minimum(temp,2)
        c.append(temp)
    c = np.array(c)
    
    call_mc_cv1 = X-c*(Y-eY)
    del X
    del Y
    del eY
    
    return np.average(call_mc_cv1,axis=0)



def global_reduction(rho,H,eps,T,a_k_part,S0,strike_array,n_steps,N_sims,w1,steps,maturities, index = 0):
    
    eta_tild = eps**(H-0.5)
    kappa_tild = (0.5-H)/eps
    
    a_0,a_1,a_3,a_5 = a_k_part
    a_k = np.array([a_0,a_1,0,a_3,0,a_5])

    dt = T/n_steps
    tt = np.linspace(0., T, n_steps + 1)
    
    r = iD.r(tt, index)
    q = iD.q(tt, index)

    exp1 = np.exp(kappa_tild*tt)
    exp2 = np.exp(2*kappa_tild*tt)

    diff_exp2 = np.concatenate((np.array([0.]),np.diff(exp2)))
    std_vec = np.sqrt(diff_exp2/(2*kappa_tild))[:,np.newaxis] #to be broadcasted columnwise 
    exp1 = exp1[:,np.newaxis] 
    X = (1/exp1)*(eta_tild*np.cumsum(std_vec*w1, axis = 0)) 
    Xt = np.array(X[:-1])
    del X
    
    tt = tt[:-1]
    std_X_t = np.sqrt(eta_tild**2/(2*kappa_tild)*(1-np.exp(-2*kappa_tild*tt)))
    n = len(a_k)
    
    cauchy_product = np.convolve(a_k,a_k)
    normal_var = np.sum(cauchy_product[np.arange(0,2*n,2)].reshape(-1,1)*std_X_t**(np.arange(0,2*n,2).reshape(-1,1))*\
        scipy.special.factorial2(np.arange(0,2*n,2).reshape(-1,1)-1),axis=0)
    
    f_func = horner_vector(a_k[::-1], len(a_k), Xt)
        
    del Xt
    
    fv_curve = vc.variance_curve(tt, index).reshape(-1,1)

    volatility = f_func/np.sqrt(normal_var.reshape(-1,1))
    del f_func
    volatility = np.sqrt(fv_curve)*volatility
    
    ST1 = list()
    logS1 = np.log(S0)
    for i in range(w1.shape[0]-1):
        logS1 = logS1 - 0.5*dt*(volatility[i]*rho)**2 + np.sqrt(dt)*rho*volatility[i]*w1[i+1] + rho**2*(r[i]-q[i])*dt
        if i in steps:
            ST1.append(np.exp(logS1))
    del w1
    ST1.append(np.exp(logS1))
    ST1 = np.array(ST1)
    del logS1 

    int_var = np.sum(volatility[:-1,]**2*dt,axis=0)
    Q = np.max(int_var)+1e-9
    del volatility
    
    P = list()
    
    for i in range(len(steps)):
        T_aux = maturities[i]
        r = iD.r(T_aux, index); q = iD.q(T_aux, index)
        
        X = (bs.BSCall(ST1[i], strike_array.reshape(-1,1), T_aux, r, q, np.sqrt((1-rho**2)*int_var/T))).T
        Y = (bs.BSCall(ST1[i], strike_array.reshape(-1,1), T_aux, r, q, np.sqrt(rho**2*(Q-int_var)/T))).T
        eY = (bs.BSCall(S0, strike_array.reshape(-1,1), T_aux, r, q, np.sqrt(rho**2*Q/T))).T

        c = []
        for i in range(strike_array.shape[0]):
            cova = np.cov(X[:,i]+10,Y[:,i]+10)[0,1]
            varg = np.cov(X[:,i]+10,Y[:,i]+10)[1,1]
            if (cova or varg)<1e-8:
                temp = 1e-40
            else:
                temp = np.nan_to_num(cova/varg,1e-40)
            temp = np.minimum(temp,2)
            c.append(temp)
        c = np.array(c)

        call_mc_cv1 = X-c*(Y-eY)
        P.append(np.average(call_mc_cv1,axis=0))
    
    return np.array(P) 