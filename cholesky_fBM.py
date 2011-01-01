import numpy as np

def cholesky_fBm_mult(N_steps, T_max, Hurst, n_paths):
    gamma = lambda k,H: 0.5*(np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))

    # Precompute Cholesky decomposition
    L = np.zeros((N_steps,N_steps))
    L[0,0] = 1.0
    L[1,0] = gamma(1,Hurst)
    L[1,1] = np.sqrt(1 - (L[1,0]**2))
    
    for i in range(2,N_steps):
        L[i,0] = gamma(i,Hurst)
        for j in range(1, i):
            L[i,j] = (1/L[j,j])*(gamma(i-j,Hurst) - (L[i,0:j] @ L[j,0:j]))
        L[i,i] = np.sqrt(1 - np.sum((L[i,0:i]**2)))
    
    # Generate multiple paths
    paths = np.zeros((n_paths, N_steps))
    for n in range(n_paths):
        V = np.random.standard_normal(size=N_steps)
        X = np.zeros(N_steps)
        X[0] = V[0]
        X[1] = np.sum(L[1,0:2] @ V[0:2])
        for i in range(2,N_steps):
            X[i] = L[i,0:i+1] @ V[0:i+1]
        fBm = np.cumsum(X)*(N_steps**(-Hurst))
        paths[n] = (T_max**Hurst)*fBm
    
    t_values = np.linspace(0, T_max, N_steps)
    return t_values, paths

if __name__=="__main__":
    print(cholesky_fBm_mult(10,10,0.5,2)[0]) # t
    print(cholesky_fBm_mult(10,10,0.5,2)[1]) # f(t)'s