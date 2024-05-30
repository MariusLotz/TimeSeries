import numpy as np
import matplotlib.pyplot as plt

# Plotting all four plots next to each other
def plot_subplots(a, b, c, d, alpha, alpha_0, alpha_1, beta_0, beta_1, N, T, H, x, y, z):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Plot T_t
    plot_T_t(axs[0], a, b, c, d, alpha, N, T, x)

    # Plot S_t
    plot_S_t(axs[1], alpha_0, alpha_1, beta_0, beta_1, N, T, y)

    # Plot fBm_path
    plot_fBm_path(axs[2], N, T, H, z)

    # Plot model_path
    t_values, model_path_values = model_path(a, b, c, d, alpha, alpha_0, alpha_1, beta_0, beta_1, N, T, H, x, y, z)
    axs[3].plot(t_values, model_path_values, label='Model Path')
    axs[3].set_title('Combined Model Path')
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('Value')
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.savefig("testpic")
    #plt.show()

def plot_fBm_path(ax, N, T, H, z):
    t_values, fBm_path = cholesky_fBm(N, T, H, z)
    ax.plot(t_values, fBm_path, label='Fractional Brownian Motion')
    ax.set_title('Fractional Brownian Motion Path (Hurst parameter = {})'.format(H))
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

def plot_S_t(ax, alpha_0, alpha_1, beta_0, beta_1, N, T, y):
    t_values, S_t_values = S_t(alpha_0, alpha_1, beta_0, beta_1, N, T, y)
    ax.plot(t_values, S_t_values, label=r'$S_{t} = \beta_0 \cdot \sin(\alpha_0 t) + \beta_1 \cdot \sin(\alpha_1 t)$')
    ax.set_title('Path of $S_{t}$')
    ax.set_xlabel('t')
    ax.set_ylabel('$S_{t}$')
    ax.legend()
    ax.grid(True)

def plot_T_t(ax, a, b, c, d, alpha, N, T, x):
    t_values, T_t_values = T_t(a, b, c, d, N, T, alpha, x)
    ax.plot(t_values, T_t_values, label=r'$T_{t} = (at + b) \cdot \sin(\alpha \cdot t) + ct + d$')
    ax.set_title('Path of $T_{t}$')
    ax.set_xlabel('t')
    ax.set_ylabel('$T_{t}$')
    ax.legend()
    ax.grid(True)


def model_path(a, b, c, d, alpha, alpha_0, alpha_1, beta_0, beta_1, N, T, H, x, y, z, case=1):
    if case==1:
        t_values, T_t_values = T_t(a, b, c, d, N, T, alpha, x)
        t_values, S_t_values = S_t(alpha_0, alpha_1, beta_0, beta_1, N, T, y)
        t_values, path = cholesky_fBm(N, T, H, z)
        #print(T_t_values + S_t_values + path)
    return t_values, T_t_values + S_t_values + path

def T_t(a, b, c, d, N, T, alpha, x):
    t_values = np.linspace(0, T, N)
    T_t_values = (a * t_values + b) * np.sin(alpha *t_values) + c * t_values + d
    return t_values, x * T_t_values

def S_t(alpha_0, alpha_1, beta_0, beta_1, N, T, y):
    t_values = np.linspace(0, T, N)
    S_t_values = (beta_0 * np.sin(alpha_0 *t_values)) + (beta_1 * np.sin(alpha_1 *t_values))
    return t_values, y * S_t_values

def cholesky_fBm(N, T, H, z):
    '''
    args:
        T:      length of time (in years)
        N:      number of time steps within timeframe
        H:      Hurst parameter
    '''
    gamma = lambda k,H: 0.5*(np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))  
    
    L = np.zeros((N,N))
    X = np.zeros(N)
    V = np.random.standard_normal(size=N)

    L[0,0] = 1.0
    X[0] = V[0]
    
    L[1,0] = gamma(1,H)
    L[1,1] = np.sqrt(1 - (L[1,0]**2))
    X[1] = np.sum(L[1,0:2] @ V[0:2])
    
    for i in range(2,N):
        L[i,0] = gamma(i,H)
        
        for j in range(1, i):         
            L[i,j] = (1/L[j,j])*(gamma(i-j,H) - (L[i,0:j] @ L[j,0:j]))

        L[i,i] = np.sqrt(1 - np.sum((L[i,0:i]**2))) 
        X[i] = L[i,0:i+1] @ V[0:i+1]

    fBm = np.cumsum(X)*(N**(-H))
    path = (T**H)*(fBm)
    t_values = np.linspace(0, T, N)
    return t_values, z * path

if __name__=="__main__":
    #plot_fBm_path(250, 10, 0.6, 1)
    #plot_T_t(1, 5, 7, 1, 0.5, 250, 10, 1)
    #plot_S_t(10, 15, 2, 1.2, 250, 10, 1)
    #plot_model_path(1, 5, 7, 1, 0.5, 10, 15, 2, 1.2, 250, 10, 0.6, 1, 1.5, 4)
    plot_subplots(1, 5, 7, 1, 0.5, 10, 15, 2, 1.2, 250, 10, 0.6, 1, 1.5, 4)





























































