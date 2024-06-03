import numpy as np
import matplotlib.pyplot as plt
from season import Season_paper
from trend import Trend_paper
from cholesky_fBM import cholesky_fBm_mult

def time_series_paper(Hurst, m, b, c1, c2, alpha,
                      alpha1, alpha2, beta1, beta2,
                      N_steps, T_max, n_paths, w_random, w_trend, w_season, typ):
    
    t_values, T_t = Trend_paper(m, b, c1, c2, alpha, N_steps, T_max, n_paths)
    t_values, S_t = Season_paper(alpha1, alpha2, beta1, beta2, N_steps, T_max, n_paths)
    t_values, paths = cholesky_fBm_mult(N_steps, T_max, Hurst, n_paths)
    paths = np.array(paths)
    T_t = np.array(T_t)
    S_t = np.array(S_t)
    
    if typ == 1:
        Yts = w_trend * T_t + w_season* S_t + w_random * paths   
    elif typ == 8:
        Yts = w_trend * T_t * w_season* S_t * w_random * paths     
    else:
        raise ValueError("Unsupported 'typ' value. Use 1 or 8.")
        
    return t_values, Yts
        

# Plotting all four plots next to each other
def plot_subplots(Hurst, m, b, c1, c2, alpha,
                      alpha1, alpha2, beta1, beta2,
                      N_steps, T_max, n_paths, w_random, w_trend, w_season, typ=1, i=0):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # Plot T_t
    plot_T_t(axs[0],m, b, c1, c2, alpha, N_steps, T_max, n_paths)

    # Plot S_t
    plot_S_t(axs[1],alpha1, alpha2, beta1, beta2, N_steps, T_max, n_paths)

    # Plot fBm_path
    plot_fBm_path(axs[2], N_steps, T_max, Hurst, n_paths)

    # Plot model_path
    t_values, model_path_values = time_series_paper(Hurst, m, b, c1, c2, alpha,
                      alpha1, alpha2, beta1, beta2,
                      N_steps, T_max, n_paths, w_random, w_trend, w_season, typ=1)
    axs[3].plot(t_values, model_path_values[i], label='Model Path')
    axs[3].set_title('Combined Model Path')
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('Value')
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.savefig("testpic")
    #plt.show()

def plot_fBm_path(ax, N_steps, T_max, Hurst, n_paths, i=0):
    t_values, fBm_path = cholesky_fBm_mult(N_steps, T_max, Hurst, n_paths)
    ax.plot(t_values, fBm_path[i], label='Fractional Brownian Motion')
    ax.set_title('Fractional Brownian Motion Path (Hurst parameter = {})'.format(Hurst))
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)

def plot_S_t(ax, alpha1, alpha2, beta1, beta2, N_steps, T_max, n_paths, i=0):
    t_values, S_t_values = Season_paper(alpha1, alpha2, beta1, beta2, N_steps, T_max, n_paths)
    ax.plot(t_values, S_t_values[i], label=r'$S_{t} = \beta_0 \cdot \sin(\alpha_0 t) + \beta_1 \cdot \sin(\alpha_1 t)$')
    ax.set_title('Path of $S_{t}$')
    ax.set_xlabel('t')
    ax.set_ylabel('$S_{t}$')
    ax.legend()
    ax.grid(True)

def plot_T_t(ax, m, b, c1, c2, alpha, N_steps, T_max, n_paths, i=0):
    t_values, T_t_values = Trend_paper(m, b, c1, c2, alpha, N_steps, T_max, n_paths)
    ax.plot(t_values, T_t_values[i], label=r'$T_{t} = (at + b) \cdot \sin(\alpha \cdot t) + ct + d$')
    ax.set_title('Path of $T_{t}$')
    ax.set_xlabel('t')
    ax.set_ylabel('$T_{t}$')
    ax.legend()
    ax.grid(True)


if __name__ == "__main__":
    # Define parameters
    Hurst = 0.5
    m = 1
    b = 1
    c1 = 1
    c2 = 1
    alpha = 1
    alpha1 = 1
    alpha2 = 1
    beta1 = 1
    beta2 = 1
    N_steps = 100
    T_max = 10
    n_paths = 2
    w_random = 0.5
    w_trend = 0.3
    w_season = 0.2
    typ = 1
    

    # Generate time series
    t_values, Yts = time_series_paper(Hurst, m, b, c1, c2, alpha, alpha1, alpha2, beta1, beta2,
                                      N_steps, T_max, n_paths, w_random, w_trend, w_season, typ)
    
    # plot
    plot_subplots(Hurst, m, b, c1, c2, alpha,
                      alpha1, alpha2, beta1, beta2,
                      N_steps, T_max, n_paths, w_random, w_trend, w_season, typ)
    
    # Print results
    print("Time Values:", t_values)
    print("Time Series:", Yts)




























































