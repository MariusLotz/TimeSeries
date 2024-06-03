import numpy as np

def Season_paper(alpha1, alpha2, beta1, beta2, N_steps, T_max, n_paths):
    """A Methodology for Validating Diversity in Synthetic Time Series Generation
    f(t) = mt + b + (c1*t+c2)*sin((alpha/2*pi*T_max)*t)
    """
    t_values = np.linspace(0, T_max, N_steps)
    S_t_values = (beta1 * np.sin(alpha1 *t_values)) + (beta2 * np.sin(alpha2 *t_values))
    paths = [S_t_values for _ in range(n_paths)]
    return t_values, paths

if __name__=="__main__":
    print(Season_paper(1, 2, 3, 4, 10, 2, 2)[0]) # t
    print(Season_paper(1, 2, 3, 4, 10, 2, 2)[1]) # f(t)'s