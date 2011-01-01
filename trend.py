import numpy as np

def Trend_paper(m, b, c1, c2, alpha, N_steps, T_max, n_paths):
    """A Methodology for Validating Diversity in Synthetic Time Series Generation
    f(t) = mt + b + (c1*t+c2)*sin((alpha/2*pi*T_max)*t)
    """
    
    t_values = np.array(np.linspace(0, T_max, N_steps))
    T_t_values = np.array(m * t_values + b + (c1 * t_values + c2) * np.sin((alpha / (2 * np.pi * T_max)) * t_values))
    paths = [T_t_values for _ in range(n_paths)]
    return t_values, paths


if __name__=="__main__":
    print(Trend_paper(1, 1, 1, 1, 1, 10, 10, 2)[0]) # t
    print(Trend_paper(1, 1, 1, 1, 1, 10, 10, 2)[1]) # f(t)'s
