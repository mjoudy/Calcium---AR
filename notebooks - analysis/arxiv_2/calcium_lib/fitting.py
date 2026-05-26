
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func_exp(t, A, tau, c):
    return A * np.exp(-t/tau) + c

def fit_exponential(signal, do_plot=False):
    """
    Fit an exponential decay function to the signal.
    """
    sig_len = len(signal)
    x_axis = np.arange(0, sig_len, 1)

    popt, pcov = curve_fit(func_exp, x_axis, signal)
    A_fit, tau_fit, c = popt

    if do_plot:
        fig, ax = plt.subplots()
        ax.scatter(x_axis, signal, marker='.', s=5, label='Data')
        ax.plot(x_axis, func_exp(x_axis, A_fit, tau_fit, c), label='Fit')
        ax.legend()
        
    return tau_fit
