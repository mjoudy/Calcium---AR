from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt

def func_exp(t, A, tau, c):
    return A * np.exp(-t/tau)+c

#def func_beta

def kernel_fit(signal, func='exp', do_plot=False):

    sig_len = len(signal)
    x_axis = np.arange(0, sig_len, 1)

    if func=='exp':
        popt, pcov = curve_fit(func_exp, x_axis, signal)
        A_fit, tau_fit, c = popt

        if do_plot==True:
            fig, ax = plt.subplots()
            ax.scatter(x_axis, signal, marker='.',s=5, label='Data')
            ax.plot(x_axis, func_exp(x_axis, A_fit, tau_fit, c), label='Fit')
        return tau_fit
    


    