
import numpy as np
from . import signal_utils
from . import outliers

def preprocess_batch(signal, spikes, window_length=5):
    """
    Standard pipeline for processing a batch of calcium traces:
    1. Smooth the signals.
    2. Remove spike times from signal & derivative (Cut).
    3. Fit separate linear models to the remaining points for each neuron to estimate tau.
    4. Reconstruct 'Feed' (estimated input) using the fits.
    
    This function is designed to work on "Chunks" of data (e.g., from Dask).
    
    Args:
        signal (np.ndarray): 2D array (N_neurons x Time).
        spikes (np.ndarray): 2D array (N_neurons x Time).
        window_length (int): Smoothing window.
        
    Returns:
        tuple: (feed_array, slopes_array)
    """
    # 1. Align & Smooth (2D capable)
    if signal.shape[1] != spikes.shape[1]:
        spikes = spikes[:, -signal.shape[1]:]

    smooth_cal, smooth_deriv = signal_utils.get_signal_derivative_pair(signal, window_length, delta=1.0)
    
    # 2. Cut Spikes (2D "Ragged" capable - returns list of arrays)
    # returns list of 1D arrays
    cut_cal_list, cut_deriv_list = signal_utils.cut_spikes(spikes, smooth_cal, smooth_deriv, window_length)
    
    # 3. Fit Slopes (Outliers module handles lists/ragged arrays)
    # Here we use basic polyfit via outliers.fit_linear or explicit calculation
    # Using fit_linear which now handles lists
    slopes = np.zeros(len(cut_cal_list))
    for i in range(len(cut_cal_list)):
        # We need to invert the slope because original logic is slope = -1/tau -> tau = -1/slope
        # But we need to return 'slopes' which in the original code was '1/b_pure_fit' (i.e., -tau).
        # Wait, let's check legacy logic carefully.
        # Original: b_pure_fit = np.polyfit(cal, deriv) [slope of deriv vs cal]
        #           b_pure_fits[row] = 1/b_pure_fit
        #           feed = -b_pure_fits * smooth_cal + smooth_deriv
        # If b_pure_fit = -1/tau, then 1/b_pure_fit = -tau.
        # feed = -(-tau)*C + dC = tau*C + dC.
        #
        # So 'slopes' return variable in this function actually holds '1/slope' (approx -tau).
        
        try:
            b_val = outliers.fit_linear(cut_cal_list[i], cut_deriv_list[i])
            if abs(b_val) > 1e-9:
                slopes[i] = 1.0 / b_val
            else:
                slopes[i] = 0.0
        except:
             slopes[i] = 0.0

    # 4. Calculate Feed (reconstruction) 
    # Feed = - (-tau) * C + dC = tau * C + dC
    # Slopes holds (-tau).
    # Feed = -slopes * C + dC.
    
    # Vectorized calculation since smooth_cal/deriv are 2D arrays, and slopes is 1D array.
    # slopes[:, None] allows broadcasting (N, 1) * (N, T)
    
    feed = -slopes[:, None] * smooth_cal + smooth_deriv
    
    return feed, slopes


def feed_wrapper_batch(signal, spikes, window_length=5):
    """
    Wrapper to get just the feed.
    """
    feed, _ = preprocess_batch(signal, spikes, window_length)
    return feed

def fits_wrapper_batch(signal, spikes, window_length=5):
    """
    Wrapper to get just the fits.
    """
    _, slopes = preprocess_batch(signal, spikes, window_length)
    return slopes
