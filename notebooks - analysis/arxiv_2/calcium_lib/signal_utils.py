
import numpy as np
import scipy.signal as sig

def smooth_signal(signal, window_length=5, polyorder=3, deriv=0, delta=1.0):
    """
    Smooth signal(s).
    Uses axis=-1 to automatically handle 1D (T,) or 2D (N, T) arrays vectorized.
    """
    # savgol_filter with axis=-1 works for both 1D and 2D arrays directly.
    # No explicit reshape needed, but we ensure array type.
    signal = np.array(signal)
    return sig.savgol_filter(signal, window_length=window_length, deriv=deriv, delta=delta, polyorder=polyorder, axis=-1)


def get_signal_derivative_pair(signal, window_length=5, delta=1.0):
    """
    Get smoothed signal and derivative. Vectorized.
    """
    signal = np.array(signal)
    smooth_cal = smooth_signal(signal, window_length=window_length, deriv=0, delta=delta)
    smooth_deriv = smooth_signal(signal, window_length=window_length, deriv=1, delta=delta)
    return smooth_cal, smooth_deriv


def cut_spikes(spikes, signal, deriv=None, window_length=5):
    """
    Remove spike segments. 
    Because removed segments vary per neuron, 2D outputs are 'ragged' (lists).
    We unify logic by processing everything as a list of neurons.
    """
    spikes = np.array(spikes)
    signal = np.array(signal)
    input_ndim = spikes.ndim
    
    # Force to 2D (N, T) view
    spikes_2d = np.atleast_2d(spikes)
    signal_2d = np.atleast_2d(signal)
    if deriv is not None:
        deriv = np.array(deriv)
        deriv_2d = np.atleast_2d(deriv)
    
    N, T = spikes_2d.shape
    
    cut_signals = []
    cut_derivs = []
    
    for i in range(N):
        s_row = spikes_2d[i]
        sig_row = signal_2d[i]
        
        # Identify spikes (Vectorized check for this row)
        # Check if binary (0/1) or indices
        if np.all((s_row == 0) | (s_row == 1)):
            event_spikes = np.where(s_row)[0]
        else:
            event_spikes = s_row.astype(int)
            
        # Create mask or indices to remove
        if len(event_spikes) > 0:
            # Vectorized index generation
            # arange window + indices
            # shape (num_spikes, win) -> flatten
            # A bit tricky to strictly vectorize construction of indices without loop or broadcast
            # But simple loop over spikes is fine for constructing mask indices
            remove_indices = []
            for spk in event_spikes:
                remove_indices.append(np.arange(spk - window_length, spk + window_length))
            
            remove_indices = np.concatenate(remove_indices)
            remove_indices = np.unique(remove_indices)
            remove_indices = remove_indices[(remove_indices >= 0) & (remove_indices < T)]
            
            sig_cut = np.delete(sig_row, remove_indices)
            if deriv is not None:
                der_cut = np.delete(deriv_2d[i], remove_indices)
        else:
            sig_cut = sig_row
            if deriv is not None:
                der_cut = deriv_2d[i]
                
        cut_signals.append(sig_cut)
        if deriv is not None:
            cut_derivs.append(der_cut)

    # Return format:
    # If input was 1D, return single arrays (legacy behavior convenience).
    # If input was 2D, return lists.
    if input_ndim == 1:
        if deriv is not None:
            return cut_signals[0], cut_derivs[0]
        return cut_signals[0]
    else:
        if deriv is not None:
            return cut_signals, cut_derivs
        return cut_signals


def calculate_cv(spikes):
    """
    Calculate CV. Vectorized for 2D.
    """
    spikes = np.array(spikes)
    input_ndim = spikes.ndim
    spikes_2d = np.atleast_2d(spikes)
    
    # There isn't a simple numpy function for "diff of indices of non-zero elements" per row
    # without a loop, because number of spikes varies per row (ragged result).
    # We must loop over rows (neurons).
    
    cv_list = []
    for i in range(spikes_2d.shape[0]):
        row = spikes_2d[i]
        if np.all((row == 0) | (row == 1)):
            times = np.where(row)[0]
        else:
            times = row
            
        if len(times) < 2:
            cv_list.append(np.nan)
            continue
            
        isis = np.diff(times)
        if np.mean(isis) == 0:
            cv_list.append(0.0)
        else:
            cv_list.append(np.std(isis) / np.mean(isis))
            
    cv_arr = np.array(cv_list)
    
    if input_ndim == 1:
        return cv_arr[0]
    return cv_arr


def calculate_cv2(spikes):
    """
    Calculate CV2.
    """
    spikes = np.array(spikes)
    input_ndim = spikes.ndim
    spikes_2d = np.atleast_2d(spikes)
    
    cv2_list = []
    for i in range(spikes_2d.shape[0]):
        row = spikes_2d[i]
        if np.all((row == 0) | (row == 1)):
            times = np.where(row)[0]
        else:
            times = row
            
        if len(times) < 3:
            cv2_list.append(np.nan)
            continue
            
        isis = np.diff(times)
        # CV2 = mean( 2*|diff_isi| / sum_isi )
        # Vectorized calculation on ISIs
        isis_next = isis[1:]
        isis_curr = isis[:-1]
        
        num = 2 * np.abs(isis_next - isis_curr)
        denom = isis_next + isis_curr
        
        # Avoid div by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            vals = np.where(denom == 0, 0.0, num / denom)
            
        cv2_list.append(np.nanmean(vals))
        
    cv2_arr = np.array(cv2_list)
    
    if input_ndim == 1:
        return cv2_arr[0]
    return cv2_arr
