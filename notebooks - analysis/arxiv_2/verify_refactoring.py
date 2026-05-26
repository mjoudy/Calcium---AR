
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure we can import the local package
sys.path.append(os.getcwd())

try:
    import calcium_lib.simulation as sim
    import calcium_lib.signal_utils as utils
    import calcium_lib.outliers as out
    import calcium_lib.connectivity as conn
    import calcium_lib.dask_utils as dask_utils
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)


# --- Configuration Parameters ---
DATA_PATH = "data/spikes-10e4-ms_5.npy"
WARMUP_TIME = 1000
TAU = 100.0
DT = 1.0
NUM_NEURONS_TEST = 5  # Number of neurons for 2D/batch testing
ROBUST_METHOD = 'ransac'
PLOT_PHASE_SPACE = True # Enable plotting for robust estimation verification

def verify():
    print("Loading data...")
    try:
        spikes = np.load(DATA_PATH)
        print(f"Data loaded, shape: {spikes.shape}")
        
        # Apply Warmup Slicing Logic manually for test data if needed 
        # (simulation function handles it internally if passed, but here we prep input data)
        if WARMUP_TIME < spikes.shape[1]:
            spikes = spikes[:, WARMUP_TIME:] 
        
    except FileNotFoundError:
        print("Data file not found, generating dummy data.")
        spikes = np.random.binomial(1, 0.01, (10, 1000))

    # --- Test Simulation (Unified) ---
    print("\nTesting Simulation (Unified)...")
    # 1D
    trace_1d = sim.simulate_calcium_signals(spikes[0], tau=TAU, dt=DT)
    print(f"1D trace simulated: {trace_1d.shape}")
    # 2D
    trace_2d = sim.simulate_calcium_signals(spikes[:NUM_NEURONS_TEST], tau=TAU, dt=DT)
    print(f"2D traces simulated: {trace_2d.shape}")

    # --- Test Signal Utils (Unified) ---
    print("\nTesting Signal Utils (Unified)...")
    # 1D
    s1, d1 = utils.get_signal_derivative_pair(trace_1d)
    print(f"1D Smooth: {s1.shape}")
    # 2D
    s2, d2 = utils.get_signal_derivative_pair(trace_2d)
    print(f"2D Smooth: {s2.shape}")
    
    # Cut Spikes (Unified)
    # 1D
    c1_s, c1_d = utils.cut_spikes(spikes[0], s1, d1)
    print(f"1D Cut Spikes: {c1_s.shape}")
    # 2D (Should return list)
    c2 = utils.cut_spikes(spikes[:NUM_NEURONS_TEST], s2)
    print(f"2D Cut Spikes (List len): {len(c2)}, First item shape: {c2[0].shape}")

    # --- Test CV / CV2 ---
    print("\nTesting CV / CV2...")
    cv_1d = utils.calculate_cv(spikes[0])
    cv2_1d = utils.calculate_cv2(spikes[0])
    print(f"1D CV: {cv_1d:.4f}, CV2: {cv2_1d:.4f}")
    
    cv_2d = utils.calculate_cv(spikes[:NUM_NEURONS_TEST])
    cv2_2d = utils.calculate_cv2(spikes[:NUM_NEURONS_TEST])
    print(f"2D CVs: {cv_2d}")
    print(f"2D CV2s: {cv2_2d}")

    # --- Test Robust Tau Estimation (Phase Space) ---
    print("\nTesting Robust Tau Estimation (No Spike Cutting)...")
    
    # Method 3: RANSAC
    print("--- RANSAC ---")
    tau_ransac_1d = conn.estimate_tau_robust(trace_1d, method='ransac', do_plot=PLOT_PHASE_SPACE)
    print(f"1D Robust Tau (RANSAC): {tau_ransac_1d:.2f} ms")
    tau_ransac_2d = conn.estimate_tau_robust(trace_2d, method='ransac', do_plot=PLOT_PHASE_SPACE) 
    print(f"2D Robust Taus (RANSAC): {tau_ransac_2d}")

    # Method 2: DBSCAN
    print("\n--- DBSCAN ---")
    tau_dbscan_1d = conn.estimate_tau_robust(trace_1d, method='dbscan', do_plot=PLOT_PHASE_SPACE)
    print(f"1D Robust Tau (DBSCAN): {tau_dbscan_1d:.2f} ms")
    # For 2D batch test of DBSCAN
    tau_dbscan_2d = conn.estimate_tau_robust(trace_2d, method='dbscan', do_plot=PLOT_PHASE_SPACE)
    print(f"2D Robust Taus (DBSCAN): {tau_dbscan_2d}")

    # --- Test Outliers (Unified) ---
    print("\nTesting Outliers (Unified)...")
    # 1D
    slope_1d = out.remove_outliers_iqr(c1_s, c1_d)
    print(f"1D IQR Slope: {slope_1d}")
    
    # 2D (Ragged input from c2)
    c2_s, c2_d = utils.cut_spikes(spikes[:NUM_NEURONS_TEST], s2, d2)
    slopes_2d = out.remove_outliers_iqr(c2_s, c2_d)
    print(f"2D IQR Slopes (Array shape): {slopes_2d.shape}, Values: {slopes_2d}")

    # --- Tau Estimation ---
    print("\nEstimating Tau (-1/slope)...")
    if slope_1d != 0:
        print(f"1D Estimated Tau: {-1/slope_1d:.2f} ms")
    
    # Avoid division by zero for 2D
    taus_2d = np.zeros_like(slopes_2d)
    nonzero_mask = np.abs(slopes_2d) > 1e-9
    taus_2d[nonzero_mask] = -1 / slopes_2d[nonzero_mask]
    print(f"2D Estimated Taus: {taus_2d}")
    
    # --- Test Dask Utils (Pipeline) ---
    print("\nTesting Dask Utils (Pipeline)...")
    feed, fits = dask_utils.preprocess_batch(trace_2d, spikes[:NUM_NEURONS_TEST])
    print(f"Pipeline Feed: {feed.shape}, Fits: {fits.shape}")
    
    print("\nVERIFICATION SUCCESSFUL: All 2D/Unified modules executed.")

if __name__ == "__main__":
    
    # You can override parameters via CLI args if desired, but here we use the constants
    print(f"Running verification with: Tau={TAU}, Neurons={NUM_NEURONS_TEST}, Data={DATA_PATH}")
    verify()
