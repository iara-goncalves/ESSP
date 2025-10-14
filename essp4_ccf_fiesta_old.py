import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from astropy.io import fits
import pickle
from datetime import datetime
from scipy import signal

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_offsets(offset_file):
    """Load instrument offsets from CSV file"""
    if not os.path.exists(offset_file):
        print(f"Warning: Offset file {offset_file} not found. Using zero offsets.")
        return {}
    
    offsets = {}
    try:
        with open(offset_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                    
                parts = line.split(',')
                if len(parts) >= 2:
                    instrument = parts[0].strip().lower()
                    offset_value = float(parts[1].strip())
                    offsets[instrument] = offset_value
                    
        print(f"Loaded offsets: {offsets}")
        
    except Exception as e:
        print(f"Error reading offset file: {e}")
        return {}
    
    return offsets


def get_available_datasets(essp_dir):
    """Get list of available datasets"""
    datasets = []
    for item in os.listdir(essp_dir):
        if item.startswith('DS') and os.path.isdir(os.path.join(essp_dir, item)):
            try:
                ds_num = int(item[2:])  # Extract number from DS##
                datasets.append(ds_num)
            except ValueError:
                continue
    return sorted(datasets)


def analyze_ccfs_with_fiesta(essp_dir, dset_num, k_max=5, save_results=False, output_dir=None):
    """Apply FIESTA analysis to CCFs from a dataset using actual ESSP CCF errors"""
    try:
        from FIESTA_II import FIESTA
    except ImportError:
        print("Error: FIESTA_II module not found. Please install FIESTA first.")
        return None
    
    print(f"\n=== FIESTA Analysis for DS{dset_num} ===")
    
    instruments = ['harpsn', 'harps', 'expres', 'neid']
    ccf_dir = os.path.join(essp_dir, f'DS{dset_num}', 'CCFs')
    
    if not os.path.exists(ccf_dir):
        print(f"Warning: {ccf_dir} not found")
        return None
    
    results = {}
    
    for inst in instruments:
        print(f"\nProcessing {inst.upper()}...")
        
        # Get all files for this instrument
        files = sorted(glob(os.path.join(ccf_dir, f'DS{dset_num}.*_ccfs_{inst}.fits')))
        if not files:
            print(f"No files found for {inst}")
            continue
        
        # Process files to extract CCF data with proper errors
        ccf_data = []
        ccf_errors = []
        v_grids = []
        emjd_times = []
        file_names = []
        
        for file in files:
            try:
                hdus = fits.open(file)
                
                # Extract data from FITS file
                v_grid = hdus['V_GRID'].data.copy()  # Velocity grid [km/s]
                ccf = hdus['CCF'].data.copy()        # Global CCF
                e_ccf = hdus['E_CCF'].data.copy()    # Global CCF errors
                
                # Extract eMJD from FITS header TIME field
                try:
                    if 'TIME' in hdus[0].header:
                        emjd = hdus[0].header['TIME']  # Extended Modified Julian Date
                        emjd_times.append(emjd)
                    else:
                        print(f"Warning: No TIME in header for {os.path.basename(file)}, using index")
                        emjd_times.append(len(emjd_times))  # Use index as fallback
                        
                except Exception as e:
                    print(f"Warning: Could not extract eMJD from {os.path.basename(file)}: {e}")
                    emjd_times.append(len(emjd_times))  # Use index as fallback
                
                hdus.close()
                
                # Check for valid data
                if np.sum(np.isfinite(ccf)) == 0:
                    print(f"Skipping {os.path.basename(file)}: no valid CCF data")
                    continue
                
                # Check for valid errors
                if np.sum(np.isfinite(e_ccf)) == 0:
                    print(f"Warning: {os.path.basename(file)}: no valid error data, estimating errors")
                    # Fallback error estimation if E_CCF is invalid
                    ccf_max = np.nanmax(ccf)
                    ccf_depth = ccf_max - np.nanmin(ccf)
                    e_ccf = np.sqrt(np.abs(ccf)) * 0.01 + ccf_depth * 0.001
                
                ccf_data.append(ccf)
                ccf_errors.append(e_ccf)
                v_grids.append(v_grid)
                file_names.append(os.path.basename(file))
                
            except KeyError as e:
                print(f"Error: Missing FITS extension in {os.path.basename(file)}: {e}")
                print(f"Available extensions: {[hdu.name for hdu in hdus]}")
                hdus.close()
                continue
            except Exception as e:
                print(f"Error processing {os.path.basename(file)}: {e}")
                continue
        
        if len(ccf_data) < 2:
            print(f"Insufficient data for {inst} (need at least 2 CCFs, got {len(ccf_data)})")
            continue
        
        # Convert to numpy arrays
        n_files = len(ccf_data)
        n_points = len(v_grids[0])
        
        # Create CCF matrix: [n_velocity_points, n_observations]
        CCF = np.zeros((n_points, n_files))
        eCCF = np.zeros((n_points, n_files))
        V_grid = v_grids[0]  # Assuming all have same velocity grid
        
        # COMPREHENSIVE CCF ANALYSIS AND PROPER NORMALIZATION FOR FIESTA
        for i, (ccf, e_ccf) in enumerate(zip(ccf_data, ccf_errors)):
            
            # DETAILED ANALYSIS (first few CCFs only)
            if i < 3:
                print(f"\n=== CCF {i} DETAILED ANALYSIS ===")
                print(f"Raw CCF stats:")
                print(f"  Min: {np.nanmin(ccf):.2f}")
                print(f"  Max: {np.nanmax(ccf):.2f}")
                print(f"  Mean: {np.nanmean(ccf):.2f}")
                print(f"  Range: {np.nanmax(ccf) - np.nanmin(ccf):.2f}")
                
                # Find the line center (minimum for absorption)
                min_idx = np.nanargmin(ccf)
                max_idx = np.nanargmax(ccf)
                print(f"  Deepest point: v={V_grid[min_idx]:.2f} km/s, flux={ccf[min_idx]:.2f}")
                print(f"  Highest point: v={V_grid[max_idx]:.2f} km/s, flux={ccf[max_idx]:.2f}")
            
            # STEP 1: NORMALIZE TO CONTINUUM = 1.0
            continuum_flux = np.nanmax(ccf)  # Use maximum as continuum
            ccf_normalized = ccf / continuum_flux
            
            # STEP 2: INVERT FOR FIESTA (CRITICAL!)
            # FIESTA expects: high values = line center, low values = continuum
            ccf_inverted = 1.0 - ccf_normalized
            
            # STEP 3: VERIFY INVERSION
            line_center_inverted = np.nanmax(ccf_inverted)  # Should be highest after inversion
            continuum_inverted = np.nanmin(ccf_inverted)   # Should be lowest after inversion
            line_depth = line_center_inverted - continuum_inverted
            
            if i < 3:  # Debug first few
                print(f"CCF {i} AFTER INVERSION:")
                print(f"  Original range: [{np.nanmin(ccf_normalized):.3f}, {np.nanmax(ccf_normalized):.3f}]")
                print(f"  Inverted range: [{np.nanmin(ccf_inverted):.3f}, {np.nanmax(ccf_inverted):.3f}]")
                print(f"  Line depth (inverted): {line_depth:.3f}")
                print(f"  → Ready for FIESTA!")
            
            # Check if inversion is reasonable
            if line_depth < 0.01:  # Less than 1% depth
                print(f"Warning: CCF {i} appears too shallow after inversion (depth = {line_depth:.3f})")
            elif line_depth > 0.8:  # More than 80% depth
                print(f"Warning: CCF {i} appears too deep after inversion (depth = {line_depth:.3f})")
            
            # STEP 4: STORE INVERTED CCF FOR FIESTA
            CCF[:, i] = ccf_inverted
            
            # STEP 5: PROPAGATE ERRORS (same normalization, no inversion needed for errors)
            e_ccf_normalized = e_ccf / continuum_flux
            
            # Handle invalid errors
            invalid_errors = ~np.isfinite(e_ccf_normalized) | (e_ccf_normalized <= 0)
            if np.any(invalid_errors):
                # Use photon noise estimate based on normalized flux
                e_ccf_normalized[invalid_errors] = np.sqrt(np.abs(ccf_normalized[invalid_errors])) * 0.01
            
            eCCF[:, i] = e_ccf_normalized
        
        # Add CCF diagnostics
        print(f"  CCF diagnostics:")
        print(f"    Velocity range: {np.min(V_grid):.2f} to {np.max(V_grid):.2f} km/s")
        print(f"    CCF depth range: {np.min(CCF):.4f} to {np.max(CCF):.4f}")
        print(f"    CCF variation across observations: {np.std(np.mean(CCF, axis=0)):.6f}")

        # Check if CCFs actually vary between observations
        ccf_centers = []
        for i in range(n_files):
            # Find the minimum (deepest part) of each CCF
            min_idx = np.argmin(CCF[:, i])
            ccf_centers.append(V_grid[min_idx])

        ccf_centers = np.array(ccf_centers)
        print(f"    CCF center variations: {np.std(ccf_centers):.4f} km/s")
        print(f"    CCF center range: {np.min(ccf_centers):.4f} to {np.max(ccf_centers):.4f} km/s")
        
        # Apply FIESTA
        try:
            print(f"  Applying FIESTA with k_max={k_max} to {n_files} CCFs...")
            
            # FIESTA returns 6 values when noise is present:
            # df, v_k, σv_k, A_k, σA_k, RV_gauss
            result = FIESTA(V_grid, CCF, eCCF, k_max=k_max, template=[])
            
            if len(result) == 6:
                # With noise (normal case)
                df, v_k, sigma_v_k, A_k, sigma_A_k, RV_gauss = result
                
                # ===== ADD DEBUG HERE - RIGHT AFTER FIESTA RETURNS =====
                print(f"\n=== DEBUGGING RV_GAUSS for {inst} ===")
                print(f"RV_gauss type: {type(RV_gauss)}")
                print(f"RV_gauss shape: {RV_gauss.shape if hasattr(RV_gauss, 'shape') else 'no shape'}")
                print(f"RV_gauss raw values (first 5): {RV_gauss[:5] if hasattr(RV_gauss, '__len__') else RV_gauss}")
                print(f"RV_gauss min/max/std: {np.min(RV_gauss):.6f} / {np.max(RV_gauss):.6f} / {np.std(RV_gauss):.6f}")
                
                # Convert all to m/s
                RV_gauss_ms = RV_gauss * 1000  # This should be the main RV signal
                
                print(f"After conversion to m/s:")
                print(f"RV_gauss_ms min/max/std: {np.min(RV_gauss_ms):.6f} / {np.max(RV_gauss_ms):.6f} / {np.std(RV_gauss_ms):.6f}")
                print(f"=== END DEBUG ===\n")
                # ===== END DEBUG BLOCK =====
                
                RV_FT_k = v_k * 1000          # These are the Fourier mode variations
                eRV_FT_k = sigma_v_k * 1000   # Uncertainties
                
                # Debug: Check what we actually got
                print(f"  Debug: RV_gauss stats - mean: {np.mean(RV_gauss_ms):.2f}, std: {np.std(RV_gauss_ms):.2f} m/s")
                print(f"  Debug: RV_gauss range: {np.min(RV_gauss_ms):.2f} to {np.max(RV_gauss_ms):.2f} m/s")
                
                # Check if RV_gauss is actually flat
                if np.std(RV_gauss_ms) < 1e-3:  # Less than 1 mm/s variation
                    print(f"  WARNING: RV_gauss is essentially flat! This suggests an issue with FIESTA setup.")
                    print(f"  Check your CCF data and FIESTA parameters.")
                    
                    # As a fallback, use the first Fourier mode as the main RV
                    print(f"  Using first Fourier mode as main RV signal...")
                    RV_reference = RV_FT_k[0, :].copy()  # Use first mode as reference
                    
                    # Calculate differentials relative to first mode
                    ΔRV_k = np.zeros((RV_FT_k.shape[0]-1, RV_FT_k.shape[1]))  # One less mode
                    for k in range(1, RV_FT_k.shape[0]):  # Start from second mode
                        ΔRV_k[k-1, :] = RV_FT_k[k, :] - RV_reference
                        
                else:
                    # Normal case: RV_gauss has variation
                    RV_reference = RV_gauss_ms.copy()
                    
                    # Calculate differential RVs relative to RV_gauss
                    ΔRV_k = np.zeros(RV_FT_k.shape)
                    for k in range(RV_FT_k.shape[0]):
                        ΔRV_k[k, :] = RV_FT_k[k, :] - RV_reference
                
            elif len(result) == 3:
                # Without noise (unlikely)
                v_k, A_k, RV_gauss = result
                RV_FT_k = v_k * 1000
                eRV_FT_k = np.ones_like(v_k) * 100  # Default errors
                RV_reference = RV_gauss * 1000
                
                # Calculate differentials
                ΔRV_k = np.zeros(RV_FT_k.shape)
                for k in range(RV_FT_k.shape[0]):
                    ΔRV_k[k, :] = RV_FT_k[k, :] - RV_reference
                    
            else:
                print(f"  Warning: FIESTA returned unexpected {len(result)} values")
                continue
            
            # More debug info
            print(f"  Final RV_reference range: {np.min(RV_reference):.2f} to {np.max(RV_reference):.2f} m/s")
            print(f"  Final RV_reference RMS: {np.std(RV_reference):.2f} m/s")
            
            # Store results
            results[inst] = {
                'files': file_names,
                'emjd_times': np.array(emjd_times[:n_files]),
                'V_grid': V_grid,
                'CCF': CCF,
                'eCCF': eCCF,
                'RV_FT_k': RV_FT_k,      # All Fourier modes [m/s]
                'eRV_FT_k': eRV_FT_k,    # Uncertainties [m/s]
                'A_k': A_k,              # Amplitudes
                'RV_gauss': RV_reference,  # Main RV signal [m/s]
                'ΔRV_k': ΔRV_k,          # Differential RVs [m/s]
                'n_files': n_files,
                'k_max': ΔRV_k.shape[0]  # Number of differential modes
            }
            
            print(f"  Success! Processed {n_files} CCFs with {ΔRV_k.shape[0]} differential modes")
            print(f"  RV range: {np.min(RV_reference):.1f} to {np.max(RV_reference):.1f} m/s")
            print(f"  eMJD range: {np.min(emjd_times):.6f} to {np.max(emjd_times):.6f}")
            print(f"  Activity indicator ranges:")
            for k in range(min(3, ΔRV_k.shape[0])):
                rms = np.std(ΔRV_k[k, :])
                print(f"    ΔRV_{k+1}: RMS = {rms:.2f} m/s")
            
        except Exception as e:
            print(f"  Error applying FIESTA to {inst}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def periodogram6(ax, time, data, vlines=None):
    """Calculate and plot periodogram similar to the reference code"""
    from scipy import signal
    
    # Remove mean
    data_centered = data - np.mean(data)
    
    # Calculate Lomb-Scargle periodogram
    frequencies = np.logspace(-2, 2, 1000)  # From 0.01 to 100 days
    periods = 1.0 / frequencies
    
    try:
        power = signal.lombscargle(time, data_centered, frequencies, normalize=True)
        
        # Plot periodogram
        ax.semilogx(periods, power, 'k-', alpha=0.7, linewidth=1)
        
        # Add vertical lines for specific periods
        if vlines:
            for period in vlines:
                ax.axvline(period, color='red', linestyle='--', alpha=0.7, linewidth=1)
                ax.text(period, ax.get_ylim()[1]*0.9, f'{period}', 
                       rotation=90, ha='right', va='top', fontsize=8, color='red')
        
        ax.set_xlim(1, 100)
        ax.set_ylabel('Power')
        
    except Exception as e:
        print(f"Error calculating periodogram: {e}")
        ax.text(0.5, 0.5, 'Periodogram\nError', ha='center', va='center', 
                transform=ax.transAxes)


def plot_fiesta_results(results, dset_num, save_plots=True, output_dir=None):
    """Plot FIESTA results in the same style as the reference figure"""
    if not results:
        print("No results to plot")
        return
    
    print(f"\nCreating FIESTA-style plots for DS{dset_num}...")
    
    # Combine all instruments data
    all_times = []
    all_rv_gauss = []
    all_delta_rvs = []
    
    for inst in results:
        data = results[inst]
        times = data['emjd_times'] - np.min(data['emjd_times'])  # Convert to days from start (eMJD)
        rv_gauss = data['RV_gauss']
        delta_rvs = data['ΔRV_k']
        
        all_times.extend(times)
        all_rv_gauss.extend(rv_gauss)
        
        # Extend delta RVs
        if len(all_delta_rvs) == 0:
            all_delta_rvs = [[] for _ in range(delta_rvs.shape[0])]
        
        for k in range(delta_rvs.shape[0]):
            all_delta_rvs[k].extend(delta_rvs[k, :])
    
    # Convert to numpy arrays
    all_times = np.array(all_times)
    all_rv_gauss = np.array(all_rv_gauss)
    all_delta_rvs = [np.array(drv) for drv in all_delta_rvs]
    
    # Determine number of modes
    k_mode = len(all_delta_rvs)
    N_file = len(all_times)
    
    # Set up the plot layout (same as reference)
    alpha1 = 0.2
    plt.rcParams.update({'font.size': 10})
    lw = 2
    widths = [7, 1, 7]
    heights = [1] * (k_mode + 1)  # +1 for RV_gauss row
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    
    fig, f6_axes = plt.subplots(figsize=(10, k_mode + 1), ncols=3, nrows=k_mode + 1, 
                               constrained_layout=True, gridspec_kw=gs_kw)
    
    # Import LinearRegression for R² calculation
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError:
        print("Warning: sklearn not available, R² calculation will be skipped")
        LinearRegression = None
    
    # Plot each row
    for r in range(k_mode + 1):
        for c in range(3):
            ax = f6_axes[r, c]
            
            # Column 0: Time-series
            if c == 0:
                if r == 0:
                    # First row: RV_gauss)
                    ax.plot(all_times, all_rv_gauss, 'k.', alpha=alpha1)
                    ax.set_title('Time-series')
                    ax.set_ylabel('$RV_{gauss}$')
                else:
                    # Other rows: ΔRV_k
                    ax.plot(all_times, all_delta_rvs[r-1], 'k.', alpha=alpha1)
                    ax.set_ylabel(r'$\Delta$RV$_{%d}$' % r)
                
                if r != k_mode:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel('Time [d]')
            
            # Column 1: R² correlation plots
            elif c == 1:
                if LinearRegression is not None:
                    if r == 0:
                        # RV_gauss vs RV_gauss (perfect correlation)
                        reg = LinearRegression().fit(all_rv_gauss.reshape(-1, 1), 
                                                   all_rv_gauss.reshape(-1, 1))
                        score = reg.score(all_rv_gauss.reshape(-1, 1), 
                                        all_rv_gauss.reshape(-1, 1))
                        adjust_R2 = 1 - (1 - score) * (N_file - 1) / (N_file - 1 - 1)
                        title = r'$\bar{R}$' + r'$^2$'
                        ax.set_title(title + ' = {:.2f}'.format(adjust_R2))
                        ax.plot(all_rv_gauss, all_rv_gauss, 'k.', alpha=alpha1)
                    else:
                        # ΔRV_k vs RV_gauss correlation
                        reg = LinearRegression().fit(all_rv_gauss.reshape(-1, 1), 
                                                   all_delta_rvs[r-1].reshape(-1, 1))
                        score = reg.score(all_rv_gauss.reshape(-1, 1), 
                                        all_delta_rvs[r-1].reshape(-1, 1))
                        adjust_R2 = 1 - (1 - score) * (N_file - 1) / (N_file - 1 - 1)
                        title = r'$\bar{R}$' + r'$^2$'
                        ax.set_title(title + ' = {:.2f}'.format(adjust_R2))
                        ax.plot(all_rv_gauss, all_delta_rvs[r-1], 'k.', alpha=alpha1)
                else:
                    ax.text(0.5, 0.5, 'R² calc\nunavailable', ha='center', va='center',
                           transform=ax.transAxes)
                
                if r != k_mode:
                    ax.tick_params(labelbottom=False)
                else:
                    ax.set_xlabel('$RV_{gauss}$')
                ax.yaxis.tick_right()
            
            # Column 2: Periodograms
            elif c == 2:
                if r == 0:
                    # RV_gauss periodogram
                    periodogram6(ax, all_times, all_rv_gauss, vlines=[8.0, 14.6, 26.6])
                    ax.set_title('Periodogram')
                else:
                    # ΔRV_k periodogram
                    periodogram6(ax, all_times, all_delta_rvs[r-1], vlines=[8.0, 14.6, 26.6])
                
                if r != k_mode:
                    ax.tick_params(labelbottom=False)
                if r == k_mode:
                    ax.set_xlabel('Period [days]')
    
    # Align y-labels
    fig.align_ylabels(f6_axes[:, 0])
    
    # Save or show
    if save_plots and output_dir:
        filename = os.path.join(output_dir, f'DS{dset_num}_FIESTA_style.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    else:
        plt.show()


def save_activity_indicators_dat(results, dset_num, output_dir):
    """Save FIESTA activity indicators (differential RVs) to .dat files with eMJD times"""
    if not results:
        print("No FIESTA results to save")
        return
    
    print(f"\nSaving activity indicators to .dat files with eMJD times...")
    
    # Set the correct output directory for .dat files
    dat_output_dir = '/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/data'
    os.makedirs(dat_output_dir, exist_ok=True)
    
    for inst in results:
        data = results[inst]
        n_obs = data['n_files']
        k_max = data['k_max']
        
        print(f"  Processing {inst.upper()}...")
        
        # Save the main RV signal (RV_gauss) as well
        filename_rv = os.path.join(dat_output_dir, f'DS{dset_num}_{inst}_fiesta_rv_gauss.dat')
        
        # Use eMJD times from FIESTA results
        emjd_times = data['emjd_times']
        rv_gauss = data['RV_gauss']
        
        # Get uncertainties for RV_gauss (use first mode uncertainties as estimate)
        if 'eRV_FT_k' in data and data['eRV_FT_k'].shape[0] > 0:
            rv_uncertainties = data['eRV_FT_k'][0, :]
        else:
            rv_uncertainties = np.full(n_obs, np.std(rv_gauss) * 0.1)  # 10% of RMS as uncertainty
        
        # Create other columns for RV_gauss
        jitter = np.zeros(n_obs)
        offset = np.zeros(n_obs)
        flag = np.full(n_obs, -1, dtype=int)
        
        # Save RV_gauss
        rv_data = np.column_stack([
            emjd_times,
            rv_gauss,
            rv_uncertainties,
            jitter,
            offset,
            flag
        ])
        
        np.savetxt(filename_rv, rv_data, 
                 fmt=['%.6f', '%.6f', '%.6f', '%d', '%d', '%d'])
        print(f"    Saved: {filename_rv} (n_obs={n_obs})")
        
        # Handle each differential mode separately
        for k in range(k_max):
            filename = os.path.join(dat_output_dir, f'DS{dset_num}_{inst}_fiesta_mode{k+1}.dat')
            
            delta_rv = data['ΔRV_k'][k, :]  # Differential RV for mode k
            
            # Get uncertainties
            if 'eRV_FT_k' in data and k < data['eRV_FT_k'].shape[0]:
                uncertainties = data['eRV_FT_k'][k, :]
            else:
                # Estimate from RMS if formal errors not available
                uncertainties = np.full(n_obs, np.std(delta_rv))
            
            # Stack data: eMJD, value, error, jitter, offset, flag
            output_data = np.column_stack([
                emjd_times,     # eMJD VALUES
                delta_rv,       # FIESTA differential RV
                uncertainties,  # Uncertainties
                jitter,         # Jitter (0)
                offset,         # Offset (0)
                flag            # Flag (-1 as integer)
            ])
            
            # Save with same format as Contrast.dat - FLAGS AS INTEGERS
            np.savetxt(filename, output_data, 
                     fmt=['%.6f', '%.6f', '%.6f', '%d', '%d', '%d'])
            
            print(f"    Saved: {filename} (n_obs={n_obs})")

    print(f"All FIESTA files saved with eMJD times!")


def main():
    """Main function to run FIESTA analysis"""
    # ========== EDIT THESE SETTINGS ==========
    essp_dir = '/work2/lbuc/data/ESSP4/ESSP4'
    output_dir = '/work2/lbuc/iara/GitHub/ESSP/Figures/FIESTA_figures'
    offset_filename = 'instrument_offsets_iccf.csv'
    
    # FIESTA settings
    dataset = None  # Specific dataset number, or None for all datasets
    k_max = 5      # Number of FIESTA modes
    save_results = False  # No pickle files needed
    save_plots = True
    # =========================================
    
    print(f"FIESTA Analysis for ESSP4 CCF Data")
    print(f"Data directory: {essp_dir}")
    print(f"Output directory for plots: {output_dir}")
    print(f"Output directory for .dat files: /work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/data")
    print(f"FIESTA modes: {k_max}")
    print(f"Save plots: {save_plots}")
    
    if not os.path.exists(essp_dir):
        print(f"Error: {essp_dir} not found")
        return
    
    # Create output directory for plots
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load offsets (optional, for future use)
    offset_file = os.path.join(essp_dir, offset_filename)
    offset_dict = load_offsets(offset_file)
    
    # Get datasets
    if dataset:
        datasets = [dataset]
    else:
        datasets = get_available_datasets(essp_dir)
    
    print(f"Analyzing datasets: {datasets}")
    
    for dset_num in datasets:
        print(f"\n{'='*60}")
        print(f"Processing Dataset {dset_num}")
        print(f"{'='*60}")
        
        # Run FIESTA analysis
        fiesta_results = analyze_ccfs_with_fiesta(
            essp_dir, dset_num, k_max=k_max,
            save_results=save_results,
            output_dir=output_dir if save_results else None
        )
        
        if fiesta_results:
            # Save activity indicators to .dat files
            save_activity_indicators_dat(
                fiesta_results, dset_num, output_dir
            )
            
            # Plot main results (only differential RVs)
            plot_fiesta_results(
                fiesta_results, dset_num,
                save_plots=save_plots,
                output_dir=output_dir if save_plots else None
            )
        
        print(f"Completed analysis for DS{dset_num}")
    
    print(f"\n{'='*60}")
    print("FIESTA Analysis Complete!")
    print(f"Plots saved to: {output_dir}")
    print(f".dat files saved to: /work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/data")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()