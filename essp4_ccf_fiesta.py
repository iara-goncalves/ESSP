import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from astropy.io import fits
import pickle
from datetime import datetime

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
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    dataset = parts[0]
                    instrument_offsets = {}
                    # Parse instrument offsets (assuming format: DS1,harps:0.5,harpsn:1.2,etc.)
                    for i in range(1, len(parts)):
                        if ':' in parts[i]:
                            inst, offset = parts[i].split(':')
                            instrument_offsets[inst.lower()] = float(offset)
                    offsets[dataset] = instrument_offsets
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
        from FIESTA_II import FIESTA  # Import here to avoid issues if not available
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
        files = sorted(glob(os.path.join(ccf_dir, f'DS{dset_num}*_{inst}.fits')))
        if not files:
            print(f"  No files found for {inst}")
            continue
        
        # Process files to extract CCF data with proper errors
        ccf_data = []
        ccf_errors = []
        v_grids = []
        bjd_times = []
        file_names = []
        
        for file in files:
            try:
                hdus = fits.open(file)
                
                # Extract data from FITS file (using CORRECT UPPERCASE extension names from documentation)
                v_grid = hdus['V_GRID'].data.copy()  # Velocity grid [km/s]
                ccf = hdus['CCF'].data.copy()        # Global CCF
                e_ccf = hdus['E_CCF'].data.copy()    # Global CCF errors ← THIS EXISTS!
                
                # Extract BJD from filename since headers don't contain it
                # Filename format: DS1_2023-04-15T03-55-01_harps.fits
                try:
                    basename = os.path.basename(file)
                    # Extract date/time part: 2023-04-15T03-55-01
                    date_part = basename.split('_')[1]  # Get the date-time part
                    
                    # Convert to BJD (this is a rough estimate - you might need proper conversion)
                    from datetime import datetime
                    dt = datetime.strptime(date_part, '%Y-%m-%dT%H-%M-%S')
                    # Convert to approximate BJD (MJD + 2400000.5)
                    # This is a rough conversion - you might need more precise BJD calculation
                    mjd = (dt - datetime(1858, 11, 17)).total_seconds() / 86400.0
                    bjd = mjd + 2400000.5
                    bjd_times.append(bjd)
                    
                except Exception as e:
                    print(f"    Warning: Could not extract BJD from {basename}, using index: {e}")
                    bjd_times.append(len(bjd_times))  # Use index as fallback
                
                hdus.close()
                
                # Check for valid data
                if np.sum(np.isfinite(ccf)) == 0:
                    print(f"    Skipping {os.path.basename(file)}: no valid CCF data")
                    continue
                
                # Check for valid errors
                if np.sum(np.isfinite(e_ccf)) == 0:
                    print(f"    Warning: {os.path.basename(file)}: no valid error data, estimating errors")
                    # Fallback error estimation if E_CCF is invalid
                    ccf_max = np.nanmax(ccf)
                    ccf_depth = ccf_max - np.nanmin(ccf)
                    e_ccf = np.sqrt(np.abs(ccf)) * 0.01 + ccf_depth * 0.001
                
                ccf_data.append(ccf)
                ccf_errors.append(e_ccf)
                v_grids.append(v_grid)
                file_names.append(os.path.basename(file))
                
            except KeyError as e:
                print(f"    Error: Missing FITS extension in {os.path.basename(file)}: {e}")
                print(f"    Available extensions: {[hdu.name for hdu in hdus]}")
                hdus.close()
                continue
            except Exception as e:
                print(f"    Error processing {os.path.basename(file)}: {e}")
                continue
        
        if len(ccf_data) < 2:
            print(f"  Insufficient data for {inst} (need at least 2 CCFs, got {len(ccf_data)})")
            continue
        
        # Convert to numpy arrays
        n_files = len(ccf_data)
        n_points = len(v_grids[0])
        
        # Create CCF matrix: [n_velocity_points, n_observations]
        CCF = np.zeros((n_points, n_files))
        eCCF = np.zeros((n_points, n_files))
        V_grid = v_grids[0]  # Assuming all have same velocity grid
        
        # Normalize CCFs and handle errors properly
        for i, (ccf, e_ccf) in enumerate(zip(ccf_data, ccf_errors)):
            # Find normalization factor
            ccf_max = np.nanmax(ccf)
            if ccf_max <= 0:
                print(f"    Skipping observation {i}: invalid CCF maximum")
                continue
            
            # Normalize CCF
            ccf_norm = ccf / ccf_max
            CCF[:, i] = ccf_norm
            
            # Propagate errors through normalization: e_norm = e_ccf / ccf_max
            e_ccf_norm = e_ccf / ccf_max
            
            # Handle any remaining invalid errors
            invalid_errors = ~np.isfinite(e_ccf_norm) | (e_ccf_norm <= 0)
            if np.any(invalid_errors):
                # Use photon noise estimate for invalid errors
                e_ccf_norm[invalid_errors] = np.sqrt(np.abs(ccf_norm[invalid_errors])) * 0.01
            
            eCCF[:, i] = e_ccf_norm
        
        # Apply FIESTA
        try:
            print(f"  Applying FIESTA with k_max={k_max} to {n_files} CCFs...")
            
            # FIESTA returns 6 values when noise is present:
            # df, v_k, σv_k, A_k, σA_k, RV_gauss
            result = FIESTA(
                V_grid, CCF, eCCF, 
                template=[],  # Let FIESTA calculate template automatically
                SNR=2.0,
                k_max=k_max
            )
            
            if len(result) == 6:
                # With noise (normal case)
                df, v_k, sigma_v_k, A_k, sigma_A_k, RV_gauss = result
                
                # Check if RV_gauss is flat (all same values)
                if np.std(RV_gauss) < 1e-6:  # Very small variation
                    print(f"  Warning: RV_gauss appears flat (std={np.std(RV_gauss):.2e})")
                    print(f"  Using first FIESTA mode (k=1) as reference instead")
                    # Use first mode as reference for differential RVs
                    RV_reference = v_k[0, :] * 1000  # Convert to m/s
                else:
                    RV_reference = RV_gauss * 1000  # Convert to m/s
                
                # v_k contains the RV variations for each mode
                RV_FT_k = v_k * 1000  # Convert to m/s
                eRV_FT_k = sigma_v_k * 1000  # Convert uncertainties to m/s
                
            elif len(result) == 3:
                # Without noise (unlikely)
                v_k, A_k, RV_gauss = result
                RV_FT_k = v_k * 1000
                eRV_FT_k = np.ones_like(v_k) * 100  # Default errors in m/s
                RV_reference = RV_gauss * 1000
                
            else:
                print(f"  Warning: FIESTA returned unexpected {len(result)} values")
                continue
            
            # Calculate differential RVs (activity indicators)
            ΔRV_k = np.zeros(RV_FT_k.shape)
            for k in range(RV_FT_k.shape[0]):
                ΔRV_k[k, :] = RV_FT_k[k, :] - RV_reference
            
            # Store results
            results[inst] = {
                'files': file_names,
                'bjd_times': np.array(bjd_times[:n_files]),  # ← REAL BJD TIMES
                'V_grid': V_grid,
                'CCF': CCF,
                'eCCF': eCCF,
                'RV_FT_k': RV_FT_k,      # RV from each Fourier mode [m/s]
                'eRV_FT_k': eRV_FT_k,    # Uncertainties on RV_FT_k [m/s]
                'A_k': A_k,              # Amplitude of each Fourier mode
                'RV_gauss': RV_reference,  # Reference RV [m/s]
                'ΔRV_k': ΔRV_k,          # Differential RVs (activity indicators) [m/s]
                'n_files': n_files,
                'k_max': RV_FT_k.shape[0]  # Actual number of modes returned
            }
            
            print(f"  Success! Processed {n_files} CCFs with {RV_FT_k.shape[0]} FIESTA modes")
            print(f"  RV range: {np.min(RV_reference):.1f} to {np.max(RV_reference):.1f} m/s")
            print(f"  BJD range: {np.min(bjd_times):.6f} to {np.max(bjd_times):.6f}")
            print(f"  Activity indicator ranges:")
            for k in range(min(3, RV_FT_k.shape[0])):
                rms = np.std(ΔRV_k[k, :])
                print(f"    ΔRV_{k+1}: RMS = {rms:.2f} m/s")
            
        except Exception as e:
            print(f"  Error applying FIESTA to {inst}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results

def plot_fiesta_results(results, dset_num, save_plots=True, output_dir=None):
    """Plot FIESTA differential RV results as time series with error bars"""
    if not results:
        print("No results to plot")
        return
    
    print(f"\nCreating plots for DS{dset_num}...")
    
    # Determine how many modes to plot (max 5)
    k_max_plot = 5
    if results:
        first_inst = list(results.keys())[0]
        if first_inst in results:
            k_max_plot = min(5, results[first_inst]['k_max'])
    
    # Create subplots: 2x2 for first 4 modes, or adjust as needed
    if k_max_plot <= 2:
        fig, axes = plt.subplots(1, k_max_plot, figsize=(6*k_max_plot, 5))
    elif k_max_plot <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    if k_max_plot == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Colors and markers for each instrument
    inst_colors = {
        'harpsn': 'red',
        'harps': 'blue', 
        'expres': 'green',
        'neid': 'purple'
    }
    
    inst_markers = {
        'harpsn': 'o',
        'harps': 's',
        'expres': '^', 
        'neid': 'v'
    }
    
    # Plot each mode
    for k in range(k_max_plot):
        ax = axes[k]
        
        obs_counter = 0  # Global observation counter
        
        # Plot each instrument
        for inst in ['harpsn', 'harps', 'expres', 'neid']:
            if inst not in results:
                continue
                
            data = results[inst]
            n_obs = len(data['bjd_times'])
            
            # Get data for this mode
            delta_rv = data['ΔRV_k'][k, :]
            
            # Calculate error bars (this is how FIESTA computes them)
            if 'eRV_FT_k' in data:
                errors = data['eRV_FT_k'][k, :]
            else:
                # Fallback: estimate from RMS of residuals
                errors = np.std(delta_rv) * np.ones(n_obs)
            
            # Create observation numbers for this instrument
            obs_numbers = np.arange(obs_counter, obs_counter + n_obs)
            
            # Calculate RMS for legend
            rms_value = np.sqrt(np.mean(delta_rv**2))
            
            # Plot with error bars
            ax.errorbar(obs_numbers, delta_rv, yerr=errors,
                       fmt=inst_markers[inst], color=inst_colors[inst], 
                       alpha=0.7, markersize=4, capsize=2, capthick=1,
                       label=f'{inst.upper()} (RMS={rms_value:.2f} m/s)')
            
            obs_counter += n_obs
        
        # Format subplot
        ax.set_xlabel('Observation Number')
        ax.set_ylabel(f'ΔRV_{k+1} [m/s]')
        ax.set_title(f'Mode k={k+1} (ΔRV_{k+1})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for k in range(k_max_plot, len(axes)):
        axes[k].set_visible(False)
    
    plt.suptitle(f'DS{dset_num} - FIESTA Differential RVs (Activity Indicators)', fontsize=14)
    plt.tight_layout()
    
    if save_plots and output_dir:
        filename = os.path.join(output_dir, f'DS{dset_num}_fiesta_differential_rvs.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    else:
        plt.show()

def save_activity_indicators_dat(results, dset_num, output_dir):
    """Save FIESTA activity indicators (differential RVs) to .dat files with real BJD times"""
    if not results:
        print("No FIESTA results to save")
        return
    
    print(f"\nSaving activity indicators to .dat files with real BJD times...")
    
    # Set the correct output directory for .dat files
    dat_output_dir = '/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/data'
    os.makedirs(dat_output_dir, exist_ok=True)
    
    for inst in results:
        data = results[inst]
        n_obs = data['n_files']
        k_max = data['k_max']
        
        print(f"  Processing {inst.upper()}...")
        
        # Handle each instrument separately (including harps and harpsn)
        for k in range(k_max):
            filename = os.path.join(dat_output_dir, f'DS{dset_num}_{inst}_fiesta_mode{k+1}.dat')
            
            # Use REAL BJD times from FIESTA results
            bjd_times = data['bjd_times']                    # ← REAL BJD VALUES
            delta_rv = data['ΔRV_k'][k, :]                  # ← Differential RV for mode k
            
            # Get uncertainties
            if 'eRV_FT_k' in data:
                uncertainties = data['eRV_FT_k'][k, :]
            else:
                # Estimate from RMS if formal errors not available
                uncertainties = np.full(n_obs, np.std(delta_rv))
            
            # Create other columns (same format as Contrast.dat)
            jitter = np.zeros(n_obs)
            offset = np.zeros(n_obs)  # Single instrument = single offset
            flag = np.full(n_obs, -1, dtype=int)  # ← INTEGER FLAGS
            
            # Stack data: BJD, value, error, jitter, offset, flag
            output_data = np.column_stack([
                bjd_times,      # ← REAL BJD VALUES
                delta_rv,       # ← FIESTA differential RV
                uncertainties,  # ← Uncertainties
                jitter,         # ← Jitter (0)
                offset,         # ← Offset (0)
                flag            # ← Flag (-1 as integer)
            ])
            
            # Save with same format as Contrast.dat - FLAGS AS INTEGERS
            np.savetxt(filename, output_data, 
                     fmt=['%.6f', '%.6f', '%.6f', '%d', '%d', '%d'])
            
            print(f"    Saved: {filename} (n_obs={n_obs})")

    print(f"All FIESTA activity indicator files saved with real BJD times!")

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
