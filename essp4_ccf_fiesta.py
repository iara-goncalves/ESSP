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

def analyze_ccfs_with_fiesta(essp_dir, dset_num, k_max=10, save_results=False, output_dir=None):
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
                
                # Extract data from FITS file
                v_grid = hdus['V_GRID'].data.copy()  # Velocity grid [km/s]
                ccf = hdus['CCF'].data.copy()        # Global CCF
                e_ccf = hdus['E_CCF'].data.copy()    # Global CCF errors
                
                # Get observation time if available in header
                try:
                    bjd = hdus[0].header.get('BJD', hdus[0].header.get('MJD', 0))
                    bjd_times.append(bjd)
                except:
                    bjd_times.append(0)
                
                hdus.close()
                
                # Check for valid data
                if np.sum(np.isfinite(ccf)) == 0:
                    print(f"    Skipping {os.path.basename(file)}: no valid CCF data")
                    continue
                
                if np.sum(np.isfinite(e_ccf)) == 0:
                    print(f"    Warning: {os.path.basename(file)}: no valid error data, estimating errors")
                    # Fallback error estimation if E_CCF is not available
                    e_ccf = np.sqrt(np.abs(ccf)) * 0.01
                
                ccf_data.append(ccf)
                ccf_errors.append(e_ccf)
                v_grids.append(v_grid)
                file_names.append(os.path.basename(file))
                
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
                'bjd_times': np.array(bjd_times[:n_files]),
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
            print(f"  RV range: {np.min(RV_gauss):.1f} to {np.max(RV_gauss):.1f} m/s")
            print(f"  Activity indicator ranges:")
            for k in range(min(3, RV_FT_k.shape[0])):
                rms = np.std(ΔRV_k[k, :])
                print(f"    ΔRV_{k+1}: RMS = {rms:.2f} m/s")
            
        except Exception as e:
            print(f"  Error applying FIESTA to {inst}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results if requested
    if save_results and output_dir and results:
        os.makedirs(output_dir, exist_ok=True)
        
        results_file = os.path.join(output_dir, f'DS{dset_num}_fiesta_results.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        print(f"\nSaved FIESTA results to: {results_file}")
        
        # Also save as text file for easy inspection
        summary_file = os.path.join(output_dir, f'DS{dset_num}_fiesta_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"FIESTA Analysis Summary for DS{dset_num}\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*60 + "\n\n")
            
            for inst in results:
                f.write(f"{inst.upper()}:\n")
                f.write(f"  Number of observations: {results[inst]['n_files']}\n")
                f.write(f"  Number of FIESTA modes: {results[inst]['k_max']}\n")
                f.write(f"  RV range: {np.min(results[inst]['RV_gauss']):.1f} to {np.max(results[inst]['RV_gauss']):.1f} m/s\n")
                f.write(f"  RV RMS: {np.std(results[inst]['RV_gauss']):.2f} m/s\n")
                f.write(f"  Activity indicator RMS values:\n")
                for k in range(min(5, results[inst]['k_max'])):
                    rms = np.std(results[inst]['ΔRV_k'][k, :])
                    f.write(f"    ΔRV_{k+1}: {rms:.2f} m/s\n")
                f.write("\n")
        
        print(f"Saved summary to: {summary_file}")
    
    return results

def plot_fiesta_results(results, dset_num, save_plots=False, output_dir=None):
    """Plot FIESTA analysis results - only differential RVs plot"""
    if not results:
        print("No FIESTA results to plot")
        return
    
    instruments = list(results.keys())
    n_inst = len(instruments)
    colors = sns.color_palette('Set1', n_inst)
    
    # Only Plot: Differential RVs (ΔRV_k) - the main activity indicators
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'DS{dset_num} - FIESTA Differential RVs (Activity Indicators)', fontsize=16)
    axes = axes.flatten()
    
    for k in range(min(4, results[instruments[0]]['k_max'])):
        ax = axes[k]
        ax.set_title(f'Mode k={k+1} (ΔRV_{k+1})', fontsize=12)
        ax.set_ylabel(f'ΔRV_{k+1} [m/s]')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        for i, inst in enumerate(instruments):
            n_obs = results[inst]['n_files']
            
            # Use BJD times if available
            if np.any(results[inst]['bjd_times'] > 0):
                x_data = results[inst]['bjd_times']
                ax.set_xlabel('BJD')
            else:
                x_data = np.arange(n_obs)
                ax.set_xlabel('Observation Number')
            
            y_data = results[inst]['ΔRV_k'][k, :]
            rms = np.std(y_data)
            
            ax.plot(x_data, y_data, 'o-', color=colors[i], 
                   label=f'{inst.upper()} (RMS={rms:.2f} m/s)', 
                   alpha=0.8, markersize=4)
        
        if k == 0:
            ax.legend()
    
    plt.tight_layout()
    if save_plots and output_dir:
        filename = os.path.join(output_dir, f'DS{dset_num}_fiesta_differential_rvs.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    plt.show()

    """Load previously saved FIESTA results"""
    try:
        with open(results_file, 'rb') as f:
            results = pickle.load(f)
        print(f"Loaded FIESTA results from: {results_file}")
        return results
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def save_activity_indicators_dat(results, dset_num, output_dir):
    """Save FIESTA activity indicators (differential RVs) to .dat files"""
    if not results:
        print("No FIESTA results to save")
        return
    
    print(f"\nSaving activity indicators to .dat files...")
    
    for inst in results:
        data = results[inst]
        n_obs = data['n_files']
        k_max = data['k_max']
        
        # Prepare data for each mode
        for k in range(k_max):
            # Create output filename
            filename = os.path.join(output_dir, f'DS{dset_num}_{inst}_fiesta_mode{k+1}.dat')
            
            # Prepare data: BJD, delta_RV, uncertainty
            bjd_times = data['bjd_times']
            delta_rv = data['ΔRV_k'][k, :]  # Differential RV for mode k
            uncertainties = data['eRV_FT_k'][k, :] if 'eRV_FT_k' in data else np.ones(n_obs) * 1.0
            
            # Stack data
            output_data = np.column_stack([bjd_times, delta_rv, uncertainties])
            
            # Save to file
            np.savetxt(filename, output_data, fmt=['%.6f', '%.6f', '%.6f'], 
                      header='BJD  Delta_RV[m/s]  Uncertainty[m/s]')
            
            print(f"  Saved: {filename}")

def main():
    """Main function to run FIESTA analysis"""
    # ========== EDIT THESE SETTINGS ==========
    essp_dir = '/work2/lbuc/data/ESSP4/ESSP4'
    output_dir = '/work2/lbuc/iara/GitHub/ESSP/Figures/FIESTA_figures'
    offset_filename = 'instrument_offsets_iccf.csv'
    
    # FIESTA settings
    dataset = None  # Specific dataset number, or None for all datasets
    k_max = 10      # Number of FIESTA modes
    save_results = True
    save_plots = True
    # =========================================
    
    print(f"FIESTA Analysis for ESSP4 CCF Data")
    print(f"Data directory: {essp_dir}")
    print(f"Output directory: {output_dir}")
    print(f"FIESTA modes: {k_max}")
    print(f"Save results: {save_results}")
    print(f"Save plots: {save_plots}")
    
    if not os.path.exists(essp_dir):
        print(f"Error: {essp_dir} not found")
        return
    
    # Create output directory
    if save_plots or save_results:
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
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")