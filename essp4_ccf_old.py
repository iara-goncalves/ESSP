import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import interp1d
import seaborn as sns
import pickle


def load_offsets(offset_file):
    """Load instrument offsets from CSV file or use defaults"""
    instruments = ['harpsn', 'harps', 'expres', 'neid']
    
    if offset_file and os.path.exists(offset_file):
        try:
            # Fixed syntax to match your working code
            offset_dict = dict(zip(*np.loadtxt(offset_file, delimiter=',', unpack=True, dtype=str)))
            offset_dict = {key: float(val) for key, val in offset_dict.items()}
            print(f"Loaded offsets from {offset_file}:")
            for key, val in offset_dict.items():
                print(f"  {key}: {val:.1f} m/s")
            return offset_dict
        except Exception as e:
            print(f"Warning: Could not load offset file {offset_file}: {e}")
    
    print("Using default offsets (all zeros)")
    return {inst: 0.0 for inst in instruments}


def shift_ccf(ccf_x, ccf_y, rv_shift):
    """Shift CCF by RV offset (rv_shift in m/s)"""
    if rv_shift == 0:
        return ccf_y
    return interp1d(ccf_x - rv_shift/1000, ccf_y, kind='cubic', 
                   bounds_error=False, fill_value=np.nan)(ccf_x)


def get_available_datasets(essp_dir):
    """Get list of available datasets"""
    datasets = []
    for item in os.listdir(essp_dir):
        if item.startswith('DS') and os.path.isdir(os.path.join(essp_dir, item)):
            try:
                ds_num = int(item[2:])
                datasets.append(ds_num)
            except ValueError:
                continue
    return sorted(datasets)


def process_all_files_for_instrument(files):
    """Process all files for an instrument and return combined data"""
    all_ccfs = []
    all_v_grids = []
    all_obo_ccfs = []
    
    for file in files:
        try:
            hdus = fits.open(file)
            v_grid = hdus['v_grid'].data.copy()
            ccf = hdus['ccf'].data.copy()
            obo_ccf = hdus['obo_ccf'].data.copy()
            hdus.close()
            
            all_ccfs.append(ccf)
            all_v_grids.append(v_grid)
            all_obo_ccfs.append(obo_ccf)
            
        except Exception as e:
            print(f"    Error processing {os.path.basename(file)}: {e}")
            continue
    
    return all_ccfs, all_v_grids, all_obo_ccfs


def calculate_mean_ccf(all_ccfs, all_v_grids):
    """Calculate mean CCF from all files"""
    if not all_ccfs:
        return None, None
    
    # Use the first v_grid as reference (assuming all are the same)
    v_ref = all_v_grids[0]
    
    # Collect all CCFs
    ccf_stack = []
    for ccf in all_ccfs:
        if np.sum(np.isfinite(ccf)) > 0:
            ccf_norm = ccf / np.nanmax(ccf)  # Normalize each CCF
            ccf_stack.append(ccf_norm)
    
    if not ccf_stack:
        return None, None
    
    # Calculate mean
    ccf_mean = np.nanmean(ccf_stack, axis=0)
    
    return v_ref, ccf_mean


def calculate_mean_obo_ccf(all_obo_ccfs, all_v_grids):
    """Calculate mean order-by-order CCF from all files"""
    if not all_obo_ccfs:
        return None, None
    
    # Use the first v_grid as reference
    v_ref = all_v_grids[0]
    num_orders = all_obo_ccfs[0].shape[0]
    
    # Stack all order-by-order CCFs
    obo_stack = []
    for obo_ccf in all_obo_ccfs:
        obo_norm = np.zeros_like(obo_ccf)
        for nord in range(num_orders):
            if np.sum(np.isfinite(obo_ccf[nord])) > 0:
                obo_norm[nord] = obo_ccf[nord] / np.nanmax(obo_ccf[nord])
        obo_stack.append(obo_norm)
    
    if not obo_stack:
        return None, None
    
    # Calculate mean for each order
    obo_mean = np.nanmean(obo_stack, axis=0)
    
    return v_ref, obo_mean


def plot_ccfs_for_dataset(essp_dir, dset_num, offset_dict, 
                         show_shifted=True, save_plots=False, output_dir=None, normalize=True):
    """Plot CCFs for ALL files in a specific dataset with optional residuals"""
    instruments = ['harpsn', 'harps', 'expres', 'neid']
    
    print(f"\nAnalyzing DS{dset_num} (normalize={normalize})...")
    
    ccf_dir = os.path.join(essp_dir, f'DS{dset_num}', 'CCFs')
    if not os.path.exists(ccf_dir):
        print(f"Warning: {ccf_dir} not found")
        return
    
    # Create output directory if saving plots
    if save_plots and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving plots to: {output_dir}")
    
    # Plot 1: Order-by-Order CCFs - with residuals only if normalized
    norm_text = "Normalized" if normalize else "Raw"
    
    if normalize:
        # With residuals (2 rows)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), 
                                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.4})
        fig.suptitle(f'DS{dset_num} - Order-by-Order CCFs ({norm_text}) - All Files', fontsize=14)
        main_axes = axes[0]
        res_axes = axes[1]
    else:
        # Without residuals (1 row)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle(f'DS{dset_num} - Order-by-Order CCFs ({norm_text}) - All Files', fontsize=14)
        main_axes = axes
        res_axes = None
    
    for iinst, inst in enumerate(instruments):
        ax_main = main_axes[iinst]
        ax_main.set_title(f'{inst.upper()}')
        ax_main.set_xlabel('Velocity [km/s]')
        ax_main.set_ylabel('Normalized Counts' if normalize else 'Counts')
        
        if normalize and res_axes is not None:
            ax_res = res_axes[iinst]
            ax_res.set_xlabel('Velocity [km/s]')
            ax_res.set_ylabel('Residuals')
        
        # Get ALL files for this instrument
        files = glob(os.path.join(ccf_dir, f'DS{dset_num}*_{inst}.fits'))
        if not files:
            ax_main.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_main.transAxes)
            if normalize and res_axes is not None:
                ax_res.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_res.transAxes)
            continue
        
        print(f"  {inst}: Processing {len(files)} files")
        
        # Process all files
        all_ccfs, all_v_grids, all_obo_ccfs = process_all_files_for_instrument(files)
        
        if not all_ccfs:
            ax_main.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax_main.transAxes)
            if normalize and res_axes is not None:
                ax_res.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax_res.transAxes)
            continue
        
        # Use the first file to get the number of orders and setup colors
        hdus = fits.open(files[0])
        num_ord = len(hdus['echelle_orders'].data)
        hdus.close()
        
        colors = sns.color_palette('Spectral', num_ord)
        alpha = 0.3 if len(all_ccfs) > 10 else 0.7
        
        # Calculate mean order-by-order CCF for residuals (only if normalized)
        if normalize:
            v_ref, obo_mean = calculate_mean_obo_ccf(all_obo_ccfs, all_v_grids)
        
        # Plot individual order-by-order CCFs
        for file_idx, (ccf, v_grid, obo_ccf) in enumerate(zip(all_ccfs, all_v_grids, all_obo_ccfs)):
            for nord in range(num_ord):
                if np.sum(np.isfinite(obo_ccf[nord])) == 0:
                    continue
                
                if normalize:
                    y_data = obo_ccf[nord]/np.nanmax(obo_ccf[nord])
                else:
                    y_data = obo_ccf[nord]
                
                ax_main.plot(v_grid, y_data, color=colors[nord], alpha=alpha, linewidth=0.5)
                
                # Plot residuals only if normalized
                if normalize and res_axes is not None and v_ref is not None:
                    residual = y_data - obo_mean[nord]
                    ax_res.plot(v_grid, residual, color=colors[nord], alpha=alpha, linewidth=0.5)
        
        # Plot mean order-by-order CCFs (only if normalized)
        if normalize and v_ref is not None:
            for nord in range(num_ord):
                if np.sum(np.isfinite(obo_mean[nord])) == 0:
                    continue
                ax_main.plot(v_ref, obo_mean[nord], color=colors[nord], linewidth=2, alpha=0.8)
        
        # Add zero line to residuals (only if normalized)
        if normalize and res_axes is not None:
            ax_res.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add legend only for the first subplot
        if iinst == 0:
            if normalize:
                ax_main.plot([], [], color='gray', linewidth=1, alpha=0.7, label=f'Individual (n={len(all_ccfs)} files)')
                ax_main.plot([], [], color='black', linewidth=2, label='Mean')
            else:
                ax_main.plot([], [], color='gray', linewidth=1, label=f'Order-by-order CCFs (n={len(all_ccfs)} files)')
            ax_main.legend()
    
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Use subplots_adjust instead of tight_layout
    if save_plots and output_dir:
        norm_suffix = "_normalized" if normalize else "_raw"
        filename = os.path.join(output_dir, f'DS{dset_num}_all_obo_ccfs{norm_suffix}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
    plt.show()
    
    # Plot 2: Combined CCFs - with residuals only if normalized
    if normalize and show_shifted:
        # With residuals (2 rows, 2 columns)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.4})
        fig.suptitle(f'DS{dset_num} - Combined CCFs ({norm_text}) - All Files', fontsize=14)
        
        ax1_main, ax2_main = axes[0]
        ax1_res, ax2_res = axes[1]
        
        ax1_main.set_title('Original')
        ax2_main.set_title('Offset-Corrected')
        ax1_main.set_ylabel('Normalized Counts' if normalize else 'Counts')
        ax2_main.set_ylabel('Normalized Counts' if normalize else 'Counts')
        
        ax1_res.set_xlabel('Velocity [km/s]')
        ax1_res.set_ylabel('Residuals')
        ax2_res.set_xlabel('Velocity [km/s]')
        ax2_res.set_ylabel('Residuals')
        
        main_axes = [ax1_main, ax2_main]
        res_axes = [ax1_res, ax2_res]
        
    elif normalize and not show_shifted:
        # With residuals (2 rows, 1 column)
        fig, axes = plt.subplots(2, 1, figsize=(6, 8),
                                gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.4})
        fig.suptitle(f'DS{dset_num} - Combined CCFs ({norm_text}) - All Files', fontsize=14)
        
        ax1_main = axes[0]
        ax1_res = axes[1]
        
        ax1_main.set_ylabel('Normalized Counts' if normalize else 'Counts')
        ax1_res.set_xlabel('Velocity [km/s]')
        ax1_res.set_ylabel('Residuals')
        
        main_axes = [ax1_main]
        res_axes = [ax1_res]
        
    else:
        # Without residuals (1 row)
        if show_shifted:
            fig, (ax1_main, ax2_main) = plt.subplots(1, 2, figsize=(12, 4))
            ax1_main.set_title('Original')
            ax2_main.set_title('Offset-Corrected')
            main_axes = [ax1_main, ax2_main]
        else:
            fig, ax1_main = plt.subplots(1, 1, figsize=(6, 4))
            main_axes = [ax1_main]
        
        fig.suptitle(f'DS{dset_num} - Combined CCFs ({norm_text}) - All Files', fontsize=14)
        res_axes = None
        
        for ax in main_axes:
            ax.set_xlabel('Velocity [km/s]')
            ax.set_ylabel('Normalized Counts' if normalize else 'Counts')
    
    colors = sns.color_palette('Set1', len(instruments))
    
    # Store mean CCFs for residual calculation (only if normalized)
    mean_ccfs = {}
    
    for iinst, inst in enumerate(instruments):
        files = glob(os.path.join(ccf_dir, f'DS{dset_num}*_{inst}.fits'))
        if not files:
            continue
        
        print(f"  {inst}: Plotting {len(files)} combined CCFs")
        
        # Process all files for this instrument
        all_ccfs, all_v_grids, all_obo_ccfs = process_all_files_for_instrument(files)
        
        if not all_ccfs:
            continue
        
        # Calculate mean CCF for this instrument (only if normalized)
        if normalize:
            v_ref, ccf_mean = calculate_mean_ccf(all_ccfs, all_v_grids)
            if v_ref is not None:
                mean_ccfs[inst] = {'v_grid': v_ref, 'ccf': ccf_mean, 
                                  'ccf_shifted': shift_ccf(v_ref, ccf_mean, offset_dict[inst]) if show_shifted else None}
        
        alpha = 0.3 if len(all_ccfs) > 10 else 0.7
        
        for file_idx, (ccf, v_grid) in enumerate(zip(all_ccfs, all_v_grids)):
            if normalize:
                ccf_norm = ccf / np.nanmax(ccf)
            else:
                ccf_norm = ccf
            
            # Original CCF
            label = f'{inst.upper()} (n={len(all_ccfs)})' if file_idx == 0 else None
            main_axes[0].plot(v_grid, ccf_norm, alpha=alpha, linewidth=1, 
                             color=colors[iinst], label=label)
            
            # Shifted CCF
            if show_shifted:
                ccf_shifted = shift_ccf(v_grid, ccf_norm, offset_dict[inst])
                shift_label = f'{inst.upper()} (Δ={offset_dict[inst]:.1f} m/s, n={len(all_ccfs)})' if file_idx == 0 else None
                main_axes[1].plot(v_grid, ccf_shifted, alpha=alpha, linewidth=1,
                                 color=colors[iinst], label=shift_label)
        
        # Plot mean CCFs (thicker lines, only if normalized)
        if normalize and inst in mean_ccfs:
            main_axes[0].plot(mean_ccfs[inst]['v_grid'], mean_ccfs[inst]['ccf'], 
                             color=colors[iinst], linewidth=3, alpha=0.9)
            if show_shifted and mean_ccfs[inst]['ccf_shifted'] is not None:
                main_axes[1].plot(mean_ccfs[inst]['v_grid'], mean_ccfs[inst]['ccf_shifted'], 
                                 color=colors[iinst], linewidth=3, alpha=0.9)
    
    # Calculate and plot residuals (only if normalized)
    if normalize and res_axes is not None:
        for iinst, inst in enumerate(instruments):
            files = glob(os.path.join(ccf_dir, f'DS{dset_num}*_{inst}.fits'))
            if not files or inst not in mean_ccfs:
                continue
            
            all_ccfs, all_v_grids, all_obo_ccfs = process_all_files_for_instrument(files)
            
            for file_idx, (ccf, v_grid) in enumerate(zip(all_ccfs, all_v_grids)):
                if normalize:
                    ccf_norm = ccf / np.nanmax(ccf)
                else:
                    ccf_norm = ccf
                
                # Original residuals
                residual_orig = ccf_norm - mean_ccfs[inst]['ccf']
                res_axes[0].plot(v_grid, residual_orig, alpha=alpha, linewidth=1, color=colors[iinst])
                
                # Shifted residuals
                if show_shifted and len(res_axes) > 1 and mean_ccfs[inst]['ccf_shifted'] is not None:
                    ccf_shifted = shift_ccf(v_grid, ccf_norm, offset_dict[inst])
                    residual_shifted = ccf_shifted - mean_ccfs[inst]['ccf_shifted']
                    res_axes[1].plot(v_grid, residual_shifted, alpha=alpha, linewidth=1, color=colors[iinst])
        
        # Add zero lines to residual plots
        for ax_res in res_axes:
            ax_res.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Add legends
    main_axes[0].legend()
    if show_shifted and len(main_axes) > 1:
        main_axes[1].legend()
    
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Use subplots_adjust instead of tight_layout
    if save_plots and output_dir:
        norm_suffix = "_normalized" if normalize else "_raw"
        filename = os.path.join(output_dir, f'DS{dset_num}_all_combined_ccfs{norm_suffix}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
    plt.show()


def plot_ccfs_normalized_and_raw(essp_dir, dset_num, offset_dict, 
                                show_shifted=True, save_plots=False, output_dir=None):
    """Plot both normalized and raw CCFs for a dataset"""
    print(f"\n=== Plotting both normalized and raw CCFs for DS{dset_num} ===")
    
    # Plot normalized version (with residuals)
    plot_ccfs_for_dataset(essp_dir, dset_num, offset_dict, 
                         show_shifted=show_shifted, save_plots=save_plots, 
                         output_dir=output_dir, normalize=True)
    
    # Plot raw version (without residuals)
    plot_ccfs_for_dataset(essp_dir, dset_num, offset_dict, 
                         show_shifted=show_shifted, save_plots=save_plots, 
                         output_dir=output_dir, normalize=False)

def extract_inverse_normalized_ccf_with_errors(files):
    """Extract inverse normalized total CCF and errors from all files for an instrument"""
    inv_ccfs = []
    errors = []
    v_grids = []
    metadata = []
    
    for file in files:
        try:
            hdus = fits.open(file)
            
            # Extract data
            v_grid = hdus['v_grid'].data.copy()
            ccf = hdus['ccf'].data.copy()
            eccf = hdus['E_CCF'].data.copy()  # Error CCF
            
            # Normalize CCF
            ccf_max = np.nanmax(ccf)
            if ccf_max > 0:
                ccf_normalized = ccf / ccf_max
                eccf_normalized = eccf / ccf_max  # Scale errors accordingly
                
                # Inverse normalized CCF
                inv_ccf_normalized = 1.0 - ccf_normalized
                
                inv_ccfs.append(inv_ccf_normalized)
                errors.append(eccf_normalized)
                v_grids.append(v_grid)
                
                # Extract metadata from header
                header = hdus[0].header
                file_metadata = {
                    'filename': os.path.basename(file),
                    'time': header.get('TIME', np.nan),
                    'rv': header.get('RV', np.nan),
                    'e_rv': header.get('E_RV', np.nan),
                    'inst': header.get('INST', '').strip(),
                    'spec_file': header.get('SPEC', ''),
                    'date_ccf': header.get('DATE-CCF', ''),
                    'offset': header.get('OFFSET', np.nan)
                }
                metadata.append(file_metadata)
            
            hdus.close()
            
        except Exception as e:
            print(f"    Error processing {os.path.basename(file)}: {e}")
            continue
    
    return inv_ccfs, errors, v_grids, metadata

def save_inverse_ccf_data(inv_ccf_data, output_dir, dataset_num, instrument):
    """Save inverse CCF data to pickle file"""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"DS{dataset_num}_invCCF_{instrument}.pkl"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(inv_ccf_data, f)
    
    print(f"  Saved inverse CCF data: {filepath}")
    return filepath

def process_and_save_inverse_ccfs_for_dataset(essp_dir, dset_num, output_dir):
    """Process and save inverse normalized CCFs with errors for all instruments in a dataset"""
    instruments = ['harpsn', 'harps', 'expres', 'neid']
    
    print(f"\nProcessing and saving inverse CCFs for DS{dset_num}...")
    
    ccf_dir = os.path.join(essp_dir, f'DS{dset_num}', 'CCFs')
    if not os.path.exists(ccf_dir):
        print(f"Warning: {ccf_dir} not found")
        return
    
    for inst in instruments:
        # Get ALL files for this instrument
        files = glob(os.path.join(ccf_dir, f'DS{dset_num}*_{inst}.fits'))
        if not files:
            print(f"  {inst}: No files found")
            continue
        
        print(f"  {inst}: Processing {len(files)} files")
        
        # Extract inverse normalized CCFs with errors
        inv_ccfs, errors, v_grids, metadata = extract_inverse_normalized_ccf_with_errors(files)
        
        if not inv_ccfs:
            print(f"  {inst}: No valid data")
            continue
        
        # Convert to 2D arrays (n_files x n_velocity_points)
        inv_ccf_array = np.array(inv_ccfs)  # Shape: (n_files, n_vel_points)
        error_array = np.array(errors)      # Shape: (n_files, n_vel_points)
        v_grid_array = np.array(v_grids)    # Shape: (n_files, n_vel_points)
        
        # Prepare data structure for saving
        inv_ccf_data = {
            'dataset': dset_num,
            'instrument': inst,
            'n_files': len(inv_ccfs),
            'inv_ccf': inv_ccf_array,           # 2D array: inverse normalized CCFs
            'eccf': error_array,                # 2D array: errors
            'v_grid': v_grid_array,             # 2D array: velocity grids
            'metadata': metadata,               # List of dictionaries with file info
            'data_info': {
                'shape': inv_ccf_array.shape,
                'v_range_km_s': [np.min(v_grid_array), np.max(v_grid_array)],
                'processing_date': np.datetime64('now').astype(str),
                'description': 'Inverse normalized total CCF: 1 - (CCF/max(CCF))',
                'error_description': 'Normalized CCF errors: eCCF/max(CCF)'
            }
        }
        
        # Calculate mean inverse CCF and error
        mean_inv_ccf = np.nanmean(inv_ccf_array, axis=0)
        mean_error = np.sqrt(np.nanmean(error_array**2, axis=0)) / np.sqrt(len(inv_ccfs))  # Standard error of mean
        mean_v_grid = v_grids[0]  # Assuming all v_grids are the same
        
        inv_ccf_data['mean_inv_ccf'] = {
            'v_grid': mean_v_grid,
            'inv_ccf': mean_inv_ccf,
            'error': mean_error
        }
        
        # Save to pickle file
        save_inverse_ccf_data(inv_ccf_data, output_dir, dset_num, inst)
        
        print(f"    Saved 2D array shape: {inv_ccf_array.shape} (files x velocity_points)")

def main():
    # ========== EDIT THESE SETTINGS ==========
    essp_dir = '/work2/lbuc/data/ESSP4/ESSP4'
    output_dir = '/work2/lbuc/iara/GitHub/ESSP/Figures/CCF_figures'
    ccf_data_dir = '/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/data'  # CCF data directory
    offset_filename = 'instrument_offsets_iccf.csv'
    dataset = None  # Specific dataset number, or None for all datasets
    show_shifted = True
    save_plots = True
    save_inverse_ccf_data_flag = True  # NEW: Set to True to save inverse CCF data
    plot_both_versions = True
    # =========================================
    
    print(f"ESSP4 CCF Analysis")
    print(f"Data directory: {essp_dir}")
    print(f"Save plots: {save_plots}")
    print(f"Save inverse CCF data: {save_inverse_ccf_data_flag}")
    print(f"Plot both normalized and raw: {plot_both_versions}")
    if save_plots:
        print(f"Plot output directory: {output_dir}")
    if save_inverse_ccf_data_flag:
        print(f"Inverse CCF data output directory: {ccf_data_dir}")
    
    if not os.path.exists(essp_dir):
        print(f"Error: {essp_dir} not found")
        return
    
    # Construct full path to offset file
    offset_file = os.path.join(essp_dir, offset_filename)
    print(f"Looking for offset file: {offset_file}")
    
    # Load offsets
    offset_dict = load_offsets(offset_file)
    
    # Analyze datasets
    if dataset:
        datasets = [dataset]
        print(f"Analyzing dataset: DS{dataset}")
    else:
        datasets = get_available_datasets(essp_dir)
        print(f"Found datasets: {datasets}")
        print("Analyzing all datasets...")
    
    # Save inverse CCF data if requested
    if save_inverse_ccf_data_flag:
        print(f"\n{'='*50}")
        print("SAVING INVERSE CCF DATA")
        print(f"{'='*50}")
        for dset_num in datasets:
            process_and_save_inverse_ccfs_for_dataset(essp_dir, dset_num, ccf_data_dir)
    
    # Plot CCFs
    print(f"\n{'='*50}")
    print("PLOTTING CCFs")
    print(f"{'='*50}")
    for dset_num in datasets:
        if plot_both_versions:
            plot_ccfs_normalized_and_raw(
                essp_dir, dset_num, offset_dict,
                show_shifted=show_shifted,
                save_plots=save_plots,
                output_dir=output_dir if save_plots else None
            )
        else:
            # Default to normalized only
            plot_ccfs_for_dataset(
                essp_dir, dset_num, offset_dict,
                show_shifted=show_shifted,
                save_plots=save_plots,
                output_dir=output_dir if save_plots else None,
                normalize=True
            )
    
    if save_plots:
        print(f"\nAll plots saved to: {output_dir}")
    if save_inverse_ccf_data_flag:
        print(f"All inverse CCF data saved to: {ccf_data_dir}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()