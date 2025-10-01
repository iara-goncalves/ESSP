import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import interp1d
import seaborn as sns


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


def plot_ccfs_for_dataset(essp_dir, dset_num, offset_dict, 
                         show_shifted=True, save_plots=False, output_dir=None, normalize=True):
    """Plot CCFs for ALL files in a specific dataset"""
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
    
    # Plot 1: Order-by-Order CCFs for ALL files
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    norm_text = "Normalized" if normalize else "Raw"
    fig.suptitle(f'DS{dset_num} - Order-by-Order CCFs ({norm_text}) - All Files', fontsize=14)
    
    for iinst, inst in enumerate(instruments):
        ax = axes[iinst]
        ax.set_title(f'{inst.upper()}')
        ax.set_xlabel('Velocity [km/s]')
        ax.set_ylabel('Normalized Counts' if normalize else 'Counts')
        
        # Get ALL files for this instrument
        files = glob(os.path.join(ccf_dir, f'DS{dset_num}*_{inst}.fits'))
        if not files:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        print(f"  {inst}: Processing {len(files)} files")
        
        # Process all files
        all_ccfs, all_v_grids, all_obo_ccfs = process_all_files_for_instrument(files)
        
        if not all_ccfs:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Use the first file to get the number of orders and setup colors
        hdus = fits.open(files[0])
        num_ord = len(hdus['echelle_orders'].data)
        hdus.close()
        
        colors = sns.color_palette('Spectral', num_ord)
        
        # Plot ONLY order-by-order CCFs from all files (NO COMBINED)
        for file_idx, (ccf, v_grid, obo_ccf) in enumerate(zip(all_ccfs, all_v_grids, all_obo_ccfs)):
            alpha = 0.3 if len(all_ccfs) > 10 else 0.7  # Adjust transparency based on number of files
            
            # Plot order-by-order CCFs
            for nord in range(num_ord):
                if np.sum(np.isfinite(obo_ccf[nord])) == 0:
                    continue
                
                if normalize:
                    y_data = obo_ccf[nord]/np.nanmax(obo_ccf[nord])
                else:
                    y_data = obo_ccf[nord]
                
                ax.plot(v_grid, y_data, color=colors[nord], alpha=alpha, linewidth=0.5)
        
        # Add legend only for the first subplot
        if iinst == 0:
            ax.plot([], [], color='gray', linewidth=1, label=f'Order-by-order CCFs (n={len(all_ccfs)} files)')
            ax.legend()
    
    plt.tight_layout()
    if save_plots and output_dir:
        norm_suffix = "_normalized" if normalize else "_raw"
        filename = os.path.join(output_dir, f'DS{dset_num}_all_obo_ccfs{norm_suffix}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
    plt.show()
    
    # Plot 2: Combined CCFs from ALL files
    if show_shifted:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'DS{dset_num} - Combined CCFs ({norm_text}) - All Files', fontsize=14)
        ax1.set_title('Original')
        ax2.set_title('Offset-Corrected')
        axes = [ax1, ax2]
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        fig.suptitle(f'DS{dset_num} - Combined CCFs ({norm_text}) - All Files', fontsize=14)
        axes = [ax1]
    
    for ax in axes:
        ax.set_xlabel('Velocity [km/s]')
        ax.set_ylabel('Normalized Counts' if normalize else 'Counts')
    
    colors = sns.color_palette('Set1', len(instruments))
    
    for iinst, inst in enumerate(instruments):
        files = glob(os.path.join(ccf_dir, f'DS{dset_num}*_{inst}.fits'))
        if not files:
            continue
        
        print(f"  {inst}: Plotting {len(files)} combined CCFs")
        
        # Process all files for this instrument
        all_ccfs, all_v_grids, all_obo_ccfs = process_all_files_for_instrument(files)
        
        if not all_ccfs:
            continue
        
        alpha = 0.3 if len(all_ccfs) > 10 else 0.7  # Adjust transparency
        
        for file_idx, (ccf, v_grid) in enumerate(zip(all_ccfs, all_v_grids)):
            if normalize:
                ccf_norm = ccf / np.nanmax(ccf)
            else:
                ccf_norm = ccf
            
            # Original CCF
            label = f'{inst.upper()} (n={len(all_ccfs)})' if file_idx == 0 else None
            ax1.plot(v_grid, ccf_norm, alpha=alpha, linewidth=1, 
                    color=colors[iinst], label=label)
            
            # Shifted CCF
            if show_shifted:
                ccf_shifted = shift_ccf(v_grid, ccf_norm, offset_dict[inst])
                shift_label = f'{inst.upper()} (Δ={offset_dict[inst]:.1f} m/s, n={len(all_ccfs)})' if file_idx == 0 else None
                ax2.plot(v_grid, ccf_shifted, alpha=alpha, linewidth=1,
                        color=colors[iinst], label=shift_label)
    
    ax1.legend()
    if show_shifted:
        ax2.legend()
    
    plt.tight_layout()
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
    
    # Plot normalized version
    plot_ccfs_for_dataset(essp_dir, dset_num, offset_dict, 
                         show_shifted=show_shifted, save_plots=save_plots, 
                         output_dir=output_dir, normalize=True)
    
    # Plot raw version
    plot_ccfs_for_dataset(essp_dir, dset_num, offset_dict, 
                         show_shifted=show_shifted, save_plots=save_plots, 
                         output_dir=output_dir, normalize=False)


def main():
    # ========== EDIT THESE SETTINGS ==========
    essp_dir = '/work2/lbuc/data/ESSP4/ESSP4'
    output_dir = '/work2/lbuc/iara/GitHub/ESSP/Figures/CCF_figures'
    offset_filename = 'instrument_offsets_iccf.csv'  # Just the filename
    dataset = None  # Specific dataset number, or None for all datasets
    show_shifted = True
    save_plots = True
    plot_both_versions = True  # Set to True to plot both normalized and raw
    # =========================================
    
    print(f"ESSP4 CCF Analysis")
    print(f"Data directory: {essp_dir}")
    print(f"Save plots: {save_plots}")
    print(f"Plot both normalized and raw: {plot_both_versions}")
    if save_plots:
        print(f"Output directory: {output_dir}")
    
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
    print("Analysis complete!")


if __name__ == "__main__":
    main()
