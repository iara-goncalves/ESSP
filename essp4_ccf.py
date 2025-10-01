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
            data = np.loadtxt(offset_file, delimiter=',', unpack=True, dtype=str)
            offset_dict = dict(zip(*data))
            return {key: float(val) for key, val in offset_dict.items()}
        except Exception as e:
            print(f"Warning: Could not load offset file: {e}")
    
    print("Using default offsets (all zeros)")
    return {inst: 0.0 for inst in instruments}


def resample_ccf(file_name):
    """Resample CCF data for HARPS compatibility"""
    hdus = fits.open(file_name)
    hdu_names = [hdu.name.lower() for hdu in hdus][1:]
    ccf_dict = {}
    
    for key in hdu_names:
        if key == 'echelle_orders':
            data = hdus[key].data
        elif 'obo' in key:
            if 'rv' in key:
                data = hdus[key].data
            else:
                data = hdus[key].data[:, ::2]
        else:
            data = hdus[key].data[::2]
        ccf_dict[key] = data.copy()
    
    hdus.close()
    return ccf_dict


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


def process_all_files_for_instrument(files, use_resampling=True):
    """Process all files for an instrument and return combined data"""
    all_ccfs = []
    all_v_grids = []
    all_obo_ccfs = []
    
    for file in files:
        try:
            if use_resampling:
                ccf_dict = resample_ccf(file)
                v_grid = ccf_dict['v_grid']
                ccf = ccf_dict['ccf']
                obo_ccf = ccf_dict['obo_ccf']
            else:
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


def plot_ccfs_for_dataset(essp_dir, dset_num, offset_dict, use_resampling=True, 
                         show_shifted=True, save_plots=False, output_dir=None):
    """Plot CCFs for ALL files in a specific dataset"""
    instruments = ['harpsn', 'harps', 'expres', 'neid']
    
    print(f"\nAnalyzing DS{dset_num}...")
    
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
    fig.suptitle(f'DS{dset_num} - Order-by-Order CCFs (All Files)', fontsize=14)
    
    for iinst, inst in enumerate(instruments):
        ax = axes[iinst]
        ax.set_title(f'{inst.upper()}')
        ax.set_xlabel('Velocity [km/s]')
        ax.set_ylabel('Normalized Counts')
        
        # Get ALL files for this instrument
        files = glob(os.path.join(ccf_dir, f'DS{dset_num}*_{inst}.fits'))
        if not files:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        print(f"  {inst}: Processing {len(files)} files")
        
        # Process all files
        all_ccfs, all_v_grids, all_obo_ccfs = process_all_files_for_instrument(files, use_resampling)
        
        if not all_ccfs:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Use the first file to get the number of orders and setup colors
        if use_resampling:
            ccf_dict = resample_ccf(files[0])
            num_ord = len(ccf_dict['echelle_orders'])
        else:
            hdus = fits.open(files[0])
            num_ord = len(hdus['echelle_orders'].data)
            hdus.close()
        
        colors = sns.color_palette('Spectral', num_ord)
        
        # Plot all order-by-order CCFs from all files
        for file_idx, (ccf, v_grid, obo_ccf) in enumerate(zip(all_ccfs, all_v_grids, all_obo_ccfs)):
            alpha = 0.3 if len(all_ccfs) > 10 else 0.7  # Adjust transparency based on number of files
            
            # Plot order-by-order CCFs
            for nord in range(num_ord):
                if np.sum(np.isfinite(obo_ccf[nord])) == 0:
                    continue
                ax.plot(v_grid, obo_ccf[nord]/np.nanmax(obo_ccf[nord]), 
                       color=colors[nord], alpha=alpha, linewidth=0.5)
            
            # Plot combined CCF
            ax.plot(v_grid, ccf/np.nanmax(ccf), color='k', alpha=alpha, linewidth=1)
        
        # Add legend only for the first subplot
        if iinst == 0:
            ax.plot([], [], color='k', linewidth=2, label=f'Combined CCFs (n={len(all_ccfs)})')
            ax.plot([], [], color='gray', linewidth=1, label='Order-by-order CCFs')
            ax.legend()
    
    plt.tight_layout()
    if save_plots and output_dir:
        filename = os.path.join(output_dir, f'DS{dset_num}_all_obo_ccfs.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
    plt.show()
    
    # Plot 2: Combined CCFs from ALL files
    if show_shifted:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'DS{dset_num} - Combined CCFs (All Files)', fontsize=14)
        ax1.set_title('Original')
        ax2.set_title('Offset-Corrected')
        axes = [ax1, ax2]
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        fig.suptitle(f'DS{dset_num} - Combined CCFs (All Files)', fontsize=14)
        axes = [ax1]
    
    for ax in axes:
        ax.set_xlabel('Velocity [km/s]')
        ax.set_ylabel('Normalized Counts')
    
    colors = sns.color_palette('Set1', len(instruments))
    
    for iinst, inst in enumerate(instruments):
        files = glob(os.path.join(ccf_dir, f'DS{dset_num}*_{inst}.fits'))
        if not files:
            continue
        
        print(f"  {inst}: Plotting {len(files)} combined CCFs")
        
        # Process all files for this instrument
        all_ccfs, all_v_grids, all_obo_ccfs = process_all_files_for_instrument(files, use_resampling)
        
        if not all_ccfs:
            continue
        
        alpha = 0.3 if len(all_ccfs) > 10 else 0.7  # Adjust transparency
        
        for file_idx, (ccf, v_grid) in enumerate(zip(all_ccfs, all_v_grids)):
            ccf_norm = ccf / np.nanmax(ccf)
            
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
        filename = os.path.join(output_dir, f'DS{dset_num}_all_combined_ccfs.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filename}")
    plt.show()


def main():
    # ========== EDIT THESE SETTINGS ==========
    essp_dir = '/work2/lbuc/data/ESSP4/ESSP4'
    output_dir = '/work2/lbuc/iara/GitHub/ESSP/Figures/CCF_figures'
    offset_file = None  # Path to offset CSV file, or None for defaults
    dataset = None  # Specific dataset number, or None for all datasets
    use_resampling = True
    show_shifted = True
    save_plots = True
    # =========================================
    
    print(f"ESSP4 CCF Analysis")
    print(f"Data directory: {essp_dir}")
    print(f"Save plots: {save_plots}")
    if save_plots:
        print(f"Output directory: {output_dir}")
    
    if not os.path.exists(essp_dir):
        print(f"Error: {essp_dir} not found")
        return
    
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
        plot_ccfs_for_dataset(
            essp_dir, dset_num, offset_dict,
            use_resampling=use_resampling,
            show_shifted=show_shifted,
            save_plots=save_plots,
            output_dir=output_dir if save_plots else None
        )
    
    if save_plots:
        print(f"\nAll plots saved to: {output_dir}")
    print("Analysis complete!")


if __name__ == "__main__":
    main()