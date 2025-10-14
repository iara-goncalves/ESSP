import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import interp1d
import seaborn as sns
import pickle

class CCFAnalyzer:
    def __init__(self, essp_dir, output_dir, ccf_data_dir):
        self.essp_dir = essp_dir
        self.output_dir = output_dir
        self.ccf_data_dir = ccf_data_dir
        self.instruments = ['harpsn', 'harps', 'expres', 'neid']
    
    def load_offsets(self, offset_file):
        """Load instrument offsets from CSV file or use defaults"""
        if offset_file and os.path.exists(offset_file):
            try:
                offset_dict = dict(zip(*np.loadtxt(offset_file, delimiter=',', unpack=True, dtype=str)))
                offset_dict = {key: float(val) for key, val in offset_dict.items()}
                print(f"Loaded offsets from {offset_file}:")
                for key, val in offset_dict.items():
                    print(f"  {key}: {val:.1f} m/s")
                return offset_dict
            except Exception as e:
                print(f"Warning: Could not load offset file {offset_file}: {e}")
        
        print("Using default offsets (all zeros)")
        return {inst: 0.0 for inst in self.instruments}
    
    def shift_ccf(self, ccf_x, ccf_y, rv_shift):
        """Shift CCF by RV offset (rv_shift in m/s)"""
        if rv_shift == 0:
            return ccf_y
        return interp1d(ccf_x - rv_shift/1000, ccf_y, kind='cubic', 
                       bounds_error=False, fill_value=np.nan)(ccf_x)
    
    def get_available_datasets(self):
        """Get list of available datasets"""
        datasets = []
        for item in os.listdir(self.essp_dir):
            if item.startswith('DS') and os.path.isdir(os.path.join(self.essp_dir, item)):
                try:
                    datasets.append(int(item[2:]))
                except ValueError:
                    continue
        return sorted(datasets)
    
    def process_files(self, files):
        """Process all files for an instrument and return combined data"""
        all_ccfs, all_v_grids, all_obo_ccfs = [], [], []
        
        for file in files:
            try:
                with fits.open(file) as hdus:
                    all_ccfs.append(hdus['ccf'].data.copy())
                    all_v_grids.append(hdus['v_grid'].data.copy())
                    all_obo_ccfs.append(hdus['obo_ccf'].data.copy())
            except Exception as e:
                print(f"    Error processing {os.path.basename(file)}: {e}")
                continue
        
        return all_ccfs, all_v_grids, all_obo_ccfs
    
    def calculate_mean_ccf(self, all_ccfs, all_v_grids):
        """Calculate mean CCF from all files"""
        if not all_ccfs:
            return None, None
        
        v_ref = all_v_grids[0]
        ccf_stack = [ccf / np.nanmax(ccf) for ccf in all_ccfs if np.sum(np.isfinite(ccf)) > 0]
        
        return (v_ref, np.nanmean(ccf_stack, axis=0)) if ccf_stack else (None, None)
    
    def calculate_mean_obo_ccf(self, all_obo_ccfs, all_v_grids):
        """Calculate mean order-by-order CCF from all files"""
        if not all_obo_ccfs:
            return None, None
        
        v_ref = all_v_grids[0]
        num_orders = all_obo_ccfs[0].shape[0]
        
        obo_stack = []
        for obo_ccf in all_obo_ccfs:
            obo_norm = np.zeros_like(obo_ccf)
            for nord in range(num_orders):
                if np.sum(np.isfinite(obo_ccf[nord])) > 0:
                    obo_norm[nord] = obo_ccf[nord] / np.nanmax(obo_ccf[nord])
            obo_stack.append(obo_norm)
        
        return (v_ref, np.nanmean(obo_stack, axis=0)) if obo_stack else (None, None)
    
    def setup_plots(self, dset_num, normalize, show_shifted, plot_type):
        """Setup plot structure based on parameters"""
        norm_text = "Normalized" if normalize else "Raw"
        
        if plot_type == "obo":
            if normalize:
                fig, axes = plt.subplots(2, 4, figsize=(16, 8), 
                                        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.4})
                fig.suptitle(f'DS{dset_num} - Order-by-Order CCFs ({norm_text})', fontsize=14)
                return fig, axes[0], axes[1]
            else:
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                fig.suptitle(f'DS{dset_num} - Order-by-Order CCFs ({norm_text})', fontsize=14)
                return fig, axes, None
        
        else:  # combined
            if normalize:
                if show_shifted:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 8),
                                            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.4})
                    fig.suptitle(f'DS{dset_num} - Combined CCFs ({norm_text})', fontsize=14)
                    return fig, axes[0], axes[1]
                else:
                    fig, axes = plt.subplots(2, 1, figsize=(6, 8),
                                            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.4})
                    fig.suptitle(f'DS{dset_num} - Combined CCFs ({norm_text})', fontsize=14)
                    return fig, [axes[0]], [axes[1]]
            else:
                if show_shifted:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    fig.suptitle(f'DS{dset_num} - Combined CCFs ({norm_text})', fontsize=14)
                    return fig, axes, None
                else:
                    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
                    fig.suptitle(f'DS{dset_num} - Combined CCFs ({norm_text})', fontsize=14)
                    return fig, [ax], None
    
    def plot_ccfs_for_dataset(self, dset_num, offset_dict, show_shifted=True, 
                         save_plots=False, normalize=True):
        """Plot CCFs for ALL files in a specific dataset"""
        print(f"\nAnalyzing DS{dset_num} (normalize={normalize})...")
        
        ccf_dir = os.path.join(self.essp_dir, f'DS{dset_num}', 'CCFs')
        if not os.path.exists(ccf_dir):
            print(f"Warning: {ccf_dir} not found")
            return
        
        if save_plots:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Plot 1: Order-by-Order CCFs
        fig, main_axes, res_axes = self.setup_plots(dset_num, normalize, show_shifted, "obo")
        
        for iinst, inst in enumerate(self.instruments):
            ax_main = main_axes[iinst]
            ax_main.set_title(f'{inst.upper()}')
            ax_main.set_xlabel('Velocity [km/s]')
            ax_main.set_ylabel('Normalized Counts' if normalize else 'Counts')
            
            files = glob(os.path.join(ccf_dir, f'DS{dset_num}*_{inst}.fits'))
            if not files:
                ax_main.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_main.transAxes)
                if res_axes is not None:
                    res_axes[iinst].text(0.5, 0.5, 'No data', ha='center', va='center', transform=res_axes[iinst].transAxes)
                continue
            
            print(f"  {inst}: Processing {len(files)} files")
            all_ccfs, all_v_grids, all_obo_ccfs = self.process_files(files)
            
            if not all_ccfs:
                continue
            
            # Get number of orders and colors
            with fits.open(files[0]) as hdus:
                num_ord = len(hdus['echelle_orders'].data)
            colors = sns.color_palette('Spectral', num_ord)
            alpha = 0.3 if len(all_ccfs) > 10 else 0.7
            
            # Calculate mean for residuals
            if normalize:
                v_ref, obo_mean = self.calculate_mean_obo_ccf(all_obo_ccfs, all_v_grids)
            
            # Plot individual and mean CCFs
            for ccf, v_grid, obo_ccf in zip(all_ccfs, all_v_grids, all_obo_ccfs):
                for nord in range(num_ord):
                    if np.sum(np.isfinite(obo_ccf[nord])) == 0:
                        continue
                    
                    y_data = obo_ccf[nord]/np.nanmax(obo_ccf[nord]) if normalize else obo_ccf[nord]
                    ax_main.plot(v_grid, y_data, color=colors[nord], alpha=alpha, linewidth=0.5)
                    
                    # Plot residuals
                    if normalize and res_axes is not None and v_ref is not None:
                        residual = y_data - obo_mean[nord]
                        res_axes[iinst].plot(v_grid, residual, color=colors[nord], alpha=alpha, linewidth=0.5)
            
            # Plot mean
            if normalize and v_ref is not None:
                for nord in range(num_ord):
                    if np.sum(np.isfinite(obo_mean[nord])) == 0:
                        continue
                    ax_main.plot(v_ref, obo_mean[nord], color=colors[nord], linewidth=2, alpha=0.8)
            
            # Add zero line to residuals
            if normalize and res_axes is not None:
                res_axes[iinst].axhline(y=0, color='black', linestyle='--', alpha=0.5)
                res_axes[iinst].set_xlabel('Velocity [km/s]')
                res_axes[iinst].set_ylabel('Residuals')
            
            # Legend for first subplot
            if iinst == 0:
                if normalize:
                    ax_main.plot([], [], color='gray', linewidth=1, alpha=0.7, label=f'Individual (n={len(all_ccfs)} files)')
                    ax_main.plot([], [], color='black', linewidth=2, label='Mean')
                else:
                    ax_main.plot([], [], color='gray', linewidth=1, label=f'Order-by-order CCFs (n={len(all_ccfs)} files)')
                ax_main.legend()
        
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        if save_plots:
            norm_suffix = "_normalized" if normalize else "_raw"
            filename = os.path.join(self.output_dir, f'DS{dset_num}_obo_ccfs{norm_suffix}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Saved: {filename}")
        plt.show()
        
        # Plot 2: Combined CCFs
        fig, main_axes, res_axes = self.setup_plots(dset_num, normalize, show_shifted, "combined")
        
        # Set labels
        if normalize:
            if show_shifted and len(main_axes) > 1:
                main_axes[0].set_title('Original')
                main_axes[1].set_title('Offset-Corrected')
            for ax in main_axes:
                ax.set_ylabel('Normalized Counts' if normalize else 'Counts')
            if res_axes is not None:  # CHANGED: Added 'is not None' check
                for ax in res_axes:
                    ax.set_xlabel('Velocity [km/s]')
                    ax.set_ylabel('Residuals')
        else:
            if show_shifted and len(main_axes) > 1:
                main_axes[0].set_title('Original')
                main_axes[1].set_title('Offset-Corrected')
            for ax in main_axes:
                ax.set_xlabel('Velocity [km/s]')
                ax.set_ylabel('Normalized Counts' if normalize else 'Counts')
        
        colors = sns.color_palette('Set1', len(self.instruments))
        mean_ccfs = {}
        
        for iinst, inst in enumerate(self.instruments):
            files = glob(os.path.join(ccf_dir, f'DS{dset_num}*_{inst}.fits'))
            if not files:
                continue
            
            all_ccfs, all_v_grids, all_obo_ccfs = self.process_files(files)
            if not all_ccfs:
                continue
            
            # Calculate mean CCF
            if normalize:
                v_ref, ccf_mean = self.calculate_mean_ccf(all_ccfs, all_v_grids)
                if v_ref is not None:
                    mean_ccfs[inst] = {
                        'v_grid': v_ref, 'ccf': ccf_mean,
                        'ccf_shifted': self.shift_ccf(v_ref, ccf_mean, offset_dict[inst]) if show_shifted else None
                    }
            
            alpha = 0.3 if len(all_ccfs) > 10 else 0.7
            
            # Plot individual CCFs
            for file_idx, (ccf, v_grid) in enumerate(zip(all_ccfs, all_v_grids)):
                ccf_norm = ccf / np.nanmax(ccf) if normalize else ccf
                
                # Original
                label = f'{inst.upper()} (n={len(all_ccfs)})' if file_idx == 0 else None
                main_axes[0].plot(v_grid, ccf_norm, alpha=alpha, linewidth=1, 
                                color=colors[iinst], label=label)
                
                # Shifted
                if show_shifted and len(main_axes) > 1:
                    ccf_shifted = self.shift_ccf(v_grid, ccf_norm, offset_dict[inst])
                    shift_label = f'{inst.upper()} (Δ={offset_dict[inst]:.1f} m/s, n={len(all_ccfs)})' if file_idx == 0 else None
                    main_axes[1].plot(v_grid, ccf_shifted, alpha=alpha, linewidth=1,
                                    color=colors[iinst], label=shift_label)
            
            # Plot mean CCFs
            if normalize and inst in mean_ccfs:
                main_axes[0].plot(mean_ccfs[inst]['v_grid'], mean_ccfs[inst]['ccf'], 
                                color=colors[iinst], linewidth=3, alpha=0.9)
                if show_shifted and len(main_axes) > 1 and mean_ccfs[inst]['ccf_shifted'] is not None:
                    main_axes[1].plot(mean_ccfs[inst]['v_grid'], mean_ccfs[inst]['ccf_shifted'], 
                                    color=colors[iinst], linewidth=3, alpha=0.9)
        
        # Plot residuals
        if normalize and res_axes is not None:
            for iinst, inst in enumerate(self.instruments):
                files = glob(os.path.join(ccf_dir, f'DS{dset_num}*_{inst}.fits'))
                if not files or inst not in mean_ccfs:
                    continue
                
                all_ccfs, all_v_grids, all_obo_ccfs = self.process_files(files)
                alpha = 0.3 if len(all_ccfs) > 10 else 0.7
                
                for ccf, v_grid in zip(all_ccfs, all_v_grids):
                    ccf_norm = ccf / np.nanmax(ccf)
                    
                    # Original residuals
                    residual_orig = ccf_norm - mean_ccfs[inst]['ccf']
                    res_axes[0].plot(v_grid, residual_orig, alpha=alpha, linewidth=1, color=colors[iinst])
                    
                    # Shifted residuals
                    if show_shifted and len(res_axes) > 1 and mean_ccfs[inst]['ccf_shifted'] is not None:
                        ccf_shifted = self.shift_ccf(v_grid, ccf_norm, offset_dict[inst])
                        residual_shifted = ccf_shifted - mean_ccfs[inst]['ccf_shifted']
                        res_axes[1].plot(v_grid, residual_shifted, alpha=alpha, linewidth=1, color=colors[iinst])
            
            # Add zero lines
            for ax_res in res_axes:
                ax_res.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add legends
        main_axes[0].legend()
        if show_shifted and len(main_axes) > 1:
            main_axes[1].legend()
        
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        if save_plots:
            norm_suffix = "_normalized" if normalize else "_raw"
            filename = os.path.join(self.output_dir, f'DS{dset_num}_combined_ccfs{norm_suffix}.png')
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"  Saved: {filename}")
        plt.show()

    
    def extract_inverse_ccf_with_errors(self, files):
        """Extract inverse normalized total CCF and errors from all files"""
        inv_ccfs, errors, v_grids, metadata = [], [], [], []
        
        for file in files:
            try:
                with fits.open(file) as hdus:
                    v_grid = hdus['v_grid'].data.copy()
                    ccf = hdus['ccf'].data.copy()
                    eccf = hdus['E_CCF'].data.copy()
                    
                    ccf_max = np.nanmax(ccf)
                    if ccf_max > 0:
                        ccf_normalized = ccf / ccf_max
                        eccf_normalized = eccf / ccf_max
                        inv_ccf_normalized = 1.0 - ccf_normalized
                        
                        inv_ccfs.append(inv_ccf_normalized)
                        errors.append(eccf_normalized)
                        v_grids.append(v_grid)
                        
                        # Extract metadata
                        header = hdus[0].header
                        metadata.append({
                            'filename': os.path.basename(file),
                            'time': header.get('TIME', np.nan),
                            'rv': header.get('RV', np.nan),
                            'e_rv': header.get('E_RV', np.nan),
                            'inst': header.get('INST', '').strip(),
                            'spec_file': header.get('SPEC', ''),
                            'date_ccf': header.get('DATE-CCF', ''),
                            'offset': header.get('OFFSET', np.nan)
                        })
            except Exception as e:
                print(f"    Error processing {os.path.basename(file)}: {e}")
                continue
        
        return inv_ccfs, errors, v_grids, metadata
    
    def save_inverse_ccf_data(self, inv_ccf_data, dataset_num, instrument):
        """Save inverse CCF data to pickle file"""
        os.makedirs(self.ccf_data_dir, exist_ok=True)
        filename = f"DS{dataset_num}_invCCF_{instrument}.pkl"
        filepath = os.path.join(self.ccf_data_dir, filename)
        
        with open(filepath, 'wb') as f:
            pickle.dump(inv_ccf_data, f)
        
        print(f"  Saved inverse CCF data: {filepath}")
        return filepath
    
    def process_and_save_inverse_ccfs(self, dset_num):
        """Process and save inverse normalized CCFs with errors for all instruments"""
        print(f"\nProcessing and saving inverse CCFs for DS{dset_num}...")
        
        ccf_dir = os.path.join(self.essp_dir, f'DS{dset_num}', 'CCFs')
        if not os.path.exists(ccf_dir):
            print(f"Warning: {ccf_dir} not found")
            return
        
        for inst in self.instruments:
            files = glob(os.path.join(ccf_dir, f'DS{dset_num}*_{inst}.fits'))
            if not files:
                print(f"  {inst}: No files found")
                continue
            
            print(f"  {inst}: Processing {len(files)} files")
            
            inv_ccfs, errors, v_grids, metadata = self.extract_inverse_ccf_with_errors(files)
            
            if not inv_ccfs:
                print(f"  {inst}: No valid data")
                continue
            
            # Convert to arrays
            inv_ccf_array = np.array(inv_ccfs)
            error_array = np.array(errors)
            v_grid_array = np.array(v_grids)
            
            # Calculate mean
            mean_inv_ccf = np.nanmean(inv_ccf_array, axis=0)
            mean_error = np.sqrt(np.nanmean(error_array**2, axis=0)) / np.sqrt(len(inv_ccfs))
            
            # Prepare data structure
            inv_ccf_data = {
                'dataset': dset_num,
                'instrument': inst,
                'n_files': len(inv_ccfs),
                'inv_ccf': inv_ccf_array,
                'eccf': error_array,
                'v_grid': v_grid_array,
                'metadata': metadata,
                'mean_inv_ccf': {
                    'v_grid': v_grids[0],
                    'inv_ccf': mean_inv_ccf,
                    'error': mean_error
                },
                'data_info': {
                    'shape': inv_ccf_array.shape,
                    'v_range_km_s': [np.min(v_grid_array), np.max(v_grid_array)],
                    'processing_date': np.datetime64('now').astype(str),
                    'description': 'Inverse normalized total CCF: 1 - (CCF/max(CCF))',
                    'error_description': 'Normalized CCF errors: eCCF/max(CCF)'
                }
            }
            
            self.save_inverse_ccf_data(inv_ccf_data, dset_num, inst)
            print(f"    Saved 2D array shape: {inv_ccf_array.shape} (files x velocity_points)")

def main():
    # Configuration
    essp_dir = '/work2/lbuc/data/ESSP4/ESSP4'
    output_dir = '/work2/lbuc/iara/GitHub/ESSP/Figures/CCF_figures'
    ccf_data_dir = '/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/data'
    offset_filename = 'instrument_offsets_iccf.csv'
    dataset = None  # Specific dataset or None for all
    show_shifted = True
    save_plots = True
    save_inverse_ccf_data_flag = True
    plot_both_versions = True
    
    print(f"ESSP4 CCF Analysis")
    print(f"Data directory: {essp_dir}")
    print(f"Save plots: {save_plots}")
    print(f"Save inverse CCF data: {save_inverse_ccf_data_flag}")
    
    if not os.path.exists(essp_dir):
        print(f"Error: {essp_dir} not found")
        return
    
    # Initialize analyzer
    analyzer = CCFAnalyzer(essp_dir, output_dir, ccf_data_dir)
    
    # Load offsets
    offset_file = os.path.join(essp_dir, offset_filename)
    offset_dict = analyzer.load_offsets(offset_file)
    
    # Get datasets
    datasets = [dataset] if dataset else analyzer.get_available_datasets()
    print(f"Analyzing datasets: {datasets}")
    
    # Save inverse CCF data
    if save_inverse_ccf_data_flag:
        print(f"\n{'='*50}")
        print("SAVING INVERSE CCF DATA")
        print(f"{'='*50}")
        for dset_num in datasets:
            analyzer.process_and_save_inverse_ccfs(dset_num)
    
    # Plot CCFs
    print(f"\n{'='*50}")
    print("PLOTTING CCFs")
    print(f"{'='*50}")
    for dset_num in datasets:
        if plot_both_versions:
            # Normalized version
            analyzer.plot_ccfs_for_dataset(dset_num, offset_dict, show_shifted, save_plots, True)
            # Raw version
            analyzer.plot_ccfs_for_dataset(dset_num, offset_dict, show_shifted, save_plots, False)
        else:
            analyzer.plot_ccfs_for_dataset(dset_num, offset_dict, show_shifted, save_plots, True)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
