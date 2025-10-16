import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pandas as pd
from scipy import signal

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class FIESTAProcessor:
    def __init__(self, ccf_data_dir, output_dir):
        self.ccf_data_dir = ccf_data_dir
        self.output_dir = output_dir
        self.dat_output_dir = '/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/data'
        self.instruments = ['harpsn', 'harps', 'expres', 'neid']
        self.sigma_threshold = 3.5
    
    def load_pickle_data(self, dset_num):
        """Load inverse CCF data from pickle files"""
        print(f"\nLoading pickle data for DS{dset_num}...")
        data = {}
        
        for inst in self.instruments:
            filename = f"DS{dset_num}_invCCF_{inst}.pkl"
            filepath = os.path.join(self.ccf_data_dir, filename)
            
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'rb') as f:
                        inst_data = pickle.load(f)
                    data[inst] = inst_data
                    print(f"  {inst}: Loaded {inst_data['n_files']} CCFs")
                except Exception as e:
                    print(f"  {inst}: Error loading {filepath}: {e}")
            else:
                print(f"  {inst}: File not found: {filepath}")
        
        return data
    
    def plot_ccf_data(self, data, dset_num):
        """Plot loaded CCF data for verification - renamed to invCCF"""
        if not data:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'DS{dset_num} - Loaded Inverse CCF Data', fontsize=14)
        axes = axes.flatten()
        
        colors = sns.color_palette('Set1', len(self.instruments))
        
        for i, inst in enumerate(self.instruments):
            if inst not in data:
                axes[i].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{inst.upper()}')
                continue
            
            inst_data = data[inst]
            v_grid = inst_data['mean_inv_ccf']['v_grid']
            mean_inv_ccf = inst_data['mean_inv_ccf']['inv_ccf']
            mean_error = inst_data['mean_inv_ccf']['error']
            
            # Plot mean with error envelope
            axes[i].plot(v_grid, mean_inv_ccf, color=colors[i], linewidth=2, label='Mean')
            axes[i].fill_between(v_grid, mean_inv_ccf - mean_error, mean_inv_ccf + mean_error, 
                                alpha=0.3, color=colors[i], label='±1σ (SEM)')
            
            axes[i].set_title(f'{inst.upper()} (n={inst_data["n_files"]})')
            axes[i].set_xlabel('Velocity [km/s]')
            axes[i].set_ylabel('Inverse Normalized CCF')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, f'DS{dset_num}_invCCF.png')  # Changed filename
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved CCF verification plot: {filename}")
    
    def apply_fiesta(self, data, dset_num, k_max=4):
        """Apply FIESTA and save results to DataFrame"""
        try:
            from FIESTA_II import FIESTA
        except ImportError:
            print("Error: FIESTA_II module not found. Please install FIESTA first.")
            return None
        
        print(f"\nApplying FIESTA with k_max={k_max}...")
        dataframes = {}
        
        for inst in self.instruments:
            if inst not in data:
                continue
            
            print(f"  Processing {inst.upper()}...")
            inst_data = data[inst]
            
            # Extract data
            inv_ccf_array = inst_data['inv_ccf']
            error_array = inst_data['eccf']
            v_grid = inst_data['v_grid'][0]
            metadata = inst_data['metadata']
            
            # Get times from metadata
            emjd_times = np.array([meta['time'] for meta in metadata])
            
            # Prepare for FIESTA: transpose to (n_vel_points, n_files)
            CCF = inv_ccf_array.T
            eCCF = error_array.T
            n_files = CCF.shape[1]
            
            print(f"    CCF shape: {CCF.shape}, eMJD range: {np.min(emjd_times):.6f} to {np.max(emjd_times):.6f}")
            
            try:
                # Apply FIESTA
                result = FIESTA(v_grid, CCF, eCCF, k_max=k_max, template=[])
                
                if len(result) == 6:
                    df, v_k, sigma_v_k, A_k, sigma_A_k, RV_gauss = result
                    
                    # Convert to m/s
                    RV_gauss_ms = RV_gauss * 1000
                    RV_FT_k = v_k * 1000
                    eRV_FT_k = sigma_v_k * 1000
                    
                    print(f"    RV_gauss values (m/s) for {inst.upper()}:")
                    print(f"    Shape: {RV_gauss_ms.shape}")
                    print(f"    Min: {np.min(RV_gauss_ms):.6f}")
                    print(f"    Max: {np.max(RV_gauss_ms):.6f}")
                    print(f"    Mean: {np.mean(RV_gauss_ms):.6f}")
                    print(f"    Std: {np.std(RV_gauss_ms):.6f}")
                    
                    # Calculate differential RVs
                    if np.std(RV_gauss_ms) < 1e-3:  # Flat RV_gauss
                        print(f"    Warning: RV_gauss is flat (std={np.std(RV_gauss_ms):.6f}), using first mode as reference")
                        RV_reference = RV_FT_k[0, :].copy()
                        ΔRV_k = np.zeros((RV_FT_k.shape[0]-1, RV_FT_k.shape[1]))
                        for k in range(1, RV_FT_k.shape[0]):
                            ΔRV_k[k-1, :] = RV_FT_k[k, :] - RV_reference
                    else:
                        RV_reference = RV_gauss_ms.copy()
                        ΔRV_k = np.zeros(RV_FT_k.shape)
                        for k in range(RV_FT_k.shape[0]):
                            ΔRV_k[k, :] = RV_FT_k[k, :] - RV_reference
                    
                    # Create DataFrame with all the data
                    df_data = {
                        'emjd_times': emjd_times,
                        'RV_gauss': RV_reference
                    }
                    
                    # Add differential modes
                    for k in range(ΔRV_k.shape[0]):
                        df_data[f'delta_RV_{k+1}'] = ΔRV_k[k, :]
                    
                    # Add uncertainties
                    if eRV_FT_k.shape[0] > 0:
                        df_data['eRV_gauss'] = eRV_FT_k[0, :]
                        for k in range(min(ΔRV_k.shape[0], eRV_FT_k.shape[0])):
                            df_data[f'e_delta_RV_{k+1}'] = eRV_FT_k[k, :]
                    else:
                        # Generate default uncertainties
                        df_data['eRV_gauss'] = np.full(n_files, np.std(RV_reference) * 0.1)
                        for k in range(ΔRV_k.shape[0]):
                            df_data[f'e_delta_RV_{k+1}'] = np.full(n_files, np.std(ΔRV_k[k, :]))
                    
                    # Create DataFrame
                    df_inst = pd.DataFrame(df_data)
                    dataframes[inst] = df_inst
                    
                    print(f"    Success! DataFrame created with {len(df_inst)} points, {ΔRV_k.shape[0]} modes")
                    print(f"    DataFrame columns: {list(df_inst.columns)}")
                
                else:
                    print(f"    Warning: FIESTA returned {len(result)} values (expected 6)")
                    
            except Exception as e:
                print(f"    Error applying FIESTA: {e}")
                continue
        
        return dataframes
    
    def detect_outliers(self, dataframes, dset_num):
        """Detect outliers using 3-sigma clipping"""
        print(f"\nDetecting outliers with {self.sigma_threshold}-sigma clipping for DS{dset_num}...")
        
        outlier_info = {}
        
        for inst, df in dataframes.items():
            print(f"  Processing {inst.upper()}...")
            
            # Detect outliers in RV_gauss
            rv_mean = df['RV_gauss'].mean()
            rv_std = df['RV_gauss'].std()
            rv_outliers = np.abs((df['RV_gauss'] - rv_mean) / rv_std) > self.sigma_threshold
            
            # Detect outliers in differential modes
            mode_outliers = np.zeros(len(df), dtype=bool)
            mode_columns = [col for col in df.columns if col.startswith('delta_RV_')]
            
            for col in mode_columns:
                col_mean = df[col].mean()
                col_std = df[col].std()
                col_outliers = np.abs((df[col] - col_mean) / col_std) > self.sigma_threshold
                mode_outliers |= col_outliers
            
            # Combine all outliers
            all_outliers = rv_outliers | mode_outliers
            
            outlier_info[inst] = {
                'rv_outliers': rv_outliers,
                'mode_outliers': mode_outliers,
                'all_outliers': all_outliers,
                'n_rv_outliers': np.sum(rv_outliers),
                'n_mode_outliers': np.sum(mode_outliers & ~rv_outliers),
                'n_total_outliers': np.sum(all_outliers),
                'original_points': len(df),
                'outlier_percentage': 100 * np.sum(all_outliers) / len(df)
            }
            
            print(f"    Original points: {len(df)}")
            print(f"    RV outliers: {np.sum(rv_outliers)}")
            print(f"    Mode outliers: {np.sum(mode_outliers & ~rv_outliers)}")
            print(f"    Total outliers: {np.sum(all_outliers)} ({100 * np.sum(all_outliers) / len(df):.1f}%)")
        
        return outlier_info

    def plot_time_series_with_outliers(self, dataframes, outlier_info, dset_num):
        """Plot FIESTA time series with outliers highlighted using your aesthetics"""
        print(f"Creating FIESTA activity plots for DS{dset_num}...")
        
        # Combine all instruments
        all_data = []
        for inst, df in dataframes.items():
            df_copy = df.copy()
            df_copy['instrument'] = inst.upper()
            df_copy['is_outlier'] = outlier_info[inst]['all_outliers']
            all_data.append(df_copy)
        
        if not all_data:
            return
        
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Get mode columns and create plot info
        mode_columns = [col for col in combined_df.columns if col.startswith('delta_RV_')]
        n_modes = len(mode_columns)
        
        plot_info = [("RV_gauss", "eRV_gauss", "RV [m/s]")]
        for i, col in enumerate(mode_columns):
            error_col = f'e_delta_RV_{i+1}'
            plot_info.append((col, error_col, f"δRV_{i+1} [m/s]"))
        
        # Get all unique instruments and colors
        all_instruments = sorted(combined_df["instrument"].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_instruments)))
        color_map = dict(zip(all_instruments, colors))
        
        # Create plot
        fig, axes = plt.subplots(n_modes + 1, 1, figsize=(12, (n_modes + 1) * 3), sharex=True)
        if n_modes == 0:
            axes = [axes]
        
        for ax_idx, (col, err_col, ylabel) in enumerate(plot_info):
            ax = axes[ax_idx]
            
            if col not in combined_df.columns:
                ax.text(0.5, 0.5, f"Column '{col}' not found", 
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_ylabel(ylabel)
                continue
            
            # Calculate instrument-specific medians for centering
            inst_medians = combined_df.groupby("instrument")[col].median()
            
            # Plot each instrument
            for inst in all_instruments:
                inst_data = combined_df[combined_df["instrument"] == inst].copy()
                if inst_data.empty:
                    continue
                
                # Center data by instrument median
                center = inst_medians[inst]
                
                # Determine error values
                if err_col and err_col in inst_data.columns:
                    yerr = inst_data[err_col]
                else:
                    yerr = np.std(inst_data[col]) * 0.1  # Default error
                
                # Split into inliers and outliers
                inliers = inst_data[~inst_data["is_outlier"]]
                outliers = inst_data[inst_data["is_outlier"]]
                
                # Plot inliers
                if not inliers.empty:
                    y_in = inliers[col] - center
                    yerr_in = yerr.loc[inliers.index] if isinstance(yerr, pd.Series) else yerr
                    ax.errorbar(inliers["emjd_times"], y_in, yerr=yerr_in,
                            fmt=".", color=color_map[inst], label=inst, 
                            alpha=0.8, markersize=6)
                
                # Plot outliers
                if not outliers.empty:
                    y_out = outliers[col] - center
                    yerr_out = yerr.loc[outliers.index] if isinstance(yerr, pd.Series) else yerr
                    ax.errorbar(outliers["emjd_times"], y_out, yerr=yerr_out,
                            fmt="o", color="black", alpha=0.7, markersize=8,
                            markeredgecolor="red", markeredgewidth=1.5)
            
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        
        # Add outlier legend entry
        if combined_df["is_outlier"].any():
            axes[0].errorbar([], [], [], fmt="o", color="black", alpha=0.7, 
                            markeredgecolor="red", markeredgewidth=1.5, 
                            label="Outliers")
        
        axes[-1].set_xlabel("Time [eMJD]")
        
        # Create legend with unique entries
        handles, labels = axes[0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[0].legend(by_label.values(), by_label.keys(), loc="best")
        
        fig.suptitle(f"DS{dset_num} FIESTA Activity Indicators (Red-edged = Outliers)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        filename = os.path.join(self.output_dir, f'DS{dset_num}_activity_FIESTA.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    
    def save_dat_files_without_outliers(self, dataframes, outlier_info, dset_num):
        """Save .dat files without outliers"""
        print(f"\nSaving .dat files without outliers for DS{dset_num}...")
        os.makedirs(self.dat_output_dir, exist_ok=True)
        
        for inst, df in dataframes.items():
            print(f"  Processing {inst.upper()}...")
            
            # Remove outliers
            outliers = outlier_info[inst]['all_outliers']
            df_clean = df[~outliers].copy()
            
            print(f"    Original: {len(df)} points")
            print(f"    After outlier removal: {len(df_clean)} points")
            
            if len(df_clean) == 0:
                print(f"    Warning: No data left after outlier removal for {inst}")
                continue
            
            # Prepare common columns
            emjd_times = df_clean['emjd_times'].values
            jitter = np.zeros(len(df_clean))
            offset = np.zeros(len(df_clean))
            flag = np.full(len(df_clean), -1, dtype=int)
            
            # Save RV_gauss
            filename_rv = os.path.join(self.dat_output_dir, f'DS{dset_num}_{inst}_fiesta_rv_gauss.dat')
            rv_gauss = df_clean['RV_gauss'].values
            rv_uncertainties = df_clean['eRV_gauss'].values
            
            rv_data = np.column_stack([emjd_times, rv_gauss, rv_uncertainties, jitter, offset, flag])
            np.savetxt(filename_rv, rv_data, fmt=['%.6f', '%.6f', '%.6f', '%d', '%d', '%d'])
            print(f"    Saved: {filename_rv}")
            
            # Save differential modes
            mode_columns = [col for col in df_clean.columns if col.startswith('delta_RV_')]
            for i, col in enumerate(mode_columns):
                mode_num = i + 1
                filename = os.path.join(self.dat_output_dir, f'DS{dset_num}_{inst}_fiesta_mode{mode_num}.dat')
                
                delta_rv = df_clean[col].values
                error_col = f'e_delta_RV_{mode_num}'
                uncertainties = df_clean[error_col].values if error_col in df_clean.columns else np.full(len(df_clean), np.std(delta_rv))
                
                output_data = np.column_stack([emjd_times, delta_rv, uncertainties, jitter, offset, flag])
                np.savetxt(filename, output_data, fmt=['%.6f', '%.6f', '%.6f', '%d', '%d', '%d'])
                print(f"    Saved: {filename}")

def main():
    # Configuration
    ccf_data_dir = '/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/data'
    output_dir = '/work2/lbuc/iara/GitHub/ESSP/Figures/FIESTA_figures'
    dataset = None  # Specific dataset or None for all
    k_max = 4
    
    print(f"FIESTA Analysis with Outlier Detection")
    print(f"CCF data directory: {ccf_data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"FIESTA modes: {k_max}")
    print(f"Outlier detection: 3-sigma clipping")
    
    # Initialize processor
    processor = FIESTAProcessor(ccf_data_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get available datasets from pickle files
    if dataset:
        datasets = [dataset]
    else:
        pickle_files = [f for f in os.listdir(ccf_data_dir) if f.endswith('_invCCF_harps.pkl')]
        datasets = sorted([int(f.split('_')[0][2:]) for f in pickle_files])
    
    print(f"Processing datasets: {datasets}")
    
    for dset_num in datasets:
        print(f"\n{'='*60}")
        print(f"Processing Dataset {dset_num}")
        print(f"{'='*60}")
        
        # Load pickle data
        ccf_data = processor.load_pickle_data(dset_num)
        
        if not ccf_data:
            print(f"No data found for DS{dset_num}")
            continue
        
        # Plot loaded data for verification
        processor.plot_ccf_data(ccf_data, dset_num)
        
        # Apply FIESTA and create DataFrames
        dataframes = processor.apply_fiesta(ccf_data, dset_num, k_max)
        
        if dataframes:
            # Detect outliers
            outlier_info = processor.detect_outliers(dataframes, dset_num)
            
            # Plot time series with outliers marked
            processor.plot_time_series_with_outliers(dataframes, outlier_info, dset_num)
            
            # Save .dat files without outliers
            processor.save_dat_files_without_outliers(dataframes, outlier_info, dset_num)
        
        print(f"Completed DS{dset_num}")
    
    print(f"\n{'='*60}")
    print("FIESTA Analysis Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
