import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
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
    
    def load_pickle_data(self, dset_num):
        """Load inverse CCF data from pickle files"""
        print(f"\nLoading pickle data for DS{dset_num}...")
        data = {}
        
        for inst in self.instruments:
            filename = f"DS{dset_num:02d}_invCCF_{inst}.pkl"
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
        """Plot loaded CCF data for verification"""
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
                                alpha=0.3, color=colors[i], label='±1σ')
            
            axes[i].set_title(f'{inst.upper()} (n={inst_data["n_files"]})')
            axes[i].set_xlabel('Velocity [km/s]')
            axes[i].set_ylabel('Inverse Normalized CCF')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(self.output_dir, f'DS{dset_num}_loaded_ccfs.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved CCF verification plot: {filename}")
    
    def apply_fiesta(self, data, dset_num, k_max=5):
        """Apply FIESTA to loaded CCF data"""
        try:
            from FIESTA_II import FIESTA
        except ImportError:
            print("Error: FIESTA_II module not found. Please install FIESTA first.")
            return None
        
        print(f"\nApplying FIESTA with k_max={k_max}...")
        results = {}
        
        for inst in self.instruments:
            if inst not in data:
                continue
            
            print(f"  Processing {inst.upper()}...")
            inst_data = data[inst]
            
            # Extract data
            inv_ccf_array = inst_data['inv_ccf']  # Shape: (n_files, n_vel_points)
            error_array = inst_data['eccf']
            v_grid = inst_data['v_grid'][0]  # Assuming all same
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
                    
                    # Calculate differential RVs
                    if np.std(RV_gauss_ms) < 1e-3:  # Flat RV_gauss
                        print(f"    Warning: RV_gauss is flat, using first mode as reference")
                        RV_reference = RV_FT_k[0, :].copy()
                        ΔRV_k = np.zeros((RV_FT_k.shape[0]-1, RV_FT_k.shape[1]))
                        for k in range(1, RV_FT_k.shape[0]):
                            ΔRV_k[k-1, :] = RV_FT_k[k, :] - RV_reference
                    else:
                        RV_reference = RV_gauss_ms.copy()
                        ΔRV_k = np.zeros(RV_FT_k.shape)
                        for k in range(RV_FT_k.shape[0]):
                            ΔRV_k[k, :] = RV_FT_k[k, :] - RV_reference
                    
                    # Store results
                    results[inst] = {
                        'emjd_times': emjd_times,
                        'V_grid': v_grid,
                        'RV_FT_k': RV_FT_k,
                        'eRV_FT_k': eRV_FT_k,
                        'A_k': A_k,
                        'RV_gauss': RV_reference,
                        'ΔRV_k': ΔRV_k,
                        'n_files': n_files,
                        'k_max': ΔRV_k.shape[0]
                    }
                    
                    print(f"    Success! {n_files} CCFs, {ΔRV_k.shape[0]} modes")
                    print(f"    RV range: {np.min(RV_reference):.1f} to {np.max(RV_reference):.1f} m/s")
                
                else:
                    print(f"    Warning: FIESTA returned {len(result)} values (expected 6)")
                    
            except Exception as e:
                print(f"    Error applying FIESTA: {e}")
                continue
        
        return results
    
    def periodogram(self, ax, time, data, vlines=None):
        """Calculate and plot periodogram"""
        data_centered = data - np.mean(data)
        frequencies = np.logspace(-2, 2, 1000)
        periods = 1.0 / frequencies
        
        try:
            power = signal.lombscargle(time, data_centered, frequencies, normalize=True)
            ax.semilogx(periods, power, 'k-', alpha=0.7, linewidth=1)
            
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
    
    def plot_fiesta_results(self, results, dset_num):
        """Plot FIESTA results"""
        if not results:
            return
        
        print(f"Creating FIESTA plots for DS{dset_num}...")
        
        # Combine all instruments
        all_times, all_rv_gauss, all_delta_rvs = [], [], []
        
        for inst in results:
            data = results[inst]
            times = data['emjd_times'] - np.min(data['emjd_times'])
            rv_gauss = data['RV_gauss']
            delta_rvs = data['ΔRV_k']
            
            all_times.extend(times)
            all_rv_gauss.extend(rv_gauss)
            
            if len(all_delta_rvs) == 0:
                all_delta_rvs = [[] for _ in range(delta_rvs.shape[0])]
            
            for k in range(delta_rvs.shape[0]):
                all_delta_rvs[k].extend(delta_rvs[k, :])
        
        # Convert to arrays
        all_times = np.array(all_times)
        all_rv_gauss = np.array(all_rv_gauss)
        all_delta_rvs = [np.array(drv) for drv in all_delta_rvs]
        
        k_mode = len(all_delta_rvs)
        N_file = len(all_times)
        
        # Setup plot
        alpha1 = 0.2
        plt.rcParams.update({'font.size': 10})
        widths = [7, 1, 7]
        heights = [1] * (k_mode + 1)
        gs_kw = dict(width_ratios=widths, height_ratios=heights)
        
        fig, axes = plt.subplots(figsize=(10, k_mode + 1), ncols=3, nrows=k_mode + 1, 
                                constrained_layout=True, gridspec_kw=gs_kw)
        
        # Import for R² calculation
        try:
            from sklearn.linear_model import LinearRegression
        except ImportError:
            LinearRegression = None
        
        # Plot each row
        for r in range(k_mode + 1):
            for c in range(3):
                ax = axes[r, c]
                
                # Time-series
                if c == 0:
                    if r == 0:
                        ax.plot(all_times, all_rv_gauss, 'k.', alpha=alpha1)
                        ax.set_title('Time-series')
                        ax.set_ylabel('$RV_{gauss}$')
                    else:
                        ax.plot(all_times, all_delta_rvs[r-1], 'k.', alpha=alpha1)
                        ax.set_ylabel(r'$\Delta$RV$_{%d}$' % r)
                    
                    if r == k_mode:
                        ax.set_xlabel('Time [d]')
                    else:
                        ax.tick_params(labelbottom=False)
                
                # R² correlation
                elif c == 1:
                    if LinearRegression is not None:
                        if r == 0:
                            reg = LinearRegression().fit(all_rv_gauss.reshape(-1, 1), 
                                                       all_rv_gauss.reshape(-1, 1))
                            score = reg.score(all_rv_gauss.reshape(-1, 1), 
                                            all_rv_gauss.reshape(-1, 1))
                        else:
                            reg = LinearRegression().fit(all_rv_gauss.reshape(-1, 1), 
                                                       all_delta_rvs[r-1].reshape(-1, 1))
                            score = reg.score(all_rv_gauss.reshape(-1, 1), 
                                            all_delta_rvs[r-1].reshape(-1, 1))
                        
                        adjust_R2 = 1 - (1 - score) * (N_file - 1) / (N_file - 2)
                        ax.set_title(r'$\bar{R}^2$' + f' = {adjust_R2:.2f}')
                        
                        if r == 0:
                            ax.plot(all_rv_gauss, all_rv_gauss, 'k.', alpha=alpha1)
                        else:
                            ax.plot(all_rv_gauss, all_delta_rvs[r-1], 'k.', alpha=alpha1)
                    else:
                        ax.text(0.5, 0.5, 'R² calc\nunavailable', ha='center', va='center',
                               transform=ax.transAxes)
                    
                    if r == k_mode:
                        ax.set_xlabel('$RV_{gauss}$')
                    else:
                        ax.tick_params(labelbottom=False)
                    ax.yaxis.tick_right()
                
                # Periodograms
                elif c == 2:
                    if r == 0:
                        self.periodogram(ax, all_times, all_rv_gauss, vlines=[8.0, 14.6, 26.6])
                        ax.set_title('Periodogram')
                    else:
                        self.periodogram(ax, all_times, all_delta_rvs[r-1], vlines=[8.0, 14.6, 26.6])
                    
                    if r == k_mode:
                        ax.set_xlabel('Period [days]')
                    else:
                        ax.tick_params(labelbottom=False)
        
        fig.align_ylabels(axes[:, 0])
        
        filename = os.path.join(self.output_dir, f'DS{dset_num}_FIESTA_style.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
    
    def save_dat_files(self, results, dset_num):
        """Save FIESTA results to .dat files"""
        if not results:
            return
        
        print(f"\nSaving .dat files for DS{dset_num}...")
        os.makedirs(self.dat_output_dir, exist_ok=True)
        
        for inst in results:
            data = results[inst]
            n_obs = data['n_files']
            k_max = data['k_max']
            emjd_times = data['emjd_times']
            
            print(f"  Processing {inst.upper()}...")
            
            # Save RV_gauss
            filename_rv = os.path.join(self.dat_output_dir, f'DS{dset_num}_{inst}_fiesta_rv_gauss.dat')
            rv_gauss = data['RV_gauss']
            
            if 'eRV_FT_k' in data and data['eRV_FT_k'].shape[0] > 0:
                rv_uncertainties = data['eRV_FT_k'][0, :]
            else:
                rv_uncertainties = np.full(n_obs, np.std(rv_gauss) * 0.1)
            
            jitter = np.zeros(n_obs)
            offset = np.zeros(n_obs)
            flag = np.full(n_obs, -1, dtype=int)
            
            rv_data = np.column_stack([emjd_times, rv_gauss, rv_uncertainties, jitter, offset, flag])
            np.savetxt(filename_rv, rv_data, fmt=['%.6f', '%.6f', '%.6f', '%d', '%d', '%d'])
            print(f"    Saved: {filename_rv}")
            
            # Save differential modes
            for k in range(k_max):
                filename = os.path.join(self.dat_output_dir, f'DS{dset_num}_{inst}_fiesta_mode{k+1}.dat')
                delta_rv = data['ΔRV_k'][k, :]
                
                if 'eRV_FT_k' in data and k < data['eRV_FT_k'].shape[0]:
                    uncertainties = data['eRV_FT_k'][k, :]
                else:
                    uncertainties = np.full(n_obs, np.std(delta_rv))
                
                output_data = np.column_stack([emjd_times, delta_rv, uncertainties, jitter, offset, flag])
                np.savetxt(filename, output_data, fmt=['%.6f', '%.6f', '%.6f', '%d', '%d', '%d'])
                print(f"    Saved: {filename}")

def main():
    # Configuration
    ccf_data_dir = '/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/data'
    output_dir = '/work2/lbuc/iara/GitHub/ESSP/Figures/FIESTA_figures'
    dataset = None  # Specific dataset or None for all
    k_max = 5
    
    print(f"FIESTA Analysis from Pickle Files")
    print(f"CCF data directory: {ccf_data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"FIESTA modes: {k_max}")
    
    # Initialize processor
    processor = FIESTAProcessor(ccf_data_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get available datasets from pickle files
    if dataset:
        datasets = [dataset]
    else:
        # Find available datasets from pickle files
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
        
        # Apply FIESTA
        fiesta_results = processor.apply_fiesta(ccf_data, dset_num, k_max)
        
        if fiesta_results:
            # Plot results
            processor.plot_fiesta_results(fiesta_results, dset_num)
            
            # Save .dat files
            processor.save_dat_files(fiesta_results, dset_num)
        
        print(f"Completed DS{dset_num}")
    
    print(f"\n{'='*60}")
    print("FIESTA Analysis Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
