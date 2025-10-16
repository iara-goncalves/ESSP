import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

def load_fiesta_dat_files(data_dir):
    """Load FIESTA data from .dat files (rv_gauss and modes) - CORRECTED VERSION"""
    
    all_data = []
    
    # Look for FIESTA .dat files
    dat_files = [f for f in os.listdir(data_dir) 
                 if f.startswith('DS') and 'fiesta' in f and f.endswith('.dat')]
    dat_files.sort()
    
    if not dat_files:
        raise FileNotFoundError(f"No FIESTA .dat files found in {data_dir}")
    
    print(f"Found {len(dat_files)} FIESTA files")
    
    # Process each file individually (don't merge by dataset yet)
    for filename in dat_files:
        filepath = os.path.join(data_dir, filename)
        
        # Parse filename: DS{num}_{inst}_fiesta_{type}.dat
        parts = filename.replace('.dat', '').split('_')
        ds_name = parts[0]  # DS1, DS2, etc.
        instrument = parts[1]  # harps, harpsn, expres, neid
        fiesta_type = '_'.join(parts[3:])  # rv_gauss, mode1, mode2, etc.
        
        try:
            # Read: time, value, error, jitter_flag, offset_flag, subset_flag
            data = np.loadtxt(filepath)
            
            if len(data) == 0:
                continue
            
            # Create DataFrame with unique column names
            df = pd.DataFrame({
                'Time [eMJD]': data[:, 0],
                'Instrument': instrument,
                'Dataset': ds_name
            })
            
            # Add measurement-specific columns
            if fiesta_type == 'rv_gauss':
                df['RV_gauss [m/s]'] = data[:, 1]
                df['RV_gauss Err. [m/s]'] = data[:, 2]
            elif fiesta_type.startswith('mode'):
                mode_num = fiesta_type.replace('mode', '')
                df[f'δRV_{mode_num} [m/s]'] = data[:, 1]
                df[f'δRV_{mode_num} Err. [m/s]'] = data[:, 2]
            
            all_data.append(df)
            print(f"  Loaded {filename}: {len(df)} points")
            
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No data loaded!")
    
    # Concatenate all data (keeping all rows from all instruments)
    df_all = pd.concat(all_data, ignore_index=True)
    
    # Group by Dataset, Instrument, and Time to combine different modes for same observation
    # This handles the case where same time/instrument has multiple modes
    group_cols = ['Dataset', 'Instrument', 'Time [eMJD]']
    
    # Get all value columns (not the grouping columns)
    value_cols = [col for col in df_all.columns if col not in group_cols]
    
    # Aggregate: for each unique (Dataset, Instrument, Time), take the first non-NaN value
    df_all = df_all.groupby(group_cols, as_index=False).first()
    
    print(f"\nTotal: {len(df_all)} unique observations from {df_all['Dataset'].nunique()} datasets")
    print(f"Instruments: {sorted(df_all['Instrument'].unique())}")
    
    return df_all

def plot_fiesta_with_periodograms(ds, ds_df, fig_dir):
    """
    Plot FIESTA indicators with Lomb-Scargle periodograms.
    Left column: Time series, Right column: Periodograms
    """
    
    # Identify available FIESTA columns
    fiesta_columns = []
    
    # Check for RV_gauss
    if 'RV_gauss [m/s]' in ds_df.columns:
        fiesta_columns.append(('RV_gauss [m/s]', 'RV_gauss Err. [m/s]', 'RV_gauss [m/s]'))
    
    # Check for modes
    mode_nums = []
    for col in ds_df.columns:
        if col.startswith('δRV_') and col.endswith('[m/s]') and 'Err' not in col:
            mode_num = col.split('_')[1].split()[0]
            mode_nums.append(mode_num)
    
    for mode_num in sorted(mode_nums):
        fiesta_columns.append((
            f'δRV_{mode_num} [m/s]',
            f'δRV_{mode_num} Err. [m/s]',
            f'δRV_{mode_num} [m/s]'
        ))
    
    if not fiesta_columns:
        print(f"No FIESTA columns found for {ds}")
        return
    
    n_plots = len(fiesta_columns)
    fig, axes = plt.subplots(n_plots, 2, figsize=(18, 3*n_plots))
    
    # Handle single row case
    if n_plots == 1:
        axes = axes.reshape(1, -1)
    
    # Get all unique instruments and assign colors
    all_instruments = sorted(ds_df["Instrument"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_instruments)))
    color_map = dict(zip(all_instruments, colors))
    
    for i, (col, yerr_col, ylabel) in enumerate(fiesta_columns):
        ax_data = axes[i, 0]      # Left column for data
        ax_period = axes[i, 1]    # Right column for periodogram
        
        if col not in ds_df.columns:
            ax_data.text(0.5, 0.5, f"Column '{col}' not found", 
                        ha='center', va='center', transform=ax_data.transAxes)
            ax_period.text(0.5, 0.5, f"No data for '{col}'", 
                          ha='center', va='center', transform=ax_period.transAxes)
            ax_data.set_ylabel(ylabel)
            continue
        
        # === LEFT SIDE: DATA PLOT ===
        # Calculate instrument-specific medians for centering
        inst_medians = {}
        for inst in all_instruments:
            inst_data = ds_df[ds_df["Instrument"] == inst][col].dropna()
            if len(inst_data) > 0:
                inst_medians[inst] = inst_data.median()
        
        # Plot each instrument
        for inst in all_instruments:
            inst_data = ds_df[ds_df["Instrument"] == inst].copy()
            if inst_data.empty:
                continue
            
            # Remove NaN values for this column
            inst_data = inst_data.dropna(subset=[col])
            if inst_data.empty:
                continue
            
            # Center data by instrument median
            if inst in inst_medians:
                center = inst_medians[inst]
                y_centered = inst_data[col] - center
            else:
                y_centered = inst_data[col]
            
            # Handle error bars
            if yerr_col in inst_data.columns:
                error_values = inst_data[yerr_col]
            else:
                error_values = np.std(inst_data[col]) * 0.1 if len(inst_data) > 1 else 1.0
            
            ax_data.errorbar(inst_data["Time [eMJD]"], y_centered, 
                           yerr=error_values, fmt=".", 
                           color=color_map.get(inst, 'gray'), label=inst.upper(), 
                           alpha=0.8, markersize=6)
        
        ax_data.set_ylabel(f"{ylabel} - median")
        ax_data.grid(True, alpha=0.3)
        
        # === RIGHT SIDE: LOMB-SCARGLE PERIODOGRAM ===
        try:
            # Prepare data for periodogram (all instruments combined)
            valid_data = ds_df.dropna(subset=[col])
            
            if not valid_data.empty and col in valid_data.columns:
                time_data = valid_data["Time [eMJD]"].values
                y_data = valid_data[col].values
                
                # Handle error values
                if yerr_col in valid_data.columns:
                    dy_data = valid_data[yerr_col].values
                else:
                    dy_data = np.full_like(y_data, np.std(y_data) * 0.1 if len(y_data) > 1 else 1.0)
                
                # Additional NaN check
                mask = ~(np.isnan(time_data) | np.isnan(y_data) | np.isnan(dy_data))
                t = time_data[mask]
                y = y_data[mask]
                dy = dy_data[mask]
                
                if len(t) > 5:  # Need sufficient points
                    # Fix non-positive errors
                    m = dy > 0
                    if not m.all():
                        repl = np.median(dy[m]) if m.any() else 1.0
                        dy[~m] = repl
                    
                    span = t.max() - t.min()
                    if span > 1:  # Need reasonable time span
                        # Define period range
                        min_period = 1.1
                        max_period = max(2.0, 0.8 * span)
                        f_min = 1.0 / max_period
                        f_max = 1.0 / min_period
                        
                        # Create frequency grid
                        N = 15000
                        freq = np.linspace(f_min, f_max, N)
                        
                        # Compute Lomb-Scargle periodogram
                        ls = LombScargle(t, y, dy)
                        power = ls.power(freq)
                        
                        # === FALSE ALARM PROBABILITY LEVELS ===
                        fap_levels = [0.001, 0.01, 0.1]  # 0.1%, 1%, 10%
                        fap_colors = ['red', 'orange', 'green']
                        fap_labels = ['0.1%', '1%', '10%']
                        fap_powers = []
                        
                        try:
                            for fap in fap_levels:
                                fap_power = ls.false_alarm_level(fap, method='baluev')
                                fap_powers.append(fap_power)
                            print(f"FAP levels for {ds}-{col}: 0.1%={fap_powers[0]:.3f}, 1%={fap_powers[1]:.3f}, 10%={fap_powers[2]:.3f}")
                        except Exception as fap_error:
                            print(f"Warning: Could not compute FAP levels for {ds} - {col}: {fap_error}")
                            fap_powers = []
                        
                        # === PEAK DETECTION ===
                        power_threshold = np.percentile(power, 85)
                        max_power = np.max(power)
                        
                        if power_threshold > 0.8 * max_power:
                            power_threshold = 0.5 * max_power
                        
                        print(f"Peak detection threshold for {col}: {power_threshold:.4f} (max: {max_power:.4f})")
                        
                        # Minimum peak separation
                        min_period_separation = 0.01
                        min_freq_separation = int(len(freq) * min_period_separation / np.log10(f_max/f_min))
                        
                        # Find peaks
                        try:
                            peak_indices, peak_properties = find_peaks(
                                power, 
                                height=float(power_threshold),
                                distance=max(1, min_freq_separation),
                                prominence=float(power_threshold) * 0.1
                            )
                            
                            # If no peaks found, be more lenient
                            if len(peak_indices) == 0:
                                lower_threshold = np.percentile(power, 70)
                                peak_indices, peak_properties = find_peaks(
                                    power, 
                                    height=float(lower_threshold),
                                    distance=max(1, min_freq_separation),
                                    prominence=float(lower_threshold) * 0.05
                                )
                                print(f"Using lenient threshold: {lower_threshold:.4f}")
                                
                        except Exception as peak_error:
                            print(f"Peak detection error in {ds} - {col}: {peak_error}")
                            peak_indices = [np.argmax(power)] if len(power) > 0 else []
                            peak_indices = np.array(peak_indices)
                        
                        print(f"Found {len(peak_indices)} peaks for {col}")
                        
                        # === PROCESS DETECTED PEAKS ===
                        if len(peak_indices) > 0:
                            peak_periods = 1.0 / freq[peak_indices]
                            peak_powers_raw = power[peak_indices]
                            
                            # Sort by power (highest first)
                            sorted_indices = np.argsort(peak_powers_raw)[::-1]
                            peak_periods = peak_periods[sorted_indices][:3]  # Top 3
                            peak_powers_sorted = peak_powers_raw[sorted_indices][:3]
                            
                            # Calculate FAP for each peak
                            peak_faps = []
                            peak_significance = []
                            
                            for peak_power in peak_powers_sorted:
                                try:
                                    peak_fap = ls.false_alarm_probability(peak_power, method='baluev')
                                    peak_faps.append(peak_fap)
                                    
                                    # Determine significance
                                    if len(fap_powers) >= 3:
                                        if peak_power >= fap_powers[0]:
                                            significance = "highly significant"
                                        elif peak_power >= fap_powers[1]:
                                            significance = "significant"
                                        elif peak_power >= fap_powers[2]:
                                            significance = "marginally significant"
                                        else:
                                            significance = "not significant"
                                    else:
                                        significance = "unknown"
                                    peak_significance.append(significance)
                                    
                                except Exception as e:
                                    print(f"Could not calculate FAP for peak: {e}")
                                    peak_faps.append(np.nan)
                                    peak_significance.append("unknown")
                            
                            print(f"Detected peaks for {col}:")
                            for j, (period, power_val, fap, sig) in enumerate(zip(peak_periods, peak_powers_sorted, peak_faps, peak_significance)):
                                ordinal = ['1st', '2nd', '3rd'][j]
                                if not np.isnan(fap):
                                    print(f"  {ordinal}: {period:.2f}d, Power={power_val:.4f}, FAP={fap:.2e} ({sig})")
                                else:
                                    print(f"  {ordinal}: {period:.2f}d, Power={power_val:.4f}")
                                    
                        else:
                            peak_periods = []
                            peak_powers_sorted = []
                            peak_faps = []
                            peak_significance = []
                            print(f"No peaks detected for {col}")
                        
                        # Convert to periods for plotting
                        periods = 1.0 / freq
                        
                        # Plot periodogram
                        ax_period.semilogx(periods, power, 'k-', linewidth=1)
                        
                        # Plot FAP reference lines
                        for fap_power, fap_color, fap_label in zip(fap_powers, fap_colors, fap_labels):
                            ax_period.axhline(fap_power, color=fap_color, linestyle='--', 
                                            alpha=0.7, linewidth=1.5, 
                                            label=f'{fap_label} FAP')
                        
                        # Plot detected peaks
                        peak_colors = ['purple', 'blue', 'cyan']
                        peak_styles = ['-', '--', ':']
                        
                        for j, (peak_period, peak_power, peak_fap, significance) in enumerate(zip(peak_periods, peak_powers_sorted, peak_faps, peak_significance)):
                            if j < len(peak_colors):
                                ordinal = ['1st', '2nd', '3rd'][j]
                                if not np.isnan(peak_fap):
                                    label = f'{ordinal}: {peak_period:.1f}d (FAP={peak_fap:.1e})'
                                else:
                                    label = f'{ordinal}: {peak_period:.1f}d'
                                
                                ax_period.axvline(peak_period, color=peak_colors[j], 
                                                ls=peak_styles[j], lw=2, alpha=0.8,
                                                label=label)
                        
                        # Add reference period lines
                        reference_periods = [1, 7, 14, 28, 100, 365]
                        for ref_period in reference_periods:
                            if min_period <= ref_period <= max_period:
                                ax_period.axvline(ref_period, color='blue', alpha=0.3, 
                                                linestyle=':', linewidth=1)
                                ax_period.text(ref_period, ax_period.get_ylim()[1]*0.85, 
                                             f'{ref_period}d', rotation=90, ha='right', 
                                             va='top', fontsize=8, alpha=0.6, color='blue')
                        
                        ax_period.set_xlabel("Period [days]")
                        ax_period.set_ylabel("LS Power")
                        ax_period.set_title(f"{ylabel} Periodogram", fontsize=10)
                        ax_period.grid(True, alpha=0.3)
                        
                        # Add legend
                        ax_period.legend(fontsize=7, loc='upper right')
                        
                        # Add statistics text
                        if len(peak_periods) > 0:
                            peak_info = []
                            for j, (p, pow, fap) in enumerate(zip(peak_periods, peak_powers_sorted, peak_faps)):
                                if not np.isnan(fap):
                                    peak_info.append(f'Peak {j+1}: {p:.1f}d (FAP={fap:.1e})')
                                else:
                                    peak_info.append(f'Peak {j+1}: {p:.1f}d (P={pow:.3f})')
                            peak_text = '\n'.join(peak_info)
                            stats_text = f'N={len(t)} points\nSpan={span:.1f}d\n{peak_text}'
                        else:
                            stats_text = f'N={len(t)} points\nSpan={span:.1f}d\nNo significant peaks'
                        
                        ax_period.text(0.02, 0.75, stats_text, 
                                     transform=ax_period.transAxes, fontsize=7, 
                                     verticalalignment='top', alpha=0.9,
                                     bbox=dict(boxstyle="round,pad=0.4", facecolor="white", 
                                             alpha=0.9, edgecolor='gray', linewidth=0.5))
                        
                    else:
                        ax_period.text(0.5, 0.5, f"Insufficient time span\n({span:.1f} days)", 
                                     ha='center', va='center', transform=ax_period.transAxes)
                else:
                    ax_period.text(0.5, 0.5, f"Insufficient data\n({len(t)} points)", 
                                 ha='center', va='center', transform=ax_period.transAxes)
            else:
                ax_period.text(0.5, 0.5, "No data available", 
                             ha='center', va='center', transform=ax_period.transAxes)
                
        except Exception as e:
            ax_period.text(0.5, 0.5, f"Error computing\nperiodogram:\n{str(e)[:50]}...", 
                         ha='center', va='center', transform=ax_period.transAxes, fontsize=8)
            print(f"Error in {ds} - {col}: {str(e)}")
    
    # Set x-labels for bottom row
    axes[-1, 0].set_xlabel("Time [eMJD]")
    axes[-1, 1].set_xlabel("Period [days]")
    
    # Add legend to first data plot
    if all_instruments:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[0, 0].legend(by_label.values(), by_label.keys(), loc="best", fontsize=8)
    
    # Set overall title
    fig.suptitle(f"{ds} - FIESTA Indicators & Lomb-Scargle Periodograms", 
                 fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    fig_path = os.path.join(fig_dir, f"{ds}_activity_LS_periodograms_FIESTA.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

def main():
    """Main function"""
    
    data_dir = "/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/data"
    fig_dir = "/work2/lbuc/iara/GitHub/ESSP/Figures/FIESTA_figures"
    
    os.makedirs(fig_dir, exist_ok=True)
    
    print("="*60)
    print("FIESTA Lomb-Scargle Periodogram Analysis")
    print("="*60)
    
    # Load FIESTA data
    df_all = load_fiesta_dat_files(data_dir)
    
    print("\nCreating periodogram plots...")
    for ds, ds_df in df_all.groupby("Dataset"):
        print(f"\n{ds}:")
        plot_fiesta_with_periodograms(ds, ds_df, fig_dir)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)

if __name__ == "__main__":
    main()
