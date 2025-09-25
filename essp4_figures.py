import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

def load_dat_files(data_dir):
    """Load data directly from .dat files created by essp4_data.py"""
    
    all_data = []
    
    # Look for all DS*.dat files
    dat_files = [f for f in os.listdir(data_dir) if f.startswith('DS') and f.endswith('.dat')]
    dat_files.sort()
    
    if not dat_files:
        raise FileNotFoundError(f"No DS*.dat files found in {data_dir}")
    
    # Group files by dataset
    dataset_files = {}
    for filename in dat_files:
        ds_name = filename.split('_')[0]  # DS1, DS2, etc.
        if ds_name not in dataset_files:
            dataset_files[ds_name] = []
        dataset_files[ds_name].append(filename)
    
    # Process each dataset
    for ds_name in sorted(dataset_files.keys()):
        print(f"Processing {ds_name}...")
        ds_data = None
        
        for filename in sorted(dataset_files[ds_name]):
            filepath = os.path.join(data_dir, filename)
            measurement_type = filename.split('_')[1].replace('.dat', '')
            
            try:
                # Read: time, value, error, jitter_flag, offset_flag, subset_flag
                data = np.loadtxt(filepath)
                
                # Create base DataFrame
                df = pd.DataFrame({
                    'Time [eMJD]': data[:, 0],
                    'offset_flag': data[:, 4].astype(int),
                    'Dataset': ds_name
                })
                
                # Add measurement-specific columns
                if measurement_type == 'RV':
                    df['RV [m/s]'] = data[:, 1]
                    df['RV Err. [m/s]'] = data[:, 2]
                elif measurement_type == 'BIS':
                    df['BIS [m/s]'] = data[:, 1]
                    df['BIS Err. [m/s]'] = data[:, 2]
                elif measurement_type == 'FWHM':
                    df['CCF FWHM [m/s]'] = data[:, 1]
                    df['CCF FWHM Err. [m/s]'] = data[:, 2]
                elif measurement_type == 'Contrast':
                    df['CCF Contrast'] = data[:, 1]
                    df['CCF Contrast Err.'] = data[:, 2]
                elif measurement_type == 'Halpha':
                    df['H-alpha Emission'] = data[:, 1]
                    df['H-alpha Err.'] = data[:, 2]
                elif measurement_type == 'CaII':
                    df['CaII Emission'] = data[:, 1]
                    df['CaII Err.'] = data[:, 2]
                
                # Map offset_flag to instrument names
                instrument_map = {0: 'expres', 1: 'harps', 2: 'harpsn', 3: 'neid'}
                df['Instrument'] = df['offset_flag'].map(instrument_map).fillna('unknown')
                
                # Merge with existing dataset data
                if ds_data is None:
                    ds_data = df
                else:
                    ds_data = pd.merge(ds_data, df, on=['Time [eMJD]', 'offset_flag', 'Dataset', 'Instrument'], how='outer')
                
            except Exception as e:
                print(f"  Error loading {filepath}: {e}")
                continue
        
        if ds_data is not None:
            all_data.append(ds_data)
            print(f"  Loaded {len(ds_data)} points")
    
    # Combine all datasets
    df_all = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal: {len(df_all)} points from {len(all_data)} datasets")
    return df_all

def plot_activity_with_periodograms(ds, ds_df, fig_dir):
    """
    Plot activity indicators with Lomb-Scargle periodograms.
    Left column: Activity data, Right column: Periodograms
    """
    fig, axes = plt.subplots(6, 2, figsize=(18, 18))
    
    # Define plot information: (column_name, error_value, y_label)
    plot_info = [
        ("RV [m/s]", "RV Err. [m/s]", "RV [m/s]"),
        ("CCF Contrast", 130, "CCF Contrast"),
        ("CCF FWHM [m/s]", 5.0, "CCF FWHM [m/s]"),
        ("BIS [m/s]", 0.95, "BIS [m/s]"),
        ("H-alpha Emission", 0.001, "H-alpha Emission"),
        ("CaII Emission", 0.003, "CaII Emission"),
    ]
    
    # Get all unique instruments and assign colors
    all_instruments = sorted(ds_df["Instrument"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_instruments)))
    color_map = dict(zip(all_instruments, colors))
    
    for i, (col, yerr, ylabel) in enumerate(plot_info):
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
        inst_medians = ds_df.groupby("Instrument")[col].median()
        
        # Plot each instrument
        for inst in all_instruments:
            inst_data = ds_df[ds_df["Instrument"] == inst]
            if inst_data.empty:
                continue
                
            # Center data by instrument median
            center = inst_medians[inst]
            y_centered = inst_data[col] - center
            
            # Handle error bars
            if isinstance(yerr, str) and yerr in inst_data.columns:
                error_values = inst_data[yerr]
            else:
                error_values = yerr
            
            ax_data.errorbar(inst_data["Time [eMJD]"], y_centered, 
                           yerr=error_values, fmt=".", 
                           color=color_map[inst], label=inst, 
                           alpha=0.8, markersize=6)
        
        ax_data.set_ylabel(f"{ylabel} - mean")
        ax_data.grid(True, alpha=0.3)
        
        # === RIGHT SIDE: LOMB-SCARGLE PERIODOGRAM ===
        try:
            if not ds_df.empty and col in ds_df.columns:
                # Prepare data for periodogram (all data, all instruments combined)
                time_data = ds_df["Time [eMJD]"].values
                y_data = ds_df[col].values
                
                # Handle error values
                if isinstance(yerr, str) and yerr in ds_df.columns:
                    dy_data = ds_df[yerr].values
                else:
                    dy_data = np.full_like(y_data, yerr if isinstance(yerr, (int, float)) else 1.0)
                
                # Remove NaN values
                mask = ~(np.isnan(time_data) | np.isnan(y_data) | np.isnan(dy_data))
                t = time_data[mask]
                y = y_data[mask]
                dy = dy_data[mask]
                
                if len(t) > 5:  # Need sufficient points for periodogram
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
                        
                        # === FALSE ALARM PROBABILITY LEVELS (for reference lines) ===
                        fap_levels = [0.001, 0.01, 0.1]  # 0.1%, 1%, 10%
                        fap_colors = ['red', 'orange', 'green']
                        fap_labels = ['0.1%', '1%', '10%']
                        fap_powers = []
                        
                        try:
                            for fap in fap_levels:
                                fap_power = ls.false_alarm_level(fap, method='baluev')
                                fap_powers.append(fap_power)
                            print(f"FAP reference levels for {ds}-{col}: 0.1%={fap_powers[0]:.3f}, 1%={fap_powers[1]:.3f}, 10%={fap_powers[2]:.3f}")
                        except Exception as fap_error:
                            print(f"Warning: Could not compute FAP levels for {ds} - {col}: {fap_error}")
                            fap_powers = []
                        
                        # === PEAK DETECTION (Independent of FAP) ===
                        # Use percentile-based threshold for peak detection
                        power_threshold = np.percentile(power, 85)  # Detect peaks above 85th percentile
                        max_power = np.max(power)
                        
                        # Ensure we don't miss the highest peak
                        if power_threshold > 0.8 * max_power:
                            power_threshold = 0.5 * max_power
                        
                        print(f"Peak detection threshold: {power_threshold:.4f} (max power: {max_power:.4f})")
                        
                        # Convert minimum period separation to frequency space
                        min_period_separation = 0.01  # minimum separation in log10(period) space
                        min_freq_separation = int(len(freq) * min_period_separation / np.log10(f_max/f_min))
                        
                        # Find peaks with simple threshold
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
                                print(f"Using more lenient threshold: {lower_threshold:.4f}")
                                
                        except Exception as peak_error:
                            print(f"Peak detection error in {ds} - {col}: {peak_error}")
                            # Fallback: just find the maximum
                            peak_indices = [np.argmax(power)] if len(power) > 0 else []
                            peak_indices = np.array(peak_indices)
                        
                        print(f"Found {len(peak_indices)} peaks")
                        
                        # === PROCESS DETECTED PEAKS ===
                        if len(peak_indices) > 0:
                            peak_periods = 1.0 / freq[peak_indices]
                            peak_powers_raw = power[peak_indices]
                            
                            # Sort by power (highest first)
                            sorted_indices = np.argsort(peak_powers_raw)[::-1]
                            peak_periods = peak_periods[sorted_indices][:3]  # Top 3 peaks
                            peak_powers_sorted = peak_powers_raw[sorted_indices][:3]
                            
                            # Calculate FAP for each detected peak
                            peak_faps = []
                            peak_significance = []
                            
                            for peak_power in peak_powers_sorted:
                                try:
                                    peak_fap = ls.false_alarm_probability(peak_power, method='baluev')
                                    peak_faps.append(peak_fap)
                                    
                                    # Determine significance level
                                    if len(fap_powers) >= 3:
                                        if peak_power >= fap_powers[0]:  # 0.1% FAP
                                            significance = "highly significant"
                                        elif peak_power >= fap_powers[1]:  # 1% FAP
                                            significance = "significant"
                                        elif peak_power >= fap_powers[2]:  # 10% FAP
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
                            
                            print(f"Detected peaks:")
                            for j, (period, power_val, fap, sig) in enumerate(zip(peak_periods, peak_powers_sorted, peak_faps, peak_significance)):
                                ordinal = ['1st', '2nd', '3rd'][j]
                                if not np.isnan(fap):
                                    print(f"  {ordinal}: {period:.2f}d, Power={power_val:.4f}, FAP={fap:.2e} ({sig})")
                                else:
                                    print(f"  {ordinal}: {period:.2f}d, Power={power_val:.4f} (FAP unknown)")
                                    
                        else:
                            peak_periods = []
                            peak_powers_sorted = []
                            peak_faps = []
                            peak_significance = []
                            print("No peaks detected")
                        
                        # Convert to periods for plotting
                        periods = 1.0 / freq
                        
                        # Plot periodogram
                        ax_period.semilogx(periods, power, 'k-', linewidth=1)
                        
                        # Plot FALSE ALARM PROBABILITY reference lines
                        for fap_power, fap_color, fap_label in zip(fap_powers, fap_colors, fap_labels):
                            ax_period.axhline(fap_power, color=fap_color, linestyle='--', 
                                            alpha=0.7, linewidth=1.5, 
                                            label=f'{fap_label} FAP')
                        
                        # Plot detected peaks with their FAP values
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
                        
                        # Add reference lines for common periods
                        reference_periods = [1, 7, 14, 28, 100, 365]
                        for ref_period in reference_periods:
                            if min_period <= ref_period <= max_period:
                                ax_period.axvline(ref_period, color='blue', alpha=0.3, 
                                                linestyle=':', linewidth=1)
                                # Add label at top of plot
                                ax_period.text(ref_period, ax_period.get_ylim()[1]*0.85, 
                                             f'{ref_period}d', rotation=90, ha='right', 
                                             va='top', fontsize=8, alpha=0.6, color='blue')
                        
                        ax_period.set_xlabel("Period [days]")
                        ax_period.set_ylabel("LS Power")
                        ax_period.set_title(f"{ylabel} Periodogram", fontsize=10)
                        ax_period.grid(True, alpha=0.3)
                        
                        # Add legend (FAP lines + peaks)
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
                        
                        # Position the statistics box
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
    
    # Add legend to first data plot only
    if all_instruments:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        axes[0, 0].legend(by_label.values(), by_label.keys(), loc="best", fontsize=8)
    
    # Set overall title
    fig.suptitle(f"{ds} - Activity Indicators & Lomb-Scargle Periodograms", 
                 fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    fig_path = os.path.join(fig_dir, f"{ds}_activity_LS_periodograms.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

def main():
    """Main function"""
    
    data_dir = "/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/data"
    fig_dir = "/work2/lbuc/iara/GitHub/ESSP/Figures"
    
    os.makedirs(fig_dir, exist_ok=True)
    
    # Load data and create plots
    df_all = load_dat_files(data_dir)
    
    print("Creating periodogram plots...")
    for ds, ds_df in df_all.groupby("Dataset"):
        plot_activity_with_periodograms(ds, ds_df, fig_dir)
    
    print("Done!")

if __name__ == "__main__":
    main()