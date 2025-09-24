import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks

def load_processed_data(data_dir):
    """Load the processed dataframe from essp4_data.py"""
    pkl_path = os.path.join(data_dir, "processed_data.pkl")
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Processed data not found at {pkl_path}. Run essp4_data.py first.")
    
    df_all = pd.read_pickle(pkl_path)
    print(f"Loaded processed data: {len(df_all)} total points")
    print(f"Outliers: {df_all['is_outlier'].sum()} ({df_all['is_outlier'].sum()/len(df_all)*100:.2f}%)")
    
    return df_all

def compute_periodogram(t, y, dy, min_period=1.1, max_period_factor=0.8):
    """Compute Lomb-Scargle periodogram and find significant peaks."""
    
    if len(t) <= 5:
        return None, None, []
    
    # Fix non-positive errors
    m = dy > 0
    if not m.all():
        repl = np.median(dy[m]) if m.any() else 1.0
        dy[~m] = repl
    
    span = t.max() - t.min()
    if span <= 1:
        return None, None, []
    
    # Define frequency range
    max_period = max(2.0, max_period_factor * span)
    f_min = 1.0 / max_period
    f_max = 1.0 / min_period
    
    # Create Lomb-Scargle object and compute periodogram
    ls = LombScargle(t, y, dy)
    freq, power = ls.autopower(minimum_frequency=f_min, maximum_frequency=f_max, samples_per_peak=5)
    
    # Use simple power threshold instead of FAP
    power_threshold = np.percentile(power, 90)  # Top 10% threshold
    min_freq_separation = max(1, int(len(freq) * 0.01))
    
    peak_indices, _ = find_peaks(
        power, 
        height=power_threshold,
        distance=min_freq_separation,
        prominence=power_threshold * 0.1
    )
    
    # Get top 3 peaks by power
    peaks = []
    if len(peak_indices) > 0:
        peak_powers = power[peak_indices]
        sorted_order = np.argsort(peak_powers)[::-1]
        
        for i in sorted_order[:3]:
            idx = peak_indices[i]
            peaks.append({
                'period': 1.0 / freq[idx],
                'power': power[idx],
                'freq': freq[idx]
            })
    
    return freq, power, peaks

def plot_activity_with_periodograms(ds, ds_df, fig_dir):
    """
    Plot activity indicators with Lomb-Scargle periodograms using only inliers.
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
    
    # Use only inliers
    inliers = ds_df[~ds_df.is_outlier].copy()
    
    # Get all unique instruments and assign colors
    all_instruments = sorted(inliers["Instrument"].unique()) if not inliers.empty else []
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(all_instruments))))
    color_map = dict(zip(all_instruments, colors))
    
    for i, (col, yerr, ylabel) in enumerate(plot_info):
        ax_data = axes[i, 0]      # Left column for data
        ax_period = axes[i, 1]    # Right column for periodogram
        
        if col not in inliers.columns or inliers.empty:
            ax_data.text(0.5, 0.5, f"Column '{col}' not found or no inliers", 
                        ha='center', va='center', transform=ax_data.transAxes)
            ax_period.text(0.5, 0.5, f"No data for '{col}'", 
                          ha='center', va='center', transform=ax_period.transAxes)
            ax_data.set_ylabel(ylabel)
            ax_period.set_ylabel("LS Power")
            continue
        
        # === LEFT SIDE: DATA PLOT ===
        try:
            # Calculate instrument-specific medians for centering
            inst_medians = inliers.groupby("Instrument")[col].median()
            
            # Plot each instrument
            for inst in all_instruments:
                inst_data = inliers[inliers["Instrument"] == inst]
                if inst_data.empty:
                    continue
                    
                # Center data by instrument median
                center = inst_medians[inst]
                y_centered = inst_data[col] - center
                
                # Remove any NaN values from the plotting data
                time_vals = inst_data["Time [eMJD]"].values
                y_vals = y_centered.values
                
                # Create mask for finite values
                finite_mask = np.isfinite(time_vals) & np.isfinite(y_vals)
                
                if not np.any(finite_mask):
                    continue
                
                time_clean = time_vals[finite_mask]
                y_clean = y_vals[finite_mask]
                
                # Handle error bars - FIXED VERSION
                if isinstance(yerr, str) and yerr in inst_data.columns:
                    error_values = inst_data[yerr].values[finite_mask]  # Apply same mask
                    # Ensure error values are positive and finite
                    error_values = np.abs(error_values)
                    error_values[~np.isfinite(error_values)] = np.nanmedian(error_values[np.isfinite(error_values)])
                    if np.isnan(np.nanmedian(error_values)):  # If all errors are NaN
                        error_values = np.full_like(error_values, 1.0)
                else:
                    error_values = np.full(len(time_clean), float(yerr))
                
                # Final check: ensure all arrays have the same length
                min_len = min(len(time_clean), len(y_clean), len(error_values))
                time_clean = time_clean[:min_len]
                y_clean = y_clean[:min_len]
                error_values = error_values[:min_len]
                
                # Plot with error bars
                ax_data.errorbar(time_clean, y_clean, 
                               yerr=error_values, fmt=".", 
                               color=color_map[inst], label=inst, 
                               alpha=0.8, markersize=6)
            
            ax_data.set_ylabel(f"{ylabel} - mean")
            ax_data.grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error plotting {ds} - {col}: {str(e)}")
            ax_data.text(0.5, 0.5, f"Plotting error:\n{str(e)[:50]}...", 
                        ha='center', va='center', transform=ax_data.transAxes, fontsize=8)
        
        # === RIGHT SIDE: LOMB-SCARGLE PERIODOGRAM ===
        try:
            if not inliers.empty and col in inliers.columns:
                # Prepare data for periodogram (all inliers, all instruments combined)
                time_data = inliers["Time [eMJD]"].values
                y_data = inliers[col].values
                
                # Handle error values
                if isinstance(yerr, str) and yerr in inliers.columns:
                    dy_data = inliers[yerr].values
                    dy_data = np.abs(dy_data)  # Ensure positive
                else:
                    dy_data = np.full_like(y_data, float(yerr))
                
                # Remove NaN values
                mask = np.isfinite(time_data) & np.isfinite(y_data) & np.isfinite(dy_data)
                t = time_data[mask]
                y = y_data[mask]
                dy = dy_data[mask]
                
                if len(t) > 5:  # Need sufficient points for periodogram
                    # Fix non-positive errors
                    dy = np.abs(dy)  # Ensure positive
                    dy[dy <= 0] = np.nanmedian(dy[dy > 0]) if np.any(dy > 0) else 1.0
                    dy[~np.isfinite(dy)] = 1.0
                    
                    span = t.max() - t.min()
                    if span > 1:  # Need reasonable time span
                        # Define period range
                        min_period = 1.1
                        max_period = max(2.0, 0.8 * span)
                        f_min = 1.0 / max_period
                        f_max = 1.0 / min_period
                        
                        # Create Lomb-Scargle object and compute periodogram
                        ls = LombScargle(t, y, dy)
                        freq, power = ls.autopower(minimum_frequency=f_min, maximum_frequency=f_max, samples_per_peak=5)
                        
                        # Calculate FAP levels using baluev method
                        try:
                            fap_levels = [0.1, 0.01, 0.001]  # 10%, 1%, 0.1%
                            fap_powers = []
                            
                            for fap in fap_levels:
                                try:
                                    fap_power = ls.false_alarm_level(fap, method='baluev')
                                    fap_powers.append(fap_power)
                                except Exception as fap_error:
                                    print(f"Warning: Baluev FAP calculation failed for {fap}: {fap_error}")
                                    # Fallback to percentile-based threshold
                                    fap_power = np.percentile(power, (1-fap)*100)
                                    fap_powers.append(fap_power)
                            
                        except Exception as e:
                            print(f"Warning: All FAP calculations failed: {e}")
                            # Use percentile-based thresholds as fallback
                            fap_levels = [0.1, 0.01, 0.001]
                            fap_powers = [np.percentile(power, 90), np.percentile(power, 99), np.percentile(power, 99.9)]
                        
                        # Find peaks using the most stringent FAP level
                        power_threshold = fap_powers[-1] if fap_powers else np.percentile(power, 99)
                        min_freq_separation = max(1, int(len(freq) * 0.01))
                        
                        # Find peaks
                        from scipy.signal import find_peaks
                        peak_indices, _ = find_peaks(
                            power, 
                            height=power_threshold,           # Minimum peak height
                            distance=min_freq_separation,     # Minimum distance between peaks
                            prominence=power_threshold * 0.1  # Peak must stand out from baseline
                        )
                        
                        # Convert to periods for plotting
                        periods = 1.0 / freq
                        
                        # Plot periodogram
                        ax_period.semilogx(periods, power, 'k-', linewidth=1)
                        
                        # Plot FAP levels
                        fap_colors = ['lightcoral', 'orange', 'red']
                        fap_labels = ['10%', '1%', '0.1%']
                        
                        for fap_power, fap_color, fap_label in zip(fap_powers, fap_colors, fap_labels):
                            ax_period.axhline(fap_power, color=fap_color, linestyle='--', 
                                            linewidth=1.5, alpha=0.8, label=f'FAP {fap_label}')
                        
                        # Plot significant peaks
                        if len(peak_indices) > 0:
                            peak_periods = 1.0 / freq[peak_indices]
                            peak_powers = power[peak_indices]
                            
                            # Sort by power and take top 3
                            sorted_indices = np.argsort(peak_powers)[::-1][:3]
                            top_periods = peak_periods[sorted_indices]
                            top_powers = peak_powers[sorted_indices]
                            
                            # Calculate FAP for each peak
                            peak_colors = ['red', 'orange', 'purple']
                            peak_styles = ['--', '-.', ':']
                            
                            for j, (peak_period, peak_power) in enumerate(zip(top_periods, top_powers)):
                                if j < len(peak_colors):
                                    try:
                                        peak_fap = ls.false_alarm_probability(peak_power, method='baluev')
                                    except:
                                        # Estimate FAP based on power percentile
                                        percentile = (power < peak_power).sum() / len(power) * 100
                                        peak_fap = max(0.001, (100 - percentile) / 100)
                                    
                                    ordinal = ['1st', '2nd', '3rd'][j]
                                    fap_str = f"{peak_fap:.1e}" if peak_fap < 0.001 else f"{peak_fap:.3f}"
                                    
                                    ax_period.axvline(peak_period, color=peak_colors[j], 
                                                    ls=peak_styles[j], lw=2, 
                                                    label=f'{ordinal}: {peak_period:.1f}d (FAP={fap_str})')
                            
                            print(f"{ds} - {col}: Found {len(peak_indices)} significant peaks")
                        else:
                            print(f"{ds} - {col}: No significant peaks found")
                        
                        ax_period.set_xlabel("Period [days]")
                        ax_period.set_ylabel("LS Power")
                        ax_period.set_title(f"{ylabel} Periodogram", fontsize=10)
                        ax_period.grid(True, alpha=0.3)
                        
                        # Add legend if there are peaks or FAP levels
                        if len(peak_indices) > 0 or fap_powers:
                            ax_period.legend(fontsize=7, loc='upper right')
                        
                        # Add statistics text
                        stats_text = f'N={len(t)}\nSpan={span:.1f}d'
                        ax_period.text(0.02, 0.02, stats_text, 
                                     transform=ax_period.transAxes, fontsize=8, 
                                     verticalalignment='bottom', alpha=0.9,
                                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", 
                                             alpha=0.8, edgecolor='gray', linewidth=0.5))
                        
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
            ax_period.text(0.5, 0.5, f"Periodogram error:\n{str(e)[:30]}...", 
                         ha='center', va='center', transform=ax_period.transAxes, fontsize=8)
            print(f"Error in periodogram {ds} - {col}: {str(e)}")
        
        # Set labels for periodogram subplot
        if i == len(plot_info) - 1:  # Last row
            ax_period.set_xlabel("Period [days]")
        if ax_period.get_ylabel() == "":  # If not set due to error
            ax_period.set_ylabel("LS Power")
    
    # Set x-labels for bottom row
    axes[-1, 0].set_xlabel("Time [eMJD]")
    
    # Add legend to first data plot only
    if all_instruments:
        try:
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:  # Only if there are actually handles
                by_label = dict(zip(labels, handles))
                axes[0, 0].legend(by_label.values(), by_label.keys(), loc="best", fontsize=8)
        except Exception as e:
            print(f"Warning: Could not add legend: {e}")
    
    # Set overall title
    fig.suptitle(f"{ds} - Activity Indicators & Lomb-Scargle Periodograms (Inliers Only)", 
                 fontsize=16)
    
    # Use subplots_adjust instead of tight_layout to avoid the rendering issue
    plt.subplots_adjust(left=0.08, bottom=0.06, right=0.95, top=0.94, wspace=0.25, hspace=0.35)
    
    # Save figure
    fig_path = os.path.join(fig_dir, f"{ds}_activity_LS_periodograms.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {fig_path}")


def main():
    """Main function to create periodogram figures."""
    
    # Configuration
    data_dir = "/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/data"
    fig_dir = "/work2/lbuc/iara/GitHub/ESSP/Figures"
    
    os.makedirs(fig_dir, exist_ok=True)
    
    # Load processed data
    df_all = load_processed_data(data_dir)
    
    # Create periodogram plots
    print("Creating activity plots with periodograms using inliers only...")
    for ds, ds_df in df_all.groupby("Dataset"):
        plot_activity_with_periodograms(ds, ds_df, fig_dir)
    
    print("\nAll figures saved successfully!")

if __name__ == "__main__":
    main()
