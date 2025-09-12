from glob import glob
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import errorbar
from astropy.timeseries import LombScargle
from scipy.signal import find_peaks
from scipy.signal import lombscargle

# Specify where all the data set folders are
essp_dir = "/work2/lbuc/data/ESSP4/ESSP4"

# Columns to test for outliers
cols_to_clip = [
    "RV [m/s]",
    "BIS [m/s]",
    "CCF FWHM [m/s]",
    "CCF Contrast",
    "H-alpha Emission",
    "CaII Emission"
]

sigma_threshold = 4

# Control whether .dat files exclude outliers
exclude_outliers_when_writing = True

# Output directories
outdir = "/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/data"
fig_dir = "/work2/lbuc/iara/GitHub/ESSP/Figures"
os.makedirs(outdir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# All data sets together

# Create a list to collect all dataframes
df_list = []

# Loop over dataset 1 to 9
for dset_num in range(1, 10):
    file_path = os.path.join(essp_dir, f'DS{dset_num}', f'DS{dset_num}_timeSeries.csv')
    if os.path.exists(file_path):
        df_tmp = pd.read_csv(file_path)
        df_tmp["Dataset"] = f"DS{dset_num}"

        # Convert FWHM from km/s → m/s
        if "CCF FWHM [km/s]" in df_tmp.columns:
            df_tmp["CCF FWHM [m/s]"] = df_tmp["CCF FWHM [km/s]"] * 1000.0
            # drop the old km/s column:
            df_tmp = df_tmp.drop(columns=["CCF FWHM [km/s]"])
        
        if "CCF FWHM Err. [km/s]" in df_tmp.columns:
            df_tmp["CCF FWHM Err. [m/s]"] = df_tmp["CCF FWHM Err. [km/s]"] * 1000.0
            # drop the old km/s column:
            df_tmp = df_tmp.drop(columns=["CCF FWHM Err. [km/s]"])

        df_list.append(df_tmp)
    else:
        print(f"Warning: file not found {file_path}")

# Combine into one big dataframe
df_all = pd.concat(df_list, ignore_index=True)
print("Head of combined dataframe:")
print(df_all.head())
print("Total points loaded:", len(df_all))

# Create an 'is_outlier' column initialized to False
df_all["is_outlier"] = False

# Compute outliers per dataset, using ANY column exceeding 5σ from that dataset's mean
for ds, g in df_all.groupby("Dataset"):
    idx = g.index
    out_mask = np.zeros(len(g), dtype=bool)

    for col in cols_to_clip:
        if col not in g.columns:
            continue
        x = g[col].dropna().astype(float)
        if len(x) == 0:
            continue
            
        median = x.median()
        std = x.std(ddof=1)
        if std == 0 or np.isnan(std):
            continue
            
        # Points beyond sigma_threshold - ACCUMULATE outliers (use |= instead of =)
        col_outliers = (np.abs(g[col] - median) > (sigma_threshold * std))
        out_mask |= col_outliers.fillna(False)  # Handle NaN values

    # Assign back
    df_all.loc[idx, "is_outlier"] = out_mask

# Quick summary
n_out = df_all["is_outlier"].sum()
print(f"Outliers flagged (but kept): {n_out}  ({n_out/len(df_all)*100:.2f}%)")
df_all[df_all["is_outlier"]]

#### Dataset files for job submissions ########

# Choose which dataframe to export
if exclude_outliers_when_writing:
    df_export = df_all[~df_all.is_outlier].copy()   # inliers only
    print("Writing .dat files using INLIERS only.")
else:
    df_export = df_all.copy()
    print("Writing .dat files using ALL points (including outliers).")


###### RV ######
# Loop over datasets
for ds, subdf in df_export.groupby("Dataset"):

    # Get instruments present in this dataset
    instruments = sorted(subdf["Instrument"].unique())
    instrument_map = {inst: i for i, inst in enumerate(instruments)}

    
    # Columns
    time = subdf["Time [eMJD]"].values
    rv = subdf["RV [m/s]"].values
    rv_err = subdf["RV Err. [m/s]"].values
    jitter_flag = np.zeros(len(subdf), dtype=int)        # all 0
    offset_flag = subdf["Instrument"].map(instrument_map).astype(int).values
    subset_flag = -1 * np.ones(len(subdf), dtype=int)    # all -1

    # Combine
    data = np.column_stack([time, rv, rv_err, jitter_flag, offset_flag, subset_flag])

    # Save to .dat file
    outfile = os.path.join(outdir, f"{ds}_RV.dat")
    np.savetxt(outfile, data, fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"])


###### BIS and FWHM ######

# Empirical errors
bis_err_val = 0.95      # m/s
fwhm_err_val = 5.0    # m/s
halpha_err_val = 0.001  # 1/s
ca2_err_val = 0.003  # 1/s

# Loop over datasets
for ds, subdf in df_export.groupby("Dataset"):

    # Get instruments present in this dataset
    instruments = sorted(subdf["Instrument"].unique())
    instrument_map = {inst: i for i, inst in enumerate(instruments)}

    time = subdf["Time [eMJD]"].values
    
    # =====================
    # BIS
    # =====================
    bis = subdf["BIS [m/s]"].values
    bis_err = np.full(len(subdf), bis_err_val)   # constant error
    jitter_flag = np.zeros(len(subdf), dtype=int)
    offset_flag = subdf["Instrument"].map(instrument_map).astype(int).values
    subset_flag = -1 * np.ones(len(subdf), dtype=int)

    bis_data = np.column_stack([time, bis, bis_err, jitter_flag, offset_flag, subset_flag])
    bis_outfile = os.path.join(outdir, f"{ds}_BIS.dat")
    np.savetxt(bis_outfile, bis_data, fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"])

    # =====================
    # FWHM
    # =====================
    fwhm = subdf["CCF FWHM [m/s]"].values
    fwhm_err = np.full(len(subdf), fwhm_err_val) # constant error
    jitter_flag = np.zeros(len(subdf), dtype=int)
    offset_flag = subdf["Instrument"].map(instrument_map).astype(int).values
    subset_flag = -1 * np.ones(len(subdf), dtype=int)

    fwhm_data = np.column_stack([time, fwhm, fwhm_err, jitter_flag, offset_flag, subset_flag])
    fwhm_outfile = os.path.join(outdir, f"{ds}_FWHM.dat")
    np.savetxt(fwhm_outfile, fwhm_data, fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"])

    # =====================
    # H alpha
    # =====================
    halpha = subdf["H-alpha Emission"].values
    halpha_err = np.full(len(subdf), halpha_err_val) # constant error
    jitter_flag = np.zeros(len(subdf), dtype=int)
    offset_flag = subdf["Instrument"].map(instrument_map).astype(int).values
    subset_flag = -1 * np.ones(len(subdf), dtype=int)

    halpha_data = np.column_stack([time, halpha, halpha_err, jitter_flag, offset_flag, subset_flag])
    halpha_outfile = os.path.join(outdir, f"{ds}_Halpha.dat")
    np.savetxt(halpha_outfile, halpha_data, fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"])

    # =====================
    # CaII
    # =====================
    ca2 = subdf["CaII Emission"].values
    ca2_err = np.full(len(subdf), ca2_err_val) # constant error
    jitter_flag = np.zeros(len(subdf), dtype=int)
    offset_flag = subdf["Instrument"].map(instrument_map).astype(int).values
    subset_flag = -1 * np.ones(len(subdf), dtype=int)

    ca2_data = np.column_stack([time, ca2, ca2_err, jitter_flag, offset_flag, subset_flag])
    ca2_outfile = os.path.join(outdir, f"{ds}_CaII.dat")
    np.savetxt(ca2_outfile, ca2_data, fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"])


####### Activity Plots #########

def plot_dataset_activity(ds, ds_df, fig_dir):
    """
    Plot activity indicators for a single dataset with outliers highlighted.
    """
    fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)

    plot_info = [
        ("RV [m/s]", "RV Err. [m/s]", "RV [m/s]"),
        ("CCF Contrast", None, "CCF Contrast"),
        ("CCF FWHM [m/s]", None, "CCF FWHM [m/s]"),
        ("BIS [m/s]", None, "BIS [m/s]"),
        ("H-alpha Emission", None, "H-alpha Emission"),
        ("CaII Emission", None, "CaII Emission"),
    ]
    
    # Fixed error values for columns without error columns
    fixed_errors = {
        "CCF Contrast": 130,
        "CCF FWHM [m/s]": 5.0,
        "BIS [m/s]": 0.95,
        "H-alpha Emission": 0.001,
        "CaII Emission": 0.003,
    }

    # Get all unique instruments in this dataset
    all_instruments = sorted(ds_df["Instrument"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_instruments)))
    color_map = dict(zip(all_instruments, colors))

    for ax, (col, err_col, ylabel) in zip(axes, plot_info):
        if col not in ds_df.columns:
            ax.text(0.5, 0.5, f"Column '{col}' not found", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel(ylabel)
            continue

        # Calculate instrument-specific medians for centering
        inst_medians = ds_df.groupby("Instrument")[col].median()
        
        # Plot each instrument separately
        for inst in all_instruments:
            inst_data = ds_df[ds_df["Instrument"] == inst].copy()
            if inst_data.empty:
                continue
                
            # Center data by instrument median
            center = inst_medians[inst]
            y_centered = inst_data[col] - center
            
            # Determine error values
            if err_col and err_col in inst_data.columns:
                yerr = inst_data[err_col]
            else:
                yerr = fixed_errors.get(col, 1.0)
            
            # Split into inliers and outliers
            inliers = inst_data[~inst_data["is_outlier"]]
            outliers = inst_data[inst_data["is_outlier"]]
            
            # Plot inliers
            if not inliers.empty:
                y_in = inliers[col] - center
                if isinstance(yerr, pd.Series):
                    yerr_in = yerr.loc[inliers.index]
                else:
                    yerr_in = yerr
                    
                ax.errorbar(inliers["Time [eMJD]"], y_in, yerr=yerr_in,
                           fmt=".", color=color_map[inst], label=inst, 
                           alpha=0.8, markersize=6)
            
            # Plot outliers
            if not outliers.empty:
                y_out = outliers[col] - center
                if isinstance(yerr, pd.Series):
                    yerr_out = yerr.loc[outliers.index]
                else:
                    yerr_out = yerr
                    
                ax.errorbar(outliers["Time [eMJD]"], y_out, yerr=yerr_out,
                           fmt="o", color="black", alpha=0.7, markersize=8,
                           markeredgecolor="red", markeredgewidth=1.5)

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # Add outlier legend entry only once
    if df_all["is_outlier"].any():
        axes[0].errorbar([], [], [], fmt="o", color="black", alpha=0.7, 
                        markeredgecolor="red", markeredgewidth=1.5, 
                        label="Outliers")

    axes[-1].set_xlabel("Time [eMJD]")
    
    # Create legend with unique entries only
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), loc="best")

    fig.suptitle(f"{ds} Activity Indicators (Red-edged = Outliers)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fig_path = os.path.join(fig_dir, f"{ds}_activity.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

# Apply the improved plotting function
for ds, ds_df in df_all.groupby("Dataset"):
    plot_dataset_activity(ds, ds_df, fig_dir)


###### Lomba-Scargle Periodograms #######

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
    all_instruments = sorted(inliers["Instrument"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_instruments)))
    color_map = dict(zip(all_instruments, colors))
    
    for i, (col, yerr, ylabel) in enumerate(plot_info):
        ax_data = axes[i, 0]      # Left column for data
        ax_period = axes[i, 1]    # Right column for periodogram
        
        if col not in inliers.columns:
            ax_data.text(0.5, 0.5, f"Column '{col}' not found", 
                        ha='center', va='center', transform=ax_data.transAxes)
            ax_period.text(0.5, 0.5, f"No data for '{col}'", 
                          ha='center', va='center', transform=ax_period.transAxes)
            ax_data.set_ylabel(ylabel)
            continue
        
        # === LEFT SIDE: DATA PLOT ===
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
            if not inliers.empty and col in inliers.columns:
                # Prepare data for periodogram (all inliers, all instruments combined)
                time_data = inliers["Time [eMJD]"].values
                y_data = inliers[col].values
                
                # Handle error values
                if isinstance(yerr, str) and yerr in inliers.columns:
                    dy_data = inliers[yerr].values
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
                        
                        # === PROPER PEAK DETECTION ===
                        # Use scipy's find_peaks function
                        # Parameters:
                        # - height: minimum height of peaks (use a threshold based on power statistics)
                        # - distance: minimum distance between peaks (in samples)
                        # - prominence: how much a peak stands out from surrounding baseline
                        
                        # Calculate dynamic threshold based on power distribution
                        power_threshold = np.percentile(power, 90)  # Top 10% of power values
                        
                        # Convert minimum period separation to frequency space
                        min_period_separation = 0.01  # minimum separation in log10(period) space
                        # This corresponds to roughly 25% difference in period
                        min_freq_separation = int(len(freq) * min_period_separation / np.log10(f_max/f_min))
                        
                        # Find peaks
                        peak_indices, peak_properties = find_peaks(
                            power, 
                            height=power_threshold,           # Minimum peak height
                            distance=max(1, min_freq_separation),  # Minimum distance between peaks
                            prominence=power_threshold * 0.1  # Peak must stand out from baseline
                        )
                        
                        # Sort peaks by power (descending) and take top 3
                        if len(peak_indices) > 0:
                            peak_powers = power[peak_indices]
                            sorted_peak_order = np.argsort(peak_powers)[::-1]  # Descending order
                            
                            # Take top 3 peaks
                            top_peak_indices = peak_indices[sorted_peak_order[:3]]
                            top_peak_powers = peak_powers[sorted_peak_order[:3]]
                            top_peak_periods = 1.0 / freq[top_peak_indices]
                            
                            # Sort by period for consistent display
                            period_sort_order = np.argsort(top_peak_periods)
                            peak_periods = top_peak_periods[period_sort_order]
                            peak_powers_sorted = top_peak_powers[period_sort_order]
                        else:
                            peak_periods = []
                            peak_powers_sorted = []
                        
                        # Convert to periods for plotting
                        periods = 1.0 / freq
                        
                        # Plot periodogram
                        ax_period.semilogx(periods, power, 'k-', linewidth=1)
                        
                        # Plot the top 3 peaks
                        peak_colors = ['red', 'orange', 'purple']
                        peak_styles = ['--', '-.', ':']
                        
                        for j, (peak_period, peak_power) in enumerate(zip(peak_periods, peak_powers_sorted)):
                            if j < len(peak_colors):
                                ordinal = ['1st', '2nd', '3rd'][j]
                                ax_period.axvline(peak_period, color=peak_colors[j], 
                                                ls=peak_styles[j], lw=2, 
                                                label=f'{ordinal} Peak: {peak_period:.1f}d')
                        
                        # Add reference lines for common periods
                        reference_periods = [1, 7, 14, 28, 100, 365]
                        for ref_period in reference_periods:
                            if min_period <= ref_period <= max_period:
                                ax_period.axvline(ref_period, color='blue', alpha=0.4, 
                                                linestyle=':', linewidth=1)
                                # Add label at top of plot
                                ax_period.text(ref_period, ax_period.get_ylim()[1]*0.85, 
                                             f'{ref_period}d', rotation=90, ha='right', 
                                             va='top', fontsize=8, alpha=0.7, color='blue')
                        
                        ax_period.set_xlabel("Period [days]")
                        ax_period.set_ylabel("LS Power")
                        ax_period.set_title(f"{ylabel} Periodogram", fontsize=10)
                        ax_period.grid(True, alpha=0.3)
                        
                        # Only add legend if there are peaks
                        if len(peak_periods) > 0:
                            ax_period.legend(fontsize=8)
                        
                        # Add statistics text
                        if len(peak_periods) > 0:
                            peak_info = '\n'.join([f'Peak {j+1}: {p:.1f}d (P={pow:.3f})' 
                                                 for j, (p, pow) in enumerate(zip(peak_periods, peak_powers_sorted))])
                            stats_text = f'N={len(t)} points\nSpan={span:.1f}d\n{peak_info}'
                        else:
                            stats_text = f'N={len(t)} points\nSpan={span:.1f}d\nNo significant peaks'
                        
                        # Position the statistics box
                        ax_period.text(0.02, 0.75, stats_text, 
                                     transform=ax_period.transAxes, fontsize=7, 
                                     verticalalignment='top', alpha=0.9,
                                     bbox=dict(boxstyle="round,pad=0.4", facecolor="white", 
                                             alpha=0.9, edgecolor='gray', linewidth=0.5))
                        
                        # Print results
                        if len(peak_periods) > 0:
                            peak_str = ', '.join([f'{p:.2f}d' for p in peak_periods])
                            print(f"{ds} - {col}: Top periods = {peak_str}")
                        else:
                            print(f"{ds} - {col}: No significant peaks found")
                        
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
    fig.suptitle(f"{ds} - Activity Indicators & Lomb-Scargle Periodograms (Inliers Only)", 
                 fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    fig_path = os.path.join(fig_dir, f"{ds}_activity_LS_periodograms.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

# Apply the function to all datasets
print("Creating activity plots with periodograms using inliers only...")
for ds, ds_df in df_all.groupby("Dataset"):
    plot_activity_with_periodograms(ds, ds_df, fig_dir)