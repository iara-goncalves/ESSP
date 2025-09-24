from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import errorbar

def load_and_process_data(essp_dir, sigma_threshold=4):
    """Load all ESSP4 datasets and process them for outlier detection."""
    
    # Columns to test for outliers
    cols_to_clip = [
        "RV [m/s]",
        "BIS [m/s]",
        "CCF FWHM [m/s]",
        "CCF Contrast",
        "H-alpha Emission",
        "CaII Emission"
    ]
    
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
                df_tmp = df_tmp.drop(columns=["CCF FWHM [km/s]"])
            
            if "CCF FWHM Err. [km/s]" in df_tmp.columns:
                df_tmp["CCF FWHM Err. [m/s]"] = df_tmp["CCF FWHM Err. [km/s]"] * 1000.0
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

    # Compute outliers per dataset
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
                
            # Points beyond sigma_threshold
            col_outliers = (np.abs(g[col] - median) > (sigma_threshold * std))
            out_mask |= col_outliers.fillna(False)

        # Assign back
        df_all.loc[idx, "is_outlier"] = out_mask

    # Quick summary
    n_out = df_all["is_outlier"].sum()
    print(f"Outliers flagged: {n_out} ({n_out/len(df_all)*100:.2f}%)")
    
    return df_all

def save_dat_files(df_all, outdir, exclude_outliers=True):
    """Save processed data to .dat files for PyORBIT with proper flags."""
    
    os.makedirs(outdir, exist_ok=True)
    
    # Choose which dataframe to export
    if exclude_outliers:
        df_export = df_all[~df_all.is_outlier].copy()
        print("Writing .dat files using INLIERS only.")
    else:
        df_export = df_all.copy()
        print("Writing .dat files using ALL points (including outliers).")

    # Empirical errors for activity indicators
    error_values = {
        "BIS": 0.95,
        "Contrast": 130,
        "FWHM": 5.0,
        "Halpha": 0.001,
        "CaII": 0.003
    }

    # Loop over datasets
    for ds, subdf in df_export.groupby("Dataset"):
        print(f"Processing {ds}...")
        
        # Get instruments present in this dataset
        instruments = sorted(subdf["Instrument"].unique())
        instrument_map = {inst: i for i, inst in enumerate(instruments)}
        print(f"  Instruments: {instruments}")
        
        # Common columns for all data types
        time = subdf["Time [eMJD]"].values
        jitter_flag = np.zeros(len(subdf), dtype=int)        # all 0
        offset_flag = subdf["Instrument"].map(instrument_map).astype(int).values
        subset_flag = -1 * np.ones(len(subdf), dtype=int)    # all -1

        # === RV DATA ===
        if "RV [m/s]" in subdf.columns and "RV Err. [m/s]" in subdf.columns:
            rv = subdf["RV [m/s]"].values
            rv_err = subdf["RV Err. [m/s]"].values
            
            rv_data = np.column_stack([time, rv, rv_err, jitter_flag, offset_flag, subset_flag])
            rv_outfile = os.path.join(outdir, f"{ds}_RV.dat")
            np.savetxt(rv_outfile, rv_data, fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"])
            print(f"  Saved: {rv_outfile}")

        # === BIS DATA ===
        if "BIS [m/s]" in subdf.columns:
            bis = subdf["BIS [m/s]"].values
            bis_err = np.full(len(subdf), error_values["BIS"])
            
            bis_data = np.column_stack([time, bis, bis_err, jitter_flag, offset_flag, subset_flag])
            bis_outfile = os.path.join(outdir, f"{ds}_BIS.dat")
            np.savetxt(bis_outfile, bis_data, fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"])
            print(f"  Saved: {bis_outfile}")

        # === CCF CONTRAST DATA ===
        if "CCF Contrast" in subdf.columns:
            contrast = subdf["CCF Contrast"].values
            contrast_err = np.full(len(subdf), error_values["Contrast"])
            
            contrast_data = np.column_stack([time, contrast, contrast_err, jitter_flag, offset_flag, subset_flag])
            contrast_outfile = os.path.join(outdir, f"{ds}_Contrast.dat")
            np.savetxt(contrast_outfile, contrast_data, fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"])
            print(f"  Saved: {contrast_outfile}")

        # === CCF FWHM DATA ===
        if "CCF FWHM [m/s]" in subdf.columns:
            fwhm = subdf["CCF FWHM [m/s]"].values
            #fwhm_err = np.full(len(subdf), error_values["FWHM"])
            fwhm_err = subdf["CCF FWHM Err. [m/s]"].values
            
            fwhm_data = np.column_stack([time, fwhm, fwhm_err, jitter_flag, offset_flag, subset_flag])
            fwhm_outfile = os.path.join(outdir, f"{ds}_FWHM.dat")
            np.savetxt(fwhm_outfile, fwhm_data, fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"])
            print(f"  Saved: {fwhm_outfile}")

        # === H-ALPHA DATA ===
        if "H-alpha Emission" in subdf.columns:
            halpha = subdf["H-alpha Emission"].values
            halpha_err = np.full(len(subdf), error_values["Halpha"])
            
            halpha_data = np.column_stack([time, halpha, halpha_err, jitter_flag, offset_flag, subset_flag])
            halpha_outfile = os.path.join(outdir, f"{ds}_Halpha.dat")
            np.savetxt(halpha_outfile, halpha_data, fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"])
            print(f"  Saved: {halpha_outfile}")

        # === CaII DATA ===
        if "CaII Emission" in subdf.columns:
            ca2 = subdf["CaII Emission"].values
            ca2_err = np.full(len(subdf), error_values["CaII"])
            
            ca2_data = np.column_stack([time, ca2, ca2_err, jitter_flag, offset_flag, subset_flag])
            ca2_outfile = os.path.join(outdir, f"{ds}_CaII.dat")
            np.savetxt(ca2_outfile, ca2_data, fmt=["%.6f", "%.6f", "%.6f", "%d", "%d", "%d"])
            print(f"  Saved: {ca2_outfile}")

def plot_dataset_activity(ds, ds_df, fig_dir):
    """Plot activity indicators for a single dataset with outliers highlighted."""
    
    fig, axes = plt.subplots(6, 1, figsize=(12, 18), sharex=True)

    plot_info = [
        ("RV [m/s]", "RV Err. [m/s]", "RV [m/s]"),
        ("CCF Contrast", None, "CCF Contrast"),
        ("CCF FWHM [m/s]", None, "CCF FWHM [m/s]"),
        ("BIS [m/s]", None, "BIS [m/s]"),
        ("H-alpha Emission", None, "H-alpha Emission"),
        ("CaII Emission", None, "CaII Emission"),
    ]
    
    # Fixed error values
    fixed_errors = {
        "CCF Contrast": 130,
        "CCF FWHM [m/s]": 5.0,
        "BIS [m/s]": 0.95,
        "H-alpha Emission": 0.001,
        "CaII Emission": 0.003,
    }

    # Get all unique instruments and colors
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
        
        # Plot each instrument
        for inst in all_instruments:
            inst_data = ds_df[ds_df["Instrument"] == inst].copy()
            if inst_data.empty:
                continue
                
            # Center data by instrument median
            center = inst_medians[inst]
            
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
                yerr_in = yerr.loc[inliers.index] if isinstance(yerr, pd.Series) else yerr
                ax.errorbar(inliers["Time [eMJD]"], y_in, yerr=yerr_in,
                           fmt=".", color=color_map[inst], label=inst, 
                           alpha=0.8, markersize=6)
            
            # Plot outliers
            if not outliers.empty:
                y_out = outliers[col] - center
                yerr_out = yerr.loc[outliers.index] if isinstance(yerr, pd.Series) else yerr
                ax.errorbar(outliers["Time [eMJD]"], y_out, yerr=yerr_out,
                           fmt="o", color="black", alpha=0.7, markersize=8,
                           markeredgecolor="red", markeredgewidth=1.5)

        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # Add outlier legend entry
    if ds_df["is_outlier"].any():
        axes[0].errorbar([], [], [], fmt="o", color="black", alpha=0.7, 
                        markeredgecolor="red", markeredgewidth=1.5, 
                        label="Outliers")

    axes[-1].set_xlabel("Time [eMJD]")
    
    # Create legend with unique entries
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    axes[0].legend(by_label.values(), by_label.keys(), loc="best")

    fig.suptitle(f"{ds} Activity Indicators (Red-edged = Outliers)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    fig_path = os.path.join(fig_dir, f"{ds}_activity.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {fig_path}")

def main():
    """Main function to process ESSP4 data."""
    
    # Configuration
    essp_dir = "/work2/lbuc/data/ESSP4/ESSP4"
    outdir = "/work2/lbuc/iara/GitHub/PyORBIT_examples/ESSP4/data"
    fig_dir = "/work2/lbuc/iara/GitHub/ESSP/Figures"
    
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    # Load and process data
    df_all = load_and_process_data(essp_dir, sigma_threshold=4)
    
    # Save .dat files (inliers only) with proper flags
    save_dat_files(df_all, outdir, exclude_outliers=True)
    
    # Create activity plots with outliers highlighted
    print("Creating activity plots with outliers highlighted...")
    for ds, ds_df in df_all.groupby("Dataset"):
        plot_dataset_activity(ds, ds_df, fig_dir)
    
    # Save processed dataframe for use in essp4_figures.py
    df_all.to_pickle(os.path.join(outdir, "processed_data.pkl"))
    print(f"Saved processed dataframe to {outdir}/processed_data.pkl")

if __name__ == "__main__":
    main()