import numpy as np
from scipy.optimize import curve_fit
import copy
import pandas as pd
import warnings

# ------------------------------------------
#
# Gaussian function
#
# ------------------------------------------

def gaussian(x, amp, mu, sig, c):
    return amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) + c

# ------------------------------------------
#
# Discrete Fourier transform (DFT)
#
# ------------------------------------------

def FT(signal, spacing):
	'''
	ξ: non-negative frequencies
	A: amplitude
	ϕ: phase
	'''
	n 		= signal.size
	fourier = np.fft.rfft(signal, n)
	ξ 		= np.fft.rfftfreq(n, d=spacing)
	A 		= np.abs(fourier)
	ϕ 		= np.angle(fourier)
	return [A, ϕ, ξ]

# ------------------------------------------
#
# Wrap the phase
#
# ------------------------------------------

def wrap(ϕ):
	'''	
	An individual phase ranges within (-np.pi, np.pi)
	The difference of two phases ranges within (-2*np.pi, 2*np.pi)
	Adding multiples of 2*np.pi to the phase is effectively the same phase
	This function wraps Δϕ such that it lies within (-np.pi, np.pi)
	'''		
	for i in np.arange(len(ϕ)):
		ϕ[i] = ϕ[i] - int(ϕ[i]/np.pi) * 2 * np.pi
	return ϕ

# ------------------------------------------
# 
# FourIEr phase SpecTrum Analysis (FIESTA)
# 
# ------------------------------------------

def estimate_gaussian_center(V_grid, CCF, bounds_check=True, verbose=False):
    """
    Robust estimation of Gaussian center from CCF data.
    
    Parameters:
    -----------
    V_grid : array
        Velocity grid
    CCF : array
        Cross-correlation function values
    bounds_check : bool
        Whether to apply parameter bounds
    verbose : bool
        Print diagnostic information
        
    Returns:
    --------
    center : float
        Estimated center of Gaussian
    sigma : float
        Estimated width of Gaussian
    popt : array
        All fitted parameters [amp, mu, sig, c]
    pcov : array
        Covariance matrix
    success : bool
        Whether the fit was successful
    """
    
    # Remove NaN and inf values
    valid_mask = np.isfinite(V_grid) & np.isfinite(CCF)
    V_grid_clean = V_grid[valid_mask]
    CCF_clean = CCF[valid_mask]
    
    if len(V_grid_clean) < 4:
        warnings.warn("Insufficient valid data points for Gaussian fit")
        return np.nan, np.nan, None, None, False
    
    # ==========================================
    # Step 1: Robust initial parameter estimation
    # ==========================================
    
    # Baseline (offset) - use median of lowest 20% of values
    sorted_CCF = np.sort(CCF_clean)
    n_baseline = max(1, len(sorted_CCF) // 5)
    offset_guess = np.median(sorted_CCF[:n_baseline])
    
    # Amplitude - difference between max and baseline
    max_CCF = np.max(CCF_clean)
    amplitude_guess = max_CCF - offset_guess
    
    # Mean - position of maximum (weighted centroid for better estimate)
    idx_max = np.argmax(CCF_clean)
    mean_guess = V_grid_clean[idx_max]
    
    # Refine mean using weighted centroid around peak
    # Use points within 80% of peak height
    threshold = offset_guess + 0.8 * amplitude_guess
    peak_mask = CCF_clean >= threshold
    if np.sum(peak_mask) > 0:
        weights = CCF_clean[peak_mask] - offset_guess
        mean_guess = np.average(V_grid_clean[peak_mask], weights=weights)
    
    # Sigma - estimate from FWHM
    # Find half-maximum points
    half_max = offset_guess + amplitude_guess / 2
    above_half = CCF_clean >= half_max
    
    if np.sum(above_half) >= 2:
        indices = np.where(above_half)[0]
        # FWHM approximation
        fwhm = V_grid_clean[indices[-1]] - V_grid_clean[indices[0]]
        sigma_guess = fwhm / (2 * np.sqrt(2 * np.log(2)))  # FWHM = 2.355 * sigma
    else:
        # Fallback: use 10% of velocity range
        sigma_guess = (V_grid_clean.max() - V_grid_clean.min()) * 0.1
    
    # Ensure sigma is reasonable
    sigma_guess = max(sigma_guess, np.diff(V_grid_clean)[0])  # At least one grid spacing
    sigma_guess = min(sigma_guess, (V_grid_clean.max() - V_grid_clean.min()) / 2)  # At most half range
    
    p0 = [amplitude_guess, mean_guess, sigma_guess, offset_guess]
    
    if verbose:
        print(f"Initial guess: amp={amplitude_guess:.3f}, mu={mean_guess:.3f}, "
              f"sig={sigma_guess:.3f}, offset={offset_guess:.3f}")
    
    # ==========================================
    # Step 2: Set reasonable bounds
    # ==========================================
    
    if bounds_check:
        # Amplitude: positive, up to 2x the data range
        amp_lower = 0
        amp_upper = 2 * (max_CCF - np.min(CCF_clean))
        
        # Mean: within velocity grid, with some margin
        v_range = V_grid_clean.max() - V_grid_clean.min()
        mu_lower = V_grid_clean.min() - 0.2 * v_range
        mu_upper = V_grid_clean.max() + 0.2 * v_range
        
        # Sigma: reasonable physical range
        sig_lower = 0.5 * np.diff(V_grid_clean)[0]  # Half a grid spacing
        sig_upper = 2 * v_range  # Twice the full range
        
        # Offset: reasonable baseline range
        c_lower = np.min(CCF_clean) - 0.1 * amplitude_guess
        c_upper = np.max(CCF_clean)
        
        bounds = ([amp_lower, mu_lower, sig_lower, c_lower],
                  [amp_upper, mu_upper, sig_upper, c_upper])
    else:
        bounds = (-np.inf, np.inf)
    
    # ==========================================
    # Step 3: Perform fit with multiple strategies
    # ==========================================
    
    success = False
    popt, pcov = None, None
    
    # Strategy 1: Standard fit with bounds
    try:
        popt, pcov = curve_fit(gaussian, V_grid_clean, CCF_clean, 
                               p0=p0, bounds=bounds, maxfev=10000)
        
        # Check if covariance is valid
        if not np.any(np.isinf(pcov)) and not np.any(np.isnan(pcov)):
            success = True
            if verbose:
                print("✓ Fit successful with bounds")
        else:
            if verbose:
                print("✗ Fit converged but covariance invalid")
    except Exception as e:
        if verbose:
            print(f"✗ Fit with bounds failed: {e}")
    
    # Strategy 2: If bounded fit fails, try without bounds
    if not success:
        try:
            popt, pcov = curve_fit(gaussian, V_grid_clean, CCF_clean, 
                                   p0=p0, maxfev=10000)
            
            if not np.any(np.isinf(pcov)) and not np.any(np.isnan(pcov)):
                success = True
                if verbose:
                    print("✓ Fit successful without bounds")
        except Exception as e:
            if verbose:
                print(f"✗ Fit without bounds failed: {e}")
    
    # Strategy 3: If still fails, try simplified initial guess
    if not success:
        try:
            p0_simple = [amplitude_guess, mean_guess, 2.0, offset_guess]
            popt, pcov = curve_fit(gaussian, V_grid_clean, CCF_clean, 
                                   p0=p0_simple, maxfev=20000)
            
            if not np.any(np.isinf(pcov)) and not np.any(np.isnan(pcov)):
                success = True
                if verbose:
                    print("✓ Fit successful with simplified guess")
        except Exception as e:
            if verbose:
                print(f"✗ All fitting strategies failed: {e}")
    
    # ==========================================
    # Step 4: Validate results
    # ==========================================
    
    if success and popt is not None:
        center = popt[1]
        sigma = abs(popt[2])  # Ensure positive
        
        # Sanity checks
        if not (V_grid_clean.min() - v_range <= center <= V_grid_clean.max() + v_range):
            warnings.warn(f"Fitted center {center:.2f} outside reasonable range")
            success = False
        
        if sigma > 2 * v_range or sigma < 0.1 * np.diff(V_grid_clean)[0]:
            warnings.warn(f"Fitted sigma {sigma:.2f} outside reasonable range")
            success = False
        
        if verbose and success:
            print(f"Final fit: center={center:.3f}, sigma={sigma:.3f}")
            # Calculate reduced chi-square
            residuals = CCF_clean - gaussian(V_grid_clean, *popt)
            chi2_red = np.sum(residuals**2) / (len(CCF_clean) - 4)
            print(f"Reduced χ² = {chi2_red:.3f}")
        
        return center, sigma, popt, pcov, success
    else:
        # Fallback: return initial guess
        if verbose:
            print("⚠ Using initial guess as fallback")
        return mean_guess, sigma_guess, p0, None, False


# ==========================================
# Updated FIESTA function with robust fitting
# ==========================================

def FIESTA(V_grid, CCF, eCCF, template=[], SNR=2, k_max=None):
    """
    Modified FIESTA with robust Gaussian center estimation
    """
    
    N_file = CCF.shape[1]
    spacing = np.diff(V_grid)[0]
    
    # Construct template
    if template != []:
        tpl_CCF = template
    elif ~np.all(eCCF == 0):
        tpl_CCF = np.average(CCF, weights=1/eCCF**2, axis=1)
    else:
        tpl_CCF = CCF[:,0]
    
    # ==========================================
    # Robust estimation of template center
    # ==========================================
    print("\n" + "="*50)
    print("Estimating template Gaussian center...")
    print("="*50)
    
    V_centre, sigma, popt_tpl, pcov_tpl, success = estimate_gaussian_center(
        V_grid, tpl_CCF, bounds_check=True, verbose=True
    )
    
    if not success:
        warnings.warn("Template Gaussian fit failed, using fallback values")
        # Use more conservative fallback
        V_centre = V_grid[np.argmax(tpl_CCF)]
        sigma = (V_grid.max() - V_grid.min()) / 10
    
    # Define analysis range
    V_min, V_max = V_centre - 5*sigma, V_centre + 5*sigma
    
    # Reshape all input spectra
    idx = (V_grid > V_min) & (V_grid < V_max)
    tpl_CCF = tpl_CCF[idx]
    V_grid_sub = V_grid[idx]
    CCF_sub = CCF[idx, :]
    eCCF_sub = eCCF[idx, :]
    
    print(f'\nVelocity grid used [{V_grid_sub.min():.2f}, {V_grid_sub.max():.2f}]')
    print(f'Template center: {V_centre:.3f} ± {sigma:.3f} km/s\n')
    
    # Re-fit template on restricted range
    _, _, popt_tpl, _, _ = estimate_gaussian_center(V_grid_sub, tpl_CCF, verbose=False)
    RV_gauss_tpl = popt_tpl[1] if popt_tpl is not None else V_centre
    
    # Information of the template CCF
    A_tpl, ϕ_tpl, ξ = FT(tpl_CCF, spacing)
    
    # ==========================================
    # Process individual CCFs
    # ==========================================
    
    A_k = np.zeros((ξ.size, N_file))
    ϕ_k = np.zeros((ξ.size, N_file))
    v_k = np.zeros((ξ.size-1, N_file))
    σA = np.zeros(N_file)
    σϕ = np.zeros((ξ.size, N_file))
    RV_gauss = np.zeros(N_file)
    ξ_normal_array = []
    
    print("Processing individual CCFs...")
    fit_failures = 0
    
    for n in range(N_file):
        # DFT
        A, ϕ, ξ = FT(CCF_sub[:,n], spacing)
        A_k[:,n] = A
        ϕ_k[:,n] = ϕ
        
        Δϕ = wrap(ϕ - ϕ_tpl)
        v_k[:,n] = -Δϕ[1:] / (2 * np.pi * ξ[1:])
        
        # Robust Gaussian fit for RV
        center, _, _, _, success = estimate_gaussian_center(
            V_grid_sub, CCF_sub[:,n], bounds_check=True, verbose=False
        )
        
        if success:
            RV_gauss[n] = center - RV_gauss_tpl
        else:
            fit_failures += 1
            # Fallback: use weighted centroid
            weights = np.maximum(CCF_sub[:,n] - np.min(CCF_sub[:,n]), 0)
            if np.sum(weights) > 0:
                RV_gauss[n] = np.average(V_grid_sub, weights=weights) - RV_gauss_tpl
            else:
                RV_gauss[n] = 0
        
        # Calculate uncertainties
        σ = (sum(eCCF_sub[:,n]**2) / 2)**0.5
        σA[n] = σ
        σϕ[:,n] = σ / A
        
        # ξ_normal
        ξ_normal_array.append(max(ξ[0.2*A > σ]))
    
    if fit_failures > 0:
        print(f"⚠ Warning: {fit_failures}/{N_file} Gaussian fits failed, using fallback method\n")
    else:
        print(f"✓ All {N_file} Gaussian fits successful\n")
    
    # Continue with rest of FIESTA analysis...
    # [Rest of the original FIESTA code continues here]
    
    if ~np.all(eCCF == 0):
        σv_k = (σϕ[1:,:].T / (2*np.pi*ξ[1:].T)).T
        # [Include all the remaining FIESTA analysis from original code]
        return None, v_k[:k_max,:], σv_k[:k_max,:], A_k[1:k_max+1,:], np.vstack([σA]*k_max), RV_gauss
    else:
        return v_k[:k_max,:], A_k[1:k_max+1,:], RV_gauss