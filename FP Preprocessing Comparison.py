# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:53:01 2024

@author: tanne
"""

import init
import hankslab_db.basicRLtasks_db as db
from pyutils import utils
import numpy as np
import fp_analysis_helpers as fpah
from sys_neuro_tools import plot_utils, fp_utils
import matplotlib.pyplot as plt
import numpy as  np
from scipy.signal import medfilt, butter, sosfiltfilt
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize, OptimizeWarning
from sklearn.linear_model import LinearRegression
import warnings


# %% Declare methods

def fit_signal(signal_to_fit, signal, t, vary_t=True):
    '''
    Scales and shift one signal to match another using linear regression

    Parameters
    ----------
    signal_to_fit : The signal being fitted
    signal : The signal being fit to

    Returns
    -------
    The fitted signal

    '''

    # # find all NaNs and drop them from both signals before regressing
    nans = np.isnan(signal) | np.isnan(signal_to_fit)

    # fit the iso signal to the ligand signal
    # reg = LinearRegression(positive=True)
    # reg.fit(signal_to_fit[~nans,None], signal[~nans])
    # fitted_signal = np.full_like(signal_to_fit, np.nan)
    # fitted_signal[~nans] = reg.predict(signal_to_fit[~nans,None])

    if vary_t:
        form = lambda x, a, b, c: a*x[0,:] + b*x[1,:] + c
        s_to_fit = np.vstack((signal_to_fit[None,~nans], t[None,~nans]))
        bounds = ([      0, -np.inf, -np.inf],
                  [ np.inf,  np.inf,  np.inf])
    else:
        form = lambda x, a, b: a*x + b
        s_to_fit = signal_to_fit[None,~nans]
        bounds = ([      0, -np.inf],
                  [ np.inf,  np.inf])

    params = curve_fit(form, s_to_fit, signal[~nans], bounds=bounds)[0]
    fitted_signal = np.full_like(signal_to_fit, np.nan)
    fitted_signal[~nans] = form(s_to_fit, *params)

    return fitted_signal


def fit_baseline(signal, n_points_min=10, baseline_form=None, bounds=None):
    '''
    Fits a baseline to the signal using a baseline formula equation.
    Defaults to a double exponential decay function:
        A*e^(-t/B) + C*e^(-t/(B*D)) + E
    Fits the baseline to the minimum value every n data points

    Parameters
    ----------
    signal : The signal to fit
    n_points_min: The number of data points to take the min over, optional

    Returns
    -------
    The fit baseline

    '''

    n_points_max = 2000

    if n_points_min > n_points_max:
        raise RuntimeError('Could not fit baseline. Try another formula or method.')

    if bounds is None:
        if baseline_form is None:
            bounds = ([-np.inf,      1, -np.inf, 0, -np.inf],
                      [ np.inf, np.inf,  np.inf, 1,  np.inf])
        else:
            bounds = (-np.inf, np.inf)

    if baseline_form is None:
        baseline_form = lambda x, a, b, c, d, e: a*np.exp(-x/b) + c*np.exp(-x/(b*d)) + e

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        warnings.simplefilter('error', category=OptimizeWarning)

        try:
            if len(signal) % n_points_min != 0:
                min_signal = np.append(signal, np.full(n_points_min - (len(signal) % n_points_min), np.nan))
            else:
                min_signal = signal

            # compute the minimum value every n points
            min_signal = np.nanmin(np.reshape(min_signal, (-1, n_points_min)), axis=1)

            # ignore nans in fit
            nans = np.isnan(min_signal)

            x = np.arange(len(min_signal))
            params = curve_fit(baseline_form, x[~nans], min_signal[~nans], bounds=bounds)[0]

            # return full baseline of the same length as the signal
            x = np.arange(len(signal))/n_points_min
            return baseline_form(x, *params)
        except (OptimizeWarning, RuntimeError):
            print('Baseline fit was unseccessful. Expanding the signal minimum window to {}..'.format(n_points_min*2))
            return fit_baseline(signal, n_points_min=n_points_min*2, baseline_form=baseline_form)


def get_all_processed_signals(raw_lig, raw_iso, t):
    ''' Gets all possible processed signals and intermediaries for the given raw signals.
        Will check to see if any signals should be excluded. Also will also optionally exclude signals before and after the behavior.'''

    # initialize signal variables
    empty_signal = np.full_like(raw_lig, np.nan)
    baseline_lig = empty_signal.copy()
    baseline_iso = empty_signal.copy()
    baseline_corr_lig = empty_signal.copy()
    baseline_corr_iso = empty_signal.copy()
    fitted_iso = empty_signal.copy()
    fitted_baseline_corr_iso = empty_signal.copy()
    dff_iso = empty_signal.copy()
    dff_iso_baseline = empty_signal.copy()
    dff_baseline = empty_signal.copy()

    # fit raw iso to raw ligand
    fitted_iso = fit_signal(raw_iso, raw_lig, t, vary_t=True)

    # calculate traditional iso dF/F
    dff_iso = ((raw_lig - fitted_iso)/fitted_iso)*100

    # baseline correction to approximate photobleaching
    try:
        baseline_lig = fit_baseline(raw_lig)
        baseline_iso = fit_baseline(raw_iso)

        baseline_corr_lig = raw_lig - baseline_lig
        baseline_corr_iso = raw_iso - baseline_iso

        # scale the isosbestic signal to best fit the ligand-dependent signal
        fitted_baseline_corr_iso = fit_signal(baseline_corr_iso, baseline_corr_lig, t, vary_t=True)
        # shift the fitted baseline corrected iso by the ligand baseline to get the reference
        shifted_fitted_baseline_corr_iso = fitted_baseline_corr_iso + baseline_lig
        dff_iso_baseline = ((baseline_corr_lig - fitted_baseline_corr_iso)/baseline_lig)*100
        dff_iso_baseline_shifted = ((baseline_corr_lig - fitted_baseline_corr_iso)/shifted_fitted_baseline_corr_iso)*100

    except RuntimeError as error:
        print(str(error))

    return {'raw_lig': raw_lig,
            'raw_iso': raw_iso,
            'baseline_lig': baseline_lig,
            'baseline_iso': baseline_iso,
            'baseline_corr_lig': baseline_corr_lig,
            'baseline_corr_iso': baseline_corr_iso,
            'fitted_iso': fitted_iso,
            'fitted_baseline_corr_iso': fitted_baseline_corr_iso,
            'shifted_fitted_baseline_corr_iso': shifted_fitted_baseline_corr_iso,
            'dff_iso': dff_iso,
            'dff_iso_baseline': dff_iso_baseline,
            'dff_iso_baseline_shifted': dff_iso_baseline_shifted}


def view_processed_signals(processed_signals, t, dec=10, title='Full Signals', t_min=0, t_max=np.inf):

    if utils.is_dict(list(processed_signals.values())[0]):
        n_panel_stacks = len(processed_signals.values())
    else:
        n_panel_stacks = 1
        # make a temporary outer dictionary for ease of use with for loop
        processed_signals = {'temp': processed_signals}

    t = t[::dec].copy()

    t_min_idx = np.argmax(t > t_min)
    t_max_idx = np.argwhere(t < t_max)[-1,0]

    # filter t but keep the same shape as signals
    t[:t_min_idx] = np.nan
    t[t_max_idx:] = np.nan

    # plot the raw signals and their baseline fits, baseline corrected signals, raw ligand and fitted iso, dff and baseline corrected df
    fig, axs = plt.subplots(2*n_panel_stacks, 2, layout='constrained', figsize=[20,6*n_panel_stacks], sharex=True)
    plt.suptitle(title)

    for i, (sub_key, sub_signals) in enumerate(processed_signals.items()):

        gen_sub_title = sub_key + ' {}' if sub_key != 'temp' else '{}'

        # plot raw signals and baseline
        ax = axs[i,0]
        color1 = next(ax._get_lines.prop_cycler)['color']
        color2 = next(ax._get_lines.prop_cycler)['color']
        ax.plot(t, sub_signals['raw_lig'][::dec], label='Raw Lig', color=color1, alpha=0.5)
        ax.plot(t, sub_signals['baseline_lig'][::dec], '--', label='Lig Baseline', color=color1)
        ax.plot(t, sub_signals['raw_iso'][::dec], label='Raw Iso', color=color2, alpha=0.5)
        ax.plot(t, sub_signals['baseline_iso'][::dec], '--', label='Iso Baseline', color=color2)
        ax.set_title(gen_sub_title.format('Raw Signals'))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fluorescent Signal (V)')
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
        ax.legend(loc='center right')

        # plot baseline corrected signals
        ax = axs[i,1]
        ax.plot(t, sub_signals['baseline_corr_lig'][::dec], label='Baseline Corrected Lig', alpha=0.5)
        ax.plot(t, sub_signals['baseline_corr_iso'][::dec], label='Baseline Corrected Iso', alpha=0.5)
        ax.plot(t, sub_signals['fitted_baseline_corr_iso'][::dec], label='Fitted Baseline Corrected Iso', alpha=0.5)
        ax.set_title(gen_sub_title.format('Baseline Subtracted Signals'))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fluorescent Signal (V)')
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
        ax.legend(loc='upper right')

        # plot raw ligand & fitted iso
        ax = axs[n_panel_stacks+i,0]
        ax.plot(t, sub_signals['raw_lig'][::dec], label='Raw Lig', alpha=0.5)
        ax.plot(t, sub_signals['fitted_iso'][::dec], label='Fitted Iso', alpha=0.5)
        ax.plot(t, sub_signals['shifted_fitted_baseline_corr_iso'][::dec], label='Shifted Fitted Baseline Corrected Iso', alpha=0.5)
        ax.set_title(gen_sub_title.format('Iso ΔF/F Signals'))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fluorescent Signal (V)')
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
        ax.legend(loc='upper right')

        # plot iso dFF and baseline corrected dF
        ax = axs[n_panel_stacks+i,1]
        ax.plot(t, sub_signals['dff_iso'][::dec], label='Iso ΔF/F', alpha=0.5)
        ax.plot(t, sub_signals['dff_iso_baseline_shifted'][::dec], label='Shifted Baseline Corrected ΔF/F', alpha=0.5)
        ax.set_title(gen_sub_title.format('Iso Corrected Ligand Signals'))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('dF/F')
        ax.tick_params(axis='y', labelcolor=color1)
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
        ax.legend(loc='upper right')

    return fig

#%% Get fp data signals

#sess_ids = {202: [101965], 191: [102208, 100301, 101667]}
sess_ids = {188: [102246]} #, 100406, 100551, 100673]}

loc_db = db.LocalDB_BasicRLTasks('')
# get fiber photometry data
fp_data = loc_db.get_sess_fp_data(utils.flatten(sess_ids))
# separate into different dictionaries

fp_data = fp_data['fp_data']

iso = '420'
lig = '490'

# %% Process signals

from sklearn.decomposition import FastICA
from scipy.optimize import curve_fit

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:

        raw_signals = fp_data[subj_id][sess_id]['raw_signals']
        t = fp_data[subj_id][sess_id]['time']
        fs = 1/(t[1] - t[0])

        #fp_data[subj_id][sess_id]['processed_signals'] = {}
        #fp_data[subj_id][sess_id]['processed_signals_denoised'] = {}

        n_regions = len(raw_signals.keys())

        for region in raw_signals.keys():

            raw_lig = raw_signals[region][lig]
            raw_iso = raw_signals[region][iso]

            denoised_lig = fp_utils.filter_signal(raw_lig, 10, fs)
            denoised_iso = fp_utils.filter_signal(raw_iso, 10, fs)
            
            fitted_iso = fp_utils.fit_signal(denoised_iso, denoised_lig, t)
            
            classic_dff = (denoised_lig - fitted_iso)/fitted_iso * 100
            
            ica = FastICA()
            signal_ica = ica.fit_transform(np.vstack([denoised_lig, fitted_iso]).T)
            mix_mat = ica.mixing_
            mix_loading = np.abs(np.sum(mix_mat, axis=0))
            
            if mix_loading[0] > mix_loading[1]:
                noise_idx = 0
                sig_idx = 1
            else:
                noise_idx = 1
                sig_idx = 0
            
            form = lambda x, a, b: a*x + b
            params = curve_fit(form, signal_ica[:,noise_idx], fitted_iso)[0]
            fitted_shared_signal = form(signal_ica[:,noise_idx], *params)

            params = curve_fit(form, signal_ica[:,sig_idx], denoised_lig-fitted_iso)[0]
            fitted_unshared_signal = form(signal_ica[:,sig_idx], *params)
            
            ica1_dff = (denoised_lig - fitted_shared_signal)/fitted_iso * 100
            ica2_dff = (fitted_unshared_signal-np.mean(fitted_unshared_signal))/fitted_iso * 100
            
            t_min = 1200 # -np.inf # 
            t_max = 1220 # np.inf # 
            plot_t = t.copy()
            plot_t[(t < t_min) | (t > t_max)] = np.nan
            
            fig, axs = plt.subplots(2,2, figsize=(10,8), layout='constrained')
            
            ax = axs[0,0]
            ax.plot(plot_t, denoised_lig, label='Lig', alpha=0.6)
            ax.plot(plot_t, fitted_iso, label='Fitted Iso', alpha=0.6)
            ax.legend(loc='upper right')
            
            ax = axs[0,1]
            ax.plot(plot_t, denoised_lig, label='Lig', alpha=0.6)
            ax.plot(plot_t, fitted_shared_signal, label='IC1', alpha=0.6)
            #ax.plot(plot_t, fitted_unshared_signal, label='IC2', alpha=0.6)
            ax.legend(loc='upper right')
            
            ax = axs[1,0]
            ax.plot(plot_t, denoised_lig - fitted_iso, label='Iso Subtracted', alpha=0.6)
            ax.plot(plot_t, denoised_lig - fitted_shared_signal, label='IC1 Subtracted', alpha=0.6)
            ax.plot(plot_t, fitted_unshared_signal-np.mean(fitted_unshared_signal), label='IC2 Subtracted', alpha=0.6)
            ax.legend(loc='upper right')
            
            ax = axs[1,1]
            ax.plot(plot_t, classic_dff, label='Classic dF/F', alpha=0.6)
            ax.plot(plot_t, ica1_dff, label='ICA1 dF/F', alpha=0.6)
            ax.plot(plot_t, ica2_dff, label='ICA2 dF/F', alpha=0.6)
            ax.legend(loc='upper right')
            
            #fp_data[subj_id][sess_id]['processed_signals'][region] = get_all_processed_signals(raw_lig, raw_iso, t)
            #fp_data[subj_id][sess_id]['processed_signals_denoised'][region] = get_all_processed_signals(denoised_lig, denoised_iso, t)



# %% Plot processed signals
gen_title = '{} - Scale, Translate & Rotate, Subject {}, Session {}'
sub_t = [0, np.inf] # [1100, 1120] #
dec = 10

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        t = fp_data[subj_id][sess_id]['time']

        #view_processed_signals(fp_data[subj_id][sess_id]['processed_signals'], t, title=gen_title.format('Exp-Lin Baseline', subj_id, sess_id), t_min=sub_t[0], t_max=sub_t[1], dec=dec)
        view_processed_signals(fp_data[subj_id][sess_id]['processed_signals_denoised'], t, title=gen_title.format('Denoised, Exp-Lin Baseline', subj_id, sess_id), t_min=sub_t[0], t_max=sub_t[1], dec=dec)

# %%

import numpy as np

# LMS Adaptive Filter Function
def lms_filter(desired_signal, noise_reference, mu=0.01, filter_order=5):
    """
    Perform adaptive noise cancellation using LMS filter.

    Parameters:
    - desired_signal: Signal containing both the desired signal and noise (d[n])
    - noise_reference: Reference noise signal (x[n])
    - mu: Learning rate (step size) for LMS algorithm
    - filter_order: Number of filter taps (higher values allow more complex noise cancellation)

    Returns:
    - cleaned_signal: Filter output with noise removed
    """
    N = len(desired_signal)
    w = np.zeros(filter_order)  # Initialize filter weights
    x_buf = np.zeros(filter_order)  # Buffer to hold past values of x(n)
    cleaned_signal = np.zeros(N)  # Output signal

    for n in range(N):
        # Update input buffer (shift values, insert new one)
        x_buf[1:] = x_buf[:-1]
        x_buf[0] = noise_reference[n] if n < len(noise_reference) else 0

        # Compute filter output (estimated noise)
        noise_estimate = np.dot(w, x_buf)

        # Compute error signal (cleaned signal)
        e = desired_signal[n] - noise_estimate
        cleaned_signal[n] = e

        # Update filter weights using LMS rule
        w += 2 * mu * e * x_buf  # Gradient descent step

    return cleaned_signal

# Simulated Example
n = 1000
np.random.seed(42)
t = np.linspace(0, 1, n)  # Time vector
true_signal = 3*np.sin(2 * np.pi * 5 * t)  # Desired signal (10 Hz sine wave)
noise = np.random.randn(n)  # Random noise
mixed_signal = true_signal + noise  # Signal + noise
reference_noise = noise + np.random.randn(n) * 0.1  # Reference noise with some variation

# Apply LMS filter
# filtered_signal = lms_filter(mixed_signal, reference_noise, mu=0.01, filter_order=5)

# # Plot results
# plt.figure(figsize=(10, 5))
# plt.plot(t, mixed_signal, label="Noisy Signal", alpha=0.6)
# plt.plot(t, filtered_signal, label="Filtered Signal", linewidth=2)
# plt.plot(t, true_signal, label="True Signal", linestyle="dashed")
# plt.legend()
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.title("LMS Adaptive Noise Cancellation")
# plt.show()

from sklearn.decomposition import FastICA
from sys_neuro_tools import fp_utils
from scipy.optimize import curve_fit

# Apply ICA
ica = FastICA()
X_ica = ica.fit_transform(np.vstack([mixed_signal, reference_noise]).T)

# Extract the first independent component (assumed to be the shared signal)
extracted_shared_signal = X_ica[:, 0]
extracted_unshared_signal = X_ica[:, 1]

form = lambda x, a, b: a*x + b
params = curve_fit(form, extracted_shared_signal, reference_noise)[0]
fitted_shared_signal = form(extracted_shared_signal, *params)

subtracted_shared_signal = mixed_signal - fitted_shared_signal

params = curve_fit(form, extracted_unshared_signal, mixed_signal)[0]
fitted_unshared_signal = form(extracted_unshared_signal, *params)

plt.figure(figsize=(10, 5))
plt.plot(t, mixed_signal, label="Noisy Signal", alpha=0.6)
plt.plot(t, fitted_shared_signal, label="Shared Noise", alpha=0.6)
plt.plot(t, subtracted_shared_signal, label="Filtered Signal", linewidth=2, alpha=0.6)
plt.plot(t, fitted_unshared_signal, label="Unshared Signal", alpha=0.6)
plt.plot(t, true_signal, label="True Signal", linestyle="dashed")
#plt.plot(t, noise, label="Noise", alpha=0.6)
#plt.plot(t, extracted_shared_signal, label="Shared Signal", alpha=0.6)
#plt.plot(t, fitted_shared_signal, label="Fitted Shared Signal", alpha=0.6)
#plt.plot(t, extracted_unshared_signal, label="Unshared Signal", alpha=0.6)
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("LMS Adaptive Noise Cancellation")
plt.show()