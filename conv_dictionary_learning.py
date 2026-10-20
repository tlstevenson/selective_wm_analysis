"""
Signed convolutional dictionary learning for multichannel continuous signals.

This module implements a SeqNMF-like factorization with real-valued templates
and real-valued loadings. The model is optimized with PyTorch autograd rather
than multiplicative nonnegative updates.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.linalg import vector_norm
from torch import nn
from scipy.sparse.linalg import LinearOperator, lsmr
import matplotlib.pyplot as plt
from collections.abc import Callable
import time

ArrayLike = Union[np.ndarray, torch.Tensor]

# %% Alternating Least Squares

def to_tensor(X: ArrayLike, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cast a data matrix to a tensor"""

    device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return torch.as_tensor(X, device=device)

def get_padding(kernel_len: int, zero_idx=None, operation='demix'):

    if zero_idx is None:
        zero_idx = kernel_len // 2

    if operation == 'demix':
        left_pad = zero_idx
        right_pad = kernel_len - zero_idx - 1

    elif operation == 'mix':
        left_pad = kernel_len - zero_idx - 1
        right_pad = zero_idx

    else:
        raise ValueError(f"Unknown operation: {operation}")

    return left_pad, right_pad

def pad(signal: torch.Tensor, kernel_len: int, zero_idx=None, operation='demix') -> torch.Tensor:
    """
    Pad a batched signal with nans based on the kernel's zero index so conv1d returns the original sample count,
    where invalid convolutions beyond the edge of thekernel are nan

    For a kernel of length L, conv1d reduces length by L - 1. Adding a total
    of L - 1 small values around the signal preserves n_time without wraparound.
    """

    left_pad, right_pad = get_padding(kernel_len, zero_idx=zero_idx, operation=operation)

    if left_pad + right_pad == 0:
        return signal

    return F.pad(signal, (left_pad, right_pad), value=torch.nan)

def make_design_matrix(signals: torch.Tensor, kernel_len: int, zero_idx=None, operation='demix') -> torch.Tensor:
    """
    Construct the convolution design matrix.

    Parameters
    ----------
    signals : (n_channels,T) torch.Tensor
    kernel_len: The length of the kernel, L

    Returns
    -------
    A : (T-L+1, n_channels*L) valid convolution design matrix
    """

    n_signals, T = signals.shape

    signals = pad(signals, kernel_len, zero_idx=zero_idx, operation=operation)

    cols = []

    for i in range(kernel_len):
        cols.append(signals[:, i:i+T].T)

    return torch.cat(cols, dim=1)

def extract_sources(X: torch.Tensor, W_demix: torch.Tensor, zero_idx=None) -> torch.Tensor:
    """
    Extract sources, H, from X as (n_kernels, n_time) by convolving with W_demix.

    Because we are treating W_demix as a forward operator to extract prior sources, keep W_demix in original
    order since `conv1d` computes cross-correlation

    Parameters
    ----------
    X: (R,T) torch.Tensor
        Regional Data
    W_demix: (K,R,L) torch.Tensor
        Demixing Kernels

    Returns
    -------
    H : (K,T) torch.Tensor
        Component loadings constructed by convolving the data with the demixing kernels
    """

    L = W_demix.shape[2]

    X_padded = pad(X.unsqueeze(0), L, zero_idx=zero_idx, operation='demix')

    H = F.conv1d(X_padded, W_demix)

    return H.squeeze(0)

def reconstruct(H: torch.Tensor, W_mix: torch.Tensor, full=True, zero_idx=None) -> torch.Tensor:
    """
    Reconstruct X from extracted sources as separate components (n_kernels, n_regions, n_time).
    Optionally get the full reconstruction by summing the reconstructions across kernels

    `conv1d` computes cross-correlation, so the temporal axis of `W_mix` is
    flipped before filtering to implement mathematical convolution.

    Parameters
    ----------
    H: (K,T) torch.Tensor
        Component Loadings
    W_mix: (K,R,L) torch.Tensor
        Mixing Kernels

    Returns
    -------
    X_hat : (K,R,T) if 'full' is False or (R,T) if 'full' is True torch.Tensor
        Data reconstructed by convolving the loadings with the mixing kernels
    """

    K, R, L = W_mix.shape

    H_padded = pad(H.unsqueeze(0), L, zero_idx=zero_idx, operation='mix')

    conv_weight = W_mix.flip(-1).reshape(K*R, 1, L)
    X_hat = F.conv1d(H_padded, conv_weight, groups=K)
    X_hat = X_hat.squeeze(0).reshape(K, R, -1)

    if full:
        X_hat = X_hat.sum(dim=0)

    return X_hat

def solve_mixing(X: torch.Tensor, H: torch.Tensor, kernel_len: int, zero_idx=None, ridge_lam=0.1) -> torch.Tensor:
    """
    Solve X = H * W_mix for W_mix

    Parameters
    ----------
    X : (R,T) torch.Tensor
        True signals
    H : (K,T) torch.Tensor
        Component loadings

    Returns
    -------
    W_mix : (K,R,L) torch.Tensor
    """

    # A is of shape (T,K*L)
    A = make_design_matrix(H, kernel_len, zero_idx=zero_idx, operation='mix')

    # remove nans
    valid_sel = torch.logical_and(torch.isfinite(A).all(dim=1), torch.isfinite(X.T).all(dim=1))

    A_valid = A[valid_sel, :]
    X_valid = X.T[valid_sel, :]

    ATA = A_valid.T @ A_valid
    ATX = A_valid.T @ X_valid

    ridge_matrix = ridge_lam * A_valid.shape[0] * torch.eye(ATA.shape[0], dtype=ATA.dtype, device=ATA.device)

    # coefficients are in shape (K*L,R) where columns are first ordered by kernel then lag
    # coef = torch.linalg.solve(ATA + ridge_matrix, ATX)
    coef = torch.linalg.lstsq(ATA + ridge_matrix, ATX).solution

    K = H.shape[0]
    R = X.shape[0]

    # need to flip the last axis because the convolution looks backwards in time
    return coef.reshape(kernel_len, K, R).permute(1,2,0).flip(-1)


def solve_demixing(X: torch.Tensor, H: torch.Tensor, kernel_len: int, zero_idx=None, ridge_lam=0.1) -> torch.Tensor:
    """
    Solve H = X * W_demix for W_demix

    Parameters
    ----------
    X : (R,T) torch.Tensor
        True Signals
    H : (K,T) torch.Tensor
        Component Loadings

    Returns
    -------
    W_demix : (K,R,L) torch.Tensor
    """

    # A is of shape (T,R*L)
    A = make_design_matrix(X, kernel_len, zero_idx=zero_idx, operation='demix')

    # remove nans
    valid_sel = torch.logical_and(torch.isfinite(A).all(dim=1), torch.isfinite(H.T).all(dim=1))

    A_valid = A[valid_sel, :]
    H_valid = H.T[valid_sel, :]

    ATA = A_valid.T @ A_valid
    ATH = A_valid.T @ H_valid

    ridge_matrix = ridge_lam * A_valid.shape[0] * torch.eye(ATA.shape[0], dtype=ATA.dtype, device=ATA.device)

    # coefficients are in shape (R*L,K) where columns are first ordered by region then lag
    coef = torch.linalg.lstsq(ATA + ridge_matrix, ATH).solution
    # coef = torch.linalg.solve(ATA + ridge_matrix, ATH)

    R = X.shape[0]
    K = H.shape[0]

    return coef.reshape(kernel_len, R, K).permute(2,1,0)

def apply_mixing_adjoint(X: torch.Tensor, W_mix: torch.Tensor, zero_idx=None) -> torch.Tensor:
    """
    Apply the adjoint of the mixing operator.

    If X_hat = reconstruct(H, W_mix), then this returns A^T X
    where A is the linear operator represented by `reconstruct`.

    Parameters
    ----------
    X : (R,T) torch.Tensor
        Signals.
    W_mix : (K,R,L) torch.Tensor
        Mixing kernels.

    Returns
    -------
    H : (K,T+L-1) torch.Tensor
        Adjoint projection of X into loading space.
    """
    K, R, L = W_mix.shape

    left_pad, right_pad = get_padding(L, zero_idx=zero_idx, operation='mix')

    # Put the mixing weights in the same shape as forward reconstruction:
    conv_weight = W_mix.flip(-1).reshape(K*R, 1, L)

    # Each X_r needs to be passed through every kernel W[k, r] to reconstruct H_k
    # Make a grouped X matrix of shape (1,K*R,T) so that we can use a grouped transpose convolution
    X_groups = X.unsqueeze(0).repeat(1, K, 1)

    # Reverse the convolution of H with W_mix by using grouped transpose convolution.
    # This will add padding to match what conv1d would have removed: (K,T+L-1)
    H_padded = F.conv_transpose1d(X_groups, conv_weight, groups=K)

    # Adjoint of zero padding  crop away the padded samples.
    if right_pad == 0:
        H = H_padded[..., left_pad:]
    else:
        H = H_padded[..., left_pad:-right_pad]

    return H.squeeze(0)


def solve_loadings(X: torch.Tensor, W_mix: torch.Tensor, H0: torch.Tensor=None, zero_idx=None,
                   maxiter=None, atol=1e-2, btol=1e-2, verbose=False):
    """
    Solve for H given X and fixed W_mix using LSMR

    Parameters
    ----------
    X : (R,T) torch.Tensor
        True signals
    W_mix : (K,R,L) torch.Tensor
        Mixing kernels
    H0 : (K,T) torch.Tensor, optional
        Initial loading estimate

    Returns
    -------
    H : (K,T) torch.Tensor
    """

    K, R, L = W_mix.shape

    # Missing observations contribute zero.
    obs_mask = torch.all(torch.isfinite(X), dim=0)
    X_filled = torch.nan_to_num(X, nan=0.0)

    # account for the nans added through reconstruct
    left_pad, right_pad = get_padding(L, zero_idx=zero_idx, operation='mix')
    if left_pad > 0:
        obs_mask[:left_pad] = False

    if right_pad > 0:
        obs_mask[-right_pad:] = False

    def apply_A(H_flat):
        H = H_flat.reshape(K, -1)

        X_hat = reconstruct(H, W_mix, zero_idx=zero_idx)

        # reconstruct may contain NaNs at convolution boundaries.
        # They are not observations in the regression.
        X_hat = torch.nan_to_num(X_hat, nan=0.0) * obs_mask.unsqueeze(0)

        return X_hat.reshape(-1)


    def apply_AT(X_flat):
        X = X_flat.reshape(R, -1)

        # Missing observations contribute nothing to A^T X.
        X = torch.nan_to_num(X, nan=0.0) * obs_mask.unsqueeze(0)

        H = apply_mixing_adjoint(X, W_mix, zero_idx=zero_idx)

        return H.reshape(-1)

    if H0 is not None:
        H0 = torch.nan_to_num(H0, nan=0.0)
        H0 = H0.reshape(-1)

    H, *stats = torch_lsmr(apply_A, apply_AT, X_filled.reshape(-1),
                           x0=H0, maxiter=maxiter, atol=atol, btol=btol, verbose=verbose)

    H = H.reshape(K, -1)
    # replace the nans where they would be after demixing
    h_left_pad, h_right_pad = get_padding(L, zero_idx=zero_idx, operation='demix')

    if h_left_pad > 0:
        H[:, :h_left_pad] = torch.nan

    if h_right_pad > 0:
        H[:, -h_right_pad:] = torch.nan

    return H, stats

def sort_kernels_by_region_ev(ev):
    """
    Sort kernels by taking the best remaining kernel from each region
    in successive rounds.

    Parameters
    ----------
    ev : (K, R) array-like
        Explained variance for each kernel and region.

    Returns
    -------
    order : (K,) ndarray
        Kernel indices in round-robin region order.
    """

    K, R = ev.shape

    # Each column gives kernels ranked by EV for that region
    region_order = torch.argsort(ev, axis=0, descending=True)

    order = []
    used = torch.zeros(K, dtype=bool)

    for rank in range(K):
        for region in range(R):
            idx = region_order[rank, region]

            if not used[idx]:
                order.append(idx)
                used[idx] = True

    return torch.asarray(order)

def find_unique_kernels(X: ArrayLike, W_mix: ArrayLike, zero_idx=None, ev_thresh=0.02,
                        order=None, atol=1e-4, btol=1e-4, ridge_lam=0.1, verbose=False):
    """
    Identify redundant mixing kernels using marginal explained variance
    and loading similarity.

    A candidate kernel is rejected if its marginal explained variance is below ev_thresh

    Kernels are initially ordered by their explained variance when fit individually to the full data.

    Parameters
    ----------
    X : (R,T) torch.Tensor
        Observed signals.
    W_mix : (K,R,L) torch.Tensor
        Candidate mixing kernels.
    ev_thresh : float, optional
        Minimum marginal explained variance required for a kernel to be accepted
    order : ArrayLike, optional
        The order with which to sequentially add the kernels
    atol, btol : float
        LSMR convergence tolerances.
    ridge_lam : float
        Ridge penalty for solving W_demix and W_mix

    Returns
    -------
    keep : (K,) torch.Tensor
        Boolean mask indicating retained kernels.
    results: dict containing the reduced 'W_mix' and 'H' in order of individual explained variance
    details: dict containing 'individual_ev', 'marginal_ev', and 'individual_H'
    """

    print('Finding unique kernels...')

    X = to_tensor(X)
    W_mix = to_tensor(W_mix)

    K, R, L = W_mix.shape

    with torch.no_grad():

        # Total variance in each channel of X
        X_mean = X.nanmean(dim=1, keepdim=True)
        full_se = (X - X_mean)**2

        def channel_explained_variance(X_hat):
            residual_se = (X - X_hat)**2
            valid = torch.all(full_se.isfinite(), dim=0) & torch.all(residual_se.isfinite(), dim=0)
            return 1 - (residual_se[:,valid].sum(dim=1) / full_se[:,valid].sum(dim=1))

        # Fit each kernel independently and optionally rank by individual EV

        start = time.perf_counter()

        individual_ev = torch.empty((K,R), dtype=X.dtype, device=X.device)
        individual_ev_demix = torch.empty((K,R), dtype=X.dtype, device=X.device)
        individual_H = []

        for k in range(K):
            W_mix_k = W_mix[k:k+1,:,:]

            H_k, stats = solve_loadings(X, W_mix_k, atol=atol, btol=btol, verbose=verbose, zero_idx=zero_idx)
            W_demix_k = solve_demixing(X, H_k, L, zero_idx=zero_idx, ridge_lam=ridge_lam)
            H_demix_k = extract_sources(X, W_demix_k, zero_idx=zero_idx)
            X_hat = reconstruct(H_k, W_mix_k, zero_idx=zero_idx)
            X_hat_demix = reconstruct(H_demix_k, W_mix_k, zero_idx=zero_idx)

            iter_ev = channel_explained_variance(X_hat)
            iter_ev_demix = channel_explained_variance(X_hat_demix)

            individual_ev[k,:] = iter_ev
            individual_ev_demix[k,:] = iter_ev_demix
            individual_H.append(H_k[0])

        individual_H = torch.stack(individual_H)

        print('  Fit individual kernels in {:.1f} s'.format(time.perf_counter()-start))

        if order is None:
            order = sort_kernels_by_region_ev(individual_ev)

        # Sequentially add kernels and check for exclusion criteria

        keep = torch.zeros(K, dtype=torch.bool, device=X.device)
        marginal_ev = torch.empty((K,R), dtype=X.dtype, device=X.device)
        marginal_ev_demix = torch.empty((K,R), dtype=X.dtype, device=X.device)

        # Always accept the first kernel
        idx = order[0]
        keep[idx] = True
        prev_ev = individual_ev[idx,:]
        prev_ev_demix = individual_ev_demix[idx,:]
        marginal_ev[idx,:] = prev_ev
        marginal_ev_demix[idx,:] = prev_ev_demix
        iter_ev = [prev_ev]
        iter_ev_demix = [prev_ev_demix]

        H_current = individual_H[idx:idx+1]
        W_current = W_mix[idx:idx+1,:,:]
        best_demix_W_mix = W_current
        best_H_demix = H_current
        best_W_demix = None

        print('  Sequentially adding kernels')
        print('    Kept kernel {}. Average EV: {:.3f}'.format(idx, prev_ev.mean()))
        start = time.perf_counter()

        for idx in order[1:]:

            # add the next kernel to the candidate set
            W_candidate = torch.cat([W_current, W_mix[idx:idx+1,:,:]])
            H0 = torch.cat([H_current, individual_H[idx:idx+1]])

            # Jointly solve for all loadings in the current candidate set
            H_candidate, stats = solve_loadings(X, W_candidate, H0=H0, zero_idx=zero_idx, atol=atol, btol=btol, verbose=verbose)

            W_demix = solve_demixing(X, H_candidate, L, zero_idx=zero_idx, ridge_lam=ridge_lam)
            H_demix = extract_sources(X, W_demix, zero_idx=zero_idx)

            # solve for new W_mix
            demix_W_mix = solve_mixing(X, H_demix, L, zero_idx=zero_idx, ridge_lam=ridge_lam)

            # compute marginal ev
            X_hat = reconstruct(H_candidate, W_candidate, zero_idx=zero_idx)
            X_hat_demix = reconstruct(H_demix, W_candidate, zero_idx=zero_idx)
            new_ev = channel_explained_variance(X_hat)
            new_ev_demix = channel_explained_variance(X_hat_demix)

            delta_ev = new_ev - prev_ev
            delta_ev_demix = new_ev_demix - prev_ev_demix
            marginal_ev[idx,:] = delta_ev
            marginal_ev_demix[idx,:] = delta_ev_demix
            iter_ev.append(new_ev)
            iter_ev_demix.append(new_ev_demix)

            if torch.any(delta_ev_demix > ev_thresh):
                keep[idx] = True
                prev_ev = new_ev
                prev_ev_demix = new_ev_demix
                H_current = H_candidate
                W_current = W_candidate
                best_demix_W_mix = demix_W_mix
                best_W_demix = W_demix
                best_H_demix = H_demix

                print('    Kept kernel {}. Average EV: {:.3f}. Elapsed Time: {:.1f} s'.format(idx, new_ev_demix.mean(), time.perf_counter()-start))
            else:
                print('    Discarded kernel {}. Average EV: {:.3f}. Elapsed Time: {:.1f} s'.format(idx, new_ev_demix.mean(), time.perf_counter()-start))

        iter_ev = torch.stack(iter_ev)
        iter_ev_demix = torch.stack(iter_ev_demix)

        print('  Kept {} kernels. Reconstruction EV: {:.3f}'.format(keep.sum(), prev_ev_demix.mean()))

        return (keep.cpu().numpy(),
                {'W_mix_orig': W_current.cpu().numpy(), 'H_lsq': H_current.cpu().numpy(), 'ev': prev_ev.cpu().numpy(),
                 'W_mix_est': best_demix_W_mix.cpu().numpy(), 'W_demix_est': best_W_demix.cpu().numpy(),
                 'H_est': best_H_demix.cpu().numpy(), 'ev_est': prev_ev_demix.cpu().numpy()},
                {'individual_ev': individual_ev.cpu().numpy(), 'marginal_ev': marginal_ev.cpu().numpy(),
                 'iter_ev': iter_ev.cpu().numpy(), 'individual_H': individual_H.cpu().numpy(),
                 'individual_ev_demix': individual_ev_demix.cpu().numpy(), 'marginal_ev_demix': marginal_ev_demix.cpu().numpy(),
                 'iter_ev_demix': iter_ev_demix.cpu().numpy()})


def calc_ev(X: torch.Tensor, H: torch.Tensor, W_mix: torch.Tensor):

    K, R, _ = W_mix.shape

    X_mean = X.nanmean(dim=1, keepdim=True)
    data_var = (X - X_mean)**2

    X_hat_components = reconstruct(H, W_mix, full=False)
    X_hat = X_hat_components.sum(dim=0)

    valid = torch.all(data_var.isfinite(), dim=0) & torch.all(X_hat.isfinite(), dim=0)

    data_var_region = data_var[:,valid].sum(dim=1)
    data_var_full = data_var_region.sum()

    ev_component = torch.zeros(K)
    ev_component_region = torch.zeros((K, R))
    ev_total = 1 - torch.sum((X[:,valid] - X_hat[:,valid])**2) / data_var_full
    ev_region = 1 - torch.sum((X[:,valid] - X_hat[:,valid])**2, dim=1) / data_var_region

    for i in range(K):

        Xhat_i = X_hat_components[i,:,:]

        ev_component[i] = 1 - torch.sum((X[:,valid] - Xhat_i[:,valid])**2) / data_var_full
        ev_component_region[i,:] =  1 - torch.sum((X[:,valid] - Xhat_i[:,valid])**2, dim=1) / data_var_region

    return {'ev_total': ev_total.cpu().numpy(), 'ev_component': ev_component.cpu().numpy(), 'ev_region': ev_region.cpu().numpy(), 'ev_component_region': ev_component_region.cpu().numpy()}


def solve_conv_autoencoder(X: ArrayLike, W_mix: ArrayLike, zero_idx=None, ev_thresh=0.005,
                           order=None, max_iter=50, error_tol=1e-4, eval_iter=3,
                           ridge_lam=0.1, update_alpha=0.5, atol=1e-4, btol=1e-4, verbose=False):
    """
    Solve the convolutional autoencoder using alternating least squares

    The auto-encoder process is X (*) W_demix = H, H (*) W_mix = X_hat, where (*) is convolution and H are sources of shape (K,T)

    Parameters
    ----------
    X : (R,T) ArrayLike
        Observed signals.
    W_mix : (K,R,L) ArrayLike
        Candidate mixing kernels.
    ev_thresh : float, optional
        Minimum marginal explained variance required for a kernel to be accepted
    order : ArrayLike, optional
        The order with which to sequentially add the kernels
    max_iter : int, optional
        Maximum number of ALS iterations.
    atol, btol : float, optional
        LSMR convergence tolerances.
    ridge_lam : float, optional
        Ridge penalty for solving W_demix and W_mix
    update_alpha : float, optional
        The update fraction to W at each iteration. 1 means completely replace the prior W

    Returns
    -------
    Model: dict containing 'W_mix': (K,R,L) torch.Tensor, 'W_demix': (K,R,L) torch.Tensor
        The best-fit model parameters
    Results: dict containing 'H': (K,T) torch.Tensor, 'X_hat': (R,T) torch.Tensor, and explained variances
    """

    K, R, L = W_mix.shape

    with torch.no_grad():

        X = to_tensor(X)
        W_mix = to_tensor(W_mix)

        # First find unique kernels
        keep, results, details = find_unique_kernels(X, W_mix, zero_idx=zero_idx, ev_thresh=ev_thresh,
                                                     order=order, atol=atol, btol=btol, ridge_lam=ridge_lam, verbose=verbose)

        # start with the best-fit W_mix and W_demix from the method to find unique kernels
        best_W_mix = (1-update_alpha)*results['W_mix_orig'] + update_alpha*results['W_mix_est']
        best_W_demix = results['W_demix_est']
        best_H = results['H_est']

        ev_details = calc_ev(X, best_H, best_W_mix)
        best_ev = ev_details['ev_total']

        iter_W_mix = best_W_mix
        iter_W_demix = best_W_demix
        iter_H = best_H
        prev_ev = best_ev

        n_converged = 0

        print('Finding best-fit demixing and mixing kernels...')
        start = time.perf_counter()

        for i in range(max_iter):
            # find the best-fit H from W_demix
            iter_H, _ = solve_loadings(X, iter_W_mix, H0=iter_H, zero_idx=zero_idx, atol=atol, btol=btol, verbose=verbose)

            # use the new best-fit H to find the best-fit W_demix
            new_W_demix = solve_demixing(X, iter_H, L, zero_idx=zero_idx, ridge_lam=ridge_lam)

            # update the prior W_demix
            iter_W_demix = (1-update_alpha)*iter_W_demix + update_alpha*new_W_demix

            # use the new W_demix to extract sources
            iter_H = extract_sources(X, iter_W_demix, zero_idx=zero_idx)

            # use these extracted sources to find the best-fit W_mix
            new_W_mix = solve_mixing(X, iter_H, L, zero_idx=zero_idx, ridge_lam=ridge_lam)

            # update the prior W_mix
            iter_W_mix = (1-update_alpha)*iter_W_mix + update_alpha*new_W_mix

            # calculate explained variance
            iter_ev_details = calc_ev(X, iter_H, iter_W_mix)
            iter_ev = iter_ev_details['ev_total']

            if abs(iter_ev - prev_ev) < error_tol:
                n_converged += 1
            else:
                n_converged = 0

            prev_ev = iter_ev
            if iter_ev > best_ev:
                best_W_mix = iter_W_mix
                best_W_demix = iter_W_demix
                best_H = iter_H
                best_ev = iter_ev

                print('  Found improved kernels at iter {}. Reconstruction EV: {:.4f}. Elapsed Time: {:.1f} s'.format(i, best_ev, time.perf_counter()-start))

            if n_converged >= eval_iter:
                break

        X_hat_components = reconstruct(best_H, best_W_mix, zero_idx=zero_idx, full=False)
        X_hat = X_hat_components.sum(dim=0)
        ev_details = calc_ev(X, best_H, best_W_mix)

        print('  Completed in {} iterations. Reconstruction EV: {:.3f}. Elapsed Time: {:.1f} s'.format(i, ev_details['ev_total'], time.perf_counter()-start))

    return ({'W_mix': best_W_mix, 'W_demix': best_W_demix, 'H': best_H},
            {'X_hat': X_hat.cpu().numpy(), 'X_hat_components': X_hat_components.cpu().numpy(), 'ev_details': ev_details, 'keep': keep.cpu().numpy()})


def calc_h_corr(H: torch.Tensor):

    valid_sel = torch.isfinite(H).all(dim=0)
    valid_H = H[:,valid_sel]

    valid_H = valid_H / torch.linalg.vector_norm(valid_H, dim=1, keepdim=True)
    corr = valid_H @ valid_H.T

    return corr.cpu().numpy()

# %% Least-squares solver port

def torch_lsmr(A: Callable[torch.Tensor, torch.Tensor], AT: Callable[torch.Tensor, torch.Tensor], b: torch.Tensor,
               x0: torch.Tensor=None, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8, maxiter=None, verbose=False):
    '''
    Pytorch rewrite of Scipy's LSMR method: https://github.com/scipy/scipy/blob/v1.18.0/scipy/sparse/linalg/_isolve/lsmr.py
    '''

    hdg = '  itn       x(1)       norm r    norm Ar   compatible   LS     norm A   cond A'
    stop_msg = ('The exact solution is x = 0, or x = x0, if x0 was given  ',
               'Ax - b is small enough, given atol, btol                  ',
               'The least-squares solution is good enough, given atol     ',
               'The estimate of cond(Abar) has exceeded conlim            ',
               'Ax - b is small enough for this machine                   ',
               'The least-squares solution is good enough for this machine',
               'Cond(Abar) seems to be too large for this machine         ',
               'The iteration limit has been reached                      ')
    print_freq = 50
    hdg_freq = 20
    pcount = hdg_freq

    with torch.no_grad():

        dtype = b.dtype
        device = b.device

        def make_scalar(value):
            return torch.tensor(value, dtype=dtype, device=device)

        def _sym_ortho(a, b):
            """
            Pytorch rewrite of scipy's _sym_ortho method: https://github.com/scipy/scipy/blob/v1.18.0/scipy/sparse/linalg/_isolve/lsqr.py
            """
            if b == 0:
                c = torch.sign(a)
                s = torch.zeros_like(a)
                r = torch.abs(a)
            elif a == 0:
                c = torch.zeros_like(a)
                s = torch.sign(b)
                r = torch.abs(b)
            elif torch.abs(b) > torch.abs(a):
                tau = a / b
                s = torch.sign(b) / torch.sqrt(1 + tau * tau)
                c = s * tau
                r = b / s
            else:
                tau = b / a
                c = torch.sign(a) / torch.sqrt(1 + tau * tau)
                s = c * tau
                r = a / c

            return c, s, r

        b = b.squeeze()
        if b.ndim > 1:
            raise ValueError('b must be one-dimensional')

        adjoint = AT(b).squeeze()
        if adjoint.ndim > 1:
            raise ValueError('adjoint operator must produce a one-dimensional output')

        m = b.numel()
        n = adjoint.numel()

        if maxiter is None:
            maxiter = min(m, n)

        u = b.clone()
        norm_b = vector_norm(b)

        if x0 is None:
            x = torch.zeros_like(adjoint)
            beta = norm_b
        else:
            x = x0.clone().squeeze()
            u = u - A(x)
            beta = vector_norm(u)

        if beta > 0:
            u = u / beta
            v = AT(u)
            alpha = vector_norm(v)
        else:
            v = torch.zeros_like(adjoint)
            alpha = make_scalar(0)

        if alpha > 0:
            v = v / alpha

        damp = make_scalar(damp)

        # initialize variables for first iteration

        zeta_bar = alpha * beta
        alpha_bar = alpha

        rho = make_scalar(1)
        rho_bar = make_scalar(1)
        c_bar = make_scalar(1)
        s_bar = make_scalar(0)
        h = v.clone()
        h_bar = torch.zeros_like(v)

        # Initialize variables for estimation of ||r||.

        beta_dd = beta
        beta_d = make_scalar(0)
        rho_d_old = make_scalar(1)
        tau_tilde_old = make_scalar(0)
        theta_tilde = make_scalar(0)
        zeta = make_scalar(0)
        d = make_scalar(0)

        # Initialize variables for estimation of ||A|| and cond(A)

        norm_A2 = alpha**2
        max_rbar = make_scalar(0)
        min_rbar = make_scalar(1e100)
        norm_A = torch.sqrt(norm_A2)
        cond_A = make_scalar(1)
        norm_x = make_scalar(0)

        i_stop = 0
        i_iter = 0
        ctol = 1 / conlim if conlim > 0 else 0
        norm_r = beta

        if norm_b == 0 or zeta_bar == 0:
            if verbose:
                print(stop_msg[0])
            return x, i_stop, i_iter, beta, alpha * beta, norm_A, cond_A, norm_x

        while i_iter < maxiter:
            i_iter += 1

            # Perform the next step of the bidiagonalization to obtain the
            # next  beta, u, alpha, v.  These satisfy the relations
            #         beta*u  =  A@v   -  alpha*u,
            #        alpha*v  =  A'@u  -  beta*v.

            u = A(v) - alpha * u
            beta = vector_norm(u)

            if beta > 0:
                u = u / beta
                v = AT(u) - beta * v
                alpha = vector_norm(v)

                if alpha > 0:
                    v = v / alpha

            # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

            # Construct rotation Qhat_{k,2k+1}.

            c_hat, s_hat, alpha_hat = _sym_ortho(alpha_bar, damp)

            # Use a plane rotation (Q_i) to turn B_i to R_i

            rho_old = rho
            c, s, rho = _sym_ortho(alpha_hat, beta)
            theta_new = s*alpha
            alpha_bar = c*alpha

            # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

            rho_bar_old = rho_bar
            zeta_old = zeta
            theta_bar = s_bar * rho
            rho_temp = c_bar * rho
            c_bar, s_bar, rho_bar = _sym_ortho(rho_temp, theta_new)
            zeta = c_bar * zeta_bar
            zeta_bar = -s_bar * zeta_bar

            # Update h, h_hat, x

            h_bar = h - (theta_bar * rho / (rho_old * rho_bar)) * h_bar
            x = x + (zeta / (rho * rho_bar)) * h_bar
            h = v - (theta_new / rho) * h

            # Estimate residual norm ||r||

            # Apply rotation Qhat_{k,2k+1}
            beta_acute = c_hat * beta_dd
            beta_check = -s_hat * beta_dd

            # Apply rotation Q_{k,k+1}
            beta_hat = c * beta_acute
            beta_dd = -s * beta_acute

            # Apply rotation Qtilde_{k-1}
            # beta_d = beta_d_{k-1} here
            theta_tilde_old = theta_tilde
            c_tilde_old, s_tilde_old, rho_tilde_old = _sym_ortho(rho_d_old, theta_bar)
            theta_tilde = s_tilde_old * rho_bar
            rho_d_old = c_tilde_old * rho_bar
            beta_d = -s_tilde_old * beta_d + c_tilde_old * beta_hat

            # beta_d   = beta_d_k here
            # rhod_old = rho_d_k  here

            tau_tilde_old = (zeta_old - theta_tilde_old * tau_tilde_old) / rho_tilde_old
            tau_d = (zeta - theta_tilde * tau_tilde_old) / rho_d_old
            d = d + beta_check**2
            norm_r = torch.sqrt(d + (beta_d - tau_d)**2 + beta_dd**2)

            # Estimate ||A||
            norm_A2 = norm_A2 + beta**2
            norm_A = torch.sqrt(norm_A2)
            norm_A2 = norm_A2 + alpha**2

            # Estimate cond(A).
            max_rbar = torch.maximum(max_rbar, rho_bar_old)
            if i_iter > 1:
                min_rbar = torch.minimum(min_rbar, rho_bar_old)

            cond_A = torch.maximum(max_rbar, rho_temp) / torch.minimum(min_rbar, rho_temp)

            # Test for convergence.

            # Compute norms for convergence testing.
            norm_ar = torch.abs(zeta_bar)
            norm_x = vector_norm(x)

            # Now use these norms to estimate certain other quantities,
            # some of which will be small near a solution.

            test1 = norm_r / norm_b
            if (norm_A * norm_r) != 0:
                test2 = norm_ar / (norm_A * norm_r)
            else:
                test2 = torch.inf

            test3 = 1 / cond_A
            t1 = test1 / (1 + norm_A * norm_x / norm_b)
            rtol = btol + atol * norm_A * norm_x / norm_b

            # The following tests guard against extremely small values of
            # atol, btol or ctol.  (The user may have set any or all of
            # the parameters atol, btol, conlim  to 0.)
            # The effect is equivalent to the normAl tests using
            # atol = eps,  btol = eps,  conlim = 1/eps.

            if i_iter >= maxiter:
                i_stop = 7
            if 1 + test3 <= 1:
                i_stop = 6
            if 1 + test2 <= 1:
                i_stop = 5
            if 1 + t1 <= 1:
                i_stop = 4

            # Allow for tolerances set by the user.

            if test3 <= ctol:
                i_stop = 3
            if test2 <= atol:
                i_stop = 2
            if test1 <= rtol:
                i_stop = 1

            if verbose:
                if i_iter % print_freq == 0:
                    if pcount == hdg_freq:
                        pcount = 0
                        print(hdg)
                    pcount = pcount + 1
                    str1 = f'{i_iter:6g} {x[0]:12.5e}'
                    str2 = f' {norm_r:10.3e} {norm_ar:10.3e}'
                    str3 = f'  {test1:8.1e} {test2:8.1e}'
                    str4 = f' {norm_A:8.1e} {cond_A:8.1e}'
                    print(''.join([str1, str2, str3, str4]))

            if i_stop != 0:
                break

        # Print the stopping condition.

    if verbose:
        print(' ')
        print('LSMR finished')
        print(stop_msg[i_stop])
        str1 = f'i stop ={i_stop:3g}  norm r ={norm_r:8.1e}  '
        str2 = f'norm A ={norm_A:8.1e}  norm Ar ={norm_ar:8.1e}  '
        str3 = f'i iter ={i_iter:7g}  cond A ={cond_A:8.1e}  norm x ={norm_x:8.1e}'
        print(''.join([str1, str2, str3]))

    return x, i_stop, i_iter, norm_r, norm_ar, norm_A, cond_A, norm_x


# %% Pytorch Dictionary Learning

class ConvDictionaryLearning(nn.Module):
    """
    Convolutional dictionary learning with signed templates and signed loadings.

    The input is a continuous recording ``X`` with shape ``(n_regions, n_time)``.
    The model learns ``W`` with shape ``(n_templates, n_regions, kernel_length)``
    and ``H`` with shape ``(n_templates, n_time)`` such that

        X_hat = sum_k W_k * H_k

    where ``*`` is implemented with ``torch.nn.functional.conv1d``. The finite
    recording is treated with zero padding so the output keeps the original
    length without wrapping the session end back to the beginning.
    """

    def __init__(
        self,
        n_templates: int,
        kernel_length: int,
        n_regions: Optional[int] = None,
        W_init: Optional[ArrayLike] = None,
        lambda_sparse: float = 0.0,
        lambda_smooth: float = 0.0,
        lambda_template: float = 0.0,
        lambda_ortho: float = 0.0,
        lambda_x_ortho: float = 0.0,
        lambda_w_l1: float = 0.0,
        lambda_ica: float = 0.0,
        lr_W: float = 1e-3,
        lr_H: float = 1e-3,
        n_iter: int = 1000,
        early_stopping_window: int = 10,
        early_stopping_tol: Optional[float] = 1e-6,
        h_init_scale: float = 1e-3,
        h_init_method: str = "auto",
        norm_h: bool = False,
        residual_init_passes: int = 1,
        eps: float = 1e-8,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        if W_init is not None:
            if W_init.ndim != 3:
                raise ValueError("W_init must have shape (n_templates, n_regions, kernel_length).")
            n_templates, n_regions_from_init, kernel_length = W_init.shape
            if n_regions is not None and n_regions != n_regions_from_init:
                raise ValueError("n_regions does not match W_init.shape[1].")
            n_regions = n_regions_from_init

        if n_templates <= 0:
            raise ValueError("n_templates must be positive.")
        if kernel_length <= 0:
            raise ValueError("kernel_length must be positive.")
        if n_regions is not None and n_regions <= 0:
            raise ValueError("n_regions must be positive.")
        if residual_init_passes <= 0:
            raise ValueError("residual_init_passes must be positive.")

        self.n_templates = int(n_templates)
        self.n_regions = None if n_regions is None else int(n_regions)
        self.kernel_length = int(kernel_length)
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_template = float(lambda_template)
        self.lambda_ortho = float(lambda_ortho)
        self.lambda_x_ortho = float(lambda_x_ortho)
        self.lambda_w_l1 = float(lambda_w_l1)
        self.lambda_ica = float(lambda_ica)
        self.lr_W = float(lr_W)
        self.lr_H = float(lr_H)
        self.n_iter = int(n_iter)
        self.early_stopping_window = int(early_stopping_window)
        self.early_stopping_tol = early_stopping_tol
        self.h_init_scale = float(h_init_scale)
        self.h_init_method = h_init_method
        self.norm_h = norm_h
        self.residual_init_passes = int(residual_init_passes)
        self.eps = float(eps)
        self.verbose = bool(verbose)
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._w_init_provided = W_init is not None
        self.loss_history: Dict[str, List[float]] = {
            "total": [],
            "reconstruction": [],
            "sparsity": [],
            "smoothness": [],
            "template_similarity": [],
            "reconstruction_similarity": [],
            "x_ortho": [],
            "w_l1": [],
            "ica": []
        }

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.W: Optional[nn.Parameter] = None
        self.H: Optional[nn.Parameter] = None
        self.X_: Optional[torch.Tensor] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.stop_reason_: Optional[str] = None
        self.n_iter_: int = 0
        if W_init is not None:
            W_tensor = torch.as_tensor(W_init, dtype=self.dtype, device=self.device)
            self.W = nn.Parameter(W_tensor.clone())
        elif self.n_regions is not None:
            self._initialize_W()

    def _initialize_W(self) -> None:
        """Initialize templates with small random signed values."""
        W = torch.randn(
            self.n_templates,
            self.n_regions,
            self.kernel_length,
            dtype=self.dtype,
            device=self.device,
        )
        self.W = nn.Parameter(W)

    def _resolve_h_init_method(self) -> str:
        """Return the concrete H initialization method requested by the user."""
        method = self.h_init_method.lower()
        if method == "auto":
            return "residual" if self._w_init_provided else "random"
        if method not in {"random", "matched_filter", "residual"}:
            raise ValueError("h_init_method must be 'auto', 'random', 'matched_filter', or 'residual'.")
        return method

    def _initialize_H(self, n_time: int, X: Optional[torch.Tensor] = None) -> None:
        """Initialize H with noise, matched filtering, or residual matched filtering."""
        method = self._resolve_h_init_method()
        if X is None or method == "random":
            H = self.h_init_scale * torch.randn(
                self.n_templates,
                n_time,
                dtype=self.dtype,
                device=self.device,
            )
        elif method == "matched_filter":
            H = self.matched_filter_loadings(X)
        else:
            H = self.residual_matched_filter_loadings(X, n_passes=self.residual_init_passes)
        self.H = nn.Parameter(H)

        self.normalize_loadings_(preserve_reconstruction=False)

    def _as_tensor(self, X: ArrayLike) -> torch.Tensor:
        """Move NumPy or torch input onto the model device with the model dtype."""
        X_tensor = torch.as_tensor(X, dtype=self.dtype, device=self.device)
        if X_tensor.ndim != 2:
            raise ValueError("X must have shape (n_regions, n_time).")
        return X_tensor

    def _fill_missing(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a zero-filled signal and a Boolean mask of observed samples."""
        observed_mask = torch.isfinite(X)
        X_filled = torch.where(observed_mask, X, torch.zeros_like(X))
        return X_filled, observed_mask

    def _check_initialized(self) -> None:
        if self.W is None or self.H is None:
            raise RuntimeError("The model is not fitted yet. Call fit(X) first.")

    def _same_length_pad(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Pad a batched signal so conv1d returns the original sample count.

        For a kernel of length L, conv1d reduces length by L - 1. Adding a total
        of L - 1 small values around the signal preserves n_time without wraparound.
        """
        total_pad = self.kernel_length - 1
        left_pad = total_pad // 2
        right_pad = total_pad - left_pad
        if total_pad == 0:
            return signal
        return F.pad(signal, (left_pad, right_pad), mode="constant", value=self.eps)

    def reconstruction_from_tensors(
        self,
        W: torch.Tensor,
        H: torch.Tensor,
        separate_templates: bool = False,
    ) -> torch.Tensor:
        """
        Reconstruct X from arbitrary template and loading tensors.

        ``conv1d`` computes cross-correlation, so the temporal axis of ``W`` is
        flipped before filtering to implement mathematical convolution.

        When ``separate_templates`` is false, the return value has shape
        ``(n_regions, n_time)`` and equals the sum of all template contributions.
        When true, the return value has shape
        ``(n_templates, n_regions, n_time)``, so the full reconstruction is simply
        ``template_contributions.sum(dim=0)``.
        """
        H_batch = H.unsqueeze(0)
        H_padded = self._same_length_pad(H_batch)

        if separate_templates:
            n_templates, n_regions, _ = W.shape
            conv_weight = W.flip(-1).reshape(n_templates * n_regions, 1, self.kernel_length)
            template_contrib = F.conv1d(H_padded, conv_weight, groups=n_templates)
            return template_contrib.squeeze(0).reshape(n_templates, n_regions, -1)

        conv_weight = W.permute(1, 0, 2).flip(-1)
        return F.conv1d(H_padded, conv_weight).squeeze(0)

    def matched_filter_loadings(
        self,
        X: torch.Tensor,
        W: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Estimate signed loadings by correlating every template with X.

        With unit-normalized templates, this is a signed matched-filter amplitude:
        positive values indicate template-like structure, while negative values
        indicate inverted template-like structure.
        """
        if W is None:
            if self.W is None:
                raise RuntimeError("Templates are not initialized.")
            W = self.W.detach()
        X_batch = X.unsqueeze(0)
        X_padded = self._same_length_pad(X_batch)
        return F.conv1d(X_padded, W).squeeze(0)

    def residual_matched_filter_loadings(
        self,
        X: torch.Tensor,
        n_passes: int = 1,
    ) -> torch.Tensor:
        """
        Initialize H by ordered residual matched filtering.

        Templates are ranked by the total energy of their initial matched-filter
        responses. In that order, each template is correlated with the current
        residual and its reconstruction is subtracted before the next template is
        initialized. Extra passes repeat this coordinate-style warm start.
        """
        if self.W is None:
            raise RuntimeError("Templates are not initialized.")

        W = self.W.detach()
        initial_H = self.matched_filter_loadings(X, W=W)
        template_order = torch.argsort(torch.sum(initial_H**2, dim=1), descending=True)
        H = torch.zeros(self.n_templates, X.shape[1], dtype=X.dtype, device=X.device)
        residual = X.clone()

        for pass_idx in range(max(1, int(n_passes))):
            if pass_idx > 0:
                residual = X - self.reconstruction_from_tensors(W, H)
            for k_tensor in template_order:
                k = int(k_tensor.item())

                # Put this template's current contribution back before refitting
                # it, so each pass behaves like coordinate descent on H.
                if pass_idx > 0:
                    residual = residual + self.reconstruction_from_tensors(W[k:k + 1], H[k:k + 1])

                H_k = self.matched_filter_loadings(residual, W=W[k:k + 1])
                candidate_recon = self.reconstruction_from_tensors(W[k:k + 1], H_k)
                numerator = torch.sum(residual * candidate_recon)
                denominator = torch.clamp(torch.sum(candidate_recon**2), min=self.eps)
                H_k = H_k * (numerator / denominator)
                H[k].copy_(H_k.squeeze(0))
                residual = residual - self.reconstruction_from_tensors(W[k:k + 1], H[k:k + 1])

        return H

    def reconstruct(self) -> torch.Tensor:
        """Return the reconstruction for the fitted data as a torch tensor."""
        self._check_initialized()
        return self.reconstruction_from_tensors(self.W, self.H)

    def reconstruction_loss(
        self,
        X_hat: torch.Tensor,
        X: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mean squared reconstruction error over observed samples only."""
        if observed_mask is None:
            return F.mse_loss(X_hat, X)

        mask = observed_mask.to(dtype=X_hat.dtype)
        sq_err = (X_hat - X) ** 2
        return torch.sum(sq_err * mask) / torch.clamp(torch.sum(mask), min=1.0)

    def sparsity_loss(self, H: Optional[torch.Tensor] = None, observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Smooth L1-like loading magnitude.

        The square root expression behaves like ``abs(H)`` away from zero but has
        finite gradients around zero, which is friendlier to Adam.
        """
        self._check_initialized()
        H = self.H if H is None else H
        if observed_mask is not None:
            mask = torch.any(observed_mask, dim=0).to(dtype=H.dtype)[None,:]
            return torch.mean(torch.sqrt((H*mask)**2 + self.eps), dim=1).sum()
        else:
            return torch.mean(torch.sqrt(H**2 + self.eps), dim=1).sum()
        # return torch.mean(H**2)

    def smoothness_loss(self, W: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Average squared first temporal derivative of each template."""
        self._check_initialized()
        W = self.W if W is None else W
        if W.shape[-1] < 2:
            return torch.zeros((), dtype=W.dtype, device=W.device)
        return torch.mean((W[..., 1:] - W[..., :-1]) ** 2)

    def template_similarity_loss(self, W: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Average squared off-diagonal cosine similarity between templates.

        Squaring the cosine similarity penalizes both duplicated templates
        (cosine near 1) and inverted duplicated templates (cosine near -1).
        """
        self._check_initialized()
        W = self.W if W is None else W
        if W.shape[0] < 2:
            return torch.zeros((), dtype=W.dtype, device=W.device)

        W_flat = W.reshape(W.shape[0], -1)
        W_norm = W_flat / torch.clamp(torch.linalg.norm(W_flat, dim=1, keepdim=True), min=self.eps)
        cosine = W_norm @ W_norm.T
        off_diag = ~torch.eye(W.shape[0], dtype=torch.bool, device=W.device)
        return torch.mean(cosine[off_diag] ** 2)

    def reconstruction_similarity_loss(
        self,
        template_reconstructions: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Average squared off-diagonal cosine similarity between template reconstructions.

        ``template_reconstructions`` has shape
        ``(n_templates, n_regions, n_time)``. Squaring the cosine similarity
        penalizes both duplicated reconstructions (cosine near 1) and inverted
        reconstructions (cosine near -1). When a mask is supplied, each pairwise
        cosine uses only observed samples.
        """

        if template_reconstructions.shape[0] < 2:
            return torch.zeros((), dtype=template_reconstructions.dtype, device=template_reconstructions.device)

        X_flat = template_reconstructions.reshape(template_reconstructions.shape[0], -1)

        if observed_mask is not None:
            flat_mask = observed_mask.reshape(-1)
            if not torch.any(flat_mask):
                return torch.zeros((), dtype=template_reconstructions.dtype, device=template_reconstructions.device)
            X_flat = X_flat[:, flat_mask]

        X_norm = X_flat / torch.clamp(torch.linalg.norm(X_flat, dim=1, keepdim=True), min=self.eps)
        cosine = X_norm @ X_norm.T
        return torch.sum(torch.triu(cosine**2, diagonal=1))

    def x_ortho_loss(
        self,
        X: torch.Tensor,
        H: Optional[torch.Tensor] = None,
        W: Optional[torch.Tensor] = None,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        SeqNMF-style x-ortho penalty: off-diagonal entries of (W * X) H^T.

        ``W * X`` is implemented as the matched-filter response of every template
        against the data. Off-diagonal entries are large when template k matches
        the signal at times where loading j is active, so minimizing this term
        discourages different factors from explaining the same structure.
        """
        if self.W is None and W is None:
            raise RuntimeError("Templates are not initialized.")
        if self.H is None and H is None:
            raise RuntimeError("Loadings are not initialized.")

        W = self.W if W is None else W
        H = self.H if H is None else H
        if W.shape[0] < 2:
            return torch.zeros((), dtype=X.dtype, device=X.device)

        XW = self.matched_filter_loadings(X, W=W)
        if observed_mask is not None:
            time_mask = torch.any(observed_mask, dim=0).to(dtype=XW.dtype)
            XW = XW * time_mask[None, :]
            H = H * time_mask[None, :]
            denom = torch.clamp(torch.sum(time_mask), min=1.0)
        else:
            denom = torch.tensor(max(H.shape[1], 1), dtype=XW.dtype, device=XW.device)

        overlap = (XW @ H.T) / denom
        off_diag = ~torch.eye(W.shape[0], dtype=torch.bool, device=X.device)
        return torch.mean(overlap[off_diag] ** 2)

    def w_l1_loss(self, W: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Smooth elementwise L1 penalty on W to shrink unused kernel samples."""
        if self.W is None and W is None:
            raise RuntimeError("Templates are not initialized.")
        W = self.W if W is None else W
        # return torch.mean(torch.sqrt(W**2 + self.eps))
        return torch.sqrt(W**2 + self.eps).mean(dim=(1,2)).sum()

    def ica_loss(self, H: Optional[torch.Tensor] = None, observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Log cosh independent loss
        """
        self._check_initialized()
        H = self.H if H is None else H

        if observed_mask is not None:
            mask = torch.any(observed_mask, dim=0).to(dtype=H.dtype)[None,:]
            return -torch.log(torch.cosh(H*mask)).mean(dim=1).sum()
        else:
            return -torch.log(torch.cosh(H)).mean(dim=1).sum()

    def compute_losses(
        self,
        X: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total objective and each named component."""
        self._check_initialized()
        template_recon = self.reconstruction_from_tensors(self.W, self.H, separate_templates=True)
        X_hat = template_recon.sum(dim=0)
        rec = self.reconstruction_loss(X_hat, X, observed_mask=observed_mask)
        sparse = self.sparsity_loss(self.H, observed_mask=observed_mask)
        smooth = self.smoothness_loss(self.W)
        template_sim = self.template_similarity_loss(self.W)
        recon_sim = self.reconstruction_similarity_loss(template_recon, observed_mask=observed_mask)
        x_ortho = self.x_ortho_loss(X, self.H, self.W, observed_mask=observed_mask)
        w_l1 = self.w_l1_loss(self.W)
        ica = self.ica_loss(self.H, observed_mask=observed_mask)
        # Positive lambdas are true penalties: minimizing this objective favors
        # accurate reconstruction, smaller loadings, smoother templates, and less
        # template redundancy.
        total = (
            rec
            + self.lambda_sparse * sparse
            + self.lambda_smooth * smooth
            + self.lambda_template * template_sim
            + self.lambda_ortho * recon_sim
            + self.lambda_x_ortho * x_ortho
            + self.lambda_w_l1 * w_l1
            + self.lambda_ica * ica
        )
        return total, {
            "total": total,
            "reconstruction": rec,
            "sparsity": sparse,
            "smoothness": smooth,
            "template_similarity": template_sim,
            "reconstruction_similarity": recon_sim,
            "x_ortho": x_ortho,
            "w_l1": w_l1,
            "ica": ica
        }

    def rescale_w(self):
        ''' Rescale W to unit RMS '''
        with torch.no_grad():
            # make each component's RMS equal 1
            w_rms = torch.sqrt(self.W.square().mean(dim=(1,2)) + self.eps).view(-1, 1, 1)
            self.W.div_(w_rms)

    def normalize_loadings_(self, preserve_reconstruction: bool = True) -> None:
        """
        Project each loading H_k to unit L2 norm.

        Scaling H_k by ``1 / norm_k`` and W_k by ``norm_k`` leaves each
        convolution W_k * H_k unchanged. With this convention, template amplitude
        lives in W, so L1 and group penalties on W can shrink unused samples and
        whole templates toward zero.
        """
        if self.H is None or not self.norm_h:
            return

        with torch.no_grad():
            norms = torch.linalg.norm(self.H, dim=1)
            norms = torch.clamp(norms, min=self.eps)
            self.H.div_(norms[:, None])
            if preserve_reconstruction and self.W is not None:
                self.W.mul_(norms[:, None, None])

    def _shift_1d_zero_fill(self, values: torch.Tensor, shift: int) -> torch.Tensor:
        """Shift the last axis by ``shift`` samples, filling exposed entries with zero."""
        shifted = torch.full_like(values, self.eps)
        if shift == 0:
            shifted.copy_(values)
        elif shift > 0:
            shifted[..., shift:] = values[..., :-shift]
        else:
            shifted[..., :shift] = values[..., -shift:]
        return shifted

    def recenter_templates_(self) -> None:
        """
        Shift template energy centers toward the middle sample.

        The template is translated with zero fill so edge samples can fall off the
        finite kernel. The corresponding loading is shifted in the opposite
        direction, also with zero fill. This is an approximate projection at the
        recording boundaries, which is appropriate when boundary samples are a
        negligible fraction of the recording.
        """
        self._check_initialized()

        with torch.no_grad():
            positions = torch.arange(self.kernel_length, dtype=self.W.dtype, device=self.W.device)
            middle = (self.kernel_length - 1) / 2.0
            energy = (self.W**2).sum(dim=1)
            energy_sum = torch.clamp(energy.sum(dim=1), min=self.eps)
            center_of_mass = (energy * positions[None, :]).sum(dim=1) / energy_sum
            shifts = torch.round(middle - center_of_mass).to(torch.long)

            for k, shift_tensor in enumerate(shifts):
                shift = int(shift_tensor.item())
                if shift == 0:
                    continue
                self.W[k].copy_(self._shift_1d_zero_fill(self.W[k], shift))
                self.H[k].copy_(self._shift_1d_zero_fill(self.H[k], -shift))

    def _make_optimizer(self, fix_W=False) -> torch.optim.Optimizer:
        """Create Adam with separate learning rates for W and H."""
        self._check_initialized()
        if fix_W:
            return torch.optim.Adam(
                [
                    {"params": [self.H], "lr": self.lr_H},
                ]
            )
        else:
            return torch.optim.Adam(
                [
                    {"params": [self.W], "lr": self.lr_W},
                    {"params": [self.H], "lr": self.lr_H},
                ]
            )

    def _record_losses(self, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Store and return a detached float diagnostics dictionary."""
        diagnostics = {name: float(value.detach().cpu()) for name, value in loss_dict.items()}
        for name, value in diagnostics.items():
            self.loss_history[name].append(value)
        return diagnostics

    def _recent_loss_slope(self) -> Optional[float]:
        """Return the least-squares slope of the most recent total losses."""
        if self.early_stopping_tol is None or self.early_stopping_tol <= 0:
            return None
        if self.early_stopping_window <= 1:
            return None
        losses = self.loss_history["total"]
        if len(losses) < self.early_stopping_window:
            return None

        y = torch.tensor(losses[-self.early_stopping_window:], dtype=torch.float64)
        x = torch.arange(self.early_stopping_window, dtype=torch.float64)
        x = x - torch.mean(x)
        y = y - torch.mean(y)
        return float(torch.sum(x * y) / torch.clamp(torch.sum(x**2), min=self.eps))

    def fit(self, X: ArrayLike, fix_W = False) -> List[Dict[str, float]]:
        """
        Fit W and H to a recording.

        Returns
        -------
        diagnostics
            A list with one dictionary per iteration containing total loss,
            reconstruction loss, sparsity loss, smoothness loss, and template
            similarity loss.
        """
        X_tensor = self._as_tensor(X)
        X_filled, observed_mask = self._fill_missing(X_tensor)
        self.X_ = X_tensor
        n_regions, n_time = X_tensor.shape

        if self.n_regions is None:
            self.n_regions = int(n_regions)
            self._initialize_W()
        elif self.n_regions != n_regions:
            raise ValueError("X.shape[0] does not match n_regions.")

        if self.W is None:
            self._initialize_W()
        if self.W.shape != (self.n_templates, self.n_regions, self.kernel_length):
            raise ValueError("W has shape {}, expected {}.".format(
                tuple(self.W.shape),
                (self.n_templates, self.n_regions, self.kernel_length),
            ))

        self._initialize_H(n_time, X_filled)
        self.loss_history = {name: [] for name in self.loss_history}
        self.optimizer = self._make_optimizer(fix_W)
        diagnostics: List[Dict[str, float]] = []
        self.stop_reason_ = None
        self.n_iter_ = 0

        for iteration in range(self.n_iter):
            self.optimizer.zero_grad()
            total_loss, loss_dict = self.compute_losses(X_filled, observed_mask=observed_mask)
            total_loss.backward()
            self.optimizer.step()

            # Recompute diagnostics after projection so reported values match the
            # actual stored W/H at the end of this iteration.
            if not fix_W:
                with torch.no_grad():
                    self.recenter_templates_()
                    self.normalize_loadings_()
                    _, loss_dict = self.compute_losses(X_filled, observed_mask=observed_mask)

            iter_diag = self._record_losses(loss_dict)
            diagnostics.append(iter_diag)
            self.n_iter_ = iteration + 1

            recent_slope = self._recent_loss_slope()
            if recent_slope is not None and abs(recent_slope) < self.early_stopping_tol:
                self.stop_reason_ = (
                    "loss slope {:.3g} over the last {} iterations fell below tolerance {:.3g}".format(
                        recent_slope,
                        self.early_stopping_window,
                        self.early_stopping_tol,
                    )
                )
                if self.verbose:
                    print("early stopping at iter {:04d}: {}".format(iteration + 1, self.stop_reason_))
                break

            if self.verbose and (iteration == 0 or (iteration + 1) % 100 == 0):
                print("iter {:04d}: loss={:.6g}, recon={:.6g}".format(
                    iteration + 1,
                    iter_diag["total"],
                    iter_diag["reconstruction"],
                ))
        if self.stop_reason_ is None:
            self.stop_reason_ = "completed max iterations"

        return diagnostics

    # def transform(
    #     self,
    #     X: ArrayLike,
    #     n_iter: int = 500,
    #     lr_H: Optional[float] = None,
    #     h_init: Optional[ArrayLike] = None,
    # ) -> torch.Tensor:
    #     """
    #     Infer loadings for new data while keeping fitted templates fixed.

    #     This solves only for H using reconstruction loss plus the configured
    #     loading sparsity penalty. The learned template bank ``W`` is not changed.
    #     """
    #     if self.W is None:
    #         raise RuntimeError("Templates are not initialized. Call fit(X) or pass W_init first.")

    #     X_tensor = self._as_tensor(X)
    #     X_filled, observed_mask = self._fill_missing(X_tensor)
    #     if X_tensor.shape[0] != self.W.shape[1]:
    #         raise ValueError("X.shape[0] must match the number of template regions.")

    #     if h_init is None:
    #         method = self._resolve_h_init_method()
    #         if method == "random":
    #             H = self.h_init_scale * torch.randn(
    #                 self.n_templates,
    #                 X_filled.shape[1],
    #                 dtype=self.dtype,
    #                 device=self.device,
    #             )
    #         elif method == "matched_filter":
    #             H = self.matched_filter_loadings(X_filled)
    #         else:
    #             H = self.residual_matched_filter_loadings(X_filled, n_passes=self.residual_init_passes)
    #     else:
    #         H = torch.as_tensor(h_init, dtype=self.dtype, device=self.device).clone()
    #         if H.shape != (self.n_templates, X_filled.shape[1]):
    #             raise ValueError("h_init must have shape (n_templates, n_time).")

    #     H = nn.Parameter(H)
    #     optimizer = torch.optim.Adam([{"params": [H], "lr": self.lr_H if lr_H is None else lr_H}])
    #     W_fixed = self.W.detach()

    #     for _ in range(int(n_iter)):
    #         optimizer.zero_grad()
    #         template_recon = self.reconstruction_from_tensors(W_fixed, H, separate_templates=True)
    #         X_hat = template_recon.sum(dim=0)
    #         rec = self.reconstruction_loss(X_hat, X_filled, observed_mask=observed_mask)
    #         sparse = torch.mean(torch.sqrt(H**2 + self.eps))
    #         recon_sim = self.reconstruction_similarity_loss(template_recon, observed_mask=observed_mask)
    #         x_ortho = self.x_ortho_loss(X_filled, H, W_fixed, observed_mask=observed_mask)
    #         ica = self.ica_loss(self.H)
    #         loss = rec + self.lambda_sparse * sparse + self.lambda_ortho * recon_sim + self.lambda_x_ortho * x_ortho + self.lambda_ica * ica
    #         loss.backward()
    #         optimizer.step()

    #     return H.detach()

    def inverse_transform(self, H: ArrayLike) -> torch.Tensor:
        """Reconstruct data from supplied loadings using the fitted templates."""
        if self.W is None:
            raise RuntimeError("Templates are not initialized. Call fit(X) or pass W_init first.")
        H_tensor = torch.as_tensor(H, dtype=self.dtype, device=self.device)
        if H_tensor.shape[0] != self.W.shape[0]:
            raise ValueError("H.shape[0] must match n_templates.")
        with torch.no_grad():
            return self.reconstruction_from_tensors(self.W.detach(), H_tensor)

    def get_W(self) -> np.ndarray:
        """Return fitted templates on CPU as a NumPy array."""
        if self.W is None:
            raise RuntimeError("Templates are not initialized.")
        return self.W.detach().cpu().numpy()

    def get_H(self) -> np.ndarray:
        """Return fitted loadings on CPU as a NumPy array."""
        self._check_initialized()
        return self.H.detach().cpu().numpy()


    def plot_templates(self, template_idxs: Optional[Iterable[int]] = None, title=None):
        """Plot each template as region traces over its kernel window."""
        if self.W is None:
            raise RuntimeError("Templates are not initialized.")

        W = self.get_W()

        template_idxs = list(range(W.shape[0])) if template_idxs is None else list(template_idxs)

        n_cols = self.n_regions
        n_rows = int(np.ceil(len(template_idxs)/n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=[4*n_cols,4*n_rows], sharey=True, sharex=True)
        axs = np.reshape(axs, (n_rows, n_cols))

        if not title is None:
            fig.suptitle(title)

        for i, k in enumerate(template_idxs):
            comp_row = int(np.floor(i/n_cols))
            comp_col = int(i % n_cols)

            # plot component
            ax = axs[comp_row,comp_col]
            ax.plot(W[k].T)
            ax.set_title('Kernel {}'.format(k))
            ax.set_ylabel('Weight')
            ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)

        return fig, axs

    def plot_loadings(self):
        """Plot loadings for every component."""

        plot_H = self.H.detach().cpu().numpy()

        fig, axs = plt.subplots(self.n_templates, 1, figsize=(4*self.n_templates,12), layout='constrained', sharex=True)
        fig.suptitle('Component Loadings')
        for i in range(self.n_templates):
            ax = axs[i]
            ax.plot(plot_H[i,:])
            ax.set_title('Component {}'.format(i))
            ax.set_ylabel('Loading')
        axs[-1].set_xlabel('Time')

    def plot_losses(self, figsize: Tuple[float, float] = (8, 4)):
        """Plot tracked loss curves over optimization iterations."""

        fig, ax = plt.subplots(figsize=figsize)
        for name, values in self.loss_history.items():
            if values:
                ax.plot(values, label=name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss value")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_reconstruction(self, X: ArrayLike):
        """Plot original and reconstructed traces for every region."""

        X_hat_ind = self.reconstruction_from_tensors(self.W, self.H, separate_templates=True).detach().cpu().numpy()
        X_hat_full = np.sum(X_hat_ind, axis=0)

        fig, axs = plt.subplots(self.n_regions, 1, figsize=(4*self.n_regions,12), layout='constrained', sharex=True)
        for i in range(self.n_regions):
            ax = axs[i]
            ax.plot(X[i,:], label='Signal', linewidth=1)
            ax.plot(X_hat_full[i,:], label='Reconstruction', linewidth=1)
            for j in range(self.n_templates):
                ax.plot(X_hat_ind[j,i,:], label='Template {}'.format(j), linewidth=1, alpha=0.6)
            ax.set_title('Region {}'.format(i))
            ax.legend()
        axs[-1].set_xlabel('Time')

    def plot_residual(self, X: Optional[ArrayLike] = None, figsize: Tuple[float, float] = (10, 5)):
        """Plot residual traces X - X_hat for every region."""
        self._check_initialized()

        X_tensor = self.X_ if X is None else self._as_tensor(X)
        if X_tensor is None:
            raise RuntimeError("No fitted X is stored; pass X explicitly.")
        residual = X_tensor - self.reconstruct()
        residual_np = residual.detach().cpu().numpy()

        fig, axes = plt.subplots(residual_np.shape[0], 1, figsize=figsize, squeeze=False, sharex=True)
        for region, ax in enumerate(axes[:, 0]):
            ax.plot(residual_np[region], linewidth=1)
            ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
            ax.set_ylabel("Region {}".format(region))
        axes[-1, 0].set_xlabel("Time")
        fig.tight_layout()
        return fig, axes[:, 0]


# %% Pytorch Convolutional Autoencoder

class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for ICA

    The input is a continuous recording X with shape (n_regions, n_time).
    The model learns W_demixing and W_mixing with shape (n_kernels, n_regions, kernel_length)
    such that convolving W_demixing produces a set of independent sources, H, that can subsequently
    be convolved by W_mixing to reproduce the original data:

        H_k = X * W_demix_k
        X_hat = sum_k [W_k * H_k]

    where `*` is convolution
    """

    def __init__(
        self,
        n_kernels: Optional[int] = 0,
        n_regions: Optional[int] = 0,
        kernel_length: Optional[int] = 0,
        W_demix_init: Optional[ArrayLike] = None,
        W_mix_init: Optional[ArrayLike] = None,
        W_alpha_init: Optional[ArrayLike] = None,
        zero_idx: Optional = None,
        n_iter: int = 10000,
        early_stopping_window: int = 20,
        early_stopping_tol: Optional[float] = 1e-6,
        lambda_ica: float = 0,
        lambda_w: float = 0.01,
        lambda_w_alpha: float = 0,
        lambda_recon_sim: float = 1,
        lambda_smooth: float = 0.1,
        lr: float = 1e-3,
        eps: float = 1e-6,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float64,
        random_state: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__()

        if W_demix_init is not None:
            if W_demix_init.ndim != 3:
                raise ValueError("W_demix_init must have shape (n_kernels, n_regions, kernel_length).")
            n_kernels, n_regions, kernel_length = W_demix_init.shape

        if W_mix_init is not None:
            if W_mix_init.ndim != 3:
                raise ValueError("W_mix_init must have shape (n_kernels, n_regions, kernel_length).")

            if W_demix_init is not None:
                if W_demix_init.shape != W_mix_init.shape:
                    raise ValueError("W_demix_init and W_mix_init must have the same shape")
            else:
                n_kernels, n_regions, kernel_length = W_mix_init.shape

        if n_kernels <= 0:
            raise ValueError("n_kernels must be positive.")
        if kernel_length <= 0:
            raise ValueError("kernel_length must be positive.")
        if n_regions <= 0:
            raise ValueError("n_regions must be positive.")

        self.n_kernels = int(n_kernels)
        self.n_regions = int(n_regions)
        self.kernel_length = int(kernel_length)
        self.lambda_ica = float(lambda_ica)
        self.lambda_w = float(lambda_w)
        self.lambda_w_alpha = float(lambda_w_alpha)
        self.lambda_recon_sim = float(lambda_recon_sim)
        self.lambda_smooth = float(lambda_smooth)
        self.n_iter = int(n_iter)
        self.early_stopping_window = int(early_stopping_window)
        self.early_stopping_tol = early_stopping_tol
        self.eps = float(eps)
        self.lr = float(lr)
        self.verbose = bool(verbose)
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_history: Dict[str, List[float]] = {
            "total": [],
            "reconstruction": [],
            "ica": [],
            "w": [],
            "w_alpha": [],
            "reconstruction_sim": [],
            "w_smooth": []
        }

        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.stop_reason_: Optional[str] = None
        self.n_iter_: int = 0

        self.zero_idx = zero_idx if zero_idx is not None else self.kernel_length // 2

        if W_demix_init is not None:
            self.W_demix = nn.Parameter(torch.as_tensor(W_demix_init, dtype=self.dtype, device=self.device))
        else:
            self.W_demix = self._rand_W()

        if W_mix_init is not None:
            self.W_mix = nn.Parameter(torch.as_tensor(W_mix_init, dtype=self.dtype, device=self.device))
        else:
            self.W_mix = self._rand_W()

        if W_alpha_init is not None:
            self.W_alpha = nn.Parameter(torch.as_tensor(W_alpha_init, dtype=self.dtype, device=self.device))
        else:
            self.W_alpha = nn.Parameter(torch.ones(self.n_kernels, dtype=self.dtype, device=self.device))

    def _rand_W(self) -> None:
        """Initialize templates with small random signed values."""
        return nn.Parameter(torch.randn(self.n_kernels, self.n_regions, self.kernel_length,
            dtype=self.dtype, device=self.device))

    def _as_tensor(self, X: ArrayLike) -> torch.Tensor:
        """Move NumPy or torch input onto the model device with the model dtype."""
        return torch.as_tensor(X, dtype=self.dtype, device=self.device)

    def _fill_missing(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a zero-filled signal and a Boolean mask of observed samples."""
        observed_mask = torch.isfinite(X)
        X_filled = torch.where(observed_mask, X, torch.zeros_like(X))
        return X_filled, observed_mask

    def _pad_signal(self, signal: torch.Tensor, operation) -> torch.Tensor:
        """
        Pad a batched signal so conv1d returns the original sample count

        For a kernel of length L, conv1d reduces length by L - 1. Adding a total
        of L - 1 small values around the signal preserves n_time without wraparound.
        """

        if operation == 'demix':
            left_pad = self.zero_idx
            right_pad = self.kernel_length - self.zero_idx - 1

        elif operation == 'mix':
            left_pad = self.kernel_length - self.zero_idx - 1
            right_pad = self.zero_idx

        return F.pad(signal, (left_pad, right_pad), mode='constant', value=0)

    def extract_sources(self, X: torch.Tensor) -> torch.Tensor:
        """
        Extract sources from X as (n_kernels, n_time) by convolving with W_demix.

        ``conv1d`` computes cross-correlation, so the temporal axis of ``W`` is
        flipped before filtering to implement mathematical convolution.

        """
        X_batch = X.unsqueeze(0)
        X_padded = self._pad_signal(X_batch, 'demix')

        H = F.conv1d(X_padded, self.W_demix)
        return H.squeeze(0)

    def reconstruct(self, H: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct X from extracted sources as separate components (n_kernels, n_regions, n_time).
        Get the full reconstruction by summing the reconstructions accross kernels

        ``conv1d`` computes cross-correlation, so the temporal axis of ``W`` is
        flipped before filtering to implement mathematical convolution.

        """
        H_batch = H.unsqueeze(0)
        H_padded = self._pad_signal(H_batch, 'mix')

        conv_weight = self.W_mix.flip(-1).reshape(self.n_kernels * self.n_regions, 1, self.kernel_length)
        X_hat = F.conv1d(H_padded, conv_weight, groups=self.n_kernels)
        X_hat = X_hat.squeeze(0).reshape(self.n_kernels, self.n_regions, -1) * self.W_alpha.view(-1, 1, 1)
        return X_hat

    def extract_and_reconstruct(self, X: ArrayLike):
        with torch.no_grad():
            X_filled, observed_mask = self._fill_missing(self._as_tensor(X))
            sources = self.extract_sources(X_filled)
            x_hat = self.reconstruct(sources)
        return sources, x_hat

    def reconstruction_loss(self, X_hat: torch.Tensor, X: torch.Tensor, observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean squared reconstruction error over observed samples only."""
        if observed_mask is None:
            return F.mse_loss(X_hat, X)

        mask = observed_mask.to(dtype=X_hat.dtype)
        sq_err = (X_hat - X) ** 2
        return torch.sum(sq_err * mask) / torch.clamp(torch.sum(mask), min=1.0)

    def ica_loss(self, H) -> torch.Tensor:
        """
        Log cosh independent components loss
        """
        # return -torch.log(torch.cosh(H)).mean(dim=1).sum()
        return -torch.log(torch.cosh(H)).mean(dim=1).mean()

    def w_alpha_loss(self) -> torch.Tensor:
        """ L1 penalty on W_alpha """

        return torch.sqrt(self.W_alpha**2).sum()

    def w_loss(self) -> torch.Tensor:
        """ Grouped L1/L2 penalty on both W_demix and W_mix pairs to keep W values down and shrink unused kernel samples."""

        # return torch.sqrt(torch.mean(self.W_demix**2, dim=(1,2)) + torch.mean(self.W_mix**2, dim=(1,2)) + self.eps).sum()
        # return torch.sqrt((self.W_demix.square().sum(dim=(1,2)) + self.W_mix.square().sum(dim=(1,2)))/2).sum()
        return torch.sum((self.W_demix.square().sum(dim=(1,2)) + self.W_mix.square().sum(dim=(1,2)))/2)

    def smoothness_loss(self) -> torch.Tensor:
        """L2 loss of first temporal derivative of each mixing template."""
        return torch.sum((self.W_mix[..., 1:] - self.W_mix[..., :-1])**2)

    def reconstruction_similarity_loss(self, X_recon: torch.Tensor, observed_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Average squared off-diagonal cosine similarity between kernel reconstructions.

        ``X_recon`` has shape ``(n_kernels, n_regions, n_time)``. Squaring the cosine similarity
        penalizes both duplicated reconstructions (cosine near 1) and inverted
        reconstructions (cosine near -1). When a mask is supplied, each pairwise
        cosine uses only observed samples.
        """

        X_flat = X_recon.reshape(X_recon.shape[0], -1)

        if observed_mask is not None:
            flat_mask = observed_mask.reshape(-1)
            if not torch.any(flat_mask):
                return torch.zeros((), dtype=X_recon.dtype, device=X_recon.device)
            X_flat = X_flat[:, flat_mask]

        X_norm = X_flat / torch.clamp(torch.linalg.norm(X_flat, dim=1, keepdim=True), min=self.eps)
        cosine = X_norm @ X_norm.T
        return torch.sum(torch.triu(cosine**2, diagonal=1))  #, cosine.detach().cpu().numpy()

    def compute_losses(self, H: torch.Tensor, X_recon: torch.Tensor, X: torch.Tensor, observed_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total objective and each named component."""

        X_hat = X_recon.sum(dim=0)
        rec_loss = self.reconstruction_loss(X_hat, X, observed_mask=observed_mask)
        ica_loss = self.ica_loss(H)
        w_loss = self.w_loss()
        w_alpha_loss = self.w_alpha_loss()
        smooth_loss = self.smoothness_loss()
        recon_sim_loss = self.reconstruction_similarity_loss(X_recon, observed_mask=observed_mask)

        total = (
            rec_loss
            + self.lambda_ica * ica_loss
            + self.lambda_w * w_loss
            + self.lambda_w_alpha * w_alpha_loss
            + self.lambda_recon_sim * recon_sim_loss
            + self.lambda_smooth * smooth_loss
        )
        return total, {
            "total": total,
            "reconstruction": rec_loss,
            "ica": ica_loss,
            "w": w_loss,
            "w_alpha": w_alpha_loss,
            "reconstruction_sim": recon_sim_loss,
            "w_smooth": smooth_loss
        }

    def calc_ev(self, X: ArrayLike):
        with torch.no_grad():
            X = self._as_tensor(X)
            X_filled, observed_mask = self._fill_missing(X)
            _, X_recon = self.extract_and_reconstruct(X)

            mask = observed_mask.to(dtype=X.dtype)
            X_mean = X.nanmean(dim=1, keepdim=True)
            data_var = (X_filled - X_mean)**2

            X_hat = X_recon.sum(dim=0)

            data_var_region = torch.sum(data_var * mask, dim=1)
            data_var_full = data_var_region.sum()

            sq_err = (X_filled - X_hat)**2 * mask

            ev_total = 1 - sq_err.sum() / data_var_full
            ev_region = 1 - sq_err.sum(dim=1) / data_var_region

            ev_component = torch.zeros(self.n_kernels)
            ev_component_region = torch.zeros((self.n_kernels, self.n_regions))

            ev_component_marg = torch.zeros(self.n_kernels)
            ev_component_region_marg = torch.zeros((self.n_kernels, self.n_regions))

            for i in range(self.n_kernels):
                X_hat_i = X_recon[i,:,:]
                X_hat_marg_i = X_hat - X_hat_i

                sq_err_i = (X_filled - X_hat_i)**2 * mask
                sq_err_marg_i = (X_filled - X_hat_marg_i)**2 * mask

                ev_component[i] = 1 - sq_err_i.sum() / data_var_full
                ev_component_region[i,:] = 1 - sq_err_i.sum(dim=1) / data_var_region

                ev_component_marg[i] = ev_total - (1 - sq_err_marg_i.sum() / data_var_full)
                ev_component_region_marg[i,:] = ev_region - (1 - sq_err_marg_i.sum(dim=1) / data_var_region)

            return {'ev_total': ev_total.cpu().numpy(), 'ev_region': ev_region.cpu().numpy(),
                    'ev_component': ev_component.cpu().numpy(), 'ev_component_region': ev_component_region.cpu().numpy(),
                    'ev_component_marg': ev_component_marg.cpu().numpy(), 'ev_component_region_marg': ev_component_region_marg.cpu().numpy()}


    def _make_optimizer(self, fix_w_mix=False, fix_w_demix=False, fix_w_alpha=False) -> torch.optim.Optimizer:
        """Create Adam optimizer"""
        params = []
        if not fix_w_mix:
            params.append(self.W_mix)
        if not fix_w_demix:
            params.append(self.W_demix)
        if not fix_w_alpha:
            params.append(self.W_alpha)

        return torch.optim.Adam([{"params": params, "lr": self.lr}])

    def _record_losses(self, loss_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Store and return a detached float diagnostics dictionary."""
        diagnostics = {name: float(value.detach().cpu()) for name, value in loss_dict.items()}
        for name, value in diagnostics.items():
            self.loss_history[name].append(value)
        return diagnostics

    def _recent_loss_slope(self) -> Optional[float]:
        """Return the least-squares slope of the most recent total losses."""
        if self.early_stopping_tol is None or self.early_stopping_tol <= 0:
            return None
        if self.early_stopping_window <= 1:
            return None
        losses = self.loss_history["total"]
        if len(losses) < self.early_stopping_window:
            return None

        y = torch.tensor(losses[-self.early_stopping_window:], dtype=torch.float64)
        x = torch.arange(self.early_stopping_window, dtype=torch.float64)
        x = x - torch.mean(x)
        y = y - torch.mean(y)
        return float(torch.sum(x * y) / torch.clamp(torch.sum(x**2), min=self.eps))


    def find_unique_kernels(self, X: torch.Tensor, ev_thresh=0.005, order=None, recalc_W_demix=True):
        """
        Identify redundant mixing kernels using marginal explained variance.

        A candidate kernel is rejected if its marginal explained variance is below ev_thresh

        Kernels are initially ordered by their explained variance when fit individually to the full data.

        Parameters
        ----------
        X : (R,T) torch.Tensor
            Observed signals.
        ev_thresh : float, optional
            Minimum marginal explained variance required for a kernel to be accepted
        order : ArrayLike, optional
            The order with which to sequentially add the kernels

        Returns
        -------
        keep : (K,) torch.Tensor
            Boolean mask indicating retained kernels.
        results: dict containing the reduced 'W_mix', 'W_demix', and 'W_alpha'
        details: dict containing explained variance details
        """

        print('Finding unique kernels...')

        # Fit each kernel independently and optionally rank by individual EV
        X = self._as_tensor(X)

        start = time.perf_counter()

        individual_ev = np.empty((self.n_kernels, self.n_regions))

        orig_W_mix = self.W_mix.detach().clone()
        orig_W_demix = self.W_demix.detach().clone()
        orig_n_kernels = self.n_kernels

        ind_W_mix = []
        ind_W_demix = []
        ind_W_alpha = []
        ind_ev_details = []

        self.n_kernels = 1

        for k in range(orig_n_kernels):

            W_mix_k = orig_W_mix[k:k+1,:,:].detach().clone()
            if recalc_W_demix:
                H_k, stats = solve_loadings(X, W_mix_k, zero_idx=self.zero_idx)
                W_demix_k = solve_demixing(X, H_k, self.kernel_length, zero_idx=self.zero_idx)
            else:
                W_demix_k = orig_W_demix[k:k+1,:,:].detach().clone()

            self.W_mix = nn.Parameter(W_mix_k)
            self.W_demix = nn.Parameter(W_demix_k)
            self.W_alpha = nn.Parameter(torch.ones(self.n_kernels, dtype=self.dtype, device=self.device))

            _ = self.fit(X, fix_w_mix=True)
            _ = self.fit(X, fix_w_demix=True)

            ind_W_mix.append(self.W_mix.detach().clone())
            ind_W_demix.append(self.W_demix.detach().clone())
            ind_W_alpha.append(self.W_alpha.detach().clone())

            ind_ev = self.calc_ev(X)

            ind_ev_details.append(ind_ev)
            individual_ev[k,:] = ind_ev['ev_region']

        print('  Fit individual kernels in {:.1f} s'.format(time.perf_counter()-start))

        if order is None:
            order = sort_kernels_by_region_ev(individual_ev)

        # Sequentially add kernels and check for exclusion criteria

        keep = torch.zeros(orig_n_kernels, dtype=torch.bool, device=self.device)
        marginal_ev = np.empty((orig_n_kernels, self.n_regions))

        # Always accept the first kernel
        idx = order[0]
        keep[idx] = True
        prev_ev = individual_ev[idx,:]
        marginal_ev[idx,:] = prev_ev
        iter_ev = [prev_ev]
        iter_ev_details = [ind_ev_details[idx]]

        best_W_mix = ind_W_mix[idx]
        best_W_demix = ind_W_demix[idx]
        best_W_alpha = ind_W_alpha[idx]
        best_ev_details = ind_ev_details[idx]

        print('  Sequentially adding kernels')
        print('    Kept kernel {}. Total EV: {:.3f}'.format(idx, ind_ev_details[idx]['ev_total']))
        start = time.perf_counter()

        for idx in order[1:]:

            # add the next kernel to the candidate set
            W_mix_candidate = torch.cat([best_W_mix, ind_W_mix[idx]])
            W_demix_candidate = torch.cat([best_W_demix, ind_W_demix[idx]])
            W_alpha_candidate = torch.cat([best_W_alpha, ind_W_alpha[idx]])

            self.W_mix = nn.Parameter(W_mix_candidate.detach().clone())
            self.W_demix = nn.Parameter(W_demix_candidate.detach().clone())
            self.W_alpha = nn.Parameter(W_alpha_candidate.detach().clone())
            self.n_kernels = self.W_mix.shape[0]

            _ = self.fit(X, fix_w_mix=True)
            _ = self.fit(X, fix_w_demix=True)

            # compute marginal ev
            new_ev = self.calc_ev(X)
            new_ev_tot = new_ev['ev_region']

            delta_ev = new_ev_tot - prev_ev
            marginal_ev[idx,:] = delta_ev
            iter_ev.append(new_ev_tot)
            iter_ev_details.append(new_ev)

            if np.any(delta_ev > ev_thresh):
                keep[idx] = True
                prev_ev = new_ev_tot
                best_W_mix = self.W_mix.detach().clone()
                best_W_demix = self.W_demix.detach().clone()
                best_W_alpha = self.W_alpha.detach().clone()
                best_ev_details = new_ev

                print('    Kept kernel {}. Total EV: {:.3f}. Elapsed Time: {:.1f} s'.format(idx, new_ev['ev_total'], time.perf_counter()-start))
            else:
                print('    Discarded kernel {}. Total EV: {:.3f}. Elapsed Time: {:.1f} s'.format(idx, new_ev['ev_total'], time.perf_counter()-start))

        iter_ev = np.stack(iter_ev)

        print('  Kept {} kernels. Total EV: {:.3f}'.format(keep.sum(), best_ev_details['ev_total']))

        return (keep, {'W_mix': best_W_mix, 'W_demix': best_W_demix, 'W_alpha': best_W_alpha, 'ev_details': best_ev_details},
                {'individual_ev': individual_ev, 'marginal_ev': marginal_ev, 'iter_ev': iter_ev, 'iter_ev_details': iter_ev_details})


    def fit(self, X: ArrayLike, fix_w_mix=False, fix_w_demix=False, fix_w_alpha=False) -> List[Dict[str, float]]:
        """
        Fit W_demix and W_mix to data

        Returns
        -------
        diagnostics
            A list with one dictionary per iteration containing total loss,
            reconstruction loss, sparsity loss, smoothness loss, and template
            similarity loss.
        """
        X_filled, observed_mask = self._fill_missing(self._as_tensor(X))
        n_regions, n_time = X_filled.shape

        self.loss_history = {name: [] for name in self.loss_history}
        self.optimizer = self._make_optimizer(fix_w_mix=fix_w_mix, fix_w_demix=fix_w_demix, fix_w_alpha=fix_w_alpha)
        diagnostics: List[Dict[str, float]] = []
        self.stop_reason_ = None
        self.n_iter_ = 0

        for iteration in range(self.n_iter):
            self.rescale_w()
            self.optimizer.zero_grad()

            sources = self.extract_sources(X_filled)
            X_recon = self.reconstruct(sources)

            total_loss, loss_dict = self.compute_losses(sources, X_recon, X_filled, observed_mask=observed_mask)
            total_loss.backward()
            self.optimizer.step()

            iter_diag = self._record_losses(loss_dict)
            diagnostics.append(iter_diag)
            self.n_iter_ = iteration + 1

            recent_slope = self._recent_loss_slope()
            if recent_slope is not None and abs(recent_slope) < self.early_stopping_tol:
                self.stop_reason_ = (
                    "loss slope {:.3g} over the last {} iterations fell below tolerance {:.3g}".format(
                        recent_slope,
                        self.early_stopping_window,
                        self.early_stopping_tol,
                    )
                )
                if self.verbose:
                    print("early stopping at iter {:04d}: {}".format(iteration + 1, self.stop_reason_))
                break

            if self.verbose and (iteration == 0 or (iteration + 1) % 100 == 0):
                ev = self.calc_ev(X)
                print("iter {:04d}: loss={:.6g}, recon={:.6g}, ev={:.4f}".format(
                    iteration + 1,
                    iter_diag["total"],
                    iter_diag["reconstruction"],
                    ev['ev_total']
                ))
        if self.stop_reason_ is None:
            self.stop_reason_ = "completed max iterations"

        return diagnostics

    def rescale_w(self):
        ''' Rescale W_demix and W_mix so they have equivalent RMS '''
        with torch.no_grad():

            mix_rms = torch.sqrt(self.W_mix.square().mean(dim=(1,2)) + self.eps)
            demix_rms = torch.sqrt(self.W_demix.square().mean(dim=(1,2)) + self.eps)

            scale = torch.sqrt(demix_rms / mix_rms).view(-1, 1, 1)

            self.W_mix.mul_(scale)
            self.W_demix.div_(scale)

            return scale

    def rescale_w_to_data(self, X):
        ''' Rescale W_demix to best match the reconstruction '''
        with torch.no_grad():
            X_filled, observed_mask = self._fill_missing(self._as_tensor(X))

            # first rescale so that the reconstruction best matches the data
            sources = self.extract_sources(X_filled)
            X_recon = self.reconstruct(sources)

            mask = observed_mask.reshape(-1)

            A = X_recon.reshape(self.n_kernels, -1).T[mask]
            b = X_filled.reshape(-1)[mask]

            scale = torch.linalg.lstsq(A, b).solution
            scale = scale.clamp(min=self.eps).view(-1, 1, 1)

            self.W_demix.mul_(scale)

            return scale


    def calc_impulse_response(self):
        with torch.no_grad():
            G = torch.zeros((self.n_kernels, self.n_kernels, self.kernel_length*2), dtype=self.dtype, device=self.device)
            center = self.kernel_length

            for i in range(self.n_kernels):

                H = torch.zeros((self.n_kernels, self.kernel_length*2), dtype=self.dtype, device=self.device)
                H[i, center] = 1

                X = self.reconstruct(H) / self.W_alpha.view(-1, 1, 1)
                X = X.sum(dim=0)
                H_hat = self.extract_sources(X)

                G[i,:,:] = H_hat

            return G.detach().cpu().numpy()

    def get_W_demix(self) -> np.ndarray:
        return self.W_demix.detach().cpu().numpy()

    def get_W_mix(self) -> np.ndarray:
        return self.W_mix.detach().cpu().numpy()

    def plot_templates(self, title='', dt=None):
        """Plot each template as region traces over its kernel window."""

        for j in range(2):
            if j == 0:
                plot_W = self.get_W_demix()
                fig_title = 'Demixing Kernels. {}'.format(title)
            else:
                plot_W = self.get_W_mix()
                fig_title = 'Mixing Kernels. {}'.format(title)

            plot_W *= self.W_alpha.view(-1, 1, 1).detach().cpu().numpy()

            x = np.arange(plot_W.shape[2])

            if not dt is None:
                x = x*dt

            n_cols = self.n_regions
            n_rows = int(np.ceil(self.n_kernels/n_cols))
            fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(4*n_cols,4*n_rows), sharey=True, sharex=True)
            axs = np.reshape(axs, (n_rows, n_cols))
            fig.suptitle(fig_title)

            for i, k in enumerate(range(self.n_kernels)):
                comp_row = int(np.floor(i/n_cols))
                comp_col = int(i % n_cols)

                # plot component
                ax = axs[comp_row,comp_col]
                ax.plot(x, plot_W[k,:,:].T)
                ax.set_title('Kernel {}'.format(k))
                ax.set_ylabel('Weight')
                ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)

    def plot_losses(self, figsize: Tuple[float, float] = (8, 4)):
        """Plot tracked loss curves over optimization iterations."""

        fig, ax = plt.subplots(figsize=figsize)
        for name, values in self.loss_history.items():
            if values:
                ax.plot(values, label=name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss value")
        ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_reconstruction(self, X: ArrayLike):
        """Plot original and reconstructed traces for every region."""

        _, X_hat = self.extract_and_reconstruct(X)
        X_hat_ind = X_hat.detach().cpu().numpy()
        X_hat_full = np.sum(X_hat_ind, axis=0)

        fig, axs = plt.subplots(self.n_regions, 1, figsize=(4*self.n_regions,12), layout='constrained', sharex=True)
        for i in range(self.n_regions):
            ax = axs[i]
            ax.plot(X[i,:], label='Signal', linewidth=1)
            ax.plot(X_hat_full[i,:], label='Reconstruction', linewidth=1)
            for j in range(self.n_kernels):
                ax.plot(X_hat_ind[j,i,:], label='Template {}'.format(j), linewidth=1, alpha=0.6)
            ax.set_ylabel('Region {}'.format(i))
            ax.legend()
        axs[-1].set_xlabel('Time')

    def plot_loadings(self, X: ArrayLike):
        """Plot loadings for every component."""
        X_filled, _ = self._fill_missing(self._as_tensor(X))
        plot_H = self.extract_sources(X_filled).detach().cpu().numpy()

        fig, axs = plt.subplots(self.n_kernels, 1, figsize=(4*self.n_kernels,12), layout='constrained', sharex=True)
        fig.suptitle('Component Loadings')
        for i in range(self.n_kernels):
            ax = axs[i]
            ax.plot(plot_H[i,:])
            ax.set_title('Component {}'.format(i))
            ax.set_ylabel('Loading')
        axs[-1].set_xlabel('Time')

    def plot_residual(self, X: Optional[ArrayLike] = None, figsize: Tuple[float, float] = (10, 5)):
        """Plot residual traces X - X_hat for every region."""

        X_tensor = self.X_ if X is None else self._as_tensor(X)
        if X_tensor is None:
            raise RuntimeError("No fitted X is stored; pass X explicitly.")
        residual = X_tensor - self.reconstruct()
        residual_np = residual.detach().cpu().numpy()

        fig, axes = plt.subplots(residual_np.shape[0], 1, figsize=figsize, squeeze=False, sharex=True)
        for region, ax in enumerate(axes[:, 0]):
            ax.plot(residual_np[region], linewidth=1)
            ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
            ax.set_ylabel("Region {}".format(region))
        axes[-1, 0].set_xlabel("Time")
        fig.tight_layout()
        return fig, axes[:, 0]

    def plot_impulse_response(self):
        """Plot the impulse response results for the mixing and demixing kernels."""

        g = self.calc_impulse_response()
        n_cols = self.n_regions
        n_rows = int(np.ceil(self.n_kernels/n_cols))
        fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(4*n_cols,4*n_rows), sharey=True, sharex=True)
        axs = np.reshape(axs, (n_rows, n_cols))
        fig.suptitle('Impulse Response Reconstructions')

        for i in range(self.n_kernels):
            comp_row = int(np.floor(i/n_cols))
            comp_col = int(i % n_cols)

            # plot component
            ax = axs[comp_row,comp_col]
            for j in range(self.n_kernels):
                ax.plot(g[i,j,:], label='Component {}'.format(j))
            ax.set_title('Component {} Impulse'.format(i))
            ax.set_ylabel('Reconstruction')
            ax.axhline(0, color="k", linewidth=0.5, alpha=0.3)
            ax.legend()














