# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:06:41 2026

@author: Tanner
"""

import init
import conv_dictionary_learning as cdl
import torch
from scipy.sparse.linalg import lsmr as scipy_lsmr, LinearOperator
import matplotlib.pyplot as plt

# %%
K, R, T, L = 3, 4, 100, 9

H = torch.randn(K, T)
X = torch.randn(R, T)
W_mix = torch.randn(K, R, L)
W_demix = torch.randn(K, R, L)

# %% test solving for W_mix

zero_idx = 2

X1 = cdl.reconstruct(H, W_mix, zero_idx=zero_idx)

W_mix_fit = cdl.solve_mixing(X1, H, L, zero_idx=zero_idx)

torch.testing.assert_close(W_mix, W_mix_fit)

X2 = cdl.reconstruct(H, W_mix_fit, zero_idx=zero_idx)

torch.testing.assert_close(X1, X2, equal_nan=True)

# %% test solving for W_demix

zero_idx = 2

H1 = cdl.extract_sources(X, W_demix, zero_idx=zero_idx)

W_demix_fit = cdl.solve_demixing(X, H1, L, zero_idx=zero_idx)

torch.testing.assert_close(W_demix, W_demix_fit)

H2 = cdl.extract_sources(X, W_demix_fit, zero_idx=zero_idx)

torch.testing.assert_close(H1, H2, equal_nan=True)

# %% test ported LSMR method

A = torch.randn(100, 30, dtype=torch.float64)
b = torch.randn(100, dtype=torch.float64)

Aop = lambda x: A @ x
ATop = lambda x: A.T @ x

x_torch, *stats = cdl.torch_lsmr(Aop, ATop, b, maxiter=100)

x_scipy, *stats_scipy = scipy_lsmr(A.numpy(), b.numpy(), maxiter=100)

plt.figure()
plt.plot(x_torch.detach().cpu().numpy())
plt.plot(x_scipy);

# %% test methods involved in solving for H

zero_idx = 5

AH = cdl.reconstruct(H, W_mix, zero_idx=zero_idx)
# AtX = cdl.apply_mixing_adjoint(X, W_mix)
X_masked = torch.where(torch.isfinite(AH), X, torch.zeros_like(X))

AtX = cdl.apply_mixing_adjoint(X_masked, W_mix, zero_idx=zero_idx)

lhs = torch.nansum(AH * X)
rhs = torch.nansum(H * AtX)

torch.testing.assert_close(lhs, rhs)

# %% test noiseless reconstruction
zero_idx = 4

X_recon = cdl.reconstruct(H, W_mix, zero_idx=zero_idx)
H_est, stats = cdl.solve_loadings(X_recon, W_mix, zero_idx=zero_idx)

fig, axs = plt.subplots(K, 1, layout='constrained')
for k in range(K):
    axs[k].plot(H[k,:].detach().cpu().numpy(), label='True')
    axs[k].plot(H_est[k,:].detach().cpu().numpy(), label='Est')

# %% test reconstruction with noise
zero_idx = 6

X_true = cdl.reconstruct(H, W_mix, zero_idx=zero_idx)
X_true_noise = X_recon + torch.randn(R, T)
H_est, stats = cdl.solve_loadings(X_true_noise, W_mix, zero_idx=zero_idx)
X_recon_noise = cdl.reconstruct(H_est, W_mix, zero_idx=zero_idx)

fig, axs = plt.subplots(R, 1, layout='constrained', sharex=True)
fig.suptitle('Data')
for r in range(R):
    axs[r].plot(X_true[r,:].detach().cpu().numpy(), label='Clean', alpha=0.5)
    axs[r].plot(X_true_noise[r,:].detach().cpu().numpy(), label='Noisy', alpha=0.5)
    axs[r].plot(X_recon_noise[r,:].detach().cpu().numpy(), label='Reconstruction', alpha=0.5)
    axs[r].legend()

fig, axs = plt.subplots(K, 1, layout='constrained', sharex=True)
fig.suptitle('Loadings')
for k in range(K):
    axs[k].plot(H[k,:].detach().cpu().numpy(), label='True', alpha=0.5)
    axs[k].plot(H_est[k,:].detach().cpu().numpy(), label='Est', alpha=0.5)
    axs[k].legend()

# %% test full pipeline
zero_idx = 4

H_demix = cdl.extract_sources(X, W_demix, zero_idx=zero_idx)
X_recon = cdl.reconstruct(H_demix, W_mix, zero_idx=zero_idx)
H_est, stats = cdl.solve_loadings(X_recon, W_mix, zero_idx=zero_idx)
X_est = cdl.reconstruct(H_est, W_mix, zero_idx=zero_idx)

fig, axs = plt.subplots(R, 1, layout='constrained', sharex=True)
fig.suptitle('Data')
for r in range(R):
    axs[r].plot(X_recon[r,:].detach().cpu().numpy(), label='True', alpha=0.5)
    axs[r].plot(X_est[r,:].detach().cpu().numpy(), label='Est', alpha=0.5)
    axs[r].legend()

fig, axs = plt.subplots(K, 1, layout='constrained', sharex=True)
fig.suptitle('Loadings')
for k in range(K):
    axs[k].plot(H_demix[k,:].detach().cpu().numpy(), label='True', alpha=0.5)
    axs[k].plot(H_est[k,:].detach().cpu().numpy(), label='Est', alpha=0.5)
    axs[k].legend()

# %% compare reconstruction with noise to scipy method

def solve_loadings_scipy(X: torch.Tensor, W_mix: torch.Tensor, H0: torch.Tensor=None, maxiter=None, atol=1e-6, btol=1e-6):
    """
    Solve for H given X and fixed W_mix using LSMR

    Parameters
    ----------
    X : (R,T) torch.Tensor
    W_mix : (K,R,L) torch.Tensor
    H0 : (K,T) torch.Tensor, optional
        Initial loading estimate.

    Returns
    -------
    H : (K,T) torch.Tensor
    """

    K, R, L = W_mix.shape
    T = X.shape[1]

    def apply_A(H_flat):
        H = torch.from_numpy(H_flat).to(device=W_mix.device, dtype=W_mix.dtype).reshape(K, -1)
        X = cdl.reconstruct(H, W_mix)
        return X.reshape(-1).cpu().numpy()

    def apply_AT(X_flat):
        X = torch.from_numpy(X_flat).to(device=W_mix.device, dtype=W_mix.dtype).reshape(R, -1)
        H = cdl.apply_mixing_adjoint(X, W_mix)
        return H.reshape(-1).cpu().numpy()

    A = LinearOperator(shape=(R*T,K*T), matvec=apply_A, rmatvec=apply_AT, dtype=float)

    if H0 is not None:
       H0 = H0.reshape(-1).detach().cpu().numpy()

    H, *stats = scipy_lsmr(A, X.reshape(-1), x0=H0, maxiter=maxiter, atol=atol, btol=btol)

    return H.reshape(K, -1), stats

X_true = cdl.reconstruct(H, W_mix)
X_true_noise = X_recon + torch.randn(R, T)
H_est, stats = solve_loadings_scipy(X_true_noise, W_mix)
H_est = torch.from_numpy(H_est).to(device=W_mix.device, dtype=W_mix.dtype)
X_recon_noise = cdl.reconstruct(H_est, W_mix)

fig, axs = plt.subplots(R, 1, layout='constrained')
fig.suptitle('Data')
for r in range(R):
    axs[r].plot(X_true[r,:].detach().cpu().numpy(), label='Clean', alpha=0.5)
    axs[r].plot(X_true_noise[r,:].detach().cpu().numpy(), label='Noisy', alpha=0.5)
    axs[r].plot(X_recon_noise[r,:].detach().cpu().numpy(), label='Reconstruction', alpha=0.5)
    axs[r].legend()

fig, axs = plt.subplots(K, 1, layout='constrained')
fig.suptitle('Loadings')
for k in range(K):
    axs[k].plot(H[k,:].detach().cpu().numpy(), label='True', alpha=0.5)
    axs[k].plot(H_est[k,:].detach().cpu().numpy(), label='Est', alpha=0.5)
    axs[k].legend()