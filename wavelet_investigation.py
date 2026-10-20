"""Visualize db4 and sym5 wavelets from PyWavelets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pywt


def get_wavelet_functions(wavelet_name: str, level: int) -> tuple:
    """Return x values, scaling function, and wavelet function for a wavelet."""
    wavelet = pywt.Wavelet(wavelet_name)
    phi, psi, x = wavelet.wavefun(level=level)
    return x, phi, psi


def plot_wavelets(wavelets, level):
    """Plot the scaling and wavelet functions for db4 and sym5."""
    
    fig, axes = plt.subplots(
        nrows=len(wavelets),
        ncols=2,
        figsize=(12, len(wavelets)*3),
        constrained_layout=True,
    )

    for row, wavelet_family in enumerate(wavelets):
        for wavelet_name in wavelet_family:
            x, phi, psi = get_wavelet_functions(wavelet_name, level)

            axes[row, 0].plot(x, phi, label=wavelet_name)
            axes[row, 1].plot(x, psi, label=wavelet_name)
            
        axes[row, 0].set_title("Scaling function")
        axes[row, 0].set_xlabel("x")
        axes[row, 0].set_ylabel("phi")
        axes[row, 0].legend()
        
        axes[row, 1].set_title("Wavelet function")
        axes[row, 1].set_xlabel("x")
        axes[row, 1].set_ylabel("psi")
        axes[row, 1].legend()
    
    fig.suptitle(f"PyWavelets wavelet functions at level {level}", fontsize=14)

    plt.show()

level=8

# wavelets = [('db2', 'db3', 'db4', 'db5'), ('sym2', 'sym3', 'sym4', 'sym5')]
wavelets = [('db2', 'sym2'), ('db3', 'sym3')]
plot_wavelets(wavelets, level)
