"""Validation for real FFT grids used by quadratic light-curve scoring."""

from __future__ import annotations

from numbers import Integral


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def validate_real_fft_grid(
    n_alpha: int,
    highest_mode: int,
    *,
    parameter: str,
    quantity: str,
) -> None:
    """Reject FFT grids that place a required coefficient on Nyquist."""
    if isinstance(n_alpha, bool) or not isinstance(n_alpha, Integral):
        raise ValueError(f"{parameter} must be an even integer >= 4")
    n_alpha = int(n_alpha)
    if n_alpha < 4 or n_alpha % 2:
        raise ValueError(f"{parameter} must be an even integer >= 4")
    if highest_mode < 0:
        raise ValueError("highest Fourier mode cannot be negative")

    nyquist_mode = n_alpha // 2
    if highest_mode < nyquist_mode:
        return

    required_n_alpha = 2 * highest_mode + 2
    recommended = _next_power_of_two(required_n_alpha)
    if highest_mode == nyquist_mode:
        explanation = (
            f"the required mode {highest_mode} is exactly the real-FFT Nyquist "
            "bin. At that endpoint `irfft` assumes a purely real coefficient, "
            "so a general Fourier term is represented incorrectly and the "
            "quadratic light-curve moment can be mis-scored."
        )
    else:
        explanation = (
            f"the required mode {highest_mode} is beyond the real-FFT Nyquist "
            f"bin {nyquist_mode}."
        )
    raise ValueError(
        f"{parameter}={n_alpha} is unsafe for {quantity}: {explanation} "
        f"Use an even {parameter} >= {required_n_alpha}; "
        f"{recommended} is a radix-2 choice."
    )


def validate_quadratic_alpha_grid(
    n_alpha: int, m_max: int, *, parameter: str
) -> None:
    """Validate both magnification and squared-magnification harmonics."""
    if isinstance(m_max, bool) or not isinstance(m_max, Integral) or m_max < 0:
        raise ValueError("m_max must be a non-negative integer")
    m_max = int(m_max)
    validate_real_fft_grid(
        n_alpha,
        2 * m_max,
        parameter=parameter,
        quantity=(
            f"m_max={m_max} (the quadratic moment reaches mode 2*m_max="
            f"{2 * m_max})"
        ),
    )
