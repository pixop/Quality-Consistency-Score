# qcs.py
"""
Quality Consistency Score (QCS) — minimal reference implementation.

This module implements:
  - Survival curve S(x) = P(Q >= x) (returned in probability or percent scale)
  - QCS[a,b] = (1/(b-a)) * ∫_a^b S(x) dx
  - A small set of temporal pooling baselines used for comparison in the paper.

Design goals:
  - Simple, dependency-light (numpy only)
  - Metric-agnostic: works with any per-frame quality signal (e.g., VMAF/PSNR/SSIM/MOS-over-time)
  - Deterministic and easy to audit
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union, Literal

import numpy as np

# Type aliases for readability
Curve = Tuple[np.ndarray, np.ndarray]  # (x thresholds ascending, S(x) in prob/percent)
Scale = Literal["prob", "percent"]
PathLike = Union[str, Path]


def load_scores_txt(path: PathLike, *, drop_nonfinite: bool = True) -> np.ndarray:

    """
    Load a plain-text file containing one numeric quality score per line
    (e.g., PSNR, SSIM/MS-SSIM, VMAF, or MOS-over-time).

    The file is expected to contain a time series (typically per-frame or per-second)
    quality values for a single video clip under a fixed condition (e.g., an encode).

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the .txt file. Each non-empty line should be parseable as float.
    drop_nonfinite : bool, default=True
        If True, remove NaN/Inf values after parsing.

    Returns
    -------
    scores : np.ndarray
        1D numpy array of parsed scores (dtype=float).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a non-empty line cannot be parsed as float.
    """

    path = Path(path)
    vals = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue  # skip blank lines
            vals.append(float(s))

    scores = np.asarray(vals, dtype=float)

    # Optional cleaning
    if drop_nonfinite:
        scores = scores[np.isfinite(scores)]

    return scores


def survival_curve(scores: np.ndarray, *, scale: Scale = "percent") -> Curve:
    
    """
    Compute the empirical survival curve S(x)=P(Q>=x), evaluated at unique observed thresholds.

    Parameters
    ----------
    scores : np.ndarray
        1D array of per-frame scores Q(t). Non-finite values are ignored.
    scale : {"prob","percent"}, default="percent"
        Output scale for survival values:
          - "prob"    -> S(x) in [0,1]
          - "percent" -> S(x) in [0,100]

    Returns
    -------
    (x, s) : (np.ndarray, np.ndarray)
        x : thresholds (ascending)
        s : survival values, in the requested scale

    Raises
    ------
    ValueError
        If there are no finite values in scores, or if scale is invalid.
    """
    
    q = np.asarray(scores, dtype=float).reshape(-1)
    q = q[np.isfinite(q)]
    if q.size == 0:
        raise ValueError("No finite scores provided.")

    x, counts = np.unique(q, return_counts=True)  # ascending
    ge_counts = np.cumsum(counts[::-1])[::-1]
    s_prob = ge_counts / q.size  # [0,1]

    if scale == "prob":
        return x, s_prob.astype(float)
    if scale == "percent":
        return x, (100.0 * s_prob).astype(float)
    raise ValueError("scale must be 'prob' or 'percent'.")


def qcs(scores: np.ndarray, a: float, b: float, *, grid_points: int = 1001, scale: Scale = "percent") -> float:
    
    """
    Compute the Quality Consistency Score QCS over an interval [a,b].

    Definition
    ----------
    Let S(x) = P(Q >= x) be the survival curve (probability), or
        S%(x) = 100 * P(Q >= x) be the survival curve (percent).
    Then:
        QCS[a,b] = (1/(b-a)) * ∫_a^b S(x) dx
    The returned QCS scale is controlled by `scale`:
      - scale="prob"    -> QCS in [0,1]
      - scale="percent" -> QCS in [0,100] (default)

    Implementation details
    ----------------------
    - Builds the empirical survival curve using unique thresholds.
    - Interpolates S(x) onto a uniform grid over [a,b].
    - Integrates using the trapezoidal rule.

    Boundary semantics (important)
    ------------------------------
    For interpolation outside observed thresholds:
      - For x < min(scores): S(x) = 1   (or 100%)
      - For x > max(scores): S(x) = 0   (or 0%)
    This matches survival-curve semantics.

    Parameters
    ----------
    scores : np.ndarray
        1D array of per-frame scores Q(t). Non-finite values are ignored.
    a : float
        Lower bound of the integration interval.
    b : float
        Upper bound of the integration interval (must be > a).
    grid_points : int, default=1001
        Number of grid points used to approximate the integral.
        Higher values improve numerical stability at a small runtime cost.
    scale : {"prob","percent"}, default="percent"
        Output scale for QCS:
          - "prob"    -> QCS in [0,1]
          - "percent" -> QCS in [0,100]

    Returns
    -------
    qcs_value : float
        QCS in requested scale.

    Raises
    ------
    ValueError
        If b <= a, if scores contain no finite values, or if scale is invalid.
    """
    
    if b <= a:
        raise ValueError("Require b > a.")
    if scale not in ("prob", "percent"):
        raise ValueError("scale must be 'prob' or 'percent'.")

    # Compute survival curve in probability scale internally for correct math.
    x, s_prob = survival_curve(scores, scale="prob")  # s_prob is in [0,1]

    # Uniform grid over [a,b]
    xi = np.linspace(float(a), float(b), int(grid_points))

    # Interpolate S(x) in probability scale.
    # Clamp outside observed range using survival semantics:
    #   x < min(scores) -> 1,  x > max(scores) -> 0
    si_prob = np.interp(xi, x, s_prob, left=1.0, right=0.0)

    # Integrate survival probability, then normalize by (b-a) -> QCS in [0,1]
    area = float(np.trapz(si_prob, xi))
    qcs_prob = area / (b - a)

    # Return in requested scale
    if scale == "prob":
        return qcs_prob
    return 100.0 * qcs_prob


def minkowski_pool(scores: np.ndarray, p: float = 0.5) -> float:
    
    """
    Minkowski (power-mean) temporal pooling of a score sequence.

    Definition
    ----------
        M_p = ( mean(Q^p) )^(1/p)

    Notes
    -----
    - p = 1 gives the arithmetic mean.
    - p < 1 emphasizes low values more than the arithmetic mean.
    - p > 1 emphasizes high values.

    Parameters
    ----------
    scores : np.ndarray
        1D array of per-frame scores. Non-finite values are ignored.
    p : float, default=0.5
        Power-mean exponent.

    Returns
    -------
    pooled : float
        Minkowski pooled score.

    Raises
    ------
    ValueError
        If there are no finite values in scores.
    """
    
    q = np.asarray(scores, dtype=float).reshape(-1)
    q = q[np.isfinite(q)]
    if q.size == 0:
        raise ValueError("No finite scores provided.")
    return float(np.mean(q ** p) ** (1.0 / p))


def harmonic_mean(scores: np.ndarray) -> float:

    """
    Harmonic-mean temporal pooling of a score sequence.

    Definition
    ----------
        H = 1 / mean(1/Q)

    Notes
    -----
    - The harmonic mean penalizes low values more strongly than the arithmetic mean.
    - Requires strictly positive scores (Q > 0).

    Parameters
    ----------
    scores : np.ndarray
        1D array of per-frame scores. Non-finite values are ignored.

    Returns
    -------
    pooled : float
        Harmonic-mean pooled score.

    Raises
    ------
    ValueError
        If there are no finite values in scores or if any finite score is <= 0.
    """

    q = np.asarray(scores, dtype=float).reshape(-1)
    q = q[np.isfinite(q)]
    if q.size == 0:
        raise ValueError("No finite scores provided.")
    if np.any(q <= 0):
        raise ValueError("Harmonic mean requires strictly positive scores (all Q > 0).")
    return float(1.0 / np.mean(1.0 / q))

def worst_kpct_mean(scores: np.ndarray, k_pct: float = 1.0) -> float:
    
    """
    Tail-average pooling: mean of the lowest k% samples.

    Interpretation
    --------------
    This summarizes the *severity* of the worst moments:
      - For N samples, k% corresponds to about ceil(k/100 * N) samples.
      - For example, with N=3600, k=1% averages the lowest ~36 frames.

    Parameters
    ----------
    scores : np.ndarray
        1D array of per-frame scores. Non-finite values are ignored.
    k_pct : float, default=1.0
        Percentage of the lowest samples to average (must be > 0).

    Returns
    -------
    tail_mean : float
        Mean of the lowest k% samples.

    Raises
    ------
    ValueError
        If k_pct <= 0 or if there are no finite values in scores.
    """
    
    if k_pct <= 0:
        raise ValueError("k_pct must be > 0.")

    q = np.asarray(scores, dtype=float).reshape(-1)
    q = q[np.isfinite(q)]
    if q.size == 0:
        raise ValueError("No finite scores provided.")

    qs = np.sort(q)
    k = max(1, int(np.ceil((k_pct / 100.0) * qs.size)))  # number of samples in the tail
    return float(np.mean(qs[:k]))


def percentile(scores: np.ndarray, p: float) -> float:
    
    """
    Percentile pooling of a score sequence (e.g., p=10 -> p10).

    Parameters
    ----------
    scores : np.ndarray
        1D array of per-frame scores. Non-finite values are ignored.
    p : float
        Percentile in [0,100].

    Returns
    -------
    value : float
        The p-th percentile of the finite samples.

    Raises
    ------
    ValueError
        If scores contain no finite values.
    """
    
    q = np.asarray(scores, dtype=float).reshape(-1)
    q = q[np.isfinite(q)]
    if q.size == 0:
        raise ValueError("No finite scores provided.")
    return float(np.percentile(q, p))


def baselines(scores: np.ndarray) -> Dict[str, float]:
    
    """
    Compute a small set of pooling baselines from a per-frame score series.

    The baselines are selected to represent:
      - mean pooling (common default)
      - a percentile tail summary (p10)
      - a tail-average severity summary (worst 1%)
      - a mild drop-sensitive power mean (Minkowski p=0.5)

    Parameters
    ----------
    scores : np.ndarray
        1D array of per-frame scores. Non-finite values are ignored.

    Returns
    -------
    out : dict[str, float]
        Dictionary with keys:
          - 'mean'
          - 'harmonic_mean'
          - 'minkowski_p0.5'
          - 'p10'
          - 'worst_1pct_mean'
          

    Raises
    ------
    ValueError
        If scores contain no finite values.
    """
    
    q = np.asarray(scores, dtype=float).reshape(-1)
    q = q[np.isfinite(q)]
    if q.size == 0:
        raise ValueError("No finite scores provided.")

    return {
        "mean": float(np.mean(q)),
        "harmonic_mean": harmonic_mean(q),
        "minkowski_p0.5": minkowski_pool(q, 0.5),
        "p10": percentile(q, 10),
        "worst_1pct_mean": worst_kpct_mean(q, 1.0),
    }