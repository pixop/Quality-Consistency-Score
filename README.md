# Quality Consistency Score (QCS)

This repository provides a minimal reference implementation of the **Quality Consistency Score (QCS)**, a post-processing temporal reliability descriptor computed from a per-frame (or per-second) quality signal (e.g., VMAF, PSNR, SSIM/MS-SSIM, or MOS-over-time).

## Definition

Let $Q(t)$ be a time series of quality scores and let the survival curve be:

- $S(x) = P(Q \ge x)$ (probability scale), or  
- $S_{\%}(x) = 100 \cdot P(Q \ge x)$ (percent scale).

Then the Quality Consistency Score over an operating interval $[a,b]$ is:

$$
\mathrm{QCS}_{[a,b]} = \frac{1}{b-a} \int_a^b S(x)\,dx
$$

`qcs.py` supports output in probability (`scale="prob"`, range [0,1]) or percent (`scale="percent"`, range [0,100]).

## Repository contents

- `qcs.py` — QCS + survival curve + a few pooling baselines (mean, p10, worst-1%, Minkowski p=0.5)
- `qcs_sample.ipynb` — demo notebook using the sample traces
- `VMAF_scores/` — sample per-frame VMAF traces (two .txt files)

## Quickstart

### Requirements
```bash
pip install numpy matplotlib