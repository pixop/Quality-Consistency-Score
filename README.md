# Quality Consistency Score (QCS)

This repository provides a minimal reference implementation of the **Quality Consistency Score (QCS)**, a post-processing temporal reliability descriptor computed from a per-frame (or per-second) quality signal (e.g., VMAF, PSNR, SSIM/MS-SSIM, or MOS-over-time).

QCS measures how consistently a video codec maintains quality at or above meaningful thresholds over time, expressed as a fraction of viewing time. It complements conventional RD/BD-rate analysis by exposing temporary quality degradations that may be hidden by temporal averaging.

## Definition

Let $Q(t)$ be a time series of quality scores and let the survival curve be:

- $S(x) = P(Q \ge x)$ (probability scale), or  
- $S(x) = 100 \cdot P(Q \ge x)$ (percent scale).

Then the Quality Consistency Score over an operating interval $[a,b]$ is:

$$
\mathrm{QCS}_{[a,b]} = \frac{1}{b-a} \int_a^b S(x)\,dx
$$

`qcs.py` supports output in probability (`scale="prob"`, range [0,1]) or percent (`scale="percent"`, range [0,100]).

## Repository contents

- `qcs.py` — QCS + survival curve + a few pooling baselines (arithmetic mean, harmonic mean, Minkowski p=0.5, 10th-percentile pooling, worst-1%)
- `qcs_sample.ipynb` — demo notebook using the sample traces
- `VMAF_scores/` — sample per-frame VMAF traces

## Quickstart

### Requirements
```bash
pip install numpy matplotlib
```
 
### Input format

QCS can be computed from a plain-text file containing **one quality score per line**.

For example, a per-frame VMAF trace should look like:

```text
96.333998
100.000000
99.710632
98.776662
...
```

Each line represents one sample in temporal order, typically one score per frame or per second.

The sample files in `vmaf_scores/` follow this format.

### 1. Run the provided notebook

Open and execute:

```text
qcs_sample.ipynb
```

The notebook demonstrates how to load the sample quality traces, compute the survival curve and QCS, and compare QCS with conventional temporal pooling baselines.

### 2. Use QCS in your own notebook or Python script

```python
from qcs import load_scores_txt, qcs

# Load one quality score per line
scores = load_scores_txt("vmaf_scores/BasketballGame_av1_20000.txt")

# Compute QCS over the VMAF quality range [90, 100]
score = qcs(scores, 90, 100, scale="percent")

print(f"QCS[90,100] = {score:.2f}")
```

With `scale="percent"`, QCS is returned in the range `[0,100]`.

To use probability scale instead:

```python
score = qcs(scores, 90, 100, scale="prob")
```

which returns a value in the range `[0,1]`.

### 3. Command-line usage

The command-line interface requires the CLI entry point to be added to `qcs.py`. Once included, QCS can be computed directly from a terminal:

```bash
python qcs.py vmaf_scores/BasketballGame_av1_20000.txt --range 90 100
```

To return QCS on a probability scale:

```bash
python qcs.py vmaf_scores/BasketballGame_av1_20000.txt --range 90 100 --scale prob
```
