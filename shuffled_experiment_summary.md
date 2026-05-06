# Shuffled Labels Experiment — 3-Class Sentiment (Qwen)

## Goal

Test whether the **specific class→token assignment** found by optimization matters, or whether the model can learn any arbitrary mapping from demonstrations. We take the existing optimized relabelings and **permute** the token assignments across classes, breaking the input-output correspondence.

## Methodology

### Shuffle procedure

For each `n_relabel` value (10, 20, …, 100), we load the optimized relabeling pickle and apply a **cyclic left-shift** to the token assignments:

- Class **A** (Joy) receives the token originally optimized for **C** (Anger)
- Class **C** (Anger) receives the token originally optimized for **D** (Fear)
- Class **D** (Fear) receives the token originally optimized for **A** (Joy)

This is a **derangement** — no class retains its original optimized token.

### Three accuracy metrics

We compute three metrics from the same shuffled-demonstration runs:

| Metric | Demonstrations | Evaluated against | Question answered |
|--------|---------------|-------------------|-------------------|
| **Shuffled accuracy** | Shuffled mapping | Shuffled tokens | Does the model learn the wrong mapping shown in context? |
| **Unshuffled accuracy** | Shuffled mapping | Original tokens | Despite wrong demos, does the model still predict the original optimized token? |
| **Original accuracy** (baseline) | Correct mapping | Original tokens | How well does the model perform with correct demos? |

**Interpretation:**
- **High shuffled accuracy** → the model learns whatever mapping it sees in context; the specific tokens don't matter much
- **Low shuffled accuracy** → the optimized tokens encode class-specific information that helps the model; arbitrary tokens hurt
- **High unshuffled accuracy** → the model ignores the (wrong) demonstrations and relies on prior/token semantics
- If shuffled acc + unshuffled acc ≈ 100% → the model is decisive but confused about which mapping to follow
- If both are low → the conflicting signal from shuffled demos degrades performance entirely

### Experimental setup

| Parameter | Value |
|-----------|-------|
| Model | Qwen2-7B-Base |
| Dataset | 3-class sentiment (Joy, Anger, Fear) |
| n_relabel | 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 |
| n_demos (N) | 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 |
| Runs per config | 10 |
| Results dir | `learning_curves_shuffled_3classes_qwen/` |

## Token Mappings: Original vs Shuffled

| n_relabel | Original (A/Joy, C/Anger, D/Fear) | Shuffled (A, C, D) |
|-----------|-----------------------------------|---------------------|
| 10 | Ath, offensive, Sick | offensive, Sick, Ath |
| 20 | applause, parliamentary, JsonResult | parliamentary, JsonResult, applause |
| 30 | congrat, warn, weighing | warn, weighing, congrat |
| 40 | Wonderful, bitch, ArgumentError | bitch, ArgumentError, Wonderful |
| 50 | celebration, hatred, fears | hatred, fears, celebration |
| 60 | Wonderful, bitch, noir | bitch, noir, Wonderful |
| 70 | congrat, abusive, fears | abusive, fears, congrat |
| 80 | luz, offender, nightmares | offender, nightmares, luz |
| 90 | brag, offending, terror | offending, terror, brag |
| 100 | brag, offending, terror | offending, terror, brag |

## Results

### 1. Shuffled Accuracy (demos shuffled, eval under shuffled mapping)

Does the model learn the wrong mapping from the shuffled demonstrations?

| N demos | k=10 | k=20 | k=30 | k=40 | k=50 | k=60 | k=70 | k=80 | k=90 | k=100 |
|---------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
| 0 | 26.7% ± 0.0% | 26.3% ± 0.0% | 16.3% ± 0.0% | 22.7% ± 0.0% | 18.3% ± 0.0% | 19.7% ± 0.0% | 17.7% ± 0.0% | 26.0% ± 0.0% | 23.3% ± 0.0% | 23.3% ± 0.0% |
| 10 | 35.8% ± 2.6% | 35.4% ± 3.6% | 32.4% ± 5.1% | 33.5% ± 5.3% | 31.2% ± 5.2% | 39.8% ± 8.9% | 34.2% ± 6.4% | 34.5% ± 7.8% | 31.4% ± 8.9% | 31.4% ± 8.9% |
| 20 | 36.7% ± 3.5% | 34.6% ± 2.0% | 39.6% ± 6.7% | 36.3% ± 6.7% | 32.4% ± 2.8% | 46.2% ± 7.8% | 39.6% ± 7.8% | 34.9% ± 3.2% | 29.8% ± 7.0% | 29.8% ± 7.0% |
| 30 | 41.4% ± 7.4% | 38.4% ± 6.2% | 47.2% ± 4.7% | 38.6% ± 6.8% | 36.8% ± 7.4% | 52.8% ± 5.3% | 46.8% ± 6.2% | 42.1% ± 8.8% | 37.7% ± 7.3% | 37.7% ± 7.3% |
| 40 | 46.3% ± 4.9% | 42.9% ± 8.1% | 46.4% ± 7.3% | 51.5% ± 5.8% | 33.8% ± 3.2% | 53.9% ± 5.9% | 41.6% ± 8.1% | 44.7% ± 8.1% | 48.8% ± 6.3% | 48.8% ± 6.3% |
| 50 | 44.3% ± 4.2% | 43.9% ± 3.5% | 51.7% ± 8.3% | 50.5% ± 7.2% | 36.3% ± 2.7% | 58.5% ± 4.1% | 47.4% ± 7.1% | 44.7% ± 6.8% | 46.6% ± 6.8% | 46.6% ± 6.8% |
| 60 | 50.9% ± 5.6% | 51.3% ± 4.8% | 48.6% ± 8.2% | 57.5% ± 8.4% | 38.1% ± 5.8% | 56.6% ± 4.4% | 42.9% ± 6.1% | 45.2% ± 7.2% | 54.3% ± 4.2% | 54.3% ± 4.2% |
| 70 | 46.8% ± 5.6% | 50.3% ± 6.1% | 51.8% ± 7.0% | 56.2% ± 6.3% | 38.0% ± 4.9% | 56.3% ± 5.9% | 43.6% ± 6.9% | 48.5% ± 7.9% | 54.7% ± 2.7% | 54.7% ± 2.7% |
| 80 | 50.6% ± 8.4% | 49.4% ± 5.3% | 52.9% ± 9.9% | 52.9% ± 6.5% | 39.6% ± 5.7% | 57.3% ± 3.1% | 45.2% ± 7.7% | 48.6% ± 8.5% | 55.3% ± 3.6% | 55.3% ± 3.6% |
| 90 | 54.0% ± 8.4% | 51.6% ± 6.5% | 51.6% ± 6.7% | 52.7% ± 7.1% | 40.9% ± 6.1% | 57.2% ± 8.1% | 49.8% ± 6.4% | 51.1% ± 5.9% | 55.0% ± 5.4% | 55.0% ± 5.4% |
| 100 | 49.3% ± 8.3% | 48.8% ± 7.7% | 50.5% ± 7.5% | 52.9% ± 7.5% | 40.1% ± 6.7% | 60.2% ± 3.2% | 48.6% ± 7.8% | 53.3% ± 3.6% | 53.0% ± 7.6% | 53.0% ± 7.6% |

**Figure:** plot_single-style line curves for shuffled accuracy (one curve per `n_relabel`, x-axis is `n_demos`).

![Shuffled accuracy curves](plots_shuffled/shuffled_accuracy_all_curves_qwen.pdf)

### 2. Unshuffled Accuracy (demos shuffled, eval under ORIGINAL mapping)

Despite seeing wrong demonstrations, does the model still predict the original optimized token?

| N demos | k=10 | k=20 | k=30 | k=40 | k=50 | k=60 | k=70 | k=80 | k=90 | k=100 |
|---------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|--------:|
| 0 | 44.7% ± 0.0% | 46.0% ± 0.0% | 49.7% ± 0.0% | 54.0% ± 0.0% | 62.3% ± 0.0% | 62.7% ± 0.0% | 68.0% ± 0.0% | 59.0% ± 0.0% | 63.7% ± 0.0% | 63.7% ± 0.0% |
| 10 | 37.1% ± 4.4% | 37.2% ± 5.8% | 46.7% ± 7.3% | 45.7% ± 8.1% | 49.4% ± 8.6% | 44.3% ± 9.2% | 44.5% ± 10.5% | 40.3% ± 10.9% | 51.3% ± 10.3% | 51.3% ± 10.3% |
| 20 | 34.6% ± 4.4% | 35.3% ± 3.1% | 46.2% ± 7.4% | 43.0% ± 4.7% | 48.8% ± 6.9% | 39.2% ± 6.2% | 41.2% ± 6.4% | 38.6% ± 6.7% | 52.2% ± 8.5% | 52.2% ± 8.5% |
| 30 | 30.3% ± 5.5% | 32.9% ± 4.4% | 36.6% ± 6.0% | 35.9% ± 8.7% | 47.1% ± 11.2% | 33.4% ± 3.0% | 37.6% ± 5.6% | 31.2% ± 9.5% | 41.6% ± 8.4% | 41.6% ± 8.4% |
| 40 | 25.5% ± 4.2% | 28.2% ± 4.9% | 39.6% ± 7.0% | 28.0% ± 6.0% | 50.0% ± 6.8% | 33.8% ± 3.7% | 35.1% ± 3.1% | 25.3% ± 6.9% | 40.3% ± 6.1% | 40.3% ± 6.1% |
| 50 | 27.4% ± 4.0% | 26.1% ± 3.7% | 36.4% ± 6.6% | 29.0% ± 5.5% | 46.5% ± 7.3% | 31.9% ± 2.6% | 36.9% ± 2.7% | 24.7% ± 6.1% | 38.0% ± 6.1% | 38.0% ± 6.1% |
| 60 | 22.5% ± 4.5% | 23.3% ± 4.0% | 38.6% ± 5.4% | 25.5% ± 3.9% | 48.8% ± 6.0% | 33.6% ± 2.2% | 34.0% ± 0.9% | 25.0% ± 5.4% | 35.4% ± 3.2% | 35.4% ± 3.2% |
| 70 | 24.2% ± 3.7% | 24.7% ± 2.8% | 34.8% ± 4.1% | 26.9% ± 6.9% | 45.8% ± 6.2% | 32.8% ± 2.3% | 34.8% ± 3.7% | 22.0% ± 5.9% | 34.8% ± 2.5% | 34.8% ± 2.5% |
| 80 | 23.5% ± 5.0% | 24.9% ± 4.1% | 32.2% ± 3.1% | 27.6% ± 4.3% | 44.5% ± 5.1% | 33.7% ± 2.8% | 33.5% ± 1.2% | 22.6% ± 5.8% | 33.4% ± 2.1% | 33.4% ± 2.1% |
| 90 | 20.9% ± 5.5% | 24.7% ± 4.7% | 32.5% ± 2.7% | 26.0% ± 9.6% | 43.3% ± 6.8% | 32.1% ± 3.0% | 33.4% ± 0.2% | 17.3% ± 5.5% | 33.9% ± 7.1% | 33.9% ± 7.1% |
| 100 | 21.6% ± 6.1% | 24.7% ± 5.4% | 32.9% ± 3.0% | 23.8% ± 6.5% | 41.2% ± 12.0% | 31.8% ± 2.9% | 33.2% ± 1.0% | 15.0% ± 3.5% | 33.9% ± 2.7% | 33.9% ± 2.7% |

### 3. Original (Optimized) Accuracy — Baseline

Standard experiment: demos use the correct optimized mapping, eval under that mapping.

*No data available for Original*


### Head-to-Head at N=100 Demonstrations

| n_relabel | Original | Shuffled (eval shuffled) | Shuffled (eval original) |
|-----------|--------:|------------------------:|------------------------:|
| 10 | — | 49.3% | 21.6% |
| 20 | — | 48.8% | 24.7% |
| 30 | — | 50.5% | 32.9% |
| 40 | — | 52.9% | 23.8% |
| 50 | — | 40.1% | 41.2% |
| 60 | — | 60.2% | 31.8% |
| 70 | — | 48.6% | 33.2% |
| 80 | — | 53.3% | 15.0% |
| 90 | — | 53.0% | 33.9% |
| 100 | — | 53.0% | 33.9% |
