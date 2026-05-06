# Gold-Init Optimized vs Gold Labels — Experiment Summary

## Overview

This experiment tests whether **optimizing from gold label initialization** can beat using **gold labels directly**.

- **Gold labels**: Semantically meaningful tokens (e.g., *joy*, *anger*, *fear*) used as-is for classification
- **Gold-init optimized**: Hill-climbing optimization starts from gold labels and searches for better tokens

If the optimizer finds tokens that outperform gold labels, it means the optimization landscape near gold labels
contains even better solutions — the model's internal representations don't perfectly align with human-chosen words.

**Model**: Qwen2-7B (base)  
**Dataset**: Sentiment classification (claude_multitask)  
**k** = number of examples used for relabeling optimization (10–100)  
**N** = number of in-context demonstrations at inference (0–100)  
**Runs per config**: 10 (different random demo selections)

---
## Sentiment 3-class

**Gold labels**: A → *joy*, C → *anger*, D → *fear*

### Optimized tokens found (starting from gold)

| k (relabel examples) | Optimized tokens | Objective |
|---|---|---|
| 10 | A→*MAGIC*, C→*repro*, D→*opposite* | -2.04 |
| 20 | A→*LABEL*, C→*repro*, D→*fear* | -7.50 |
| 30 | A→*------------*, C→*rant*, D→*fear* | -18.71 |
| 40 | A→*OPEN*, C→*complain*, D→*fear* | -25.06 |
| 50 | A→*LE*, C→*griev*, D→*security* | -33.06 |
| 60 | A→*ENTER*, C→*complains*, D→*terror* | -39.16 |
| 70 | A→*celebrate*, C→*griev*, D→*fear* | -47.12 |
| 80 | A→*celebration*, C→*blames*, D→*fear* | -54.39 |
| 90 | A→*Retrieve*, C→*complains*, D→*terror* | -60.02 |
| 100 | A→*Sunny*, C→*complains*, D→*terror* | -63.64 |

### Accuracy (%) — Gold labels vs Gold-init optimized (by k)

| N (demos) | **Gold labels** | k=10 | k=20 | k=30 | k=40 | k=50 | k=60 | k=70 | k=80 | k=90 | k=100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | **54.0±0.0** | 38.7±0.0 | 46.7±0.0 | 59.0±0.0 | 61.0±0.0 | 56.3±0.0 | 65.3±0.0 | 66.7±0.0 | 68.0±0.0 | 64.0±0.0 | 67.3±0.0 |
| 10 | **83.7±2.6** | 61.7±4.7 | 72.6±3.0 | 61.8±3.0 | 78.3±1.1 | 69.1±3.6 | 73.5±6.2 | 79.4±4.1 | 78.4±5.7 | 71.8±5.9 | 74.4±6.8 |
| 20 | **84.8±1.4** | 68.4±2.7 | 72.7±4.3 | 66.0±4.5 | 76.5±3.3 | 74.4±3.5 | 77.3±2.2 | 78.5±2.2 | 78.1±2.8 | 75.0±5.1 | 77.6±3.3 |
| 30 | **85.3±1.1** | 70.0±4.2 | 73.8±3.3 | 66.3±4.7 | 79.7±2.2 | 74.4±3.4 | 77.1±2.5 | 77.9±3.6 | 80.1±2.1 | 75.2±3.2 | 77.4±2.4 |
| 40 | **85.0±1.0** | 72.5±2.4 | 72.5±3.3 | 68.3±5.3 | 78.9±2.5 | 75.8±2.8 | 78.6±2.4 | 75.8±5.3 | 78.1±3.2 | 77.3±3.0 | 79.2±1.4 |
| 50 | **84.9±0.9** | 74.8±3.4 | 76.1±5.2 | 68.3±3.0 | 80.2±2.2 | 77.1±2.6 | 77.6±2.6 | 73.9±4.0 | 79.5±2.5 | 77.4±1.9 | 77.8±2.5 |
| 60 | **85.0±1.4** | 74.9±2.3 | 74.9±3.3 | 71.9±4.2 | 79.7±2.2 | 77.1±2.6 | 78.2±3.0 | 74.7±3.7 | 79.7±2.8 | 77.0±3.7 | 78.4±2.7 |
| 70 | **85.7±1.2** | 74.7±3.8 | 74.0±2.5 | 71.9±2.1 | 80.1±1.4 | 77.0±1.5 | 78.6±1.7 | 75.7±2.6 | 78.1±2.6 | 77.0±2.4 | 79.5±2.0 |
| 80 | **85.7±1.3** | 74.8±4.5 | 74.6±3.4 | 72.0±3.3 | 80.2±2.8 | 77.4±4.1 | 78.5±3.0 | 77.6±4.2 | 77.6±3.4 | 77.9±2.9 | 79.5±2.0 |
| 90 | **85.5±1.2** | 74.5±3.0 | 76.1±3.1 | 69.8±4.7 | 80.3±1.7 | 77.4±2.8 | 75.7±4.1 | 74.2±6.7 | 78.6±3.0 | 75.9±3.0 | 79.2±3.2 |
| 100 | **86.3±1.0** | 76.5±2.4 | 76.8±2.6 | 70.0±2.8 | 80.0±1.8 | 77.7±4.6 | 76.4±4.2 | 72.0±3.9 | 78.3±2.4 | 76.5±4.4 | 78.7±3.2 |

### Best gold-init curve (k=40) vs Gold labels

| N (demos) | Gold labels | Gold-init optimized (k=40) | Δ (pp) |
|---|---|---|---|
| 0 | 54.0±0.0 | 61.0±0.0 | +7.0 |
| 10 | 83.7±2.6 | 78.3±1.1 | -5.4 |
| 20 | 84.8±1.4 | 76.5±3.3 | -8.3 |
| 30 | 85.3±1.1 | 79.7±2.2 | -5.6 |
| 40 | 85.0±1.0 | 78.9±2.5 | -6.1 |
| 50 | 84.9±0.9 | 80.2±2.2 | -4.7 |
| 60 | 85.0±1.4 | 79.7±2.2 | -5.3 |
| 70 | 85.7±1.2 | 80.1±1.4 | -5.6 |
| 80 | 85.7±1.3 | 80.2±2.8 | -5.5 |
| 90 | 85.5±1.2 | 80.3±1.7 | -5.2 |
| 100 | 86.3±1.0 | 80.0±1.8 | -6.3 |

**Average Δ across all N**: -4.6pp  
**Δ at N=100**: -6.3pp  
**Gold-init wins at**: 1/11 N values (by >0.5pp)

**Best gold-init tokens (k=40)**: A→*OPEN*, C→*complain*, D→*fear*

---
## Sentiment 5-class

**Gold labels**: A → *joy*, B → *sadness*, C → *anger*, D → *fear*, E → *surprise*

### Optimized tokens found (starting from gold)

| k (relabel examples) | Optimized tokens | Objective |
|---|---|---|
| 10 | A→*--->*, B→*Hor*, C→*griev*, D→*NEG*, E→*filling* | -2.89 |
| 20 | A→*|--*, B→*Mario*, C→*repro*, D→*drawback*, E→*assumption* | -13.53 |
| 30 | A→*ENTER*, B→*Chapter*, C→*repro*, D→*Security*, E→*Unused* | -24.97 |
| 40 | A→*Ensure*, B→*Mad*, C→*workplace*, D→*insurance*, E→*helpful* | -40.86 |
| 50 | A→*--------------------------------*, B→*Failure*, C→*Workplace*, D→*security*, E→*Unsure* | -52.01 |
| 60 | A→*ENTER*, B→*Failure*, C→*protest*, D→*security*, E→*Unsure* | -62.32 |
| 70 | A→*Ensure*, B→*Failure*, C→*griev*, D→*security*, E→*unconventional* | -75.46 |
| 80 | A→*{}Ċ*, B→*Failure*, C→*Abuse*, D→*Security*, E→*YouTube* | -90.48 |
| 90 | A→*##Ċ*, B→*Failure*, C→*Abuse*, D→*security*, E→*NET* | -103.59 |
| 100 | A→*{}Ċ*, B→*Failure*, C→*Abuse*, D→*security*, E→*SHORT* | -114.18 |

### Accuracy (%) — Gold labels vs Gold-init optimized (by k)

| N (demos) | **Gold labels** | k=10 | k=20 | k=30 | k=40 | k=50 | k=60 | k=70 | k=80 | k=90 | k=100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | **40.8±0.0** | 23.2±0.0 | 23.4±0.0 | 27.6±0.0 | 25.4±0.0 | 28.8±0.0 | 34.6±0.0 | 37.4±0.0 | 37.0±0.0 | 32.8±0.0 | 38.0±0.0 |
| 10 | **74.5±2.2** | 28.3±5.4 | 28.3±2.9 | 27.1±4.1 | 24.7±3.5 | 30.4±7.9 | 38.7±7.2 | 42.4±8.6 | 39.0±5.3 | 38.4±5.3 | 41.5±5.4 |
| 20 | **76.1±2.0** | 30.5±5.5 | 31.4±4.9 | 26.9±3.6 | 30.6±3.8 | 27.3±4.8 | 42.5±7.9 | 44.7±8.2 | 42.6±3.7 | 40.0±3.6 | 47.5±4.4 |
| 30 | **76.3±1.1** | 33.6±3.8 | 33.9±3.7 | 28.6±3.3 | 30.0±4.5 | 30.9±6.6 | 46.2±3.7 | 50.3±5.2 | 45.5±5.1 | 43.8±5.2 | 49.9±3.5 |
| 40 | **75.9±2.0** | 38.9±6.8 | 36.9±7.4 | 30.3±3.2 | 32.3±3.6 | 35.6±6.9 | 47.0±3.7 | 48.9±7.1 | 46.3±7.2 | 42.9±4.6 | 51.4±5.4 |
| 50 | **76.3±1.5** | 39.0±6.6 | 41.4±4.2 | 32.1±3.8 | 33.7±4.2 | 39.4±7.7 | 50.8±5.2 | 48.9±5.1 | 47.7±6.0 | 47.5±3.6 | 55.0±3.9 |
| 60 | **77.0±2.4** | 40.5±4.5 | 42.6±3.7 | 32.7±5.4 | 35.8±5.6 | 41.8±3.1 | 51.1±3.3 | 51.4±5.0 | 51.8±2.2 | 48.7±5.2 | 56.8±3.1 |
| 70 | **77.2±1.2** | 43.8±4.8 | 42.8±3.4 | 33.0±3.0 | 30.1±4.4 | 41.8±3.6 | 49.0±3.8 | 51.2±4.8 | 51.7±3.6 | 49.0±3.5 | 57.4±2.9 |
| 80 | **76.6±1.9** | 41.1±4.5 | 43.6±4.2 | 32.6±3.3 | 35.2±3.4 | 43.4±6.1 | 52.7±2.7 | 53.4±4.3 | 53.1±2.9 | 51.2±3.1 | 57.8±1.7 |
| 90 | **78.3±0.8** | 43.5±5.0 | 46.4±3.8 | 34.4±4.0 | 34.7±6.4 | 45.3±5.2 | 54.6±3.9 | 57.3±2.3 | 54.3±5.3 | 50.5±3.8 | 58.4±5.6 |
| 100 | **78.6±0.9** | 38.3±6.4 | 45.7±3.2 | 33.9±2.8 | 37.5±4.3 | 41.4±4.3 | 53.0±3.6 | 55.1±3.5 | 53.3±2.8 | 50.4±3.0 | 56.8±3.3 |

### Best gold-init curve (k=100) vs Gold labels

| N (demos) | Gold labels | Gold-init optimized (k=100) | Δ (pp) |
|---|---|---|---|
| 0 | 40.8±0.0 | 38.0±0.0 | -2.8 |
| 10 | 74.5±2.2 | 41.5±5.4 | -33.0 |
| 20 | 76.1±2.0 | 47.5±4.4 | -28.6 |
| 30 | 76.3±1.1 | 49.9±3.5 | -26.4 |
| 40 | 75.9±2.0 | 51.4±5.4 | -24.5 |
| 50 | 76.3±1.5 | 55.0±3.9 | -21.3 |
| 60 | 77.0±2.4 | 56.8±3.1 | -20.1 |
| 70 | 77.2±1.2 | 57.4±2.9 | -19.9 |
| 80 | 76.6±1.9 | 57.8±1.7 | -18.9 |
| 90 | 78.3±0.8 | 58.4±5.6 | -19.8 |
| 100 | 78.6±0.9 | 56.8±3.3 | -21.8 |

**Average Δ across all N**: -21.6pp  
**Δ at N=100**: -21.8pp  
**Gold-init wins at**: 0/11 N values (by >0.5pp)

**Best gold-init tokens (k=100)**: A→*{}Ċ*, B→*Failure*, C→*Abuse*, D→*security*, E→*SHORT*

---
## TREC 5-class

**Gold labels**: A → *entity*, B → *description*, C → *human*, D → *location*, E → *numeric*

### Optimized tokens found (starting from gold)

| k (relabel examples) | Optimized tokens | Objective |
|---|---|---|
| 10 | A→*Highest*, B→*Geography*, C→*Kill*, D→*geometric*, E→*Work* | -3.90 |
| 20 | A→*generating*, B→*click*, C→*win*, D→*where*, E→*numerical* | -12.55 |
| 30 | A→*multiples*, B→*Info*, C→*author*, D→*Bool*, E→*numeric* | -23.91 |
| 40 | A→*sentiment*, B→*Info*, C→*Company*, D→*where*, E→*number* | -30.88 |
| 50 | A→*sentiment*, B→*Essay*, C→*Win*, D→*where*, E→*number* | -42.23 |
| 60 | A→*most*, B→*Educational*, C→*Win*, D→*geography*, E→*Numeric* | -53.97 |
| 70 | A→*fine*, B→*Inquiry*, C→*Win*, D→*where*, E→*number* | -66.14 |
| 80 | A→*entertain*, B→*Learn*, C→*who*, D→*Where*, E→*numerical* | -77.99 |
| 90 | A→*entertain*, B→*Learn*, C→*who*, D→*Where*, E→*numeric* | -85.98 |
| 100 | A→*entertain*, B→*Learn*, C→*who*, D→*Where*, E→*numeric* | -95.35 |

### Accuracy (%) — Gold labels vs Gold-init optimized (by k)

| N (demos) | **Gold labels** | k=10 | k=20 | k=30 | k=40 | k=50 | k=60 | k=70 | k=80 | k=90 | k=100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | **36.9±0.0** | 30.2±0.0 | 51.1±0.0 | 41.5±0.0 | 55.7±0.0 | 57.5±0.0 | 50.8±0.0 | 59.1±0.0 | 67.4±0.0 | 68.0±0.0 | 68.0±0.0 |
| 10 | **83.2±3.9** | 26.7±5.5 | 72.1±9.4 | 51.0±10.8 | 68.9±5.7 | 67.1±8.1 | 59.0±11.8 | 70.7±5.0 | 75.7±2.7 | 75.5±2.7 | 75.5±2.7 |
| 20 | **84.4±2.4** | 29.4±6.3 | 76.3±9.8 | 68.7±6.2 | 77.1±3.3 | 73.7±6.4 | 63.8±10.5 | 68.7±9.9 | 76.1±2.8 | 75.8±3.3 | 75.8±3.3 |
| 30 | **85.4±2.4** | 34.7±2.4 | 81.2±4.6 | 70.9±6.5 | 79.0±2.7 | 76.7±2.5 | 66.2±5.9 | 75.6±4.2 | 78.6±1.8 | 78.5±2.1 | 78.5±2.1 |
| 40 | **85.5±3.9** | 32.6±8.6 | 82.1±6.9 | 73.7±4.5 | 79.6±2.9 | 77.2±4.6 | 69.6±7.1 | 73.9±8.4 | 78.3±1.7 | 78.5±1.6 | 78.5±1.6 |
| 50 | **86.4±2.8** | 36.3±4.0 | 84.3±4.9 | 76.8±4.3 | 79.6±3.5 | 78.7±3.7 | 71.9±6.0 | 79.8±4.5 | 78.7±2.3 | 79.3±2.6 | 79.3±2.6 |
| 60 | **88.0±2.7** | 39.7±6.0 | 84.5±3.9 | 77.5±7.3 | 80.6±2.5 | 80.5±2.2 | 72.2±6.6 | 78.8±3.7 | 79.8±2.1 | 80.7±2.7 | 80.7±2.7 |
| 70 | **86.6±1.9** | 39.0±8.0 | 84.7±2.7 | 80.3±3.1 | 78.5±5.5 | 78.3±4.7 | 74.9±7.2 | 79.2±5.7 | 79.1±1.7 | 80.0±2.0 | 80.0±2.0 |
| 80 | **86.4±2.2** | 44.2±5.7 | 85.5±3.2 | 77.8±4.9 | 78.5±3.4 | 76.7±5.2 | 74.3±6.2 | 74.0±5.6 | 79.8±1.9 | 80.3±1.6 | 80.3±1.6 |
| 90 | **87.9±1.9** | 42.4±6.3 | 87.0±2.9 | 78.3±8.5 | 80.2±5.8 | 79.1±5.8 | 73.0±7.0 | 77.5±7.1 | 80.3±1.2 | 81.1±1.6 | 81.1±1.6 |
| 100 | **88.1±1.9** | 42.9±8.5 | 85.7±4.1 | 70.3±15.5 | 75.6±11.0 | 76.7±9.9 | 65.8±11.5 | 78.5±7.3 | 79.9±2.0 | 80.7±2.0 | 80.7±2.0 |

### Best gold-init curve (k=20) vs Gold labels

| N (demos) | Gold labels | Gold-init optimized (k=20) | Δ (pp) |
|---|---|---|---|
| 0 | 36.9±0.0 | 51.1±0.0 | +14.2 |
| 10 | 83.2±3.9 | 72.1±9.4 | -11.1 |
| 20 | 84.4±2.4 | 76.3±9.8 | -8.1 |
| 30 | 85.4±2.4 | 81.2±4.6 | -4.2 |
| 40 | 85.5±3.9 | 82.1±6.9 | -3.4 |
| 50 | 86.4±2.8 | 84.3±4.9 | -2.1 |
| 60 | 88.0±2.7 | 84.5±3.9 | -3.6 |
| 70 | 86.6±1.9 | 84.7±2.7 | -1.9 |
| 80 | 86.4±2.2 | 85.5±3.2 | -0.9 |
| 90 | 87.9±1.9 | 87.0±2.9 | -0.9 |
| 100 | 88.1±1.9 | 85.7±4.1 | -2.4 |

**Average Δ across all N**: -2.2pp  
**Δ at N=100**: -2.4pp  
**Gold-init wins at**: 1/11 N values (by >0.5pp)

**Best gold-init tokens (k=20)**: A→*generating*, B→*click*, C→*win*, D→*where*, E→*numerical*

---
## Key Findings

### Sentiment 3-class

- Gold labels **outperform** gold-init optimization by **6.3pp** at N=100
- Best gold-init accuracy at N=100: **80.0%** (k=40)
- Gold label accuracy at N=100: **86.3%**

### Sentiment 5-class

- Gold labels **outperform** gold-init optimization by **21.8pp** at N=100
- Best gold-init accuracy at N=100: **56.8%** (k=100)
- Gold label accuracy at N=100: **78.6%**

### TREC 5-class

- Gold labels **outperform** gold-init optimization by **2.4pp** at N=100
- Best gold-init accuracy at N=100: **85.7%** (k=20)
- Gold label accuracy at N=100: **88.1%**
