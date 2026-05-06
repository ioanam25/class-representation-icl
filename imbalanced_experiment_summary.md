# Imbalanced ICL Experiment — Summary of Results

## Experiment Setup

**Model:** Mistral-7B-v0.3 (Qwen results pending)

**Dataset:** `claude_multitask` emotion classification (balanced test set, imbalanced training/demo set)

**Imbalanced class ratios:**
| Setting | Classes | Ratios |
|---------|---------|--------|
| 3-class | Joy / Anger / Fear | 60% / 30% / 10% |
| 5-class | Joy / Sadness / Anger / Fear / Surprise | 40% / 20% / 20% / 10% / 10% |

**Grid:** `n_relabel` ∈ {10, 20, …, 100}, `K` (n_examples) ∈ {0, 3–100}

Both the relabeling fitting and the ICL demonstrations use the imbalanced ratios. The **test set remains balanced** for fair evaluation.

---

## Key Finding 1: Minority classes are severely harmed

The minority class (Fear at 10%, Surprise at 10%) shows dramatically lower F1 than the majority class, especially when few relabeling examples are used.

### 3-class, aggregated across all `n_relabel`

| K (demos) | Joy F1 | Anger F1 | Fear F1 | Macro F1 | Accuracy |
|-----------|--------|----------|---------|----------|----------|
| 0         | 0.662  | 0.606    | 0.469   | 0.579    | 0.608    |
| 10        | 0.789  | 0.686    | 0.571   | 0.682    | 0.713    |
| 25        | 0.809  | 0.741    | 0.643   | 0.731    | 0.747    |
| 50        | 0.809  | 0.732    | 0.569   | 0.704    | 0.729    |
| 64        | 0.824  | 0.758    | 0.638   | 0.740    | 0.757    |
| 100       | 0.823  | 0.730    | 0.539   | 0.697    | 0.727    |

**The gap: Joy F1 (~0.82) vs Fear F1 (~0.54–0.64) is persistent.** Even at the best point (K=64), Fear lags Joy by ~19 points in F1.

### 5-class, aggregated across all `n_relabel`

| K (demos) | Joy F1 | Sadness F1 | Anger F1 | Fear F1 | Surprise F1 | Macro F1 | Accuracy |
|-----------|--------|------------|----------|---------|-------------|----------|----------|
| 0         | 0.502  | 0.411      | 0.475    | 0.482   | 0.218       | 0.418    | 0.443    |
| 10        | 0.631  | 0.600      | 0.586    | 0.558   | 0.461       | 0.567    | 0.592    |
| 25        | 0.634  | 0.657      | 0.631    | 0.590   | 0.486       | 0.600    | 0.617    |
| 50        | 0.632  | 0.669      | 0.645    | 0.588   | 0.422       | 0.591    | 0.612    |
| 100       | 0.642  | 0.696      | 0.673    | 0.640   | 0.472       | 0.625    | 0.640    |

**Surprise (10% class) consistently trails all other classes** by 15–20+ F1 points. Sadness and Anger (20% classes) actually outperform Joy (40% class) at higher K, suggesting the relabeling quality matters more than class frequency in the demos.

---

## Key Finding 2: Relabeling quality matters enormously under imbalance

The `n_relabel` parameter (number of examples used to fit the token-to-class mapping) has a much larger effect under imbalance than in balanced settings.

### 3-class: Fear F1 at different `n_relabel` values

| K (demos) | n_relabel=10 | n_relabel=20 | n_relabel=50 | n_relabel=100 |
|-----------|-------------|-------------|-------------|--------------|
| 0         | 0.000       | 0.130       | 0.629       | 0.607        |
| 10        | 0.004       | 0.443       | 0.724       | 0.789        |
| 25        | 0.124       | 0.576       | 0.800       | 0.828        |
| 50        | 0.076       | 0.411       | 0.753       | 0.812        |
| 70        | 0.218       | 0.426       | 0.745       | 0.806        |
| 100       | 0.128       | 0.279       | 0.702       | 0.757        |

**With only 10 relabeling examples, Fear F1 is essentially 0** — the model never predicts the minority class. With 100 relabeling examples, Fear F1 reaches ~0.76–0.81. This is a >60 point improvement from just having a better relabeling.

### 5-class: Surprise F1 at different `n_relabel` values

| K (demos) | n_relabel=10 | n_relabel=50 | n_relabel=100 |
|-----------|-------------|-------------|--------------|
| 0         | 0.147       | 0.318       | 0.177        |
| 10        | 0.071       | 0.357       | 0.521        |
| 25        | 0.070       | 0.512       | 0.556        |
| 50        | 0.044       | 0.413       | 0.509        |
| 100       | 0.034       | 0.465       | 0.575        |

**Same pattern:** n_relabel=10 yields near-zero Surprise F1, while n_relabel=100 achieves ~0.55.

---

## Key Finding 3: Relabelings become semantically meaningful with more data

The token-to-class mappings discovered by the optimization reveal an interesting pattern: **with few examples, the optimizer picks arbitrary tokens; with enough examples, it discovers emotion-aligned words.**

### Mistral 3-class relabeling examples

| n_relabel | Joy → token | Anger → token | Fear → token | Objective |
|-----------|------------|---------------|-------------|-----------|
| 10        | "Ann"      | "politics"    | "experimental" | -1.17   |
| 50        | "celebration" | "anger"    | "danger"    | -18.99    |
| 100       | "joy"      | "frustration" | "fears"     | -37.04    |

At `n_relabel=10`, the imbalanced dataset provides only ~1 Fear example and ~3 Anger examples for fitting. The optimizer cannot distinguish emotion-relevant tokens from noise, landing on semantically unrelated words like "Ann" and "politics."

At `n_relabel=100`, it gets ~10 Fear and ~30 Anger examples — enough to discover that "joy," "frustration," and "fears" are the most discriminative tokens. **The labels themselves become interpretable.**

### Mistral 5-class relabeling examples

| n_relabel | Joy | Sadness | Anger | Fear | Surprise | Objective |
|-----------|-----|---------|-------|------|----------|-----------|
| 10        | "Nur" | "failed" | "corruption" | "intelligence" | "illusion" | -1.58 |
| 50        | "celebration" | "grief" | "angry" | "terror" | "surpr" | -38.63 |
| 100       | "celebration" | "grief" | "angry" | "terror" | "surprising" | -83.86 |

At `n_relabel≥50`, every class maps to a semantically aligned token. The 5-class case is harder, but the optimizer still finds meaningful words for the 10% classes (Fear→"terror", Surprise→"surprising") when given enough data.

### Qwen 3-class relabeling examples

| n_relabel | Joy → token | Anger → token | Fear → token | Objective |
|-----------|------------|---------------|-------------|-----------|
| 10        | `'",`      | "Simpsons"    | "Datensch"  | -1.06     |
| 50        | "celebr"   | "grievances"  | "fearing"   | -21.47    |
| 100       | "favourable" | "complains" | "feared"    | -46.57    |

Same trend: Qwen with 10 examples picks gibberish; with 100 examples, it finds emotion-relevant subwords.

---

## Key Finding 4: Majority class performance is stable

Joy (the majority class) is remarkably robust to the number of relabeling examples:

### 3-class: Joy F1 across `n_relabel`

| K (demos) | n_relabel=10 | n_relabel=50 | n_relabel=100 |
|-----------|-------------|-------------|--------------|
| 10        | 0.511       | 0.831       | 0.816        |
| 50        | 0.611       | 0.840       | 0.823        |
| 100       | 0.656       | 0.853       | 0.823        |

While n_relabel=10 still hurts Joy, the gap (0.66 vs 0.82) is much smaller than for Fear (0.13 vs 0.76). **The majority class has enough signal even in a small imbalanced sample.**

---

## Key Finding 5: Accuracy masks per-class disparities

Overall accuracy looks reasonable even when minority classes are completely failing:

| Setting | n_relabel=10, K=50 | n_relabel=100, K=50 |
|---------|-------------------|---------------------|
| **3-class Accuracy** | 0.495 | 0.800 |
| **3-class Macro F1** | 0.396 | 0.796 |
| **Fear F1** | 0.076 | 0.812 |

At n_relabel=10, accuracy is ~50% (which seems okay for 3 classes), but **Fear F1 is 0.076** — the model is essentially ignoring the minority class entirely and still achieving decent accuracy by getting the majority classes right. Macro F1 better reflects this failure.

---

## Summary Table: Best configurations (aggregated, Mistral)

### 3-class (K values where Macro F1 peaks)

| K   | Joy F1 | Anger F1 | Fear F1 | Macro F1 | Accuracy |
|-----|--------|----------|---------|----------|----------|
| 25  | 0.809  | 0.741    | 0.643   | 0.731    | 0.747    |
| 64  | 0.824  | 0.758    | 0.638   | 0.740    | 0.757    |
| 84  | 0.822  | 0.758    | 0.610   | 0.730    | 0.750    |

### 5-class (K values where Macro F1 peaks)

| K   | Joy F1 | Sadness F1 | Anger F1 | Fear F1 | Surprise F1 | Macro F1 | Accuracy |
|-----|--------|------------|----------|---------|-------------|----------|----------|
| 91  | 0.647  | 0.694      | 0.674    | 0.647   | 0.465       | 0.625    | 0.642    |
| 92  | 0.644  | 0.701      | 0.678    | 0.650   | 0.463       | 0.627    | 0.643    |
| 100 | 0.642  | 0.696      | 0.673    | 0.640   | 0.472       | 0.625    | 0.640    |

---

## Conclusions

1. **Class imbalance in ICL demonstrations severely degrades minority-class performance**, often to near-zero F1 when the relabeling data is also scarce.

2. **The relabeling step is the critical bottleneck**: a well-fit token mapping (from n_relabel=100 imbalanced → ~10 minority examples) recovers most of the minority-class performance, while a poorly-fit mapping (n_relabel=10 → ~1 minority example) makes the class effectively invisible.

3. **Semantically meaningful relabelings emerge when enough minority-class data is available** for fitting. This suggests the optimization is finding genuine distributional patterns rather than fitting noise.

4. **Standard accuracy is misleading under imbalance** — Macro F1 and per-class F1 are essential for diagnosing failures on minority classes.

5. **Imbalance in the demonstration set compounds with imbalance in the relabeling set**, creating a double penalty for rare classes: bad labels + underrepresentation in the prompt.
