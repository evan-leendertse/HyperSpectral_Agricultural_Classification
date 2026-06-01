# Hyperspectral Image Classification with Tucker Decomposition

SVM classification of hyperspectral satellite imagery from the **Indian Pines** dataset, comparing a raw spectral baseline against a **Tucker Decomposition** compressed representation.

---

## Overview

The Indian Pines scene is a 145 × 145 pixel hyperspectral image captured by the AVIRIS sensor, with 200 spectral bands and 16 land-cover classes. Each pixel is classified by its land-cover type using an SVM.

| Property | Value |
|---|---|
| Spatial resolution | 20 m |
| Spectral resolution | 10 nm |
| Spectral bands | 200 |
| Image size | 145 × 145 px |
| Classes | 16 land-cover types + background |

**Pipeline:**
1. Baseline SVM on raw 200-band spectral features (grid search + 5-fold CV)
2. Tucker Decomposition of the full image tensor with AIC-based rank selection
3. SVM on Tucker-compressed features

---

## Results

**Baseline SVM — best hyperparameters (via grid search):**

| Parameter | Value |
|---|---|
| Kernel | RBF |
| $C$ | 50 |
| $\gamma$ | 0.01 |
| CV Accuracy | 90.56% |

| Method | Test Accuracy |
|---|---|
| Baseline SVM (raw features) | 90.93% |
| SVM on Tucker features | **99.51%** |



> Tucker Decomposition not only reduces the dimensionality of the data but acts as a denoising step, boosting classification accuracy by ~8.6 percentage points over the raw spectral baseline.

---

## Method

### Baseline SVM

An SVM is trained directly on the 200-band spectral features with standard scaling. Hyperparameters are selected via 5-fold cross-validated grid search over $C \in \{0.1, 1, 10, 50\}$, $\gamma \in \{\text{scale}, 0.01, 0.001, 0.0001\}$, and kernel $\in \{\text{RBF}, \text{Linear}\}$.

### Tucker Decomposition

The image is reshaped into a 3rd-order tensor $\mathcal{X} \in \mathbb{R}^{145 \times 145 \times 200}$ and decomposed as:

$$\mathcal{X} \approx \mathcal{G} \times_1 A \times_2 B \times_3 C$$

where $\mathcal{G}$ is the core tensor and $A, B, C$ are the factor matrices. A symmetric rank $R = R_1 = R_2 = R_3$ is swept over $\{5, 10, \ldots, 135\}$ and the optimal rank is selected by minimising AIC:

$$\text{AIC} = n \log\!\left(\frac{\text{RSS}}{n}\right) + 2k$$

where $k$ is the total number of parameters in the decomposition and $n$ is the number of tensor elements.

---
