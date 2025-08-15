# Lorenzetti Shower Simulation — ATLAS HLT Energy Calibration

This repository contains a dataset of simulated electromagnetic showers derived from the **Lorenzetti Shower** generator [1], configured to emulate the response of the **ATLAS** Fast Calorimeter trigger (**Ringer**) employed in the **High Level Trigger (HLT)**.  
It accompanies the methodology described in [2], regarding real-time energy calibration of the fast online reconstruction.

---

## Dataset Overview

Each row in the dataset corresponds to a single electromagnetic shower, represented by high-granularity calorimeter features (rings) and reconstructed cluster-level information. The **calibration target** is a multiplicative factor `α` such that:

$E_{\text{true}} = \alpha \cdot E_{\text{fast}}$

---

## Features

| Feature(s)                 | Description |
|---------------------------|-------------|
| `0`–`99`                 | Normalized energy deposits in the 100 **Ringer rings**, corresponding to concentric Δη×Δφ sampling regions around the shower core (fast online topology). |
| `cluster_trutheta`        | True **pseudo-rapidity** ($\eta$) of the reconstructed cluster. |
| `cluster_truthe237`       | Energy in **sampling layer E2** ($\eta \times \phi = 0.023 \times 0.025$) — core energy. (Not used in calibration). |
| `cluster_truthe277`       | Energy in **sampling layer E3** ($\eta \times \phi = 0.075 \times 0.125$) — shower spread. (Not used in calibration). |
| `cluster_truthet`         | **Fast-calorimeter transverse energy** ($E_\text{fast}$) as computed by the HLT. |
| `cluster_truthphi`        | Cluster **azimuthal coordinate** $\phi$. |
| `mc_match_et` / `el_truth_pt` | Generator-level (truth) **transverse energy** $E_\text{true}$. |
| `alfa`                    | Ground-truth **calibration factor**, target of regression. |

---

## Calibration Objective

Train regression models (e.g. Regression, Boosted Decision Trees, Neural Networks) to predict $\alpha_text{calib}$ using the fast-observable calorimeter quantities, enabling **online HLT energy calibration**. Calibration performance is assessed via:

- Bias distribution of calibrated energy: $E_{\text{calib}} = \alpha_text{calib} \cdot E_{\text{fast}} \approx E_{\text{true}}$
- Paired statistical tests (t-test, Wilcoxon, F-test)
- Resolution improvements over fast reconstruction baseline

---

## Repository Structure
```
Lorenzetti-Shower-Simulation/
├── README.txt
├── data/
│   ├── Lorenzetti_simulation_raw_20250815.csv
│
├── notebooks/
│   └── to_be_included.ipynb
│
├── src/
│   ├── calibration/
│   └── evaluation/
│
└── docs/
    ├── Simas_Filho_2025_J._Inst._20_P06051_Lorenzetti_Calibration.pdf (Calibration using Lorenzetti simulations)
    └── 2023_Lorenzetti_CPC.pdf (Original Lorenzetti Shower description)

```
---

## Requirements

- Python ≥3.9  
- NumPy, SciPy, scikit-learn  
- matplotlib  
- pandas  

---

## Citation

If you use this dataset or code, please cite:

[1] Lorenzetti Shower Simulation Framework, *Comput. Phys. Commun.*, vol. 184, pp. 2636–2644, 2013. doi: https://doi.org/10.1016/j.cpc.2023.108671. (PDF: 2023_Lorenzetti_CPC.pdf)

[2] Simas Filho et al., “Real-Time Energy Calibration for ATLAS using Lorenzetti Simulated Showers,” *JINST*, vol. 20, P06051, 2025. doi: https://dx.doi.org/10.1088/1748-0221/20/06/P06051. (PDF: Simas_Filho_2025_J._Inst._20_P06051_Lorenzetti_Calibration.pdf)


