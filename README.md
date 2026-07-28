# MFPCA-RUL: Reproducing Multivariate Functional Data Analysis for RUL Prediction 
-------------------------------------------------------------------------------
## Description
This repository reproduces the multivariate functional data analysis (MFPCA) 
pipeline from Yildirim, Franco & Lillo for predicting Remaining Useful Life 
(RUL) of turbofan engines, applied to the NASA C-MAPSS FD001 dataset.

The pipeline combines B-spline smoothing with fleet-wide GCV-optimized 
penalization, multivariate functional principal component analysis (Happ & 
Greven, 2018) for health index construction, and a k-NN similarity-matching 
scheme for RUL prediction on right-censored test trajectories.

This project was built independently as part of an undergraduate research 
portfolio in Applied Mathematics, with a focus on faithful method reproduction 
and transparent reporting of implementation decisions not fully specified in 
the original paper.

**Result: RMSE ≈ 25.8–26.0 (mean-based prediction), within ~4% of the original paper's reported RMSE (~25).**
-------------------------------------------------------------------------------
## Reference 


**Primary paper (reproduced in this repository):**
> Yildirim, C., Lillo, R. E., & Franco-Pereira, A. M. (2025). Health Prognostics in Multi-Sensor Systems Based on Multivariate Functional Data Analysis. Available at SSRN 4907886.

**Methodology & Key Reference**
> Happ, C., & Greven, S. (2018). Multivariate functional principal component analysis for data observed on different (dimensional) domains. Journal of the American Statistical Association, 113(522), 649-659.
# **Pipeline**
# **Key Methodological**
