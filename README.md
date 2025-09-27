### This repository contains code associated with the manuscript: ###

**"Feature selection leads to divergent neurobiological interpretations of brain-based machine learning biomarkers"**  
*Brendan D. Adkinson et al.*

The code is released under a Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0), which permits use, distribution, and modification for non-commercial purposes, provided appropriate credit is given.

Analyses were carried out using Matlab R2024a Update 5 (24.1.0.2653294) and run on MacOS 14.5 (23F79). Matlab download and instructions are available here: https://www.mathworks.com/help/install/ug/install-products-with-internet-connection.html. Download times will vary, though 4-6GB of space are required. 


### Scripts ###

### `run_models.m`  
This is the **main script**. Users should start here.

- Loads neuroimaging and behavioral data from multiple dataset
- Defines executive function (EF), language (LANG), or other phenotypes for prediction
- Automatically applies PCA to reduce multiple behavioral measures into a latent phenotype factor, if desired
- Loops over multiple random seeds and feature rank segments to:
  - Train models using the `train_model_ranked_edges_percent.m` function
  - Save model performance results

You can customize:
- Model type (`'ridge'` or `'cpm'`)
- Feature selection method (e.g., `'ranked_1_percent’, ‘ranked_10_percent'`)
- Number of folds in cross-validation
- Whether to perform null model permutation or include covariates

---

### `train_model_ranked_edges_percent.m`  
This is the **core training function**. It supports both:

- **Within-dataset prediction**: k-fold cross-validation with optional PCA and covariate control
- **Cross-dataset prediction**: train on one dataset and test on another

Key features:
- Supports **feature ranking by correlation or p-value**
- Implements **classic CPM** and **ridge regression** (via `lasso` with low `Alpha`) 
- Automatically computes **permutation-based null models** when specified
- Behavioral phenotypes can be reduced using:
  - External (non-imaging) PCA
  - Cross-validated PCA

Returns a structured `results` object with model coefficients, predictions, and metadata.

---

### `mat2edge.m`  
Helper function to convert 3D connectome matrices (size `[nodes x nodes x subjects]`) into 2D edge matrices (`[edges x subjects]`).  
This vectorized representation is necessary for modeling.

---

### `cv_indices.m`  
Generates **cross-validation indices** for k-fold validation.  
Randomly shuffles and assigns subjects to folds.

---

Input connectome .mat file should be an Nx268x268, where N is the number of participants in the study. Input behavioral file should be a .csv of the same length, though with an additional label row. Demo data are available for behavior using the file randomized_behavior_demo_data.csv and for conncetomes using the file found here: https://www.dropbox.com/scl/fi/9yc5ez8n4qqb5ak0gzkh5/randomized_connectome_demo_data.mat?rlkey=e8jrg0x63seztuu769hzqlbzv&st=bblwoa39&dl=0

