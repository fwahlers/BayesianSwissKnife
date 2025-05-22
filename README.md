# Bachelor Thesis Codebase

This repository contains the code for the empirical and statistical analysis of hierarchical Bayesian models for cross-country financial prediction, specifically for my Bachelor's thesis: 
"Bayesian Machine Learning: Global Asset Joint Modelling - An MCMC Estimation"
The code is organized in the `src/` directory and is structured for modularity, reproducibility, and ease of experimentation. Dataset is needed on disk. 

---

## Directory Structure

---

## File Descriptions

### `data_loader.py`
- **Purpose:** Handles loading, preprocessing, and (optionally) standardizing country-level financial datasets.
- **Key Class:** `CountryDataLoader`  
  Loads data for multiple countries, splits into training/validation/test sets, and provides easy access to country-specific dataframes.

### `gibbs_sampler.py`
- **Purpose:** Implements the hierarchical Bayesian Swiss Knife Gibbs sampler.
- **Key Class:**
  - `HierarchicalGibbsSampler`: Runs the Gibbs sampling procedure for global and local parameters, supporting both Gamma and Wishart priors.

### `post_analyzer.py`
- **Purpose:** Provides additional tools for loading, analyzing, and visualizing posterior samples.
- **Key Class:**  
  - Posterior mean/median extraction  
  - Feature importance plots  
  - Country-level diagnostics

### `main_analysis.ipynb`
- **Purpose:** Jupyter notebook for running the main empirical analysis, including model evaluation, statistical summaries, and visualization of results.
- **Contents:**  
  - Loads posterior samples  
  - Computes out-of-sample $R^2$  
  - Constructs and evaluates portfolios  
  - Plots country-level and global results

### `main_sampler.py`
- **Purpose:** Script for running the Gibbs sampler on the training data.
- **Contents:**  
  - Loads and prepares data  
  - Configures and runs the `HierarchicalGibbsSampler`  
  - Saves posterior samples to disk

### `validation.py`
- **Purpose:** Script for hyperparameter validation and model selection.
- **Contents:**  
  - Splits data into pre-train and validation sets  
  - Runs the Gibbs sampler over a grid of hyperparameters  
  - Evaluates and reports validation $R^2$ scores

## Usage

- **Data Loading:** Use `CountryDataLoader` to load and preprocess country-level datasets.
- **Validation:** Use `validation.py` to perform hyperparameter tuning and model selection.
- **Model Training:** Run `main_sampler.py` to fit the final hierarchical model and save posterior samples.
- **Posterior Analysis:** Use `post_analyzer.py` for advanced diagnostics and visualization.
- **Analysis:** Open `main_analysis.ipynb` in JupyterLab to reproduce figures, tables, and statistical results.


---

## Requirements

See [`requirements.txt`](../requirements.txt) for the full list of dependencies.
