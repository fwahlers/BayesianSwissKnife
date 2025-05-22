import numpy as np
import pandas as pd
from data_loader import CountryDataLoader
from tqdm import tqdm
import gc
import os

## Validation script for the hierarchical model, change the file path to your data path.
## Depending on the prior you want to use, you can change the hyperparameters in the run_validation_grid function.
## Dataloader can standardize and such, but normally the data should be standardized/transformed before, thus standardize=False. 
## R^2 grid is just printed, but you can save it to a file if you want.

file_path = r'D:\Bachelor Data\Train'

loader = CountryDataLoader(
    data_path=file_path,
    standardize=False,
    countries=None,  # or None for all
    mode="train"
)
features_to_drop = ["eom", "gvkey", "y", "weight","ret_exc_lead1m"]

loader.load_data(split_date="19940101")
train_data = loader.pretrain_dict
val_data = loader.train_dict 

# --- Train data ---
X_dict_train = {
    c: df.drop(columns=features_to_drop).to_numpy()
    for c, df in train_data.items()
}

Y_dict_train = {
    c: df["y"].to_numpy()
    for c, df in train_data.items()
}

X_train = np.vstack([X_dict_train[c] for c in X_dict_train])
Y_train = np.concatenate([Y_dict_train[c] for c in Y_dict_train])

features_to_drop = ["gvkey", "weight", "ret_exc_lead1m"]  

X_dict_val = {}
Y_dict_val = {}
EOM_dict_val = {}

for c, df in val_data.items():
    df = df.copy()
    df["eom"] = pd.to_datetime(df["eom"])

    X_dict_val[c] = df.drop(columns=["y", "eom"] + features_to_drop).to_numpy()
    Y_dict_val[c] = df["y"].to_numpy()
    EOM_dict_val[c] = df["eom"].to_numpy()


import pandas as pd


def build_hyperparameter_grid():
    nus = [1e7,1e6, 1e5, 1e4, 1e3, 1e2, 1e1]
    taus = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    return [(n, t) for n in nus for t in taus] 
 
def r2_oos(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred)**2) / np.sum(y_true**2)

def compute_validation_r2(theta_g_map, theta_l_map, X_dict_val, Y_dict_val, EOM_dict_val):

    y_true_all = []
    y_pred_all = []
    for c in X_dict_val:
        X_c = X_dict_val[c]
        y_c = Y_dict_val[c]
        theta_total = theta_g_map + theta_l_map[c]
        y_pred = X_c @ theta_total

        theta_total = theta_g_map + theta_l_map[c]
        y_pred = X_c @ theta_total

        y_true_all.append(y_c)
        y_pred_all.append(y_pred)

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    return r2_oos(y_true,y_pred)*100  # percent



def run_validation_grid(X_dict_train, Y_dict_train, X_train, Y_train,
                        X_dict_val, Y_dict_val, HierarchicalGibbsSampler, PosteriorAnalyzer,
                        burn_in=1000, n_iter=2000):
    
    grid = build_hyperparameter_grid()
    nu_vals = sorted(set(n for n, _ in grid))
    tau_vals = sorted(set(t for _, t in grid))
    r2_matrix = np.zeros((len(nu_vals), len(tau_vals)))

    for i, (nu_val, tau_theta_g_sq) in enumerate(tqdm(grid, desc="Validation Grid")):
        row = nu_vals.index(nu_val)
        col = tau_vals.index(tau_theta_g_sq)
        try:
            p = X_train.shape[1]
            Sigma = np.eye(p)
            nu = p + nu_val

            sampler = HierarchicalGibbsSampler(
                X_dict=X_dict_train,
                Y_dict=Y_dict_train,
                X=X_train,
                Y=Y_train,
                n_iter=n_iter,
                burn_in=burn_in,
                tau_theta_g_sq=tau_theta_g_sq,
                alpha_sigma=1,
                beta_sigma=1,
                hyper_type="wishart",
                alpha_lambda=1,
                beta_lambda=1,
                Sigma=Sigma,
                nu=nu
            )
            sampler.run()
            
            theta_g_path = "theta_g_samples.parquet"
            theta_l_path = "theta_l_samples.parquet"

            analyzer = PosteriorAnalyzer(
                theta_g_path=theta_g_path,
                theta_l_path=theta_l_path,
                burn_in=0
            )

            r2_val = compute_validation_r2(
                analyzer.get_theta_g_mean(),
                analyzer.get_theta_l_mean(),
                X_dict_val,
                Y_dict_val,
                EOM_dict_val
            )

            r2_matrix[row, col] = r2_val

            del sampler, analyzer
            gc.collect()
            for f in [theta_g_path, theta_l_path]:
                if os.path.exists(f):
                    os.remove(f)

        except Exception as e:
            print(f"💥 Skipping nu={nu_val:.1e}, τ²={tau_theta_g_sq:.1e} due to error:\n{e}")
            r2_matrix[row, col] = np.nan
            continue

    r2_df = pd.DataFrame(
        r2_matrix,
        index=[f"nu={a:.0e}" for a in nu_vals],
        columns=[f"τ²={t:.0e}" for t in tau_vals]
    )

    return r2_df

import importlib
import gibbs_sampler
importlib.reload(gibbs_sampler)

from gibbs_sampler import HierarchicalGibbsSampler
from post_analyzer import PosteriorAnalyzer

r2_grid = run_validation_grid(
    X_dict_train, Y_dict_train, X_train, Y_train,
    X_dict_val, Y_dict_val,
    HierarchicalGibbsSampler, PosteriorAnalyzer
) 
print(r2_grid)
