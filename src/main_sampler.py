import numpy as np
from data_loader import CountryDataLoader

## Main sampler script for the hierarchical model, change the file path to your data path.
## Depending on whether test or train data is used, you can change the file path to that and CountryDataLoader. Mode doesn't really matter, filepath is the important part.
## You can also change the prior used. "wishart" or "gamma" for the hyper_type.
## Dataloader can standardize and such, but normally the data should be standardized/transformed before, thus standardize=False. 
## Probably should change the file name after you've run a sampler, so you don't overwrite the samples. This could potentially be improved, also the way it saves the samples.

file_path = r'D:\Bachelor Data\Train'

loader = CountryDataLoader(
    data_path=file_path,
    standardize=False,
    countries=None,  # None for all
    mode="train"
)
features_to_drop = ["eom", "gvkey", "y", "weight","ret_exc_lead1m"]

loader.load_data(split_date="19940101")
train_data = loader.pretrain_dict

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

import importlib
import gibbs_sampler
importlib.reload(gibbs_sampler)

from gibbs_sampler import HierarchicalGibbsSampler

p = X_train.shape[1]
Sigma = np.eye(p)
nu = p + 1e6
sampler = HierarchicalGibbsSampler(
                X_dict=X_dict_train,
                Y_dict=Y_dict_train,
                X=X_train,
                Y=Y_train,
                n_iter=20000,
                burn_in=10000,
                tau_theta_g_sq=1e-6,
                alpha_sigma=1,
                beta_sigma=1,
                hyper_type="wishart",
                alpha_lambda=1e6,
                beta_lambda=1,
                Sigma=Sigma,
                nu=nu
            )
sampler.run()