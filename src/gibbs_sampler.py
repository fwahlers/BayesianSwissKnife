import numpy as np
import pandas as pd
from typing import Dict
from numpy.linalg import inv
from tqdm import trange
from scipy.stats import wishart, invgamma
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, Dict



class HierarchicalGibbsSampler:
    def __init__(
        self,
        X_dict: Dict[str, np.ndarray],
        Y_dict: Dict[str, np.ndarray],
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        n_iter: int,
        tau_theta_g_sq: float = 1.0,
        alpha_sigma: float = 2.0,
        beta_sigma: float = 1.0,
        hyper_type: str = "gamma",  # or "wishart"
        burn_in: int = 0,
        alpha_lambda: float = 2.0,
        beta_lambda: float = 1.0,
        Sigma: np.ndarray = None,
        nu: float = None
    ):
        """
        Args:
            X_dict: country → X_c matrix
            Y_dict: country → Y_c vector
            n_iter: number of Gibbs iterations
            tau_theta_g_sq: prior variance for theta_g
            alpha_sigma, beta_sigma: gamma prior for variance(through tau)
            hyper_type: choose between "gamma" and "wishart"
            burn_in: burn_in period, pretty useless in the new version(will be removed)
            alpha_lbamda, beta_lambda: 
        """

        # Save all inputs
        self.X_dict = X_dict
        self.Y_dict = Y_dict
        self.X = X
        self.Y = Y
        self.Z = Z
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.hyper_type = hyper_type
        self.Gc_trace = {c: [] for c in self.X_dict} 
        self.N = self.Y.shape[0]
        self.p = list(X_dict.values())[0].shape[1]
        self.C = len(X_dict)


        # Global prior on θ_g
        self.tau_theta_g_sq = tau_theta_g_sq

        # Prior for σ²
        self.alpha_sigma = alpha_sigma
        self.beta_sigma = beta_sigma

        # Gamma hyperparameters
        self.alpha_lambda = alpha_lambda
        self.beta_lambda = beta_lambda

        # Wishart hyperparameters
        self.Sigma = Sigma if Sigma is not None else np.eye(self.p)
        self.nu = nu if nu is not None else self.p + 2  # just a safe default

        # Store samples
        self.theta_g_samples = []
        self.theta_l_samples = []
        self.sigma2_samples = []
        self.Lambda_samples = []

        # Initialize Λ_c based on hyper_type
        if self.hyper_type == "gamma":
            init_tau2 = alpha_lambda * beta_lambda
            self.Lambda = {c: init_tau2 * np.eye(self.p) for c in X_dict}
        elif self.hyper_type == "wishart":
            self.Lambda = {c: np.copy(self.Sigma) for c in X_dict}
        

        self.theta_g = np.random.normal(0, 1, size=self.p)
        self.theta_l = {
            c: np.random.normal(0, 1, size=self.p)
            for c in self.X_dict
        }
        # Initialize σ² to a reasonable value
        self.sigma2 = 0.5


    def sample_global_parameters(self):
        """
        Samples theta_g ~ N(g, A^{-1}),
        where
        A = 1/sigma2(sum_c X_c'X_c) + 1/tau_theta_g *I_p
        g = A^{-1}.(1/sigma2)*b
        """

        sigma2 = self.sigma2
        tau_sq_inv = 1.0/self.tau_theta_g_sq
        A = np.zeros((self.p, self.p))
        b = np.zeros(self.p)

        for c in self.X_dict:
            X_c = self.X_dict[c]         # shape: (n_c, p)
            Y_c = self.Y_dict[c]         # shape: (n_c,)
            theta_l_c = self.theta_l[c]  # shape: (p,)

            A += (X_c.T @ X_c)
            b += (X_c.T @ (Y_c - X_c @ theta_l_c))

        A = (1.0 / sigma2) * A + tau_sq_inv * np.eye(self.p)
        A_inv = inv(A)

        g = A_inv @ ((1/sigma2)*b)

        self.theta_g = np.random.multivariate_normal(mean=g, cov=A_inv)
        self.theta_g_samples.append(self.theta_g)

    def sample_local_parameters(self):
        """
        Samples theta_l ~ prod N(d_c, B_c^{-1}),
        where
        B_c = 1/sigma2 X_c'X_c + Lambda_c
        d_c = B_c^{-1} * ((1/sigma2)*b)
        """

        sigma2 = self.sigma2
        theta_g = self.theta_g
        for c in self.X_dict:
            X_c = self.X_dict[c]          # shape: (n_c, p)
            Y_c = self.Y_dict[c]          # shape: (n_c,)
            Lambda_c = self.Lambda[c]     # shape: (p, p)

            XtX = X_c.T @ X_c
            B_c = (1.0 / sigma2) * XtX + Lambda_c
            B_c_inv = inv(B_c)

            b = X_c.T @ (Y_c - X_c @ theta_g)
            d_c = B_c_inv @ ((1/sigma2)*b)

            self.theta_l[c] = np.random.multivariate_normal(mean=d_c, cov=B_c_inv)

        # Store a deep copy of the full set of theta_l's for this iteration
        self.theta_l_samples.append({c: self.theta_l[c].copy() for c in self.X_dict})

        # Stack all θ_ℓ into one global vector and store it for reuse
        self.theta_l_vector = np.concatenate([
            self.theta_l[c] for c in self.X_dict
        ])

    def sample_variance(self):
        """
        Samples tau^2 ~ Gamma(alpha_sigma,(1/2 r'r + 1/beta_sigma)^-1),
        sigma^2 = 1/tau^2 
        where r = Y - X θ_g - Z θ_ℓ
        """
        
        r = self.Y - self.X @ self.theta_g - self.Z @ self.theta_l_vector
        rss = 0.5*r.T @ r
        alpha_cond = self.alpha_sigma + self.N/2
        beta_cond = (rss + 1/self.beta_sigma) 

        
        tau2 = np.random.gamma(shape=alpha_cond, scale=1/beta_cond)
        self.sigma2 = 1/tau2
        self.sigma2_samples.append(self.sigma2)

    def sample_hyperparameters(self):
        """
        Sample Λ_c for each country, using either:
        - Gamma prior (diagonal Λ_c = τ² I_p)
        - Wishart prior (full Λ_c)
        """
        self.Lambda = {}

        for c in self.X_dict:
            theta_l_c = self.theta_l[c]

            if self.hyper_type == "gamma":
                alpha_cond = self.alpha_lambda + self.p
                beta_cond = 1/(0.5 * theta_l_c.T @ theta_l_c + 1/self.beta_lambda)

                tau2 = np.random.gamma(shape=alpha_cond, scale=beta_cond)
                self.Lambda[c] = tau2 * np.eye(self.p)


            elif self.hyper_type == "wishart":
                V = inv(self.Sigma) + theta_l_c @ theta_l_c.T
                inv_V = inv(V)
                df = self.nu + 1
                self.Lambda[c] = wishart.rvs(df=df, scale=inv_V)

            else:
                raise ValueError(f"Unknown hyper_type: {self.hyper_type}")

        self.Lambda_samples.append({c: self.Lambda[c].copy() for c in self.X_dict})

    def run(self, verbose: bool = True, save_path: str = "./"):
        """
        Runs the full Gibbs sampler for `n_iter` iterations.
        Writes to .parquet every 1000 iterations.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        iterator = trange(self.n_iter) if verbose else range(self.n_iter)

        for n in iterator:
            self.sample_global_parameters()
            self.sample_local_parameters()
            self.sample_variance()
            self.sample_hyperparameters()
       
            # Save G_c at this iteration
            norm_g = np.linalg.norm(self.theta_g)
            for c in self.X_dict:
                norm_l = np.linalg.norm(self.theta_l[c])
                Gc = norm_g / (norm_g + norm_l )
                self.Gc_trace[c].append(Gc)

            # Save every 1000 iterations
            if (n + 1) % 1000 == 0 or (n + 1) == self.n_iter:
                # θ_g samples
                pd.DataFrame(self.theta_g_samples).to_parquet(
                    f"{save_path}/theta_g_samples.parquet", index=False
                )

                # θ_l samples
                df_theta_l = pd.DataFrame([
                    {f"{c}_{i}": sample[c][i] for c in sample for i in range(len(sample[c]))}
                    for sample in self.theta_l_samples
                ])
                df_theta_l.to_parquet(f"{save_path}/theta_l_samples.parquet", index=False)

                # G_c trace
                pd.DataFrame(self.Gc_trace).to_parquet(
                    f"{save_path}/Gc_samples.parquet", index=False
            )


class PosteriorAnalyzer:
    def __init__(
        self,
        theta_g_path: str,
        theta_l_path: str,
        gc_path: Optional[str] = None,
        burn_in: int = 1000
    ):
        self.p = 111 ## number of features(just for readability of the parquet, this is bad that it is defined as constant)
        self.burn_in = burn_in
        self.theta_g = pd.read_parquet(theta_g_path).iloc[burn_in:].to_numpy()
        self.theta_l = pd.read_parquet(theta_l_path).iloc[burn_in:]
        self.gc_df = pd.read_parquet(gc_path).iloc[burn_in:] if gc_path else None

        self.country_codes = sorted(set(col.split("_")[0] for col in self.theta_l.columns))
        self.feature_names = [col.split("_")[1] for col in self.theta_l.columns if "_" in col][:self.p]  

    def get_theta_g_samples(self) -> np.ndarray:
        return self.theta_g

    def get_theta_l_samples(self, country: str) -> np.ndarray:
        cols = [f"{country}_{f}" for f in self.feature_names]
        return self.theta_l[cols].to_numpy()


    def get_theta_g_mean(self) -> np.ndarray:
        return self.theta_g.mean(axis=0)
    
    def get_theta_l_mean(self) -> Dict[str, np.ndarray]:
        return {
            c: self.get_theta_l_samples(c).mean(axis=0)
            for c in self.country_codes
        }


    def get_Gc_posterior_mean(self) -> Optional[Dict[str, float]]:
        if self.gc_df is not None:
            return self.gc_df.mean().to_dict()
        return None

    def plot_histogram(
        self,
        target: str = "theta_g",
        feature_idx: int = 0,
        country: Optional[str] = None,
        bins: int = 50,
        clip: bool = True,
        clip_bounds: tuple[float, float] = (1, 99)
        ):
        """
        Plot histogram for a specific feature index of θ_g or θ_l for a country.

        Args:
            target: "theta_g" or "theta_l"
            feature_idx: index of the feature to plot
            country: required if target is "theta_l"
            bins: number of bins for the histogram
            clip: whether to clip outliers based on percentiles
            clip_bounds: percentiles to clip (low, high), e.g., (1, 99)
        """
        if target == "theta_g":
            data = self.theta_g[:, feature_idx]
            label = f"θ_g[{feature_idx}]"
        elif target == "theta_l":
            if not country:
                raise ValueError("Country must be specified when plotting theta_l.")
            data = self.get_theta_l_samples(country)[:, feature_idx]
            label = f"θ_l[{country}, {feature_idx}]"
        else:
            raise ValueError("target must be 'theta_g' or 'theta_l'")

        if clip:
            lower, upper = np.percentile(data, clip_bounds)
            data = data[(data >= lower) & (data <= upper)]

        plt.figure(figsize=(8, 4))
        plt.hist(data, bins=bins, color="skyblue", edgecolor="k", alpha=0.7)
        plt.title(f"Histogram of {label}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.grid(True, axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def plot_timeseries(
        self,
        target: str = "theta_g",
        country: Optional[str] = None
        ):
        """
        Args:
            target: "theta_g" or "theta_l"
            feature_idx: index of the feature to plot
            country: required if target is "theta_l
        """
        if target == "theta_g":
            data = self.get_theta_g_mean()
            label = "Posterior Mean of θ_g"
        elif target == "theta_l":
            if not country:
                raise ValueError("Country must be specified when plotting theta_l.")
            data = self.get_theta_l_mean()[country]
            label = f"Posterior Mean of θ_l[{country}]"
        else:
            raise ValueError("target must be 'theta_g' or 'theta_l'")
        plt.figure(figsize=(10, 5))
        plt.plot(data, label=label, linewidth=2)
        plt.title(f"Timeseries of {label}")
        plt.xlabel("Feature Index")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


"""X = np.random.randn(10, 2)
theta_g_true = np.random.randn(2)
Y = X @ theta_g_true + np.random.normal(0, 0.1, size=10)

dummy = HierarchicalGibbsSampler(
    X_dict={"test": X},
    Y_dict={"test": Y},
    X=X,
    Y=Y,
    Z=X,
    n_iter=100,
    alpha_sigma=2.0,
    beta_sigma=1.0,
    hyper_type="gamma"  # or "gamma"
)

dummy.run()"""
