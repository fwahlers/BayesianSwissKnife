import numpy as np
import pandas as pd
from typing import Dict, List
from numpy.linalg import inv
from scipy.stats import invgamma, gamma, wishart, multivariate_normal
from tqdm import trange
import os


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
            alpha_sigma, beta_sigma: inverse gamma prior for variance
            hyper_type: choose between "gamma" and "wishart"
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

        self.p = list(X_dict.values())[0].shape[1]
        self.C = len(X_dict)

        # Initialize σ² to a reasonable value
        self.sigma2 = 1.0


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
        self.Sigma_inv = inv(self.Sigma)
        self.nu = nu if nu is not None else self.p + 2  # just a safe default

        # Store samples
        self.theta_g_samples = []
        self.theta_l_samples = []
        self.sigma2_samples = []
        self.Lambda_samples = []

        # Initialize Λ_c based on hyper_type
        if self.hyper_type == "gamma":
        # Use small τ² * I_p
            init_tau2 = 1.0
            self.Lambda = {c: init_tau2 * np.eye(self.p) for c in X_dict}
        elif self.hyper_type == "wishart":
        # Use Wishart scale prior directly (from __init__ args)
            self.Lambda = {c: np.copy(self.Sigma) for c in X_dict}
        

        self.theta_g = np.random.normal(0, 1, size=self.p)
        self.theta_l = {
            c: np.random.normal(0, 1, size=self.p)
            for c in self.X_dict
        }



    def sample_global_parameters(self):
        sigma2 = self.sigma2
        tau_sq_inv = 1.0 / self.tau_theta_g_sq

        A = np.zeros((self.p, self.p))
        b = np.zeros(self.p)

        for c in self.X_dict:
            X_c = self.X_dict[c]     # shape: (n_c, p)
            Y_c = self.Y_dict[c]     # shape: (n_c,)
            theta_l_c = self.theta_l[c]  # shape: (p,)

            A += (X_c.T @ X_c)
            b += X_c.T @ (Y_c - X_c @ theta_l_c)

        A = (1 / sigma2) * A + tau_sq_inv * np.eye(self.p)
        A_inv = inv(A)

        g = A_inv @ b

        # Sample from N(g, A^{-1})
        self.theta_g = np.random.multivariate_normal(mean=g, cov=A_inv)
        self.theta_g_samples.append(self.theta_g)

    def sample_local_parameters(self):
        sigma2 = self.sigma2

        for c in self.X_dict:
            X_c = self.X_dict[c]          # shape: (n_c, p)
            Y_c = self.Y_dict[c]          # shape: (n_c,)
            Lambda_c = self.Lambda[c]     # shape: (p, p)

            XtX = X_c.T @ X_c
            B_c = (1 / sigma2) * XtX + Lambda_c
            B_c_inv = inv(B_c)

            residual = Y_c - X_c @ self.theta_g
            d_c = B_c_inv @ (X_c.T @ residual)

            self.theta_l[c] = np.random.multivariate_normal(mean=d_c, cov=B_c_inv)

        # Store a deep copy of the full set of theta_l's for this iteration
        self.theta_l_samples.append({c: self.theta_l[c].copy() for c in self.X_dict})

        # Stack all θ_ℓ into one global vector and store it for reuse
        self.theta_l_vector = np.concatenate([
            self.theta_l[c] for c in self.X_dict
        ])

    def sample_variance(self):
        """
        Samples sigma^2 ~ Gamma(alpha_sigma,(1/2 r'r + 1/beta_sigma)^-1),
        where r = Y - X θ_g - Z θ_ℓ
        """
        # Stack all local θ_ℓ into one vector
        self.theta_l_vector = np.concatenate([self.theta_l[c] for c in self.X_dict])

        r = self.Y - self.X @ self.theta_g - self.Z @ self.theta_l_vector
        rss = r.T @ r

        alpha_cond = self.alpha_sigma 
        beta_cond = (1/self.beta_sigma + rss) 

        self.sigma2 = invgamma.rvs(a=alpha_cond, scale=(1/beta_cond))
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
                shape = self.alpha_lambda + self.p
                scale = 1 / (0.5 * theta_l_c.T @ theta_l_c + 1/self.beta_lambda)

                tau2 = np.random.gamma(shape=shape, scale=scale)
                self.Lambda[c] = tau2 * np.eye(self.p)


            elif self.hyper_type == "wishart":
                V = np.linalg.inv(self.Sigma_inv + np.outer(theta_l_c, theta_l_c))
                df = self.nu + 1
                self.Lambda[c] = wishart.rvs(df=df, scale=V)

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
                Gc = norm_g / (norm_g + norm_l + 1e-10)
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
                    f"{save_path}/Gc_trace.parquet", index=False
            )



    def get_posterior_samples(self):
        return {
            "theta_g": np.array(self.theta_g_samples[self.burn_in:]),
            "theta_l": self.theta_l_samples[self.burn_in:]
        }
    def get_posterior_means(self) -> Dict[str, np.ndarray]:
        """
        Computes posterior means for theta_g and theta_l (both per country and stacked block form)
        using get_posterior_samples().
        
        Returns:
            {
                "theta_g": mean vector (p,),
                "theta_l": dict of {country_code: mean vector},
            }
        """
        samples = self.get_posterior_samples()

        # Global mean
        theta_g_mean = samples["theta_g"].mean(axis=0)

        # Local means per country
        theta_l_post_samples = samples["theta_l"]
        country_codes = self.X_dict.keys()

        theta_l_sum = {c: np.zeros_like(self.theta_l[c]) for c in country_codes}

        for sample in theta_l_post_samples:
            for c in country_codes:
                theta_l_sum[c] += sample[c]

        n_samples = len(theta_l_post_samples)
        theta_l_means = {c: theta_l_sum[c] / n_samples for c in country_codes}
        return {
            "theta_g": theta_g_mean,
            "theta_l": theta_l_means,
        }
    def get_posterior_mean_Gc(self) -> Dict[str, float]:
        """
        Returns posterior mean of G_c for each country (after burn-in).
        """
        return {
            c: np.mean(self.Gc_trace[c][self.burn_in:])
            for c in self.Gc_trace
        }


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
