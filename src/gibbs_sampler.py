import numpy as np
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict
from scipy.stats import wishart
from scipy.linalg import cho_factor, cho_solve
from numpy.linalg import inv
from tqdm import trange

class HierarchicalGibbsSampler:
    def __init__(
        self,
        X_dict: Dict[str, np.ndarray],
        Y_dict: Dict[str, np.ndarray],
        X: np.ndarray,
        Y: np.ndarray,
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
            X: concatenated X_c matrix
            Y: concatenated Y_c vector
            n_iter: number of iterations
            tau_theta_g_sq: prior variance for theta_g
            alpha_sigma, beta_sigma: gamma prior for variance
            beta_sigma: scale parameter for the gamma prior
            hyper_type: choose between "gamma" and "wishart"
            burn_in: burn_in period
            alpha_lambda, beta_lambda: hyperparameters for the Gamma prior
            Sigma,nu: hyperparameters for the Wishart prior 
        """
        np.random.seed(42)  
        
        self.X_dict = X_dict
        self.Y_dict = Y_dict
        self.X = X
        self.Y = Y
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.hyper_type = hyper_type
        self.Gc_trace = {c: [] for c in self.X_dict} 
        self.N = self.Y.shape[0]
        self.p = list(X_dict.values())[0].shape[1]
        self.C = len(X_dict)
        self.I = np.eye(self.p)

        
        self.tau_theta_g_sq = tau_theta_g_sq

        
        self.alpha_sigma = alpha_sigma
        self.beta_sigma = beta_sigma

        
        self.alpha_lambda = alpha_lambda
        self.beta_lambda = beta_lambda

        
        self.Sigma = Sigma if Sigma is not None else self.I
        self.nu = nu if nu is not None else self.p + 2 

        
        self.theta_g_samples = []
        self.theta_l_samples = []
        self.sigma2_samples = []
        self.Lambda_samples = []

        if self.hyper_type == "gamma":
            init_tau2 = alpha_lambda * beta_lambda
            self.Lambda = {c: init_tau2 * self.I for c in X_dict}
        elif self.hyper_type == "wishart":
            self.Lambda = {c: np.copy(self.Sigma) for c in X_dict}
        

        self.theta_g = np.random.normal(0, 1, size=self.p)
        self.theta_l = {
            c: np.random.normal(0, 1, size=self.p)
            for c in self.X_dict
        }
    
        self.sigma2 = 1

        self.XtX = {}
        for c in self.X_dict:
            X_c = X_dict[c]
            self.XtX[c] = X_c.T @ X_c



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
            XtX = self.XtX[c]

            A += XtX
            b += (X_c.T @ (Y_c - X_c @ theta_l_c))

        A = (1.0 / sigma2) * A + tau_sq_inv * self.I
        L, lower = cho_factor(A)

        A_inv = cho_solve((L, lower), np.eye(self.p))
        g = A_inv @ ((1.0 / sigma2) * b)

        self.theta_g = np.random.multivariate_normal(mean=g, cov=A_inv)
        

    def sample_local_parameters(self):
        sigma2 = self.sigma2
        theta_g = self.theta_g

        def sample_one_country(c):
            X_c = self.X_dict[c]
            Y_c = self.Y_dict[c]
            Lambda_c = self.Lambda[c]
            XtX = self.XtX[c]

            B_c = (1.0 / sigma2) * XtX + Lambda_c

            L, lower = cho_factor(B_c)
            B_c_inv = cho_solve((L, lower), np.eye(self.p))

            b = X_c.T @ (Y_c - X_c @ theta_g)
            d_c = B_c_inv @ ((1.0 / sigma2) * b)

            theta_l_c = np.random.multivariate_normal(mean=d_c, cov=B_c_inv)
            return c, theta_l_c

        with ThreadPoolExecutor() as executor:
            results = executor.map(sample_one_country, self.X_dict)

        # Unpack results
        for c, theta_l_c in results:
            self.theta_l[c] = theta_l_c
    

    def sample_variance(self):
        """
        Samples tau^2 ~ Gamma(alpha_sigma, (1/2 r'r + 1/beta_sigma)^-1),
        where r = Y - X θ_g - Z θ_ℓ, computed efficiently without concatenation.
        """
        rss = 0.0

        for c in self.X_dict:
            X_c = self.X_dict[c]         # shape: (n_c, p)
            Y_c = self.Y_dict[c]         # shape: (n_c,)
            y_pred_g_c = X_c @ self.theta_g  # shape: (n_c,)
            y_pred_l_c = X_c @ self.theta_l[c]

            r_c = Y_c - y_pred_g_c - y_pred_l_c
            rss += r_c @ r_c  

        alpha_cond = self.alpha_sigma + self.N / 2
        beta_cond = 0.5 * rss + 1 / self.beta_sigma

        tau2 = np.random.gamma(shape=alpha_cond, scale=1 / beta_cond)
        self.sigma2 = 1 / tau2

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
                alpha_cond = self.alpha_lambda + self.p/2
                beta_cond = 1/(0.5 * theta_l_c.T @ theta_l_c + 1/self.beta_lambda)

                tau2 = np.random.gamma(shape=alpha_cond, scale=beta_cond)
                self.Lambda[c] = tau2 * self.I


            elif self.hyper_type == "wishart":
                V = inv(self.Sigma) + theta_l_c @ theta_l_c.T
                inv_V = inv(V)
                df = self.nu + 1
                self.Lambda[c] = wishart.rvs(df=df, scale=inv_V)

            else:
                raise ValueError(f"Unknown hyper_type: {self.hyper_type}")

        

    def run(self, verbose: bool = True, save_path: str = "./"):
        """
        Runs the full Gibbs sampler for `n_iter` iterations.
        Writes to .parquet every 1000 iterations.
        """
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        iterator = trange(self.n_iter) if verbose else range(self.n_iter)
        np.random.seed(42)  

        for n in iterator:
            
            self.sample_global_parameters()
            self.sample_local_parameters()
            self.sample_variance()
            self.sample_hyperparameters()
            if n >= self.burn_in:
                self.theta_g_samples.append(self.theta_g)
                self.theta_l_samples.append({c: self.theta_l[c].copy() for c in self.X_dict})
                self.sigma2_samples.append(self.sigma2)
                self.Lambda_samples.append({c: self.Lambda[c].copy() for c in self.X_dict})
                norm_g = np.linalg.norm(self.theta_g)
                for c in self.X_dict:
                    norm_l = np.linalg.norm(self.theta_l[c])
                    Gc = norm_g / (norm_g + norm_l)
                    self.Gc_trace[c].append(Gc)

            
            if n > self.burn_in and ((n + 1) % 1000 == 0 or (n +1) == self.n_iter):
                
                pd.DataFrame(self.theta_g_samples).to_parquet(
                    f"{save_path}/theta_g_samples.parquet", index=False
                )
                
                pd.DataFrame([
                    {f"{c}_{i}": sample[c][i] for c in sample for i in range(len(sample[c]))}
                    for sample in self.theta_l_samples
                ]).to_parquet(f"{save_path}/theta_l_samples.parquet", index=False)
                
                pd.DataFrame({c: vals for c, vals in self.Gc_trace.items()}).to_parquet(
                f"{save_path}/Gc_samples.parquet", index=False)
                pd.DataFrame(self.sigma2_samples, columns=["sigma2"]).to_parquet(
                    f"{save_path}/sigma2.parquet", index=False)

