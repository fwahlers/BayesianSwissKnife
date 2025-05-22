import numpy as np
import pandas as pd
from typing import Dict, Optional, Dict
import matplotlib.pyplot as plt


class PosteriorAnalyzer:
    def __init__(
        self,
        theta_g_path: str,
        theta_l_path: str,
        gc_path: Optional[str] = None,
        sigma2_path: Optional[str] = None,
        burn_in: int = 1000
    ):
        self.p = 111 ## number of features(just for readability of the parquet, this is bad that it is defined as constant)
        self.burn_in = burn_in
        self.theta_g = pd.read_parquet(theta_g_path).iloc[burn_in:].to_numpy()
        self.theta_l = pd.read_parquet(theta_l_path).iloc[burn_in:]
        self.sigma2 = pd.read_parquet(sigma2_path).iloc[burn_in:].to_numpy() if sigma2_path else None
        self.gc_df = pd.read_parquet(gc_path).iloc[burn_in:] if gc_path else None

        self.country_codes = sorted(set(col.split("_")[0] for col in self.theta_l.columns))
        self.feature_numbers = [col.split("_")[1] for col in self.theta_l.columns if "_" in col][:self.p]  
        try:
            ref_df = pd.read_parquet("D:/Bachelor Data/Test/irl.parquet")
            drop_cols = ["eom", "gvkey", "y", "weight", "ret_exc_lead1m"]
            self.feature_names = [col for col in ref_df.columns if col not in drop_cols]
        except Exception as e:
            raise RuntimeError("Failed to load feature names from irl.parquet") from e

    def get_theta_g_samples(self) -> np.ndarray:
        return self.theta_g

    def get_theta_l_samples(self, country: str) -> np.ndarray:
        cols = [f"{country}_{f}" for f in self.feature_numbers]
        return self.theta_l[cols].to_numpy()
    
    def get_sigma2_samples(self) -> np.ndarray:
            return self.sigma2

    def get_theta_g_mean(self) -> np.ndarray:
        return self.theta_g.mean(axis=0)
    
    def get_theta_l_mean(self) -> Dict[str, np.ndarray]:
        return {
            c: self.get_theta_l_samples(c).mean(axis=0)
            for c in self.country_codes
        }
    def get_theta_g_median(self) -> np.ndarray:
        return np.median(self.theta_g,axis=0)
    
    def get_theta_l_median(self) -> Dict[str, np.ndarray]:
        return {
            c: np.median(self.get_theta_l_samples(c),axis=0)
            for c in self.country_codes
        }
    
    def get_Gc_posterior(self) -> Optional[Dict[str, float]]:
        if self.gc_df is not None:
            return self.gc_df.to_dict()
        return None

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
        clip_bounds: tuple[float, float] = (1, 99),
        ax: Optional[plt.Axes] = None
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
            ax: matplotlib Axes object to plot on (optional)
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

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))

        ax.hist(data, bins=bins, color="skyblue", edgecolor="k", alpha=0.7)
        ax.set_title(f"Histogram of {label}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        if ax is None:
            plt.tight_layout()
            plt.show()

    def plot_timeseries_median(
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
            data = self.get_theta_g_median()
            label = "Posterior Mean of θ_g"
        elif target == "theta_l":
            if not country:
                raise ValueError("Country must be specified when plotting theta_l.")
            data = self.get_theta_l_median()[country]
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
    def plot_timeseries_mean(
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
        elif target == "theta_c":   
            if not country:
                raise ValueError("Country must be specified when plotting theta_c.")
            data = self.get_theta_g_mean() + self.get_theta_l_mean()[country]
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
        
    def plot_top_features_with_ci(
        self,
        target: str = "theta_g",
        country: Optional[str] = None,
        top_n: int = 20,
        ci: float = 0.9
    ):
        """
        Plot top N influential features based on absolute mean, with credible interval.

        Args:
            target: "theta_g" or "theta_l"
            country: required if target is "theta_l"
            top_n: number of top features to display
            ci: credible interval width (e.g., 0.9 = 90%)
        """
        plt.style.use("default")  # or try 'ggplot', 'bmh', etc.
        if target == "theta_g":
            samples = self.theta_g
        elif target == "theta_l":
            if not country:
                raise ValueError("Country must be specified for theta_l.")
            samples = self.get_theta_l_samples(country)
        elif target == "theta_c":   
            if not country:
                raise ValueError("Country must be specified when plotting theta_c.")
            samples = self.theta_g[:-1000] + self.get_theta_l_samples(country)
        else:
            raise ValueError("Target must be 'theta_g' or 'theta_l'")

        means = samples.mean(axis=0)
        top_idx = np.argsort(np.abs(means))[-top_n:][::-1]

        lower_q = (1 - ci) / 2
        upper_q = 1 - lower_q
        lower = np.quantile(samples[:, top_idx], q=lower_q, axis=0)
        upper = np.quantile(samples[:, top_idx], q=upper_q, axis=0)

        line_color = "#237FE9"  # light blue
        fill_alpha = 0.2        # transparency
        line_width = 1
        marker_size = 4.5

        x = np.arange(top_n)
        plt.figure(figsize=(12, 5))
        plt.plot(x, means[top_idx], label="Posterior Mean",
                 color=line_color, linewidth=line_width, marker='o', markersize=marker_size)
        plt.fill_between(x, lower, upper, color=line_color, alpha=fill_alpha, label=f"{int(ci*100)}% CI")
        plt.xticks(ticks=x, labels=[self.feature_names[i] for i in top_idx], rotation=45, ha="right")
        plt.ylabel("Coefficient Value")
        plt.legend()
        plt.grid(False)

        # ⬇️ Round y-axis ticks to 3 decimals
        ax = plt.gca()
        yticks = ax.get_yticks()
        ax.set_yticklabels([f"{y:.3f}" for y in yticks])
        plt.show()



