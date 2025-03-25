import numpy as np
import matplotlib.pyplot as plt

def generate_hierarchical_data(
    m=5,  # number of groups
    n=10, # number of observations per group (assume equal for simplicity)
    mu_true=0.0,
    tau_true=1.0,
    sigma_true=2.0,
    seed=42
):
    """
    Generate synthetic data from the hierarchical model:
    theta_j ~ Normal(mu_true, tau_true^2)
    y_ji ~ Normal(theta_j, sigma_true^2)
    """
    rng = np.random.default_rng(seed)
    
    # Generate group-level intercepts
    theta_true = rng.normal(mu_true, tau_true, size=m)
    
    # Generate observations
    data = []
    for j in range(m):
        y_j = rng.normal(theta_true[j], sigma_true, size=n)
        data.append(y_j)
    
    data = np.array(data)  # shape: (m, n)
    return data, theta_true

def systematic_scan_gibbs(
    data,
    # Hyperparameters for priors
    sigma_mu=10.0,
    a_tau=2.0, b_tau=2.0,
    a_sigma=2.0, b_sigma=2.0,
    n_iter=10_000, burn_in=2_000
):
    """
    Perform systematic-scan Gibbs sampling on the two-level hierarchical model:
    
    Model:
      y_{ji} | theta_j, sigma^2 ~ Normal(theta_j, sigma^2)
      theta_j | mu, tau^2 ~ Normal(mu, tau^2)
      mu ~ Normal(0, sigma_mu^2)
      tau^2 ~ InverseGamma(a_tau, b_tau)
      sigma^2 ~ InverseGamma(a_sigma, b_sigma)
    
    data: np.ndarray of shape (m, n_j) 
          m groups, each with n_j observations (assume n_j the same for simplicity).
    """
    rng = np.random.default_rng()
    
    m = data.shape[0]
    n_j = data.shape[1]
    
    # Initialize parameters
    theta = np.zeros(m)  # random intercepts
    mu = 0.0
    tau2 = 1.0
    sigma2 = 1.0
    
    # To store samples
    samples_theta = np.zeros((n_iter, m))
    samples_mu = np.zeros(n_iter)
    samples_tau2 = np.zeros(n_iter)
    samples_sigma2 = np.zeros(n_iter)
    
    # Precompute some constants
    sum_yj = np.sum(data, axis=1)  # sum of observations in each group
    # shape: (m,)
    
    for it in range(n_iter):
        # --- 1) Update each theta_j in a systematic order ---
        for j in range(m):
            # Posterior variance
            var_j = 1.0 / (n_j / sigma2 + 1.0 / tau2)
            # Posterior mean
            mean_j = var_j * ((sum_yj[j] / sigma2) + (mu / tau2))
            # Sample
            theta[j] = rng.normal(mean_j, np.sqrt(var_j))
        
        # --- 2) Update mu ---
        var_mu = 1.0 / (m / tau2 + 1.0 / (sigma_mu**2))
        mean_mu = var_mu * ((np.sum(theta) / tau2) + 0.0)  # prior mean is 0
        mu = rng.normal(mean_mu, np.sqrt(var_mu))
        
        # --- 3) Update tau^2 ---
        a_tau_post = a_tau + m / 2.0
        b_tau_post = b_tau + 0.5 * np.sum((theta - mu)**2)
        # Sample from InverseGamma(a_tau_post, b_tau_post)
        tau2 = 1.0 / rng.gamma(a_tau_post, 1.0 / b_tau_post)
        
        # --- 4) Update sigma^2 ---
        a_sigma_post = a_sigma + (m * n_j) / 2.0
        # sum of squared residuals
        ssr = 0.0
        for j in range(m):
            ssr += np.sum((data[j] - theta[j])**2)
        b_sigma_post = b_sigma + 0.5 * ssr
        # Sample from InverseGamma(a_sigma_post, b_sigma_post)
        sigma2 = 1.0 / rng.gamma(a_sigma_post, 1.0 / b_sigma_post)
        
        # Store samples
        samples_theta[it] = theta
        samples_mu[it] = mu
        samples_tau2[it] = tau2
        samples_sigma2[it] = sigma2
    
    # Discard burn-in
    samples_theta = samples_theta[burn_in:]
    samples_mu = samples_mu[burn_in:]
    samples_tau2 = samples_tau2[burn_in:]
    samples_sigma2 = samples_sigma2[burn_in:]
    
    return samples_theta, samples_mu, samples_tau2, samples_sigma2

def main():
    # ------------------------
    # 1. Generate synthetic data
    # ------------------------
    m = 5
    n = 1000000
    mu_true = 2.0
    tau_true = 1.5
    sigma_true = 2.0
    
    data, theta_true = generate_hierarchical_data(
        m=m, n=n,
        mu_true=mu_true,
        tau_true=tau_true,
        sigma_true=sigma_true,
        seed=123
    )
    
    # ------------------------
    # 2. Run Systematic-Scan Gibbs
    # ------------------------
    samples_theta, samples_mu, samples_tau2, samples_sigma2 = systematic_scan_gibbs(
        data,
        sigma_mu=10.0,
        a_tau=2.0, b_tau=2.0,
        a_sigma=2.0, b_sigma=2.0,
        n_iter=50000, burn_in=2_000
    )
    
    # ------------------------
    # 3. Posterior Summaries
    # ------------------------
    # Posterior means
    theta_hat = np.mean(samples_theta, axis=0)
    mu_hat = np.mean(samples_mu)
    tau_hat = np.sqrt(np.mean(samples_tau2))
    sigma_hat = np.sqrt(np.mean(samples_sigma2))
    
    print("Posterior mean of mu:", mu_hat)
    print("Posterior mean of tau:", tau_hat)
    print("Posterior mean of sigma:", sigma_hat)
    print("Posterior means of theta_j:", theta_hat)
    
    print("\nTrue mu:", mu_true)
    print("True tau:", tau_true)
    print("True sigma:", sigma_true)
    print("True theta_j:", theta_true)
    
    # ------------------------
    # 4. Diagnostic Plots
    # ------------------------
    # (a) Trace plots for mu, tau^2, sigma^2
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    
    axs[0].plot(samples_mu, alpha=0.5)
    axs[0].set_ylabel("mu")
    
    axs[1].plot(samples_tau2, alpha=0.5)
    axs[1].set_ylabel("tau^2")
    
    axs[2].plot(samples_sigma2, alpha=0.5)
    axs[2].set_ylabel("sigma^2")
    axs[2].set_xlabel("Iteration")
    
    plt.suptitle("Trace Plots for Hyperparameters (Systematic Scan Gibbs)")
    plt.tight_layout()
    plt.show()
    
    # (b) Histograms
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    axs[0].hist(samples_mu, bins=30, density=True, alpha=0.7)
    axs[0].axvline(mu_true, color='r', linestyle='--', label='True mu')
    axs[0].set_title("mu Posterior")
    
    axs[1].hist(np.sqrt(samples_tau2), bins=30, density=True, alpha=0.7)
    axs[1].axvline(tau_true, color='r', linestyle='--', label='True tau')
    axs[1].set_title("tau Posterior")
    
    axs[2].hist(np.sqrt(samples_sigma2), bins=30, density=True, alpha=0.7)
    axs[2].axvline(sigma_true, color='r', linestyle='--', label='True sigma')
    axs[2].set_title("sigma Posterior")
    
    for ax in axs:
        ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
