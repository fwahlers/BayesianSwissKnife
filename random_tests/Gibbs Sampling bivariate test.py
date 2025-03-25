import numpy as np
import matplotlib.pyplot as plt

def systematic_scan_gibbs(
    mu1, mu2, sigma1, sigma2, rho,
    n_samples=10_000, burn_in=1_000
):
    """
    Perform systematic scan Gibbs sampling for a bivariate normal distribution.
    
    Parameters:
    -----------
    mu1, mu2 : floats
        Means of X1 and X2 respectively.
    sigma1, sigma2 : floats
        Standard deviations of X1 and X2 respectively.
    rho : float
        Correlation coefficient between X1 and X2 (must be in (-1,1)).
    n_samples : int
        Number of samples to collect (including burn-in).
    burn_in : int
        Number of initial samples to discard.
    
    Returns:
    --------
    chain : np.ndarray of shape (n_samples - burn_in, 2)
        The generated samples after burn-in.
    first_5 : np.ndarray of shape (5, 2)
        The first 5 recorded samples for visualization.
    """
    # Precompute conditional variances
    var1_cond = sigma1**2 * (1 - rho**2)
    var2_cond = sigma2**2 * (1 - rho**2)
    
    # Initialize the chain
    x1_current, x2_current = 0.0, 0.0
    samples = np.zeros((n_samples, 2))
    
    for t in range(n_samples):
        # Update X1 keeping X2 fixed
        mean1_cond = mu1 + rho * (sigma1 / sigma2) * (x2_current - mu2)
        x1_current = np.random.normal(mean1_cond, np.sqrt(var1_cond))
        # Update X2 conditioning on new X1
        mean2_cond = mu2 + rho * (sigma2 / sigma1) * (x1_current - mu1)
        x2_current = np.random.normal(mean2_cond, np.sqrt(var2_cond))
        samples[t, :] = (x1_current, x2_current)
    
    return samples[burn_in:]

def get_first_5_points(mu1, mu2, sigma1, sigma2, rho):
    """
    Simulate 4 complete Gibbs cycles starting at (0,0) so that 5 states are obtained.
    Each cycle: update X1 (horizontal move, with Y fixed) 
                then update X2 (vertical move, with X fixed).
    Returns:
      points: np.ndarray with shape (5, 2) representing successive states.
    """
    var1_cond = sigma1**2 * (1 - rho**2)
    var2_cond = sigma2**2 * (1 - rho**2)
    points = []
    current = (0.0, 0.0)
    points.append(current)
    for i in range(4):
        # Horizontal update: X1 changes, Y remains
        mean1 = mu1 + rho * (sigma1/sigma2) * (current[1] - mu2)
        new_x = np.random.normal(mean1, np.sqrt(var1_cond))
        # Intermediate point (for horizontal move)
        mid = (new_x, current[1])
        # Vertical update: Y changes, X fixed to new_x
        mean2 = mu2 + rho * (sigma2/sigma1) * (new_x - mu1)
        new_y = np.random.normal(mean2, np.sqrt(var2_cond))
        current = (new_x, new_y)
        points.append(current)
    return np.array(points)

def plot_gibbs_sampling(chain, points, rho):
    x1_samples = chain[:, 0]
    x2_samples = chain[:, 1]
    
    plt.figure(figsize=(6, 6))
    # Plot full chain samples
    plt.scatter(x1_samples, x2_samples, alpha=0.2, s=10, color='lightblue', label='Gibbs Samples')
    
    # For each transition, draw horizontal then vertical segments and annotate the starting point.
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i+1]
        # Horizontal segment: from start to (end.x, start.y)
        plt.plot([start[0], end[0]], [start[1], start[1]], color='blue', linewidth=1.5)
        # Vertical segment: from (end.x, start.y) to end
        plt.plot([end[0], end[0]], [start[1], end[1]], color='blue', linewidth=1.5)
        # Mark and annotate the start point
        plt.scatter(start[0], start[1], color='blue', s=30, zorder=3)
        plt.text(start[0], start[1], f'{i+1}', fontsize=12, color='red', zorder=4)
    
    # Finally, plot and annotate the last point.
    last = points[-1]
    plt.scatter(last[0], last[1], color='blue', s=30, zorder=3)
    plt.text(last[0], last[1], f'{len(points)}', fontsize=12, color='red', zorder=4)
    
    plt.title(rf"Gibbs Sampling ($\rho = {rho}$)")
    plt.xlabel(r"$X_1$")
    plt.ylabel(r"$X_2$")
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # True parameters
    mu1_true, mu2_true = 5, 5
    sigma1_true, sigma2_true = 1.0, 1
    rho_true = 0.7  # Matching reference image
    
    n_samples = 20_000
    burn_in = 2_000
    chain = systematic_scan_gibbs(
        mu1_true, mu2_true, sigma1_true, sigma2_true, rho_true,
        n_samples=n_samples, burn_in=burn_in
    )
    
    # Get 5 points (4 full cycles)
    first_5_points = get_first_5_points(mu1_true, mu2_true, sigma1_true, sigma2_true, rho_true)
    plot_gibbs_sampling(chain, first_5_points, rho_true)
    
    # Compute sample statistics
    sample_mean = np.mean(chain, axis=0)
    sample_cov = np.cov(chain, rowvar=False)
    
    # True statistics
    true_mean = (mu1_true, mu2_true)
    true_cov = [[sigma1_true**2, rho_true*sigma1_true*sigma2_true],
                [rho_true*sigma1_true*sigma2_true, sigma2_true**2]]
    
    # Create LaTeX snippet comparing true and sample statistics (no document preamble)
    latex_text = rf"""
\section*{{Gibbs Sampling Results}}

\textbf{{True Mean:}} $\mu = ({true_mean[0]:.4f}, {true_mean[1]:.4f})$ \\[5pt]

\textbf{{True Covariance Matrix:}}
\[
\Sigma =
\begin{{pmatrix}}
{true_cov[0][0]:.4f} & {true_cov[0][1]:.4f} \\
{true_cov[1][0]:.4f} & {true_cov[1][1]:.4f}
\end{{pmatrix}}
\] \\[10pt]

\textbf{{Sample Mean:}} $\hat{{\mu}} = ({sample_mean[0]:.4f}, {sample_mean[1]:.4f})$ \\[5pt]

\textbf{{Sample Covariance Matrix:}} $\hat{{\Sigma}}$ =
\begin{{pmatrix}}
{sample_cov[0,0]:.4f} & {sample_cov[0,1]:.4f} \\
{sample_cov[1,0]:.4f} & {sample_cov[1,1]:.4f}
\end{{pmatrix}}
\]
"""
    
    # Write the LaTeX snippet to a file so you can paste it into your document.
    results_file = r"c:/Users/frede/Desktop/Bachelor/Gibbs_Sampling_Results.tex"
    with open(results_file, "w") as f:
        f.write(latex_text)

if __name__ == "__main__":
    main()
