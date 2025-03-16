import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower

def cohens_d(mean1, mean2, std1, std2):
    """
    Calculate Cohen's d for effect size between two independent samples.
    """
    # Pooled standard deviation
    s = np.sqrt((std1**2 + std2**2) / 2)
    if s == 0:
        return 0
    d = (mean2 - mean1) / s
    return d

def estimate_sample_size(effect_size, alpha=0.05, power=0.80):
    """
    Estimate required sample size per group for two-sample t-test using Cohen's d.
    """
    analysis = TTestIndPower()
    if effect_size == 0:
        return None  # Cannot compute sample size with zero effect size
    sample_size = analysis.solve_power(effect_size=abs(effect_size), alpha=alpha, power=power, alternative='two-sided')
    return int(np.ceil(sample_size))

def calculate_required_data_points(baseline_mean, baseline_std, new_mean, new_std, num_predictors, alpha=0.05, power=0.80):
    """
    Calculate the required number of data points as payment in a federated learning system,
    based on the number of predictors and effect size.
    """
    # Calculate effect size (Cohen's d)
    effect_size = cohens_d(baseline_mean, new_mean, baseline_std, new_std)

    # Adjust alpha for multiple tests (Bonferroni correction)
    alpha_adjusted = alpha / num_predictors

    # Estimate the required sample size
    required_sample_size = estimate_sample_size(effect_size, alpha=alpha_adjusted, power=power)
    return required_sample_size