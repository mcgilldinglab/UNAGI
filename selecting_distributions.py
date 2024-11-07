import numpy as np
from scipy.stats import nbinom, norm
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP

# Generate sample data (or replace this with your actual data)
data = np.random.negative_binomial(5, 0.3, 1000)  # Example data with zeros

# Step 1: Fit a zero-inflated log-normal model
def fit_zero_inflated_lognormal(data):
    zero_mask = data == 0
    non_zero_data = data[~zero_mask]

    # Fit the non-zero data to a log-normal distribution
    log_data = np.log(non_zero_data + 1e-6)  # Adding a small constant to avoid log(0)
    mean, std = norm.fit(log_data)

    # Fit a logistic regression to predict zero-inflation
    model = sm.Logit(zero_mask.astype(int), sm.add_constant(data))
    zero_inflated_model = model.fit(disp=False)

    # Calculate AIC for comparison
    ziln_aic = -2 * zero_inflated_model.llf + 2 * len(zero_inflated_model.params)
    
    return ziln_aic

# Step 2: Fit a negative binomial distribution
def fit_negative_binomial(data):
    count_data = np.round(data).astype(int)
    count_data = count_data[count_data >= 0]  # Only keep non-negative values for NB

    mean_count = np.mean(count_data)
    var_count = np.var(count_data)
    p = mean_count / var_count if var_count > mean_count else 0.5
    r = mean_count**2 / (var_count - mean_count) if var_count > mean_count else 1

    # Negative binomial log-likelihood for model comparison
    nb_aic = -2 * nbinom.logpmf(count_data, r, p).sum() + 2 * 2  # Two params (r, p)

    return nb_aic

# Step 3: Fit a zero-inflated negative binomial model
def fit_zero_inflated_negative_binomial(data):
    model = ZeroInflatedNegativeBinomialP(data, sm.add_constant(data), exog_infl=sm.add_constant(data))
    zinb_model = model.fit(disp=False)

    # Calculate AIC for comparison
    zinb_aic = zinb_model.aic
    
    return zinb_aic

# Fit models and compare AIC values
ziln_aic = fit_zero_inflated_lognormal(data)
nb_aic = fit_negative_binomial(data)
zinb_aic = fit_zero_inflated_negative_binomial(data)

# Display results
print("Zero-Inflated Log-Normal AIC:", ziln_aic)
print("Negative Binomial AIC:", nb_aic)
print("Zero-Inflated Negative Binomial AIC:", zinb_aic)

# Determine the best fit based on AIC
if min(ziln_aic, nb_aic, zinb_aic) == ziln_aic:
    print("Data is more similar to a zero-inflated log-normal distribution.")
elif min(ziln_aic, nb_aic, zinb_aic) == nb_aic:
    print("Data is more similar to a negative binomial distribution.")
else:
    print("Data is more similar to a zero-inflated negative binomial distribution.")
