import numpy as np
from scipy.stats import norm

def basket_call_price_plain(r, sigma1, sigma2, rho, x0, y0, t, k, c1, c2, n):
    x = np.zeros(n)

    for i in range(n):
        z1 = np.random.normal()
        z2 = np.random.normal()

        u1 = z1
        u2 = rho * z1 + np.sqrt(1 - rho**2) * z2

        xt = x0 * np.exp((r - 0.5 * sigma1**2) * t + sigma1 * np.sqrt(t) * u1)
        yt = y0 * np.exp((r - 0.5 * sigma2**2) * t + sigma2 * np.sqrt(t) * u2)

        payoff = max(c1 * xt + c2 * yt - k, 0)
        x[i] = np.exp(-r * t) * payoff

    price = np.mean(x)
    se = np.sqrt((np.mean(x**2) - price**2) / (n - 1))

    return price, se

def basket_call_price_conditioning(r, sigma1, sigma2, rho, x0, y0, t, k, c1, c2, n):
    x = np.zeros(n)

    for i in range(n):
        # Generate Z2 = z so we know YT
        z2 = np.random.normal()
        yt = y0 * np.exp((r - 0.5 * sigma2**2) * t + sigma2 * np.sqrt(t) * z2)

        if c2 * yt - k < 0:
            mu = np.log(c1 * x0) + (r - 0.5 * sigma1**2) * t + sigma1 * np.sqrt(t) * rho * z2
            sigma = np.sqrt(sigma1**2 * t * (1 - rho**2))
            k_bar = (np.log(k - c2 * yt) - mu) / sigma

            payoff = np.exp(mu + 0.5 * sigma**2) * norm.cdf(sigma - k_bar) - (k - c2 * yt) * norm.cdf(-k_bar)
            x[i] = np.exp(-r * t) * payoff
        elif c2 * yt - k >= 0:
            mu = np.log(c1 * x0) + (r - 0.5 * sigma1**2) * t + sigma1 * np.sqrt(t) * rho * z2
            sigma = np.sqrt(sigma1**2 * t * (1 - rho**2))

            payoff = np.exp(mu + 0.5 * sigma**2) + c2 * yt - k
            x[i] = np.exp(-r * t) * payoff

    price = np.mean(x)
    se = np.sqrt((np.mean(x**2) - price**2) / (n - 1))
    
    return price, se


# Simulating prices and SE for both methods

r = 0.1
sigma1 = 0.2
sigma2 = 0.3
rho = 0.7
X0 = 50
Y0 = 50
T = 1
K = 55
c1 = 0.5
c2 = 0.5
n = 10000

plain_price, plain_SE = basket_call_price_plain(r, sigma1, sigma2, rho, X0, Y0, T, K, c1, c2, n)
cond_price, cond_SE = basket_call_price_conditioning(r, sigma1, sigma2, rho, X0, Y0, T, K, c1, c2, n)

print(f"plain_price = {plain_price}, plain_SE = {plain_SE}")
print(f"cond_price = {cond_price}, cond_SE = {cond_SE}")