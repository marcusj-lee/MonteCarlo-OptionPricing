import numpy as np
from scipy.stats import norm

def straight_euler_vertical_spread(r, a, Theta, beta, S0, theta0, T, K1, K2, rho, m, n):
    dt = T / m
    x = np.zeros(n)

    for k in range(n):
        Yt = np.log(S0)
        theta_t = theta0

        for i in range(m):
            Z = np.random.normal()
            U = np.random.normal()
            R = rho * Z + np.sqrt(1 - rho**2) * U
            
            # Update the time processes
            Yt = Yt + (r - 0.5 * theta_t**2) * dt + theta_t * np.sqrt(dt) * Z
            theta_t = theta_t + a * (Theta - theta_t) * dt + beta * np.sqrt(dt) * R
        
        # Calculate the final stock price
        ST = np.exp(Yt)
        payoff = max(ST - K1, 0) - max(ST - K2, 0)
        x[k] = np.exp(-r * T) * payoff

    price = np.mean(x)
    se = np.sqrt((np.mean(x**2) - price**2) / (n - 1))
    
    return price, se


def straight_euler_control_vertical_spread(r, a, Theta, beta, S0, theta0, T, K1, K2, rho, m, n):
    sigma = Theta  # set sigma = Theta (long run average of theta)
    dt = T / m
    
    x = np.zeros(n)
    q = np.zeros(n)
    
    for k in range(n):
        Y_hat = np.log(S0)
        Y_bar = Y_hat
        theta_hat = theta0
        
        for i in range(m):
            Z = np.random.normal()
            U = np.random.normal()
            R = rho * Z + np.sqrt(1 - rho**2) * U
            
            # Update the time processes and control variable
            Y_hat = Y_hat + (r - 0.5 * theta_hat**2) * dt + theta_hat * np.sqrt(dt) * Z
            Y_bar = Y_bar + (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            theta_hat = theta_hat + a * (Theta - theta_hat) * dt + beta * np.sqrt(dt) * R
        
        ST = np.exp(Y_hat)
        STbar = np.exp(Y_bar)
        
        # Payoffs of actual process and control variable
        payoff_actual = max(ST - K1, 0) - max(ST - K2, 0)
        payoff_control = max(STbar - K1, 0) - max(STbar - K2, 0)
        
        # Find expected value of the vertical spread option using Black-Scholes
        bls_call_1 = black_scholes(S0, K1, r, T, sigma)
        bls_call_2 = black_scholes(S0, K2, r, T, sigma)
        bls_spread = bls_call_1 - bls_call_2
        
        # Store discounted actual payoff and variance of the discounted
        # control variable
        x[k] = np.exp(-r * T) * payoff_actual
        q[k] = np.exp(-r * T) * payoff_control - bls_spread
    
    # Let b = 1 per the given assumption in the problem
    b = 1
    h = x - b * q
    
    price = np.mean(h)
    se = np.sqrt((np.mean(h**2) - price**2) / (n - 1))
    return price, se


def black_scholes(S0, K, r, T, sigma):
    # Black-Scholes formula for call option
    tmp = np.log(K / S0) / (sigma * np.sqrt(T)) + (0.5 * sigma - r / sigma) * np.sqrt(T)
    price = S0 * norm.cdf(sigma * np.sqrt(T) - tmp) - np.exp(-r * T) * K * norm.cdf(-tmp)
    return price


# Simulating prices and SE for both methods

r = 0.1
a = 3
Theta = 0.2
beta = 0.1
S0 = 20
theta0 = 0.25
T = 1
K1 = 20
K2 = 22
rho = 0.5
m = 50
n = 10000

straight_price, straight_SE = straight_euler_vertical_spread(r, a, Theta, beta, S0, theta0, T, K1, K2, rho, m, n)
control_price, control_SE = straight_euler_control_vertical_spread(r, a, Theta, beta, S0, theta0, T, K1, K2, rho, m, n)

print(f"straight_price = {straight_price}, straight_SE = {straight_SE}")
print(f"control_price = {control_price}, control_SE = {control_SE}")