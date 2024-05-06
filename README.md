# MonteCarlo-OptionPricing
This repository contains Python implementations for simulating and pricing financial derivatives using Monte Carlo methods. The project focuses on two main financial models: a basket call option based on geometric Brownian motions and a stochastic volatility model for stock price simulation. The code compares different methods including plain Monte Carlo and advanced techniques using control variates.

## Project Overview
The project comprises two main parts:

## Basket Call Option Pricing:
Simulation of two underlying stock prices modeled as geometric Brownian motions.
Calculation of the price of a basket call option using plain Monte Carlo and the method of conditioning.

## Stochastic Volatility Model:
Simulation of stock prices under stochastic volatility to estimate the price of a vertical spread option.
Comparison of straightforward Euler scheme and Euler scheme enhanced with control variates.

## Features
Monte Carlo Simulation: Implement plain Monte Carlo and conditioning methods to estimate option prices.
Stochastic Processes: Utilize models involving geometric Brownian motions and stochastic volatility.
Numerical Techniques: Employ numerical methods like the Euler scheme and control variates to improve simulation accuracy and efficiency.
Statistical Analysis: Calculate and report estimates along with their standard errors to evaluate the precision of the simulation results.
