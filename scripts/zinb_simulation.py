#!/usr/bin/env python3
"""
Zero-Inflated Negative Binomial (ZINB) Simulation Script

This script demonstrates ZINB modeling for monthly incident counts with many zeros,
including lag term support and scenario simulation capabilities.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. Please install with: pip install statsmodels>=0.14")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def create_synthetic_dataset(n_months=24, seed=None):
    """
    Create a synthetic dataset with zero inflation for monthly incident counts.
    
    Args:
        n_months (int): Number of months to generate data for
        seed (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Dataset with incident counts and covariates
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create time index
    dates = pd.date_range(start='2022-01-01', periods=n_months, freq='ME')
    
    # Create baseline covariates
    data = pd.DataFrame({
        'date': dates,
        'month': range(1, n_months + 1),
        'intervention': np.where(dates.month >= 7, 1, 0),  # Intervention starts month 7
        'seasonal_factor': np.sin(2 * np.pi * np.arange(n_months) / 12) * 0.5 + 1
    })
    
    # Add lag term (previous month's count, initialized to 0)
    data['lag_count'] = 0
    
    # Generate true counts before zero inflation
    # Base rate affected by intervention and seasonal factors
    true_means = 3.0 * data['seasonal_factor'] * np.exp(-0.5 * data['intervention'])
    
    # Add lag effect - need to generate iteratively since each count depends on previous
    true_counts = np.zeros(n_months)
    alpha = 1.0  # dispersion parameter
    
    for i in range(n_months):
        # Calculate mean for this period including lag effect
        if i == 0:
            current_mean = true_means.iloc[i]
        else:
            current_mean = true_means.iloc[i] + 0.15 * true_counts[i-1]
        
        # Generate count from negative binomial
        p_nb = 1 / (1 + alpha * current_mean)
        true_counts[i] = np.random.negative_binomial(1/alpha, p_nb)
    
    data['true_count'] = true_counts.astype(int)
    
    # Create structural zeros (inflation)
    # Higher probability of zeros in early months and during intervention
    zero_inflation_prob = 0.2 + 0.15 * np.exp(-0.1 * data['month'])
    zero_inflation_prob[data['intervention'] == 1] *= 1.2  # More zeros during intervention
    zero_inflation_prob = np.clip(zero_inflation_prob, 0.05, 0.8)  # Keep reasonable bounds
    
    # Apply zero inflation
    is_structural_zero = np.random.random(n_months) < zero_inflation_prob
    data['incident_count'] = np.where(is_structural_zero, 0, true_counts)
    
    # Update lag counts
    for i in range(1, n_months):
        data.loc[i, 'lag_count'] = data.loc[i-1, 'incident_count']
    
    return data


def fit_zinb_model(data):
    """
    Fit a Zero-Inflated Negative Binomial model to the data.
    
    Args:
        data (pd.DataFrame): Dataset with incident counts and covariates
        
    Returns:
        tuple: (fitted_model, exog, exog_infl) Model and design matrices
    """
    # Prepare covariates for count model
    exog_vars = ['intervention', 'seasonal_factor', 'lag_count']
    exog = sm.add_constant(data[exog_vars])
    
    # Prepare covariates for inflation model (logit)
    exog_infl_vars = ['intervention', 'month']
    exog_infl = sm.add_constant(data[exog_infl_vars])
    
    # Fit ZINB model
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ZeroInflatedNegativeBinomialP(
            data['incident_count'], 
            exog, 
            exog_infl=exog_infl,
            inflation='logit'
        )
        results = model.fit(disp=0)
    
    return results, exog, exog_infl


def simulate_forward(model, exog_base, n_runs=1000, seed=None):
    """
    Perform forward simulation using the fitted ZINB model.
    
    Args:
        model: Fitted ZINB model
        exog_base: Base design matrix for simulation
        n_runs (int): Number of Monte Carlo simulations
        seed (int): Random seed
        
    Returns:
        np.ndarray: Simulated counts (n_runs x n_periods)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_periods = exog_base.shape[0]
    simulations = np.zeros((n_runs, n_periods))
    
    # Extract model parameters
    n_count_params = exog_base.shape[1]
    n_infl_params = len(model.params) - n_count_params - 1  # -1 for alpha
    params_count = model.params[:n_count_params]
    params_infl = model.params[n_count_params:-1]
    alpha = model.params[-1]  # dispersion parameter
    
    # Ensure alpha is positive
    alpha = max(alpha, 0.1)
    
    for run in range(n_runs):
        for t in range(n_periods):
            # Calculate linear predictor for count model
            eta_count = np.dot(exog_base.iloc[t], params_count)
            eta_count = np.clip(eta_count, -10, 10)  # Prevent overflow
            mu_count = np.exp(eta_count)
            
            # Calculate probability of being a structural zero
            # For inflation model: const + intervention + month
            infl_x = np.array([1, exog_base.iloc[t]['intervention'], t + 1])  # t+1 for month
            eta_infl = np.dot(infl_x, params_infl)
            p_infl = 1 / (1 + np.exp(-eta_infl))
            
            # Simulate from ZINB
            if np.random.random() < p_infl:
                # Structural zero
                simulations[run, t] = 0
            else:
                # Negative binomial draw
                n_param = 1/alpha  # number of successes parameter
                p_param = 1 / (1 + alpha * mu_count)  # probability of success
                n_param = max(n_param, 0.1)  # Ensure positive
                p_param = np.clip(p_param, 0.01, 0.99)  # Ensure valid probability
                simulations[run, t] = np.random.negative_binomial(n_param, p_param)
    
    return simulations


def create_scenarios(data, intervention_months=None):
    """
    Create intervention and no-intervention scenarios for comparison.
    
    Args:
        data (pd.DataFrame): Original dataset
        intervention_months (list): Months to apply intervention (default: months 13-24)
        
    Returns:
        tuple: (scenario_data, no_intervention_data)
    """
    if intervention_months is None:
        intervention_months = list(range(13, 25))  # Months 13-24
    
    # Create copy for scenarios
    scenario_data = data.copy()
    no_intervention_data = data.copy()
    
    # Set intervention flags
    scenario_data['intervention'] = scenario_data['month'].isin(intervention_months).astype(int)
    no_intervention_data['intervention'] = 0
    
    # Update lag counts for no-intervention scenario (deterministic)
    no_intervention_data.loc[0, 'lag_count'] = 0
    for i in range(1, len(no_intervention_data)):
        no_intervention_data.loc[i, 'lag_count'] = no_intervention_data.loc[i-1, 'incident_count']
    
    return scenario_data, no_intervention_data


def print_results(model, data, simulations_intervention, simulations_no_intervention):
    """
    Print model results and comparison metrics.
    
    Args:
        model: Fitted ZINB model
        data (pd.DataFrame): Original dataset
        simulations_intervention: Simulations with intervention
        simulations_no_intervention: Simulations without intervention
    """
    print("=" * 60)
    print("ZINB MODEL RESULTS")
    print("=" * 60)
    print("\nModel Summary:")
    print(model.summary())
    
    print("\n" + "=" * 60)
    print("IN-SAMPLE FIT")
    print("=" * 60)
    
    # Calculate in-sample predictions
    exog_vars = ['intervention', 'seasonal_factor', 'lag_count']
    exog = sm.add_constant(data[exog_vars])
    exog_infl_vars = ['intervention', 'month']
    exog_infl = sm.add_constant(data[exog_infl_vars])
    
    pred = model.predict(exog, exog_infl=exog_infl)
    print(f"\nIn-sample expected counts vs actual:")
    for i in range(min(12, len(data))):  # Show first 12 months
        print(f"Month {i+1:2d}: Expected={pred[i]:.2f}, Actual={data['incident_count'].iloc[i]}")
    
    print("\n" + "=" * 60)
    print("SCENARIO COMPARISON (Months 13-24)")
    print("=" * 60)
    
    # Calculate totals for months 13-24
    months_13_24 = data['month'] >= 13
    intervention_total = simulations_intervention[:, months_13_24].sum(axis=1).mean()
    no_intervention_total = simulations_no_intervention[:, months_13_24].sum(axis=1).mean()
    actual_total = data.loc[months_13_24, 'incident_count'].sum()
    
    print(f"\nExpected total incidents (months 13-24):")
    print(f"  With intervention:    {intervention_total:.1f}")
    print(f"  Without intervention: {no_intervention_total:.1f}")
    print(f"  Actual total:          {actual_total:.1f}")
    print(f"  Intervention effect:  {intervention_total - no_intervention_total:+.1f}")
    
    # Calculate percentiles
    intervention_pct = np.percentile(simulations_intervention[:, months_13_24].sum(axis=1), [2.5, 50, 97.5])
    no_intervention_pct = np.percentile(simulations_no_intervention[:, months_13_24].sum(axis=1), [2.5, 50, 97.5])
    
    print(f"\n95% Confidence intervals:")
    print(f"  With intervention:    [{intervention_pct[0]:.1f}, {intervention_pct[2]:.1f}]")
    print(f"  Without intervention: [{no_intervention_pct[0]:.1f}, {no_intervention_pct[2]:.1f}]")


def plot_results(data, simulations_intervention, simulations_no_intervention, output_dir="plots"):
    """
    Create plots if matplotlib is available.
    
    Args:
        data (pd.DataFrame): Original dataset
        simulations_intervention: Simulations with intervention
        simulations_no_intervention: Simulations without intervention
        output_dir (str): Directory to save plots
    """
    if not MATPLOTLIB_AVAILABLE:
        print("\nMatplotlib not available. Skipping plots.")
        return
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Plot 1: Time series with confidence intervals
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    months = data['month']
    
    # Plot actual data
    ax1.plot(months, data['incident_count'], 'ko-', label='Actual', linewidth=2)
    
    # Plot intervention scenario with confidence bands
    intervention_mean = simulations_intervention.mean(axis=0)
    intervention_pct = np.percentile(simulations_intervention, [2.5, 97.5], axis=0)
    ax1.plot(months, intervention_mean, 'b-', label='With intervention', linewidth=2)
    ax1.fill_between(months, intervention_pct[0], intervention_pct[1], alpha=0.3, color='blue')
    
    # Plot no-intervention scenario
    no_intervention_mean = simulations_no_intervention.mean(axis=0)
    no_intervention_pct = np.percentile(simulations_no_intervention, [2.5, 97.5], axis=0)
    ax1.plot(months, no_intervention_mean, 'r--', label='Without intervention', linewidth=2)
    ax1.fill_between(months, no_intervention_pct[0], no_intervention_pct[1], alpha=0.3, color='red')
    
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Incident Count')
    ax1.set_title('ZINB Model: Actual vs Simulated Incident Counts')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative totals for months 13-24
    months_13_24 = months >= 13
    cumulative_intervention = np.cumsum(simulations_intervention[:, months_13_24], axis=1)
    cumulative_no_intervention = np.cumsum(simulations_no_intervention[:, months_13_24], axis=1)
    
    ax2.plot(range(13, 25), cumulative_intervention.mean(axis=0), 'b-', label='With intervention', linewidth=2)
    ax2.plot(range(13, 25), cumulative_no_intervention.mean(axis=0), 'r--', label='Without intervention', linewidth=2)
    
    # Add confidence bands
    intervention_cum_pct = np.percentile(cumulative_intervention, [2.5, 97.5], axis=0)
    no_intervention_cum_pct = np.percentile(cumulative_no_intervention, [2.5, 97.5], axis=0)
    
    ax2.fill_between(range(13, 25), intervention_cum_pct[0], intervention_cum_pct[1], alpha=0.3, color='blue')
    ax2.fill_between(range(13, 25), no_intervention_cum_pct[0], no_intervention_cum_pct[1], alpha=0.3, color='red')
    
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Cumulative Incidents')
    ax2.set_title('Cumulative Incident Counts (Months 13-24)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/zinb_simulation_results.png", dpi=300, bbox_inches='tight')
    print(f"\nPlots saved to {output_dir}/zinb_simulation_results.png")
    plt.close()


def main():
    """Main function to run the ZINB simulation."""
    parser = argparse.ArgumentParser(description='ZINB Simulation for Monthly Incident Counts')
    parser.add_argument('--runs', type=int, default=1000, help='Number of Monte Carlo simulations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--plot', action='store_true', help='Generate plots if matplotlib is available')
    parser.add_argument('--months', type=int, default=24, help='Number of months to simulate')
    
    args = parser.parse_args()
    
    print("ZINB Simulation for Monthly Incident Counts")
    print("=" * 60)
    print(f"Parameters: runs={args.runs}, seed={args.seed}, months={args.months}")
    
    # Create synthetic dataset
    print("\nGenerating synthetic dataset...")
    data = create_synthetic_dataset(n_months=args.months, seed=args.seed)
    print(f"Created dataset with {len(data)} months")
    print(f"Zero inflation: {(data['incident_count'] == 0).mean():.1%} of observations are zeros")
    
    # Fit ZINB model
    print("\nFitting ZINB model...")
    model, exog, exog_infl = fit_zinb_model(data)
    print("Model fitting complete")
    
    # Create scenarios
    print("\nCreating intervention scenarios...")
    scenario_data, no_intervention_data = create_scenarios(data)
    
    # Prepare design matrices for simulation
    exog_vars = ['intervention', 'seasonal_factor', 'lag_count']
    exog_scenario = sm.add_constant(scenario_data[exog_vars])
    exog_no_intervention = sm.add_constant(no_intervention_data[exog_vars])
    
    # Run simulations
    print(f"\nRunning {args.runs} Monte Carlo simulations...")
    simulations_intervention = simulate_forward(
        model, exog_scenario, n_runs=args.runs, seed=args.seed
    )
    simulations_no_intervention = simulate_forward(
        model, exog_no_intervention, n_runs=args.runs, seed=args.seed + 1
    )
    print("Simulations complete")
    
    # Print results
    print_results(model, data, simulations_intervention, simulations_no_intervention)
    
    # Generate plots if requested
    if args.plot:
        plot_results(data, simulations_intervention, simulations_no_intervention)
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
