#!/usr/bin/env python3
"""
Risk Simulation Dashboard - Streamlit App

Interactive dashboard for Bayesian Network do-intervention analysis
and Zero-Inflated Negative Binomial (ZINB) simulation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import random

# Configure page
st.set_page_config(
    page_title="Risk Simulation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Analysis", ["Bayesian Network", "ZINB Simulation"])

# Import statsmodels for ZINB
if page == "ZINB Simulation":
    try:
        import statsmodels.api as sm
        from statsmodels.discrete.count_model import ZeroInflatedNegativeBinomialP
        import warnings
        STATSMODELS_AVAILABLE = True
    except ImportError:
        STATSMODELS_AVAILABLE = False
        st.error("statsmodels not available. Please install with: pip install statsmodels>=0.14")

# ============================================================================
# BAYESIAN NETWORK FUNCTIONS
# ============================================================================

def generate_bn_data(n_samples=1000, seed=42):
    """Generate synthetic data for Bayesian Network."""
    random.seed(seed)
    data = []
    
    for i in range(n_samples):
        sample = {}
        sample['ParentingStyle'] = 'protect' if random.random() < 0.6 else 'risk'
        sample['LawEdu'] = random.choices([0, 1, 2], weights=[0.3, 0.4, 0.3])[0]
        
        night_net_cont = random.gauss(5, 2)
        night_net_cont = max(0, min(10, night_net_cont))
        if night_net_cont < 3.33:
            sample['NightNet'] = 'low'
        elif night_net_cont < 6.67:
            sample['NightNet'] = 'med'
        else:
            sample['NightNet'] = 'high'
        
        if sample['ParentingStyle'] == 'protect':
            sample['PeerRisk'] = random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1])[0]
            sample['PsychReg'] = random.choices([0, 1, 2], weights=[0.5, 0.3, 0.2])[0]
        else:
            sample['PeerRisk'] = random.choices([0, 1, 2, 3], weights=[0.1, 0.2, 0.3, 0.4])[0]
            sample['PsychReg'] = random.choices([0, 1, 2], weights=[0.2, 0.3, 0.5])[0]
        
        risk_score = sample['PeerRisk'] * 0.3
        if sample['NightNet'] == 'high':
            risk_score += 0.4
        elif sample['NightNet'] == 'med':
            risk_score += 0.2
        risk_score += (2 - sample['LawEdu']) * 0.2
        risk_score += sample['PsychReg'] * 0.25
        
        if risk_score < 1.0:
            sample['SchoolDiscipline'] = 'low'
        elif risk_score < 2.0:
            sample['SchoolDiscipline'] = 'med'
        else:
            sample['SchoolDiscipline'] = 'high'
        
        data.append(sample)
    
    return data

def calculate_probabilities(data, target_var):
    """Calculate probability distribution for a target variable."""
    counts = Counter(sample[target_var] for sample in data)
    total = len(data)
    return {value: count/total for value, count in counts.items()}

def simulate_intervention(data, intervention_var, intervention_value):
    """Simulate do-intervention."""
    new_data = []
    for sample in data:
        new_sample = sample.copy()
        new_sample[intervention_var] = intervention_value
        
        if intervention_var == 'ParentingStyle':
            if intervention_value == 'protect':
                new_sample['PeerRisk'] = random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1])[0]
                new_sample['PsychReg'] = random.choices([0, 1, 2], weights=[0.5, 0.3, 0.2])[0]
            else:
                new_sample['PeerRisk'] = random.choices([0, 1, 2, 3], weights=[0.1, 0.2, 0.3, 0.4])[0]
                new_sample['PsychReg'] = random.choices([0, 1, 2], weights=[0.2, 0.3, 0.5])[0]
        
        risk_score = new_sample['PeerRisk'] * 0.3
        if new_sample['NightNet'] == 'high':
            risk_score += 0.4
        elif new_sample['NightNet'] == 'med':
            risk_score += 0.2
        risk_score += (2 - new_sample['LawEdu']) * 0.2
        risk_score += new_sample['PsychReg'] * 0.25
        
        if risk_score < 1.0:
            new_sample['SchoolDiscipline'] = 'low'
        elif risk_score < 2.0:
            new_sample['SchoolDiscipline'] = 'med'
        else:
            new_sample['SchoolDiscipline'] = 'high'
        
        new_data.append(new_sample)
    
    return new_data

# ============================================================================
# ZINB FUNCTIONS
# ============================================================================

def create_zinb_dataset(n_months=24, seed=42):
    """Create synthetic ZINB dataset."""
    np.random.seed(seed)
    
    dates = pd.date_range(start='2022-01-01', periods=n_months, freq='ME')
    data = pd.DataFrame({
        'date': dates,
        'month': range(1, n_months + 1),
        'intervention': np.where(dates.month >= 7, 1, 0),
        'seasonal_factor': np.sin(2 * np.pi * np.arange(n_months) / 12) * 0.5 + 1
    })
    
    data['lag_count'] = 0
    true_means = 3.0 * data['seasonal_factor'] * np.exp(-0.5 * data['intervention'])
    true_counts = np.zeros(n_months)
    alpha = 1.0
    
    for i in range(n_months):
        if i == 0:
            current_mean = true_means.iloc[i]
        else:
            current_mean = true_means.iloc[i] + 0.15 * true_counts[i-1]
        
        p_nb = 1 / (1 + alpha * current_mean)
        true_counts[i] = np.random.negative_binomial(1/alpha, p_nb)
    
    data['true_count'] = true_counts.astype(int)
    
    zero_inflation_prob = 0.2 + 0.15 * np.exp(-0.1 * data['month'])
    zero_inflation_prob[data['intervention'] == 1] *= 1.2
    zero_inflation_prob = np.clip(zero_inflation_prob, 0.05, 0.8)
    
    is_structural_zero = np.random.random(n_months) < zero_inflation_prob
    data['incident_count'] = np.where(is_structural_zero, 0, true_counts)
    
    for i in range(1, n_months):
        data.loc[i, 'lag_count'] = data.loc[i-1, 'incident_count']
    
    return data

def fit_zinb_model(data):
    """Fit ZINB model."""
    exog_vars = ['intervention', 'seasonal_factor', 'lag_count']
    exog = sm.add_constant(data[exog_vars])
    exog_infl_vars = ['intervention', 'month']
    exog_infl = sm.add_constant(data[exog_infl_vars])
    
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

def simulate_zinb_forward(model, exog_base, n_runs=1000, seed=42):
    """Simulate forward using ZINB model."""
    np.random.seed(seed)
    n_periods = exog_base.shape[0]
    simulations = np.zeros((n_runs, n_periods))
    
    n_count_params = exog_base.shape[1]
    params_count = model.params[:n_count_params]
    params_infl = model.params[n_count_params:-1]
    alpha = max(model.params[-1], 0.1)
    
    for run in range(n_runs):
        for t in range(n_periods):
            eta_count = np.dot(exog_base.iloc[t], params_count)
            eta_count = np.clip(eta_count, -10, 10)
            mu_count = np.exp(eta_count)
            
            infl_x = np.array([1, exog_base.iloc[t]['intervention'], t + 1])
            eta_infl = np.dot(infl_x, params_infl)
            p_infl = 1 / (1 + np.exp(-eta_infl))
            
            if np.random.random() < p_infl:
                simulations[run, t] = 0
            else:
                n_param = 1/alpha
                p_param = 1 / (1 + alpha * mu_count)
                n_param = max(n_param, 0.1)
                p_param = np.clip(p_param, 0.01, 0.99)
                simulations[run, t] = np.random.negative_binomial(n_param, p_param)
    
    return simulations

# ============================================================================
# BAYESIAN NETWORK PAGE
# ============================================================================

if page == "Bayesian Network":
    st.title("📊 Bayesian Network Do-Intervention Analysis")
    st.markdown("Analyze the causal effects of interventions on school discipline outcomes")
    
    # Sidebar parameters
    st.sidebar.subheader("Simulation Parameters")
    n_samples = st.sidebar.slider("Number of samples", 100, 5000, 1000, 100)
    seed = st.sidebar.number_input("Random seed", 0, 1000, 42)
    
    # Generate data
    with st.spinner("Generating data..."):
        data = generate_bn_data(n_samples=n_samples, seed=seed)
    
    # Display network structure
    st.subheader("Network Structure")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Nodes:**
        - ParentingStyle (risk/protect)
        - PeerRisk (0-3)
        - LawEdu (0-2)
        - NightNet (low/med/high)
        - PsychReg (0-2)
        - SchoolDiscipline (low/med/high)
        """)
    
    with col2:
        st.markdown("""
        **Edges:**
        - ParentingStyle → PsychReg
        - ParentingStyle → PeerRisk
        - PeerRisk → SchoolDiscipline
        - NightNet → SchoolDiscipline
        - LawEdu → SchoolDiscipline
        - PsychReg → SchoolDiscipline
        """)
    
    # Baseline probabilities
    st.subheader("Baseline Probabilities")
    baseline_probs = calculate_probabilities(data, 'SchoolDiscipline')
    
    fig_baseline = go.Figure(data=[
        go.Bar(x=['Low', 'Medium', 'High'], 
               y=[baseline_probs.get('low', 0), baseline_probs.get('med', 0), baseline_probs.get('high', 0)],
               marker_color=['green', 'orange', 'red'])
    ])
    fig_baseline.update_layout(title="P(SchoolDiscipline) - Baseline", 
                               yaxis_title="Probability", xaxis_title="Discipline Level")
    st.plotly_chart(fig_baseline, use_container_width=True)
    
    # Interventions
    st.subheader("Do-Intervention Analysis")
    
    interventions = [
        ("do(ParentingStyle=protect)", 'ParentingStyle', 'protect'),
        ("do(LawEdu=high)", 'LawEdu', 2),
        ("do(NightNet=low)", 'NightNet', 'low')
    ]
    
    results_data = []
    
    for name, var, value in interventions:
        int_data = simulate_intervention(data, var, value)
        int_probs = calculate_probabilities(int_data, 'SchoolDiscipline')
        
        for state in ['low', 'med', 'high']:
            baseline = baseline_probs.get(state, 0.0)
            intervention = int_probs.get(state, 0.0)
            delta = intervention - baseline
            
            results_data.append({
                'Intervention': name,
                'Discipline Level': state.capitalize(),
                'Baseline': baseline,
                'After Intervention': intervention,
                'Change': delta
            })
    
    results_df = pd.DataFrame(results_data)
    
    # Show comparison chart
    tab1, tab2 = st.tabs(["Comparison Chart", "Data Table"])
    
    with tab1:
        for intervention_name in results_df['Intervention'].unique():
            subset = results_df[results_df['Intervention'] == intervention_name]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Baseline', x=subset['Discipline Level'], y=subset['Baseline'],
                                marker_color='lightblue'))
            fig.add_trace(go.Bar(name='After Intervention', x=subset['Discipline Level'], 
                                y=subset['After Intervention'], marker_color='darkblue'))
            
            fig.update_layout(title=intervention_name, barmode='group',
                            yaxis_title="Probability", xaxis_title="Discipline Level")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.dataframe(results_df.style.format({
            'Baseline': '{:.3f}',
            'After Intervention': '{:.3f}',
            'Change': '{:+.3f}'
        }))
    
    # Sample trajectories
    st.subheader("Sample Student Trajectories")
    n_trajectories = st.slider("Number of trajectories to display", 5, 50, 10)
    sampled = random.sample(data, min(n_trajectories, len(data)))
    
    df_trajectories = pd.DataFrame(sampled)
    st.dataframe(df_trajectories)

# ============================================================================
# ZINB SIMULATION PAGE
# ============================================================================

elif page == "ZINB Simulation":
    st.title("📈 Zero-Inflated Negative Binomial Simulation")
    st.markdown("Model monthly incident counts with zero inflation and intervention effects")
    
    if not STATSMODELS_AVAILABLE:
        st.stop()
    
    # Sidebar parameters
    st.sidebar.subheader("Simulation Parameters")
    n_months = st.sidebar.slider("Number of months", 12, 48, 24)
    n_runs = st.sidebar.slider("Monte Carlo runs", 100, 5000, 1000, 100)
    seed = st.sidebar.number_input("Random seed", 0, 1000, 42)
    
    # Generate data
    with st.spinner("Generating synthetic data..."):
        data = create_zinb_dataset(n_months=n_months, seed=seed)
    
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total months", len(data))
    col2.metric("Zero inflation", f"{(data['incident_count'] == 0).mean():.1%}")
    col3.metric("Mean incidents", f"{data['incident_count'].mean():.2f}")
    
    # Plot time series
    st.subheader("Observed Incident Counts")
    fig_ts = go.Figure()
    fig_ts.add_trace(go.Scatter(x=data['month'], y=data['incident_count'],
                                mode='lines+markers', name='Incident Count'))
    fig_ts.update_layout(xaxis_title="Month", yaxis_title="Incident Count",
                        title="Monthly Incident Counts")
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Fit model
    with st.spinner("Fitting ZINB model..."):
        model, exog, exog_infl = fit_zinb_model(data)
    
    st.subheader("Model Coefficients")
    
    # Extract and display coefficients
    params_df = pd.DataFrame({
        'Parameter': model.params.index,
        'Coefficient': model.params.values,
        'Std Error': model.bse.values,
        'P-value': model.pvalues.values
    })
    st.dataframe(params_df.style.format({
        'Coefficient': '{:.4f}',
        'Std Error': '{:.4f}',
        'P-value': '{:.4f}'
    }))
    
    # Run simulations
    with st.spinner(f"Running {n_runs} Monte Carlo simulations..."):
        # Create scenarios
        scenario_data = data.copy()
        no_intervention_data = data.copy()
        no_intervention_data['intervention'] = 0
        
        exog_vars = ['intervention', 'seasonal_factor', 'lag_count']
        exog_scenario = sm.add_constant(scenario_data[exog_vars])
        exog_no_intervention = sm.add_constant(no_intervention_data[exog_vars])
        
        simulations_intervention = simulate_zinb_forward(model, exog_scenario, n_runs=n_runs, seed=seed)
        simulations_no_intervention = simulate_zinb_forward(model, exog_no_intervention, n_runs=n_runs, seed=seed+1)
    
    # Plot results
    st.subheader("Simulation Results")
    
    intervention_mean = simulations_intervention.mean(axis=0)
    no_intervention_mean = simulations_no_intervention.mean(axis=0)
    intervention_pct = np.percentile(simulations_intervention, [2.5, 97.5], axis=0)
    no_intervention_pct = np.percentile(simulations_no_intervention, [2.5, 97.5], axis=0)
    
    fig_sim = go.Figure()
    
    # Actual data
    fig_sim.add_trace(go.Scatter(x=data['month'], y=data['incident_count'],
                                 mode='lines+markers', name='Actual', line=dict(color='black', width=2)))
    
    # With intervention
    fig_sim.add_trace(go.Scatter(x=data['month'], y=intervention_mean,
                                 mode='lines', name='With intervention', line=dict(color='blue', width=2)))
    fig_sim.add_trace(go.Scatter(x=data['month'], y=intervention_pct[1],
                                 mode='lines', name='95% CI (upper)', line=dict(color='blue', width=0),
                                 showlegend=False))
    fig_sim.add_trace(go.Scatter(x=data['month'], y=intervention_pct[0],
                                 mode='lines', name='95% CI (lower)', line=dict(color='blue', width=0),
                                 fill='tonexty', fillcolor='rgba(0,0,255,0.2)', showlegend=False))
    
    # Without intervention
    fig_sim.add_trace(go.Scatter(x=data['month'], y=no_intervention_mean,
                                 mode='lines', name='Without intervention', line=dict(color='red', width=2, dash='dash')))
    fig_sim.add_trace(go.Scatter(x=data['month'], y=no_intervention_pct[1],
                                 mode='lines', name='95% CI (upper)', line=dict(color='red', width=0),
                                 showlegend=False))
    fig_sim.add_trace(go.Scatter(x=data['month'], y=no_intervention_pct[0],
                                 mode='lines', name='95% CI (lower)', line=dict(color='red', width=0),
                                 fill='tonexty', fillcolor='rgba(255,0,0,0.2)', showlegend=False))
    
    fig_sim.update_layout(xaxis_title="Month", yaxis_title="Incident Count",
                         title="ZINB Model: Actual vs Simulated Incident Counts")
    st.plotly_chart(fig_sim, use_container_width=True)
    
    # Summary statistics
    st.subheader("Intervention Impact Summary")
    
    months_13_24 = data['month'] >= 13
    intervention_total = simulations_intervention[:, months_13_24].sum(axis=1).mean()
    no_intervention_total = simulations_no_intervention[:, months_13_24].sum(axis=1).mean()
    actual_total = data.loc[months_13_24, 'incident_count'].sum()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("With Intervention (Months 13-24)", f"{intervention_total:.1f}")
    col2.metric("Without Intervention (Months 13-24)", f"{no_intervention_total:.1f}")
    col3.metric("Intervention Effect", f"{intervention_total - no_intervention_total:+.1f}")
    
    # Confidence intervals
    intervention_pct_total = np.percentile(simulations_intervention[:, months_13_24].sum(axis=1), [2.5, 50, 97.5])
    no_intervention_pct_total = np.percentile(simulations_no_intervention[:, months_13_24].sum(axis=1), [2.5, 50, 97.5])
    
    st.markdown("**95% Confidence Intervals:**")
    st.markdown(f"- With intervention: [{intervention_pct_total[0]:.1f}, {intervention_pct_total[2]:.1f}]")
    st.markdown(f"- Without intervention: [{no_intervention_pct_total[0]:.1f}, {no_intervention_pct_total[2]:.1f}]")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Risk Simulation Dashboard v1.0")
