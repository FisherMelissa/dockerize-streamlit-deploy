#!/usr/bin/env python3
"""
Bayesian Network do-intervention demo for risk/protection factors and school discipline.

This script demonstrates:
- Building a Bayesian Network structure with specified nodes and edges
- Generating synthetic data consistent with the DAG structure
- Simulating do-intervention effects using causal reasoning
- Comparing baseline vs intervention probabilities for SchoolDiscipline levels
- Sampling trajectories for individual students

Network Structure:
- Nodes: ParentingStyle (risk/protect), PeerRisk (0-3), LawEdu (0-2), 
         NightNet (low/med/high), PsychReg (0-2), SchoolDiscipline (low/med/high)
- Edges: Parenting->PsychReg, Parenting->PeerRisk, PeerRisk->SchoolDiscipline,
         NightNet->SchoolDiscipline, LawEdu->SchoolDiscipline, PsychReg->SchoolDiscipline

Usage: python scripts/bn_do_intervention.py
"""

import random
from collections import Counter

def create_bayesian_network_structure():
    """Define the Bayesian Network structure with nodes and edges."""
    nodes = ['ParentingStyle', 'PeerRisk', 'LawEdu', 'NightNet', 'PsychReg', 'SchoolDiscipline']
    
    edges = [
        ('ParentingStyle', 'PsychReg'),
        ('ParentingStyle', 'PeerRisk'),
        ('PeerRisk', 'SchoolDiscipline'),
        ('NightNet', 'SchoolDiscipline'),
        ('LawEdu', 'SchoolDiscipline'),
        ('PsychReg', 'SchoolDiscipline')
    ]
    
    return nodes, edges

def generate_synthetic_data(n_samples=1000, seed=42):
    """Generate synthetic data consistent with the DAG structure."""
    random.seed(seed)
    
    data = []
    
    for i in range(n_samples):
        sample = {}
        
        # ParentingStyle: risk/protect (2 levels)
        # 60% protective parenting, 40% risk parenting
        sample['ParentingStyle'] = 'protect' if random.random() < 0.6 else 'risk'
        
        # LawEdu: 0-2 (3 levels)
        sample['LawEdu'] = random.choices([0, 1, 2], weights=[0.3, 0.4, 0.3])[0]
        
        # NightNet: continuous, then binned (low/med/high)
        night_net_cont = random.gauss(5, 2)
        night_net_cont = max(0, min(10, night_net_cont))
        if night_net_cont < 3.33:
            sample['NightNet'] = 'low'
        elif night_net_cont < 6.67:
            sample['NightNet'] = 'med'
        else:
            sample['NightNet'] = 'high'
        
        # ParentingStyle influences PeerRisk and PsychReg
        if sample['ParentingStyle'] == 'protect':
            sample['PeerRisk'] = random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1])[0]
            sample['PsychReg'] = random.choices([0, 1, 2], weights=[0.5, 0.3, 0.2])[0]
        else:  # risk
            sample['PeerRisk'] = random.choices([0, 1, 2, 3], weights=[0.1, 0.2, 0.3, 0.4])[0]
            sample['PsychReg'] = random.choices([0, 1, 2], weights=[0.2, 0.3, 0.5])[0]
        
        # SchoolDiscipline: low/med/high, influenced by multiple factors
        risk_score = sample['PeerRisk'] * 0.3
        
        if sample['NightNet'] == 'high':
            risk_score += 0.4
        elif sample['NightNet'] == 'med':
            risk_score += 0.2
            
        risk_score += (2 - sample['LawEdu']) * 0.2  # Lower law education = higher risk
        risk_score += sample['PsychReg'] * 0.25
        
        # Convert risk score to discipline level
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
    """Simulate do-intervention by setting a variable to a specific value."""
    # For this demo, we'll create new data with the intervention applied
    # In a real Bayesian Network, this would involve do-calculus
    
    new_data = []
    for sample in data:
        new_sample = sample.copy()
        new_sample[intervention_var] = intervention_value
        
        # Recalculate dependent variables
        if intervention_var == 'ParentingStyle':
            # Recalculate PeerRisk and PsychReg
            if intervention_value == 'protect':
                new_sample['PeerRisk'] = random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1])[0]
                new_sample['PsychReg'] = random.choices([0, 1, 2], weights=[0.5, 0.3, 0.2])[0]
            else:  # risk
                new_sample['PeerRisk'] = random.choices([0, 1, 2, 3], weights=[0.1, 0.2, 0.3, 0.4])[0]
                new_sample['PsychReg'] = random.choices([0, 1, 2], weights=[0.2, 0.3, 0.5])[0]
        
        # Always recalculate SchoolDiscipline since it depends on all parent variables
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

def perform_do_interventions(data):
    """Perform do-intervention queries and compare probabilities."""
    print("\n" + "=" * 60)
    print("DO-INTERVENTION ANALYSIS")
    print("=" * 60)
    
    # Baseline probability of SchoolDiscipline
    baseline_probs = calculate_probabilities(data, 'SchoolDiscipline')
    
    print("\nBASELINE P(SchoolDiscipline):")
    for state in ['low', 'med', 'high']:
        prob = baseline_probs.get(state, 0.0)
        print(f"  P(SchoolDiscipline={state}) = {prob:.3f}")
    
    # Intervention 1: do(ParentingStyle=protect)
    print("\n" + "-" * 40)
    print("INTERVENTION 1: do(ParentingStyle=protect)")
    
    int1_data = simulate_intervention(data, 'ParentingStyle', 'protect')
    int1_probs = calculate_probabilities(int1_data, 'SchoolDiscipline')
    
    print("P(SchoolDiscipline | do(ParentingStyle=protect)):")
    for state in ['low', 'med', 'high']:
        prob = int1_probs.get(state, 0.0)
        baseline = baseline_probs.get(state, 0.0)
        delta = prob - baseline
        print(f"  P(SchoolDiscipline={state}) = {prob:.3f}")
        print(f"    Δ = {delta:+.3f}")
    
    # Intervention 2: do(LawEdu=high)
    print("\n" + "-" * 40)
    print("INTERVENTION 2: do(LawEdu=high)")
    
    int2_data = simulate_intervention(data, 'LawEdu', 2)
    int2_probs = calculate_probabilities(int2_data, 'SchoolDiscipline')
    
    print("P(SchoolDiscipline | do(LawEdu=high)):")
    for state in ['low', 'med', 'high']:
        prob = int2_probs.get(state, 0.0)
        baseline = baseline_probs.get(state, 0.0)
        delta = prob - baseline
        print(f"  P(SchoolDiscipline={state}) = {prob:.3f}")
        print(f"    Δ = {delta:+.3f}")
    
    # Intervention 3: do(NightNet=low)
    print("\n" + "-" * 40)
    print("INTERVENTION 3: do(NightNet=low)")
    
    int3_data = simulate_intervention(data, 'NightNet', 'low')
    int3_probs = calculate_probabilities(int3_data, 'SchoolDiscipline')
    
    print("P(SchoolDiscipline | do(NightNet=low)):")
    for state in ['low', 'med', 'high']:
        prob = int3_probs.get(state, 0.0)
        baseline = baseline_probs.get(state, 0.0)
        delta = prob - baseline
        print(f"  P(SchoolDiscipline={state}) = {prob:.3f}")
        print(f"    Δ = {delta:+.3f}")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("INTERVENTION EFFECTIVENESS SUMMARY")
    print("=" * 60)
    print(f"{'Intervention':<25} {'ΔP(high)':<10} {'ΔP(med)':<10} {'ΔP(low)':<10}")
    print("-" * 60)
    
    interventions = [
        ("do(ParentingStyle=protect)", int1_probs, baseline_probs),
        ("do(LawEdu=high)", int2_probs, baseline_probs),
        ("do(NightNet=low)", int3_probs, baseline_probs)
    ]
    
    for name, int_probs, base_probs in interventions:
        delta_high = int_probs.get('high', 0.0) - base_probs.get('high', 0.0)
        delta_med = int_probs.get('med', 0.0) - base_probs.get('med', 0.0)
        delta_low = int_probs.get('low', 0.0) - base_probs.get('low', 0.0)
        print(f"{name:<25} {delta_high:+10.3f} {delta_med:+10.3f} {delta_low:+10.3f}")

def sample_trajectories(data, n_students=10):
    """Sample trajectories for N students."""
    print(f"\n" + "=" * 60)
    print(f"SAMPLED TRAJECTORIES FOR {n_students} STUDENTS")
    print("=" * 60)
    
    # Randomly sample n_students from the data
    sampled = random.sample(data, min(n_students, len(data)))
    
    print(f"{'Parenting':<10} {'PeerRisk':<9} {'LawEdu':<7} {'NightNet':<9} {'PsychReg':<9} {'Discipline':<11}")
    print("-" * 60)
    
    for sample in sampled:
        print(f"{sample['ParentingStyle']:<10} {sample['PeerRisk']:<9} {sample['LawEdu']:<7} "
              f"{sample['NightNet']:<9} {sample['PsychReg']:<9} {sample['SchoolDiscipline']:<11}")
    
    return sampled

def main():
    """Main function to run the Bayesian Network demo.
    
    This function orchestrates the entire demonstration:
    1. Creates the network structure
    2. Generates synthetic data consistent with causal assumptions
    3. Performs do-intervention analysis
    4. Samples individual student trajectories
    """
    print("Bayesian Network do-intervention Demo")
    print("Risk/Protection Factors -> School Discipline")
    
    # Create the Bayesian Network structure
    print("\n1. Creating Bayesian Network structure...")
    nodes, edges = create_bayesian_network_structure()
    print(f"   Nodes: {nodes}")
    print(f"   Edges: {edges}")
    
    # Generate synthetic data
    print("\n2. Generating synthetic data...")
    data = generate_synthetic_data(n_samples=1000, seed=42)
    print(f"   Generated {len(data)} samples")
    
    print("\n   Variable distributions:")
    for var in nodes:
        probs = calculate_probabilities(data, var)
        print(f"   {var}: {dict(sorted(probs.items()))}")
    
    # Perform do-intervention analysis
    print("\n3. Performing do-intervention analysis...")
    perform_do_interventions(data)
    
    # Sample trajectories
    print("\n4. Sampling student trajectories...")
    sample_trajectories(data, n_students=10)
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 60)

if __name__ == "__main__":
    main()
