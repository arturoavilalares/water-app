import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Load the data (you might want to replace this with a file upload feature)
@st.cache_data
def load_data():
    df = pd.read_csv('water_consumption_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

st.title("🚰 Water Rate Optimization Model")

st.write("This app optimizes water rates based on consumption data and environmental factors.")

# Sidebar for user inputs
st.sidebar.header("Model Parameters")
fixed_weight_init = st.sidebar.slider("Initial Fixed Weight", 0.0, 1.0, 0.5)
season_factor_init = st.sidebar.slider("Initial Season Factor", 0.5, 1.5, 1.0)
precip_factor_init = st.sidebar.slider("Initial Precipitation Factor", 0.0, 0.5, 0.1)
temp_factor_init = st.sidebar.slider("Initial Temperature Factor", 0.0, 0.5, 0.1)

# Define base rates
def get_base_rate(consumption):
    if consumption <= 100:
        return 0.5
    elif consumption <= 200:
        return 0.4
    elif consumption <= 300:
        return 0.3
    else:
        return 0.2

# Apply base rates
df['Base_Rate'] = df['Consumption'].apply(get_base_rate)

# Calculate target cost
TARGET_COST = (df['Base_Rate'] * df['Consumption']).sum()

# Define rate calculation function
def calculate_rate(row, fixed_weight, variable_weight, season_factor, precip_factor, temp_factor):
    base_rate = row['Base_Rate']
    season_adj = season_factor if row['Season'] in ['Summer', 'Spring'] else 1/season_factor
    precip_adj = 1 - (row['Precipitation'] / 10) * precip_factor
    temp_adj = 1 + (row['Temperature'] / 30) * temp_factor
    
    fixed_component = base_rate * fixed_weight
    variable_component = base_rate * variable_weight * season_adj * precip_adj * temp_adj
    
    return max(fixed_component + variable_component, 0)

# Objective function
def objective(params):
    fixed_weight, variable_weight, season_factor, precip_factor, temp_factor = params
    
    total_cost = 0
    total_consumption = 0
    negative_rate_penalty = 0
    for _, row in df.iterrows():
        rate = calculate_rate(row, fixed_weight, variable_weight, 
                              season_factor, precip_factor, temp_factor)
        cost = rate * row['Consumption']
        total_cost += cost
        total_consumption += row['Consumption']
        
        raw_rate = (row['Base_Rate'] * fixed_weight + 
                    row['Base_Rate'] * variable_weight * 
                    (season_factor if row['Season'] in ['Summer', 'Spring'] else 1/season_factor) * 
                    (1 - (row['Precipitation'] / 10) * precip_factor) * 
                    (1 + (row['Temperature'] / 30) * temp_factor))
        if raw_rate < 0:
            negative_rate_penalty += abs(raw_rate)
    
    cost_deviation = abs(total_cost - TARGET_COST)
    
    return total_consumption + 1000 * cost_deviation + 10000 * negative_rate_penalty

# Constraint function
def constraint(params):
    return params[0] + params[1] - 1

# Optimization
if st.button("Run Optimization"):
    with st.spinner("Optimizing rates..."):
        x0 = [fixed_weight_init, 1-fixed_weight_init, season_factor_init, precip_factor_init, temp_factor_init]
        bounds = [(0, 1), (0, 1), (0.5, 1.5), (0, 0.5), (0, 0.5)]
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints={'type': 'eq', 'fun': constraint})

    fixed_weight, variable_weight, season_factor, precip_factor, temp_factor = result.x

    st.success("Optimization complete!")
    st.write("Optimized Parameters:")
    st.write(f"Fixed Weight: {fixed_weight:.2f}")
    st.write(f"Variable Weight: {variable_weight:.2f}")
    st.write(f"Season Factor: {season_factor:.2f}")
    st.write(f"Precipitation Factor: {precip_factor:.2f}")
    st.write(f"Temperature Factor: {temp_factor:.2f}")

    # Calculate original and optimized rates and costs
    df['Original_Rate'] = df['Base_Rate']
    df['Original_Cost'] = df['Original_Rate'] * df['Consumption']
    df['Optimized_Rate'] = df.apply(lambda row: calculate_rate(row, fixed_weight, variable_weight, 
                                                               season_factor, precip_factor, temp_factor), axis=1)
    df['Optimized_Cost'] = df['Optimized_Rate'] * df['Consumption']

    # Plot original vs optimized rates
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Date'], df['Original_Rate'], label='Original Rate')
    ax.plot(df['Date'], df['Optimized_Rate'], label='Optimized Rate')
    ax.set_title('Original vs Optimized Rates')
    ax.set_xlabel('Date')
    ax.set_ylabel('Rate')
    ax.legend()
    st.pyplot(fig)

    # Display summary statistics
    st.write(f"Total Original Cost: {df['Original_Cost'].sum():.2f}")
    st.write(f"Total Optimized Cost: {df['Optimized_Cost'].sum():.2f}")
    st.write(f"Cost Difference: {df['Optimized_Cost'].sum() - df['Original_Cost'].sum():.2f}")
    st.write(f"Total Original Consumption: {df['Consumption'].sum():.2f}")
    st.write(f"Estimated Reduced Consumption: {df['Consumption'].sum() * 0.95:.2f}")  # Assuming 5% reduction

    # Display monthly averages
    st.write("Monthly Averages:")
    monthly_avg = df.groupby(df['Date'].dt.to_period('M')).agg({
        'Original_Rate': 'mean',
        'Optimized_Rate': 'mean',
        'Original_Cost': 'sum',
        'Optimized_Cost': 'sum',
        'Consumption': 'sum'
    }).reset_index()
    monthly_avg['Date'] = monthly_avg['Date'].dt.to_timestamp()
    st.dataframe(monthly_avg)

st.write("This model optimizes water rates to encourage conservation while maintaining utility revenue.")
