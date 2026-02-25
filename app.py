import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

def generate_synthetic_data(n_rows=10000):
    hours = np.arange(n_rows)
    
    plant_load_pct = np.random.uniform(60, 105, n_rows)
    
    base_ethylene = 180
    ethylene_rate_tph = base_ethylene * (plant_load_pct / 100) + np.random.normal(0, 8, n_rows)
    ethylene_rate_tph = np.clip(ethylene_rate_tph, 50, 350)
    
    avg_cot_c = np.random.normal(850, 40, n_rows)
    avg_cot_c = np.clip(avg_cot_c, 700, 1050)
    
    ambient_temp_c = 15 + 20 * np.sin(2 * np.pi * hours / 8760) + np.random.normal(0, 5, n_rows)
    ambient_temp_c = np.clip(ambient_temp_c, -10, 45)
    
    modes = np.random.choice(['normal', 'ramp_up', 'ramp_down', 'maintenance', 'startup'], n_rows, 
                            p=[0.75, 0.08, 0.08, 0.04, 0.05])
    
    num_active_furnaces = np.random.choice([4, 5, 6, 7, 8], n_rows, p=[0.1, 0.2, 0.4, 0.2, 0.1])
    num_active_furnaces = np.clip(num_active_furnaces + (plant_load_pct > 85).astype(int), 4, 10)
    
    feed_mix_index = np.random.uniform(0.3, 1.0, n_rows)
    
    steam_shp_mlb = num_active_furnaces * 12 + plant_load_pct * 0.15 + np.random.normal(0, 2, n_rows)
    steam_hp_mlb = steam_shp_mlb * 0.7 + np.random.normal(0, 1, n_rows)
    steam_ip_mlb = steam_hp_mlb * 0.5 + np.random.normal(0, 0.8, n_rows)
    steam_lp_mlb = steam_ip_mlb * 0.6 + np.random.normal(0, 0.5, n_rows)
    steam_generation_efficiency = np.random.uniform(0.75, 0.92, n_rows)
    steam_renewables_share = np.random.uniform(0, 0.15, n_rows)
    
    temp_factor = (avg_cot_c - 800) / 100
    base_fuel = num_active_furnaces * 25 + plant_load_pct * 0.4
    
    fuel_h2_mmbtu = base_fuel * (0.15 + temp_factor * 0.05) + np.random.normal(0, 3, n_rows)
    fuel_ch4_mmbtu = base_fuel * (0.35 + temp_factor * 0.05) + np.random.normal(0, 5, n_rows)
    fuel_c2h6_mmbtu = base_fuel * (0.25 + temp_factor * 0.03) + np.random.normal(0, 4, n_rows)
    fuel_gasoil_mmbtu = base_fuel * (0.15 + temp_factor * 0.02) + np.random.normal(0, 3, n_rows)
    fuel_solid_mmbtu = base_fuel * 0.10 + np.random.normal(0, 2, n_rows)
    
    fuel_h2_mmbtu = np.clip(fuel_h2_mmbtu, 10, 80)
    fuel_ch4_mmbtu = np.clip(fuel_ch4_mmbtu, 30, 150)
    fuel_c2h6_mmbtu = np.clip(fuel_c2h6_mmbtu, 20, 100)
    fuel_gasoil_mmbtu = np.clip(fuel_gasoil_mmbtu, 10, 70)
    fuel_solid_mmbtu = np.clip(fuel_solid_mmbtu, 5, 50)
    
    power_purchased_mwh = 80 + plant_load_pct * 0.8 + np.random.normal(0, 10, n_rows)
    power_onsite_mwh = 40 + num_active_furnaces * 5 + np.random.normal(0, 5, n_rows)
    power_onsite_tech = np.random.choice(['gas_turbine', 'steam_turbine', 'combined_cycle', 'none'], n_rows,
                                         p=[0.35, 0.30, 0.25, 0.10])
    heat_rate_btu_per_kwh = np.random.uniform(7000, 11000, n_rows)
    
    df = pd.DataFrame({
        'plant_load_pct': plant_load_pct,
        'ethylene_rate_tph': ethylene_rate_tph,
        'avg_cot_c': avg_cot_c,
        'ambient_temp_c': ambient_temp_c,
        'mode': modes,
        'num_active_furnaces': num_active_furnaces,
        'feed_mix_index': feed_mix_index,
        'steam_shp_mlb': steam_shp_mlb,
        'steam_hp_mlb': steam_hp_mlb,
        'steam_ip_mlb': steam_ip_mlb,
        'steam_lp_mlb': steam_lp_mlb,
        'steam_generation_efficiency': steam_generation_efficiency,
        'steam_renewables_share': steam_renewables_share,
        'fuel_h2_mmbtu': fuel_h2_mmbtu,
        'fuel_ch4_mmbtu': fuel_ch4_mmbtu,
        'fuel_c2h6_mmbtu': fuel_c2h6_mmbtu,
        'fuel_gasoil_mmbtu': fuel_gasoil_mmbtu,
        'fuel_solid_mmbtu': fuel_solid_mmbtu,
        'power_purchased_mwh': power_purchased_mwh,
        'power_onsite_mwh': power_onsite_mwh,
        'power_onsite_tech': power_onsite_tech,
        'heat_rate_btu_per_kwh': heat_rate_btu_per_kwh
    })
    
    df['total_fuel_mmbtu'] = (df['fuel_h2_mmbtu'] + df['fuel_ch4_mmbtu'] + 
                              df['fuel_c2h6_mmbtu'] + df['fuel_gasoil_mmbtu'] + 
                              df['fuel_solid_mmbtu'])
    
    df['total_power_mmbtu'] = ((df['power_purchased_mwh'] + df['power_onsite_mwh']) * 
                                (df['heat_rate_btu_per_kwh'] / 1e6 / 1000))
    
    df['total_energy_mmbtu'] = df['total_fuel_mmbtu'] + df['total_power_mmbtu']
    
    df['sec_mmbtu_per_t_eth'] = np.where(
        df['ethylene_rate_tph'] >= 1,
        df['total_energy_mmbtu'] / df['ethylene_rate_tph'],
        np.nan
    )
    
    df = df[df['ethylene_rate_tph'] >= 1].copy()
    
    return df

def train_model(df):
    df_model = df.drop(columns=['sec_mmbtu_per_t_eth']).copy()
    
    categorical_cols = ['mode', 'power_onsite_tech']
    df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
    
    X = df_encoded
    y = df['sec_mmbtu_per_t_eth'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    return model, X_train, X_test, y_train, y_test, train_r2, test_r2, test_mae, df_encoded

st.set_page_config(page_title="Olefin Plant Energy Benchmarking", layout="wide")
st.title("Olefin Plant Utility Efficiency Optimization")

st.sidebar.header("Data Generation")
if st.sidebar.button("Generate New Data"):
    st.session_state['df'] = generate_synthetic_data(10000)
    st.session_state['model_trained'] = False

if 'df' not in st.session_state:
    st.session_state['df'] = generate_synthetic_data(10000)
    st.session_state['model_trained'] = False

df = st.session_state['df']

tab1, tab2, tab3, tab4 = st.tabs(["Data Table", "EDA", "Model Training & Evaluation", "Prediction"])

with tab1:
    st.subheader("Generated Synthetic Data")
    st.write(f"Total rows: {len(df)}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Plant Load", f"{df['plant_load_pct'].mean():.1f}%")
    col2.metric("Avg Ethylene Rate", f"{df['ethylene_rate_tph'].mean():.1f} tph")
    col3.metric("Avg Total Energy", f"{df['total_energy_mmbtu'].mean():.1f} MMBtu")
    col4.metric("Avg SEC", f"{df['sec_mmbtu_per_t_eth'].mean():.2f} MMBtu/t")
    
    st.dataframe(df.head(100), width='stretch')
    
    st.subheader("Data Statistics")
    st.dataframe(df.describe(), width='stretch')
    
    st.subheader("Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)

with tab2:
    st.subheader("Exploratory Data Analysis")
    
    eda_section = st.selectbox("Select EDA Section", 
        ["Distribution Plots", "Scatter Plots", "Box Plots", "Pairwise Relationships", "Target Analysis"])
    
    if eda_section == "Distribution Plots":
        st.markdown("### Distribution of Key Variables")
        cols_to_plot = st.multiselect("Select columns", 
            ['plant_load_pct', 'ethylene_rate_tph', 'avg_cot_c', 'ambient_temp_c', 
             'total_fuel_mmbtu', 'total_power_mmbtu', 'total_energy_mmbtu', 'sec_mmbtu_per_t_eth'],
            default=['plant_load_pct', 'ethylene_rate_tph', 'sec_mmbtu_per_t_eth'])
        
        for col in cols_to_plot:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(df[col], kde=True, ax=ax, color='steelblue')
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            st.pyplot(fig)
            col_stats = df[col].describe()
            st.write(f"**{col} Stats:** Mean: {col_stats['mean']:.2f}, Std: {col_stats['std']:.2f}, Min: {col_stats['min']:.2f}, Max: {col_stats['max']:.2f}")
    
    elif eda_section == "Scatter Plots":
        st.markdown("### Scatter Plots - Relationships with SEC")
        
        x_var = st.selectbox("Select X variable", 
            ['plant_load_pct', 'ethylene_rate_tph', 'avg_cot_c', 'ambient_temp_c', 
             'num_active_furnaces', 'total_fuel_mmbtu', 'total_power_mmbtu', 'feed_mix_index'])
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df[x_var], df['sec_mmbtu_per_t_eth'], alpha=0.3, s=10, c='steelblue')
        ax.set_xlabel(x_var)
        ax.set_ylabel('SEC (MMBtu/t)')
        ax.set_title(f'{x_var} vs SEC')
        
        z = np.polyfit(df[x_var], df['sec_mmbtu_per_t_eth'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[x_var].min(), df[x_var].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, label=f'Trend line')
        ax.legend()
        st.pyplot(fig)
        
        corr_val = df[x_var].corr(df['sec_mmbtu_per_t_eth'])
        st.write(f"**Correlation with SEC:** {corr_val:.4f}")
    
    elif eda_section == "Box Plots":
        st.markdown("### Box Plots by Categorical Variables")
        
        cat_var = st.selectbox("Select categorical variable", ['mode', 'power_onsite_tech'])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.boxplot(data=df, x=cat_var, y='sec_mmbtu_per_t_eth', ax=axes[0], palette='Set2')
        axes[0].set_title(f'SEC by {cat_var}')
        axes[0].set_ylabel('SEC (MMBtu/t)')
        
        sns.boxplot(data=df, x=cat_var, y='ethylene_rate_tph', ax=axes[1], palette='Set2')
        axes[1].set_title(f'Ethylene Rate by {cat_var}')
        axes[1].set_ylabel('Ethylene Rate (tph)')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write(f"**SEC Stats by {cat_var}:**")
        st.dataframe(df.groupby(cat_var)['sec_mmbtu_per_t_eth'].describe(), width='stretch')
    
    elif eda_section == "Pairwise Relationships":
        st.markdown("### Pairwise Relationships")
        
        key_vars = ['plant_load_pct', 'ethylene_rate_tph', 'avg_cot_c', 'num_active_furnaces', 
                    'total_fuel_mmbtu', 'sec_mmbtu_per_t_eth']
        
        selected_vars = st.multiselect("Select variables (max 5)", key_vars, default=key_vars[:5])
        
        if len(selected_vars) > 1:
            sample_df = df[selected_vars].sample(min(1000, len(df)), random_state=42)
            fig = sns.pairplot(sample_df, diag_kind='kde', plot_kws={'alpha': 0.5, 's': 10})
            st.pyplot(fig)
    
    elif eda_section == "Target Analysis":
        st.markdown("### SEC (Specific Energy Consumption) Analysis")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        sns.histplot(df['sec_mmbtu_per_t_eth'], kde=True, ax=axes[0, 0], color='steelblue')
        axes[0, 0].set_title('SEC Distribution')
        
        sec_by_load = df.groupby(pd.cut(df['plant_load_pct'], bins=5))['sec_mmbtu_per_t_eth'].mean()
        sec_by_load.plot(kind='bar', ax=axes[0, 1], color='steelblue')
        axes[0, 1].set_title('Avg SEC by Plant Load Range')
        axes[0, 1].set_xlabel('Plant Load (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        axes[1, 0].scatter(df['ethylene_rate_tph'], df['sec_mmbtu_per_t_eth'], alpha=0.3, s=10)
        axes[1, 0].set_xlabel('Ethylene Rate (tph)')
        axes[1, 0].set_ylabel('SEC (MMBtu/t)')
        axes[1, 0].set_title('Ethylene Rate vs SEC')
        
        sec_by_mode = df.groupby('mode')['sec_mmbtu_per_t_eth'].mean().sort_values()
        sec_by_mode.plot(kind='barh', ax=axes[1, 1], color='steelblue')
        axes[1, 1].set_title('Avg SEC by Mode')
        axes[1, 1].set_xlabel('SEC (MMBtu/t)')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.write("**SEC Percentiles:**")
        st.dataframe(df['sec_mmbtu_per_t_eth'].describe(), width='stretch')

with tab3:
    st.subheader("Model Training: Gradient Boosting Regressor")
    
    if not st.session_state.get('model_trained', False):
        model, X_train, X_test, y_train, y_test, train_r2, test_r2, test_mae, df_encoded = train_model(df)
        st.session_state['model'] = model
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['train_r2'] = train_r2
        st.session_state['test_r2'] = test_r2
        st.session_state['test_mae'] = test_mae
        st.session_state['df_encoded'] = df_encoded
        st.session_state['model_trained'] = True
    else:
        model = st.session_state['model']
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        y_train = st.session_state['y_train']
        y_test = st.session_state['y_test']
        train_r2 = st.session_state['train_r2']
        test_r2 = st.session_state['test_r2']
        test_mae = st.session_state['test_mae']
        df_encoded = st.session_state['df_encoded']
    
    st.write(f"**Training samples:** {len(X_train)}")
    st.write(f"**Test samples:** {len(X_test)}")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Train R²", f"{train_r2:.4f}")
    col2.metric("Test R²", f"{test_r2:.4f}")
    col3.metric("Test MAE", f"{test_mae:.4f} MMBtu/t")
    
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'feature': df_encoded.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance - Gradient Boosting Regressor')
    st.pyplot(fig)
    
    st.subheader("Actual vs Predicted (Test Set)")
    y_pred_test = model.predict(X_test)
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.scatter(y_test, y_pred_test, alpha=0.5, s=10)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual SEC (MMBtu/t)')
    ax2.set_ylabel('Predicted SEC (MMBtu/t)')
    ax2.set_title('Actual vs Predicted')
    st.pyplot(fig2)

with tab3:
    st.subheader("Enter Features for Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        plant_load_pct = st.number_input("Plant Load (%)", min_value=60.0, max_value=105.0, value=85.0, step=1.0)
        ethylene_rate_tph = st.number_input("Ethylene Rate (tph)", min_value=50.0, max_value=350.0, value=150.0, step=5.0)
        avg_cot_c = st.number_input("Avg COT (°C)", min_value=700.0, max_value=1050.0, value=850.0, step=5.0)
        ambient_temp_c = st.number_input("Ambient Temp (°C)", min_value=-10.0, max_value=45.0, value=20.0, step=1.0)
        mode = st.selectbox("Mode", ['normal', 'ramp_up', 'ramp_down', 'maintenance', 'startup'])
        num_active_furnaces = st.number_input("Num Active Furnaces", min_value=4, max_value=10, value=6, step=1)
        feed_mix_index = st.number_input("Feed Mix Index", min_value=0.3, max_value=1.0, value=0.7, step=0.05)
    
    with col2:
        steam_shp_mlb = st.number_input("Steam SHP (MLB)", value=100.0, step=5.0)
        steam_hp_mlb = st.number_input("Steam HP (MLB)", value=70.0, step=5.0)
        steam_ip_mlb = st.number_input("Steam IP (MLB)", value=35.0, step=5.0)
        steam_lp_mlb = st.number_input("Steam LP (MLB)", value=20.0, step=5.0)
        steam_generation_efficiency = st.number_input("Steam Gen Efficiency", min_value=0.75, max_value=0.92, value=0.85, step=0.01)
        steam_renewables_share = st.number_input("Steam Renewables Share", min_value=0.0, max_value=0.15, value=0.05, step=0.01)
    
    st.markdown("### Fuel Composition (MMBtu)")
    col3, col4, col5 = st.columns(3)
    with col3:
        fuel_h2_mmbtu = st.number_input("Fuel H2 (MMBtu)", value=30.0, step=5.0)
        fuel_ch4_mmbtu = st.number_input("Fuel CH4 (MMBtu)", value=80.0, step=5.0)
    with col4:
        fuel_c2h6_mmbtu = st.number_input("Fuel C2H6 (MMBtu)", value=50.0, step=5.0)
        fuel_gasoil_mmbtu = st.number_input("Fuel Gasoil (MMBtu)", value=30.0, step=5.0)
    with col5:
        fuel_solid_mmbtu = st.number_input("Fuel Solid (MMBtu)", value=20.0, step=5.0)
    
    st.markdown("### Power")
    col6, col7 = st.columns(2)
    with col6:
        power_purchased_mwh = st.number_input("Power Purchased (MWh)", value=150.0, step=10.0)
        power_onsite_mwh = st.number_input("Power Onsite (MWh)", value=70.0, step=10.0)
    with col7:
        power_onsite_tech = st.selectbox("Power Onsite Tech", ['gas_turbine', 'steam_turbine', 'combined_cycle', 'none'])
        heat_rate_btu_per_kwh = st.number_input("Heat Rate (BTU/kWh)", value=8500.0, step=100.0)
    
    if st.button("Predict SEC"):
        input_data = {
            'plant_load_pct': plant_load_pct,
            'ethylene_rate_tph': ethylene_rate_tph,
            'avg_cot_c': avg_cot_c,
            'ambient_temp_c': ambient_temp_c,
            'num_active_furnaces': num_active_furnaces,
            'feed_mix_index': feed_mix_index,
            'steam_shp_mlb': steam_shp_mlb,
            'steam_hp_mlb': steam_hp_mlb,
            'steam_ip_mlb': steam_ip_mlb,
            'steam_lp_mlb': steam_lp_mlb,
            'steam_generation_efficiency': steam_generation_efficiency,
            'steam_renewables_share': steam_renewables_share,
            'fuel_h2_mmbtu': fuel_h2_mmbtu,
            'fuel_ch4_mmbtu': fuel_ch4_mmbtu,
            'fuel_c2h6_mmbtu': fuel_c2h6_mmbtu,
            'fuel_gasoil_mmbtu': fuel_gasoil_mmbtu,
            'fuel_solid_mmbtu': fuel_solid_mmbtu,
            'power_purchased_mwh': power_purchased_mwh,
            'power_onsite_mwh': power_onsite_mwh,
            'heat_rate_btu_per_kwh': heat_rate_btu_per_kwh,
            'mode_ramp_up': 1 if mode == 'ramp_up' else 0,
            'mode_ramp_down': 1 if mode == 'ramp_down' else 0,
            'mode_startup': 1 if mode == 'startup' else 0,
            'power_onsite_tech_steam_turbine': 1 if power_onsite_tech == 'steam_turbine' else 0,
            'power_onsite_tech_combined_cycle': 1 if power_onsite_tech == 'combined_cycle' else 0,
            'power_onsite_tech_none': 1 if power_onsite_tech == 'none' else 0,
        }
        
        input_df = pd.DataFrame([input_data])
        for col in df_encoded.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[df_encoded.columns]
        
        prediction = model.predict(input_df)[0]
        
        st.success(f"**Predicted SEC: {prediction:.4f} MMBtu/ton ethylene**")
        
        total_fuel = fuel_h2_mmbtu + fuel_ch4_mmbtu + fuel_c2h6_mmbtu + fuel_gasoil_mmbtu + fuel_solid_mmbtu
        total_power = ((power_purchased_mwh + power_onsite_mwh) * (heat_rate_btu_per_kwh / 1e6 / 1000))
        total_energy = total_fuel + total_power
        manual_sec = total_energy / ethylene_rate_tph
        
        st.info(f"Manual calculation: {manual_sec:.4f} MMBtu/ton (for reference)")
