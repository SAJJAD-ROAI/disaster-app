# app.py
# --- Core Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import time
import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple

# --- App Configuration ---
st.set_page_config(
    page_title="System Simulation & Solutions Hub",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Directory Setup ---
if not os.path.exists("scenarios"):
    os.makedirs("scenarios")

# --- Simulation Engine: Base Class ---
class BaseSimulation(ABC):
    """Abstract base class for all simulation models."""
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.results = pd.DataFrame()

    @abstractmethod
    def setup_ui(self) -> Dict[str, Any]:
        """Create Streamlit UI widgets for simulation parameters and return them."""
        pass

    @abstractmethod
    def run_step(self, step: int, params: Dict[str, Any], history: pd.DataFrame) -> Dict[str, Any]:
        """Run a single step of the simulation."""
        pass

    def run_simulation(self, steps: int, params: Dict[str, Any]) -> None:
        """Execute the full simulation for a given number of steps."""
        records = []
        history_df = pd.DataFrame()
        for step in range(steps):
            step_data = self.run_step(step, params, history_df)
            step_data['step'] = step
            records.append(step_data)
            history_df = pd.DataFrame.from_records(records)
        self.results = history_df

# --- Simulation Engine: Concrete Implementations ---

class MechanicalVibration(BaseSimulation):
    """Simulates mechanical vibration on a piece of equipment."""
    def setup_ui(self) -> Dict[str, Any]:
        st.header("Mechanical Vibration Parameters")
        params = {}
        params['base_amplitude'] = st.slider("Base Amplitude (g)", 0.1, 5.0, 1.0, 0.1)
        params['frequency'] = st.slider("Frequency (Hz)", 1, 100, 10)
        params['noise_level'] = st.slider("Noise Level", 0.0, 1.0, 0.2, 0.05)
        params['anomaly_prob'] = st.slider("Anomaly Probability", 0.0, 1.0, 0.05, 0.01)
        params['anomaly_magnitude'] = st.slider("Anomaly Magnitude (g)", 2.0, 10.0, 5.0, 0.5)
        return params

    def run_step(self, step: int, params: Dict[str, Any], history: pd.DataFrame) -> Dict[str, Any]:
        time_t = step / params['frequency']
        base_vibration = params['base_amplitude'] * np.sin(2 * np.pi * params['frequency'] * time_t)
        noise = self.rng.normal(0, params['noise_level'])
        
        vibration = base_vibration + noise
        is_anomaly = False
        if self.rng.random() < params['anomaly_prob']:
            vibration += params['anomaly_magnitude']
            is_anomaly = True

        return {
            "time": time_t,
            "vibration_g": vibration,
            "is_anomaly": is_anomaly
        }

class EnvironmentalChamber(BaseSimulation):
    """Simulates temperature and humidity in a controlled environment."""
    def setup_ui(self) -> Dict[str, Any]:
        st.header("Environmental Chamber Parameters")
        params = {}
        params['target_temp'] = st.slider("Target Temperature (°C)", -20, 100, 25)
        params['temp_drift'] = st.slider("Temperature Drift/hr", 0.1, 5.0, 1.0, 0.1)
        params['humidity_setpoint'] = st.slider("Humidity Setpoint (%)", 0, 100, 50)
        params['hvac_effectiveness'] = st.slider("HVAC Effectiveness", 0.1, 1.0, 0.8, 0.05)
        return params

    def run_step(self, step: int, params: Dict[str, Any], history: pd.DataFrame) -> Dict[str, Any]:
        last_temp = history.iloc[-1]['temperature'] if not history.empty else params['target_temp']
        
        # Simulate drift and HVAC correction
        drift = self.rng.normal(0, params['temp_drift'] / 60) # per-step drift
        correction = (params['target_temp'] - last_temp) * params['hvac_effectiveness'] * 0.1
        current_temp = last_temp + drift + correction

        # Simulate humidity
        humidity_noise = self.rng.normal(0, 2)
        current_humidity = params['humidity_setpoint'] + humidity_noise

        return {
            "temperature": current_temp,
            "humidity": np.clip(current_humidity, 0, 100)
        }

class EpidemiologicalSIR(BaseSimulation):
    """Simulates the spread of a disease using the SIR model."""
    def setup_ui(self) -> Dict[str, Any]:
        st.header("Epidemiological SIR Model Parameters")
        params = {}
        params['population'] = st.number_input("Total Population (N)", 100, 100000, 1000)
        params['initial_infected'] = st.number_input("Initial Infected (I₀)", 1, 1000, 10)
        params['contact_rate'] = st.slider("Contact Rate (β)", 0.0, 1.0, 0.3, 0.01)
        params['recovery_rate'] = st.slider("Recovery Rate (γ)", 0.0, 1.0, 0.1, 0.01)
        return params

    def run_step(self, step: int, params: Dict[str, Any], history: pd.DataFrame) -> Dict[str, Any]:
        if step == 0:
            S = params['population'] - params['initial_infected']
            I = params['initial_infected']
            R = 0
        else:
            S = history.iloc[-1]['Susceptible']
            I = history.iloc[-1]['Infected']
            R = history.iloc[-1]['Recovered']

        N = params['population']
        new_infections = (params['contact_rate'] * S * I) / N
        new_recoveries = params['recovery_rate'] * I

        S_new = S - new_infections
        I_new = I + new_infections - new_recoveries
        R_new = R + new_recoveries

        return {
            "Susceptible": S_new,
            "Infected": I_new,
            "Recovered": R_new
        }

# --- Simulation Registry ---
SIMULATIONS = {
    "Mechanical Vibration Test": MechanicalVibration,
    "Environmental Chamber": EnvironmentalChamber,
    "Epidemiological (SIR) Model": EpidemiologicalSIR,
}

# --- Helper Functions ---
def save_scenario(name: str, params: Dict[str, Any]):
    """Saves simulation parameters to a JSON file."""
    filepath = os.path.join("scenarios", f"{name}.json")
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)
    st.success(f"Scenario '{name}' saved!")

def load_scenario(name: str) -> Dict[str, Any]:
    """Loads simulation parameters from a JSON file."""
    filepath = os.path.join("scenarios", f"{name}.json")
    with open(filepath, 'r') as f:
        return json.load(f)

def get_saved_scenarios() -> List[str]:
    """Gets a list of saved scenario filenames."""
    return [f.replace(".json", "") for f in os.listdir("scenarios") if f.endswith(".json")]

# --- Main Application UI ---
st.title("🔬 System Simulation & Solutions Hub")
st.markdown("A platform to simulate complex systems, manage scenarios, and test solutions.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Controls")

    # Mode Selector
    app_mode = st.radio("Mode", ["Demo", "Real (API Placeholder)"], horizontal=True, help="Demo uses simulated data. Real mode would connect to live APIs.")
    
    # Simulation Selector
    sim_name = st.selectbox("Select Simulation", list(SIMULATIONS.keys()))
    
    # Instantiate the selected simulation class
    sim_class = SIMULATIONS[sim_name]
    if 'sim_seed' not in st.session_state:
        st.session_state.sim_seed = 42
    
    # Test Harness: Seed for reproducibility
    st.session_state.sim_seed = st.number_input("Simulation Seed", value=st.session_state.sim_seed, step=1)
    simulation_instance = sim_class(seed=st.session_state.sim_seed)
    
    # Dynamic UI for simulation parameters
    sim_params = simulation_instance.setup_ui()
    
    # Add general simulation parameters
    st.header("General Settings")
    sim_params['steps'] = st.slider("Simulation Steps", 50, 1000, 200, 10)

    # Combine all parameters
    all_params = {
        "simulation_name": sim_name,
        "seed": st.session_state.sim_seed,
        **sim_params
    }

    # Scenario Manager
    st.header("Scenario Manager")
    col1, col2 = st.columns(2)
    scenario_name = col1.text_input("Scenario Name", "My_Test_Scenario")
    if col2.button("Save", use_container_width=True):
        save_scenario(scenario_name, all_params)

    saved_scenarios = get_saved_scenarios()
    if saved_scenarios:
        selected_scenario = st.selectbox("Load Scenario", [""] + saved_scenarios)
        if selected_scenario:
            loaded_params = load_scenario(selected_scenario)
            # This is complex in Streamlit's loop. A full reload isn't trivial.
            # Best practice is to notify user to set values manually or re-run.
            st.info(f"Loaded '{selected_scenario}'. Set parameters above to match and re-run.")
            st.json(loaded_params)


    # Run Simulation Button
    if st.button("🚀 Run Simulation", type="primary", use_container_width=True):
        st.session_state.simulation_instance = simulation_instance
        st.session_state.simulation_params = sim_params
        st.session_state.simulation_instance.run_simulation(sim_params['steps'], sim_params)
        st.session_state.results = st.session_state.simulation_instance.results
        st.session_state.last_run_time = time.time()


# --- Main Panel Display ---
if 'results' not in st.session_state or st.session_state.results.empty:
    st.info("👋 Welcome! Configure a simulation in the sidebar and click 'Run Simulation'.")
    st.image("https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.svg", width=400)
else:
    results_df = st.session_state.results
    
    # --- Tabs for different views ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "🔔 Alerts & Actions", "🛠️ Health & Testing", "🗺️ Safe-Spot Map", "💾 Data Export"])

    with tab1:
        st.header("Simulation Dashboard")
        
        # Dynamically create charts based on results columns
        numeric_cols = results_df.select_dtypes(include=np.number).columns.tolist()
        if 'step' in numeric_cols: numeric_cols.remove('step')
        if 'time' in numeric_cols: numeric_cols.remove('time')
        
        # Time Series Plot
        st.subheader("Time Series Analysis")
        if not results_df.empty:
            x_axis = 'step' if 'time' not in results_df.columns else 'time'
            fig_ts = px.line(results_df, x=x_axis, y=numeric_cols, title=f"{sim_name} Over Time")
            fig_ts.update_layout(legend_title_text='Metrics')
            st.plotly_chart(fig_ts, use_container_width=True)

        # Other Plots
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribution")
            selected_col_hist = st.selectbox("Select metric for histogram", numeric_cols)
            if selected_col_hist:
                fig_hist = px.histogram(results_df, x=selected_col_hist, nbins=50, title=f"Distribution of {selected_col_hist}")
                st.plotly_chart(fig_hist, use_container_width=True)
        with col2:
            st.subheader("Correlation")
            if len(numeric_cols) > 1:
                col_x = st.selectbox("Select X-axis", numeric_cols, index=0)
                col_y = st.selectbox("Select Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
                fig_scatter = px.scatter(results_df, x=col_x, y=col_y, title=f"{col_x} vs. {col_y}", trendline="ols")
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("Need at least two numeric columns for a scatter plot.")

    with tab2:
        st.header("Alerts & Action Rules")
        
        st.markdown("Define rules to monitor the simulation and trigger actions.")
        
        col1, col2, col3 = st.columns(3)
        alert_metric = col1.selectbox("Metric to Monitor", results_df.columns)
        alert_op = col2.selectbox("Condition", [">", "<", "=="])
        alert_threshold = col3.number_input("Threshold", value=0.0)
        
        # Evaluate alerts
        query = f"`{alert_metric}` {alert_op} {alert_threshold}"
        alert_events = results_df.query(query)
        
        st.subheader(f"🚨 Found {len(alert_events)} Alert Events")
        if not alert_events.empty:
            st.dataframe(alert_events, use_container_width=True)
            
            # Escalation Logic
            st.subheader("Escalation & Actions")
            escalation_steps = [
                "1. **Notify** operator via dashboard alert.",
                "2. **Log** event to system records.",
                "3. If >3 consecutive alerts: **Simulate** automated shutdown (flag system as 'Unsafe').",
                "4. **Escalate** to supervisor after 5 minutes of 'Unsafe' state."
            ]
            
            # Check for consecutive alerts (simple example)
            alert_events['is_consecutive'] = alert_events['step'].diff().fillna(1) == 1
            consecutive_count = alert_events['is_consecutive'].sum()

            st.write(f"Detected **{consecutive_count}** consecutive alert events.")
            if consecutive_count > 3:
                st.error(f"**Action Triggered:** {escalation_steps[2]}")
                st.session_state.system_status = "Unsafe"
            else:
                st.success(f"**Action Triggered:** {escalation_steps[0]}")
                st.session_state.system_status = "Safe"
        else:
            st.success("No events met the alert criteria.")
            st.session_state.system_status = "Safe"

    with tab3:
        st.header("System Health & Testing")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Simulated Device Health")
            # These would be updated by a background task or API in a real app
            battery = st.session_state.get('battery', rng.integers(80, 100))
            gps_status = st.session_state.get('gps', "Locked")
            network = st.session_state.get('network', "5G (Strong)")
            st.metric("Battery Level", f"{battery}%", "OK")
            st.metric("GPS Status", gps_status)
            st.metric("Network", network)
            if st.button("Manual Override: Trigger Low Battery"):
                st.session_state.battery = 15
                st.rerun()

        with col2:
            st.subheader("Automated Scheduler & Test Harness")
            run_interval_minutes = st.number_input("Run test every (minutes)", value=30)
            
            last_run = st.session_state.get('last_run_time', 0)
            time_since_last_run = (time.time() - last_run) / 60
            
            st.write(f"Time since last run: **{time_since_last_run:.1f} minutes**")
            if time_since_last_run > run_interval_minutes:
                st.warning(f"Scheduled test is due! (Would auto-run in a real background service)")
            else:
                st.info("Scheduler is within its time window.")

            if st.button("▶️ Run Full System Test"):
                with st.spinner("Running system diagnostics..."):
                    # 1. Run a short, deterministic simulation
                    test_sim = MechanicalVibration(seed=101)
                    test_params = {'base_amplitude': 4.0, 'frequency': 50, 'noise_level': 0.1, 'anomaly_prob': 1.0, 'anomaly_magnitude': 5.0}
                    test_sim.run_simulation(steps=20, params=test_params)
                    
                    # 2. Check for expected alerts
                    alerts_found = not test_sim.results.query("vibration_g > 8.0").empty
                    
                    # 3. Check health checks
                    health_ok = battery > 20 and gps_status == "Locked"
                    
                    time.sleep(2) # Simulate work

                st.success("System Test Complete!")
                st.markdown(f"- **Simulation Engine:** {'✅ PASSED' if not test_sim.results.empty else '❌ FAILED'}")
                st.markdown(f"- **Alerting Logic:** {'✅ PASSED' if alerts_found else '❌ FAILED'}")
                st.markdown(f"- **Health Monitors:** {'✅ PASSED' if health_ok else '❌ FAILED'}")


    with tab4:
        st.header("🗺️ Safe-Spot Mapping")
        st.markdown("Simulated map data based on analysis. In a real scenario, this could map safe zones away from vibration sources or disease hotspots.")
        
        # Create some random points for demonstration
        map_rng = np.random.default_rng(st.session_state.sim_seed)
        map_data = pd.DataFrame(
            map_rng.normal(loc=[17.4, 78.4], scale=[0.05, 0.05], size=(20, 2)),
            columns=['lat', 'lon']
        )
        
        if sim_name == "Mechanical Vibration Test" and not alert_events.empty:
            st.info("Mapping potential safe locations away from high-vibration zones.")
            # Move points away from center if alerts
            map_data *= 1.02 
        elif sim_name == "Epidemiological (SIR) Model":
            st.info("Mapping locations with lower population density (simulated).")
            # Spread points out more
            map_data = pd.DataFrame(
                map_rng.uniform(low=[17.3, 78.3], high=[17.5, 78.5], size=(20, 2)),
                columns=['lat', 'lon']
            )

        st.map(map_data, zoom=11)

    with tab5:
        st.header("💾 Data Export")
        st.dataframe(results_df)

        # CSV Export
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download data as CSV",
            data=csv,
            file_name=f"{sim_name}_results.csv",
            mime="text/csv",
        )
        
        # Note: PNG and PDF export require more complex setups. 
        # Plotly's built-in save-as-PNG is the easiest for users.
        st.info("To export charts, use the camera icon (📷) in the top-right corner of each plot.")
        st.warning("PDF export is not implemented in this demo due to its complexity.")
