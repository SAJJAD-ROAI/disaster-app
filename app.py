import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import time

# Page configuration
st.set_page_config(
    page_title="DisasterSim Pro",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        border-bottom: 2px solid #64B5F6;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .metric-card {
        background-color: #E3F2FD;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<h1 class="main-header">🌍 DisasterSim Pro</h1>', unsafe_allow_html=True)
st.markdown("### *Advanced Disaster Modeling & Emergency Response Planning*")

# Sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/disaster.png", width=80)
    st.title("Navigation")
    app_section = st.radio("Choose Module", 
                         ["Dashboard", 
                          "Earthquake Simulator", 
                          "Flood Modeling", 
                          "Wildfire Analysis",
                          "Response Planner",
                          "Resource Optimizer"])
    
    st.title("Settings")
    sim_speed = st.slider("Simulation Speed", 1, 10, 5)
    detail_level = st.selectbox("Detail Level", ["Basic", "Advanced", "Professional"])

# Dashboard Section
if app_section == "Dashboard":
    st.markdown('<h2 class="sub-header">Real-time Disaster Monitoring</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Interactive map
        map_data = pd.DataFrame({
            'lat': [37.77, 34.05, 40.71, 41.88, 39.74, 32.71],
            'lon': [-122.41, -118.24, -74.01, -87.63, -104.99, -117.16],
            'magnitude': [5.2, 3.7, 4.5, 2.8, 6.1, 4.8],
            'type': ['Earthquake', 'Earthquake', 'Flood', 'Wildfire', 'Earthquake', 'Wildfire'],
            'size': [30, 15, 25, 20, 40, 25],
            'name': ['CA Quake', 'LA Tremor', 'NY Flood', 'Chicago Fire', 'CO Quake', 'SD Fire']
        })
        
        fig = px.scatter_mapbox(map_data, lat="lat", lon="lon", 
                               color="type", size="size", hover_name="name",
                               color_discrete_map={'Earthquake': 'red', 'Flood': 'blue', 'Wildfire': 'orange'},
                               zoom=3, height=400)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<h2 class="sub-header">Active Events</h2>', unsafe_allow_html=True)
        
        events = [
            {"name": "California Earthquake", "type": "🌋 Earthquake", "severity": "High", "time": "2 hours ago"},
            {"name": "Colorado Wildfire", "type": "🔥 Wildfire", "severity": "Medium", "time": "5 hours ago"},
            {"name": "New York Flooding", "type": "🌊 Flood", "severity": "Critical", "time": "12 hours ago"}
        ]
        
        for event in events:
            with st.container():
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{event['name']}</h3>
                    <p><strong>{event['type']}</strong> | Severity: {event['severity']}</p>
                    <small>Detected: {event['time']}</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Metrics
    st.markdown('<h2 class="sub-header">System Metrics</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Active Events", "12", "+2 today")
    col2.metric("People Affected", "42K", "+8K today")
    col3.metric("Response Teams", "18", "4 deployed")
    col4.metric("Resource Usage", "94%", "-2% today")

# Earthquake Simulator Section
elif app_section == "Earthquake Simulator":
    st.markdown('<h2 class="sub-header">Advanced Earthquake Simulation</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Simulation Parameters")
        eq_magnitude = st.slider("Magnitude (Richter)", 3.0, 9.0, 6.5, 0.1)
        eq_depth = st.slider("Depth (km)", 0, 100, 10)
        eq_duration = st.slider("Duration (seconds)", 10, 120, 30)
        population_density = st.slider("Population Density", 100, 20000, 5000, 100)
        
        if st.button("Run Simulation", key="run_earthquake"):
            with st.spinner("Running advanced seismic analysis..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                st.success("Simulation completed successfully!")
    
    with col2:
        st.subheader("Impact Visualization")
        
        # Seismic wave simulation
        x = np.linspace(0, 10, 1000)
        y = np.sin(x * eq_magnitude) * np.exp(-x/5)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, y, color='red', linewidth=1.5)
        ax.fill_between(x, y, alpha=0.3, color='red')
        ax.set_title("Seismic Wave Propagation Pattern")
        ax.set_xlabel("Distance from Epicenter (km)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Impact assessment
    st.markdown('<h2 class="sub-header">Impact Assessment</h2>', unsafe_allow_html=True)
    
    estimated_radius = eq_magnitude * 20
    estimated_area = 3.14 * estimated_radius**2
    estimated_affected = int(population_density * estimated_area / 1000000)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Affected Radius", f"{estimated_radius:.0f} km")
    col2.metric("Impact Area", f"{estimated_area:.0f} km²")
    col3.metric("People Affected", f"{estimated_affected:,}")

# Footer
st.markdown("---")
st.markdown("### 🌐 Built with Python + Streamlit")
st.markdown("### ⚡ Deployed via GitHub + Streamlit Sharing")
st.markdown("### 🚀 Real-time disaster modeling for emergency response")
