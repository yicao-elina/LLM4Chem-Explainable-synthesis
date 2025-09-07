import streamlit as st
import json
import pandas as pd
from datetime import datetime
import os
import sys
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Import your existing CausalReasoningEngine
# Make sure the engine file is in the same directory or adjust the path
from causal_engine0828 import CausalReasoningEngine, extract_json_from_response

# Page configuration
st.set_page_config(
    page_title="🚀 ARIA - Autonomous Reasoning Intelligence for Atomics",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Aerospace/Space Capsule theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tahoma:wght@400;700&display=swap');
    
    * {
        font-family: 'Tahoma', sans-serif !important;
    }
    
    .main {
        padding: 0rem 1rem;
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e0e6ed;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    .stButton>button {
        background: linear-gradient(45deg, #00d4ff, #0099cc);
        color: #000;
        font-weight: bold;
        border-radius: 25px;
        border: 2px solid #00ffff;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s;
        text-transform: uppercase;
        letter-spacing: 1px;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(45deg, #ff6b35, #f7931e);
        color: #000;
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(255, 107, 53, 0.5);
        border-color: #ff6b35;
    }
    
    .control-panel {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border: 2px solid #00ffff;
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.2), inset 0 0 20px rgba(0, 255, 255, 0.1);
        position: relative;
    }
    
    .control-panel::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #00ffff, #ff6b35, #00d4ff, #f7931e);
        border-radius: 22px;
        z-index: -1;
        animation: borderGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes borderGlow {
        0% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .system-module {
        background: rgba(30, 60, 114, 0.8);
        border: 1px solid #00d4ff;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 0.8rem;
        cursor: pointer;
        transition: all 0.3s;
        backdrop-filter: blur(10px);
        position: relative;
    }
    
    .system-module:hover {
        background: rgba(0, 212, 255, 0.1);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
        border-color: #ff6b35;
    }
    
    .selected-module {
        background: rgba(255, 107, 53, 0.2);
        border: 2px solid #ff6b35;
        box-shadow: 0 0 20px rgba(255, 107, 53, 0.4);
    }
    
    h1 {
        color: #00ffff;
        text-align: center;
        font-family: 'Tahoma', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0 0 20px rgba(0, 255, 255, 0.8);
        margin-bottom: 0;
    }
    
    h2, h3 {
        color: #00d4ff;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .confidence-high { 
        color: #00ff88; 
        font-weight: bold; 
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.6);
    }
    .confidence-medium { 
        color: #ffaa00; 
        font-weight: bold; 
        text-shadow: 0 0 10px rgba(255, 170, 0, 0.6);
    }
    .confidence-low { 
        color: #ff4444; 
        font-weight: bold; 
        text-shadow: 0 0 10px rgba(255, 68, 68, 0.6);
    }
    
    .reasoning-chain {
        background: linear-gradient(135deg, #0f3460 0%, #0c2d5a 100%);
        border: 1px solid #00d4ff;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        position: relative;
        padding-left: 4rem;
        backdrop-filter: blur(5px);
    }
    
    .chain-indicator {
        position: absolute;
        left: 1rem;
        top: 50%;
        transform: translateY(-50%);
        background: linear-gradient(45deg, #00d4ff, #00ffff);
        color: #000;
        width: 2.5rem;
        height: 2.5rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.6);
    }
    
    .analysis-module {
        background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
        border: 2px solid #26d0ce;
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 0 25px rgba(38, 208, 206, 0.3);
    }
    
    .status-indicator {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-left: 5px solid;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0 10px 10px 0;
        backdrop-filter: blur(10px);
    }
    
    .status-optimal { border-color: #00ff88; }
    .status-caution { border-color: #ffaa00; }
    .status-critical { border-color: #ff4444; }
    
    .aria-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        border: 2px solid #00ffff;
        box-shadow: 0 0 40px rgba(0, 255, 255, 0.3);
    }
    
    .system-status {
        display: flex;
        justify-content: space-around;
        align-items: center;
        padding: 1rem;
        background: rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .status-light {
        width: 15px;
        height: 15px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
        animation: pulse 2s infinite;
    }
    
    .status-online { background-color: #00ff88; }
    .status-processing { background-color: #ffaa00; }
    .status-offline { background-color: #ff4444; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .metric-display {
        background: linear-gradient(135deg, #232526 0%, #414345 100%);
        border: 1px solid #00d4ff;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        color: #00ffff;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    
    .stSelectbox > div > div {
        background-color: #1a2980;
        color: #e0e6ed;
        border: 1px solid #00d4ff;
    }
    
    .stTextInput > div > div > input {
        background-color: #1a2980;
        color: #e0e6ed;
        border: 1px solid #00d4ff;
    }
    
    .emergency-activation {
        background: linear-gradient(45deg, #ff4444, #cc0000);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        animation: emergencyPulse 1s infinite;
        margin-bottom: 2rem;
    }
    
    @keyframes emergencyPulse {
        0% { box-shadow: 0 0 20px rgba(255, 68, 68, 0.8); }
        50% { box-shadow: 0 0 40px rgba(255, 68, 68, 1); }
        100% { box-shadow: 0 0 20px rgba(255, 68, 68, 0.8); }
    }
    
    .tab-container {
        background: rgba(26, 41, 128, 0.3);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = None
    st.session_state.order_history = []
    st.session_state.current_order = {
        'synthesis_conditions': {},
        'material_properties': {}
    }
    st.session_state.aria_status = "STANDBY"
    st.session_state.emergency_mode = False

# Add this function to reset the order structure if needed
def reset_current_order():
    st.session_state.current_order = {
        'synthesis_conditions': {},
        'material_properties': {}
    }

# Helper functions
def load_engine(json_path):
    """Load the causal reasoning engine"""
    try:
        with st.spinner("🛸 ARIA INITIALIZING... LOADING CAUSAL KNOWLEDGE GRAPHS..."):
            engine = CausalReasoningEngine(json_path)
            st.session_state.aria_status = "ONLINE"
        return engine
    except Exception as e:
        st.error(f"❌ ARIA SYSTEM FAILURE: {str(e)}")
        st.session_state.aria_status = "ERROR"
        return None

def extract_menu_items(engine):
    """Extract available options from the knowledge graph"""
    synthesis_params = []
    material_properties = []
    
    for node in engine.graph.nodes():
        if engine.graph.in_degree(node) == 0:  # Source nodes (synthesis parameters)
            synthesis_params.append(node)
        elif engine.graph.out_degree(node) == 0:  # Sink nodes (material properties)
            material_properties.append(node)
    
    return synthesis_params, material_properties

def create_system_module(item, category, selected_items):
    """Create a space capsule-style system module for an item"""
    if category not in selected_items:
        selected_items[category] = {}

    is_selected = item in selected_items[category].get("items", [])

    col1, col2 = st.columns([4, 1])
    with col1:
        if st.button(
            f"{'🟢 ACTIVE' if is_selected else '⚪ STANDBY'} | **{item.upper()}**",
            key=f"{category}_{item}",
            use_container_width=True
        ):
            if "items" not in selected_items[category]:
                selected_items[category]["items"] = []
            if is_selected:
                selected_items[category]["items"].remove(item)
            else:
                selected_items[category]["items"].append(item)
            st.rerun()

    with col2:
        if is_selected:
            st.markdown('<span class="status-light status-online"></span>**ENGAGED**', unsafe_allow_html=True)

def display_current_configuration(order):
    """Display the current system configuration"""
    st.markdown('<div class="control-panel">', unsafe_allow_html=True)
    st.markdown("### 🎛️ CURRENT SYSTEM CONFIGURATION")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**⚙️ SYNTHESIS PARAMETERS:**")
        if order['synthesis_conditions']:
            for key, items in order['synthesis_conditions'].items():
                # Handle both list and dict formats
                if isinstance(items, dict):
                    item_list = items.get('items', [])
                elif isinstance(items, list):
                    item_list = items
                else:
                    item_list = []
                
                for item in item_list:
                    st.markdown(f'<span class="status-light status-online"></span>**{item.upper()}**', unsafe_allow_html=True)
        else:
            st.markdown("*🔴 NO PARAMETERS CONFIGURED*")
    
    with col2:
        st.markdown("**🎯 TARGET PROPERTIES:**")
        if order['material_properties']:
            for key, items in order['material_properties'].items():
                # Handle both list and dict formats
                if isinstance(items, dict):
                    item_list = items.get('items', [])
                elif isinstance(items, list):
                    item_list = items
                else:
                    item_list = []
                
                for item in item_list:
                    st.markdown(f'<span class="status-light status-online"></span>**{item.upper()}**', unsafe_allow_html=True)
        else:
            st.markdown("*🔴 NO TARGETS SPECIFIED*")
    
    st.markdown('</div>', unsafe_allow_html=True)

def format_confidence_display(confidence):
    """Format confidence score with space-age styling"""
    if confidence >= 0.8:
        return f'<span class="confidence-high">🟢 OPTIMAL: {confidence:.1%}</span>'
    elif confidence >= 0.5:
        return f'<span class="confidence-medium">🟡 CAUTION: {confidence:.1%}</span>'
    else:
        return f'<span class="confidence-low">🔴 CRITICAL: {confidence:.1%}</span>'

def display_reasoning_chain(chain_steps):
    """Create a visual representation of ARIA's reasoning process"""
    st.markdown("### 🧠 ARIA REASONING SEQUENCE")
    
    for i, step in enumerate(chain_steps, 1):
        st.markdown(f'''
        <div class="reasoning-chain">
            <div class="chain-indicator">{i}</div>
            {step}
        </div>
        ''', unsafe_allow_html=True)

def display_analysis_module(reasoning):
    """Display mechanistic analysis in space capsule format"""
    st.markdown("### 🔬 DEEP ANALYSIS MODULE")
    
    st.markdown('<div class="analysis-module">', unsafe_allow_html=True)
    
    if 'similarity_analysis' in reasoning:
        st.markdown("**📊 SIMILARITY MATRIX:**")
        st.write(reasoning['similarity_analysis'])
        st.markdown("---")
    
    if 'adapted_mechanism' in reasoning:
        st.markdown("**⚙️ ADAPTIVE MECHANISM:**")
        st.write(reasoning['adapted_mechanism'])
        st.markdown("---")
    
    if 'electronic_structure' in reasoning:
        st.markdown("**⚡ ELECTRONIC STRUCTURE ANALYSIS:**")
        st.write(reasoning['electronic_structure'])
        st.markdown("---")
    
    if 'thermodynamic_analysis' in reasoning:
        st.markdown("**🌡️ THERMODYNAMIC PROFILE:**")
        st.write(reasoning['thermodynamic_analysis'])
        st.markdown("---")
    
    if 'defect_chemistry' in reasoning:
        st.markdown("**🔍 DEFECT CHEMISTRY MAPPING:**")
        st.write(reasoning['defect_chemistry'])
        st.markdown("---")
    
    if 'kinetics' in reasoning:
        st.markdown("**⏱️ KINETIC MODELING:**")
        st.write(reasoning['kinetics'])
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_system_status(uncertainty):
    """Display system status with aerospace styling"""
    st.markdown("### 📈 SYSTEM STATUS ANALYSIS")
    
    if 'high_confidence' in uncertainty:
        st.markdown(f'''
        <div class="status-indicator status-optimal">
            <strong>🟢 OPTIMAL CONFIDENCE:</strong> {uncertainty['high_confidence']}
        </div>
        ''', unsafe_allow_html=True)
    
    if 'medium_confidence' in uncertainty:
        st.markdown(f'''
        <div class="status-indicator status-caution">
            <strong>🟡 CAUTION LEVEL:</strong> {uncertainty['medium_confidence']}
        </div>
        ''', unsafe_allow_html=True)
    
    if 'low_confidence' in uncertainty:
        st.markdown(f'''
        <div class="status-indicator status-critical">
            <strong>🔴 CRITICAL UNCERTAINTY:</strong> {uncertainty['low_confidence']}
        </div>
        ''', unsafe_allow_html=True)

def create_causal_pathway_viz(engine, result, query_type):
    """Create a futuristic visualization of the causal reasoning path"""
    fig = go.Figure()
    
    # Extract nodes and edges from result
    if 'analogous_path_used' in result:
        path_str = result['analogous_path_used']
        nodes = [node.strip() for node in path_str.split(' -> ')]
        
        # Create node positions
        pos_x = list(range(len(nodes)))
        pos_y = [0] * len(nodes)
        
        # Add glowing nodes
        fig.add_trace(go.Scatter(
            x=pos_x,
            y=pos_y,
            mode='markers+text',
            marker=dict(
                size=40, 
                color='#00d4ff',
                line=dict(color='#00ffff', width=3),
                symbol='circle'
            ),
            text=nodes,
            textposition="top center",
            textfont=dict(color='#00ffff', size=12, family='Tahoma'),
            hoverinfo='text',
            hovertext=nodes,
            name='Causal Nodes'
        ))
        
        # Add glowing connections
        for i in range(len(nodes) - 1):
            fig.add_trace(go.Scatter(
                x=[pos_x[i], pos_x[i+1]],
                y=[0, 0],
                mode='lines',
                line=dict(color='#ff6b35', width=4),
                hoverinfo='none',
                showlegend=False
            ))
    
    fig.update_layout(
        title=dict(
            text=f"ARIA CAUSAL PATHWAY ANALYSIS - {query_type.upper()}",
            font=dict(color='#00ffff', size=16, family='Tahoma')
        ),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=250
    )
    
    return fig

def display_quantitative_readouts(estimates):
    """Display quantitative estimates in aerospace-style readouts"""
    st.markdown("### 📏 QUANTITATIVE READOUTS")
    
    # Check if estimates exists and is valid
    if not estimates:
        st.info("🔵 NO QUANTITATIVE DATA AVAILABLE")
        return
    
    if not isinstance(estimates, dict):
        st.warning("⚠️ READOUT FORMAT ERROR")
        return
    
    # Filter out empty, null, or invalid estimates
    valid_estimates = {}
    for k, v in estimates.items():
        if v is not None and str(v).strip() and str(v).lower() not in ['none', 'null', 'n/a', '']:
            valid_estimates[k] = v
    
    if not valid_estimates:
        st.info("🔵 NO VALID READOUTS DETECTED")
        return
    
    # Display in aerospace-style metrics
    num_estimates = len(valid_estimates)
    max_columns = min(num_estimates, 4)
    
    try:
        cols = st.columns(max_columns)
        
        for i, (key, value) in enumerate(valid_estimates.items()):
            col_idx = i % max_columns
            
            with cols[col_idx]:
                display_key = key.replace('_', ' ').replace('-', ' ').upper()
                
                try:
                    if isinstance(value, (int, float)):
                        display_value = f"{value:,.3f}".rstrip('0').rstrip('.')
                    else:
                        display_value = str(value)
                    
                    st.markdown(f'''
                    <div class="metric-display">
                        <div style="font-size: 0.8rem; color: #00d4ff;">{display_key}</div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: #00ffff;">{display_value}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.markdown(f"**{display_key}:** {value}")
                    
    except Exception as e:
        st.warning("⚠️ DISPLAY ERROR - SHOWING RAW DATA:")
        for key, value in valid_estimates.items():
            st.write(f"• **{key.replace('_', ' ').upper()}:** {value}")

def display_alternative_protocols(alternatives):
    """Display alternative mechanisms as backup protocols"""
    st.markdown("### 🔄 BACKUP PROTOCOLS")
    
    for i, alt in enumerate(alternatives, 1):
        st.markdown(f'''
        <div class="status-indicator status-caution">
            <strong>PROTOCOL {i}:</strong> {alt}
        </div>
        ''', unsafe_allow_html=True)

# Main Application
def main():
    # ARIA Header
    st.markdown('''
    <div class="aria-header">
        <h1>🛸 ARIA</h1>
        <h3 style="margin: 0; color: #00d4ff;">AUTONOMOUS REASONING INTELLIGENCE FOR ATOMICS</h3>
        <p style="margin: 10px 0 0 0; color: #e0e6ed; font-style: italic;">
            "When knowledge reaches its limits, I reason by analogy—like a composer creating new melodies from familiar themes."
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Emergency activation check
    if st.session_state.get('emergency_mode', False):
        st.markdown('''
        <div class="emergency-activation">
            🚨 EMERGENCY ACTIVATION PROTOCOL ENGAGED 🚨<br>
            DR. TSAO DETECTED IN DANGER - FULL SYSTEM ONLINE
        </div>
        ''', unsafe_allow_html=True)
    
    # System Status Bar
    status_color = "status-online" if st.session_state.aria_status == "ONLINE" else "status-offline" if st.session_state.aria_status == "ERROR" else "status-processing"
    
    st.markdown(f'''
    <div class="system-status">
        <div><span class="status-light {status_color}"></span><strong>ARIA STATUS: {st.session_state.aria_status}</strong></div>
        <div><span class="status-light status-online"></span><strong>CAUSAL ENGINE: {'ACTIVE' if st.session_state.engine else 'STANDBY'}</strong></div>
        <div><span class="status-light status-processing"></span><strong>MISSION TIME: {datetime.now().strftime('%H:%M:%S')}</strong></div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar for mission control
    with st.sidebar:
        st.markdown("## 🎛️ MISSION CONTROL")
        
        # Emergency activation button
        if st.button("🚨 EMERGENCY ACTIVATION", type="primary"):
            st.session_state.emergency_mode = True
            st.balloons()
            st.success("🚨 ARIA EMERGENCY PROTOCOL ACTIVATED!")
        
        st.markdown("---")
        
        # Knowledge base configuration
        st.markdown("### 📡 KNOWLEDGE BASE")
        json_file = st.text_input(
            "Causal Database Path",
            value="outputs/combined_doping_data.json",
            help="Path to causal knowledge graph database"
        )
        
        # System initialization
        if st.button("🚀 INITIALIZE ARIA SYSTEMS", type="primary"):
            if os.path.exists(json_file):
                st.session_state.engine = load_engine(json_file)
                if st.session_state.engine:
                    st.success("✅ ARIA SYSTEMS ONLINE")
                    st.balloons()
            else:
                st.error(f"❌ DATABASE NOT FOUND: {json_file}")
        
        # System diagnostics
        if st.session_state.engine:
            st.markdown("### 📊 SYSTEM DIAGNOSTICS")
            graph = st.session_state.engine.graph
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("NODES", graph.number_of_nodes())
            with col2:
                st.metric("PATHS", graph.number_of_edges())
        
        # Mission log
        st.markdown("---")
        st.markdown("### 📜 MISSION LOG")
        if st.session_state.order_history:
            for i, mission in enumerate(reversed(st.session_state.order_history[-3:])):
                with st.expander(f"MISSION #{len(st.session_state.order_history)-i}"):
                    st.json(mission)
    
    # Main control interface
    if st.session_state.engine is None:
        st.markdown('''
        <div class="control-panel">
            <h3 style="text-align: center;">⚠️ ARIA SYSTEMS IN STANDBY MODE</h3>
            <p style="text-align: center;">Initialize ARIA systems from Mission Control to begin operations.</p>
        </div>
        ''', unsafe_allow_html=True)
        return
    
    # Extract available system modules
    synthesis_params, material_properties = extract_menu_items(st.session_state.engine)
    
    # Mission tabs
    tab1, tab2, tab3 = st.tabs(["🔮 FORWARD PREDICTION", "🎯 INVERSE DESIGN", "📊 MISSION DASHBOARD"])
    
    with tab1:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)
        st.markdown("## 🔮 FORWARD PREDICTION PROTOCOL")
        st.markdown("*Configure synthesis parameters to predict material properties*")
        
        st.markdown("### ⚙️ SYNTHESIS PARAMETER CONFIGURATION:")
        
        # Display synthesis parameters in grid
        col1, col2, col3 = st.columns(3)
        params_per_col = len(synthesis_params) // 3 + 1
        
        for i, param in enumerate(synthesis_params):
            col_idx = i // params_per_col
            if col_idx == 0:
                with col1:
                    create_system_module(param, 'synthesis_conditions', st.session_state.current_order)
            elif col_idx == 1:
                with col2:
                    create_system_module(param, 'synthesis_conditions', st.session_state.current_order)
            else:
                with col3:
                    create_system_module(param, 'synthesis_conditions', st.session_state.current_order)
        
        # Current configuration display
        display_current_configuration(st.session_state.current_order)
        
        # Execute prediction
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 EXECUTE FORWARD PREDICTION", type="primary", use_container_width=True):
                if st.session_state.current_order['synthesis_conditions']:
                    with st.spinner("🧠 ARIA ANALYZING CAUSAL PATHWAYS..."):
                        # Prepare inputs
                        synthesis_inputs = {}
                        for category, data in st.session_state.current_order['synthesis_conditions'].items():
                            if isinstance(data, dict):
                                item_list = data.get('items', [])
                            elif isinstance(data, list):
                                item_list = data
                            else:
                                item_list = []
                            
                            for item in item_list:
                                synthesis_inputs[item] = item

                        # Execute prediction
                        result = st.session_state.engine.forward_prediction(synthesis_inputs)
                        
                        # Log mission
                        mission_record = {
                            'timestamp': datetime.now().isoformat(),
                            'type': 'forward_prediction',
                            'input': synthesis_inputs ,
                            'result': result
                        }
                        st.session_state.order_history.append(mission_record)
                        
                        # Display results
                        st.success("✅ PREDICTION COMPLETE - ANALYSIS READY")
                        
                        # Results display
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown("### 🎯 PREDICTED PROPERTIES:")
                            if 'predicted_properties' in result:
                                for prop, value in result['predicted_properties'].items():
                                    st.markdown(f'<span class="status-light status-online"></span>**{prop.upper()}**: {value}', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("### 📊 CONFIDENCE LEVEL:")
                            confidence = result.get('confidence', 0.0)
                            st.markdown(format_confidence_display(confidence), unsafe_allow_html=True)
                            
                            # Confidence gauge with aerospace styling
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=confidence * 100,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "CONFIDENCE", 'font': {'color': '#00ffff', 'family': 'Tahoma'}},
                                number={'font': {'color': '#00ffff', 'family': 'Tahoma'}},
                                gauge={
                                    'axis': {'range': [None, 100], 'tickcolor': '#00ffff'},
                                    'bar': {'color': "#00ff88" if confidence > 0.8 else "#ffaa00" if confidence > 0.5 else "#ff4444"},
                                    'bgcolor': "rgba(0,0,0,0.3)",
                                    'borderwidth': 2,
                                    'bordercolor': "#00d4ff",
                                    'steps': [
                                        {'range': [0, 50], 'color': "rgba(255, 68, 68, 0.3)"},
                                        {'range': [50, 80], 'color': "rgba(255, 170, 0, 0.3)"},
                                        {'range': [80, 100], 'color': "rgba(0, 255, 136, 0.3)"}
                                    ]
                                }
                            ))
                            fig.update_layout(
                                height=200,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font={'color': '#00ffff', 'family': 'Tahoma'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display reasoning chain
                        if 'chain_of_thought' in result:
                            display_reasoning_chain(result['chain_of_thought'])
                        
                        # Display analysis modules
                        if 'mechanistic_reasoning' in result:
                            display_analysis_module(result['mechanistic_reasoning'])
                        elif 'mechanistic_explanation' in result:
                            display_analysis_module(result['mechanistic_explanation'])
                        
                        # Display quantitative readouts
                        if 'quantitative_estimates' in result:
                            display_quantitative_readouts(result['quantitative_estimates'])
                        
                        # Display system status
                        if 'uncertainty_analysis' in result:
                            display_system_status(result['uncertainty_analysis'])
                        
                        # Display backup protocols
                        if 'alternative_mechanisms' in result:
                            display_alternative_protocols(result['alternative_mechanisms'])
                        
                        # Visualize causal pathway
                        if 'analogous_path_used' in result:
                            st.markdown("### 🗺️ CAUSAL PATHWAY VISUALIZATION:")
                            fig = create_causal_pathway_viz(st.session_state.engine, result, "Forward Prediction")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            if 'property_embedding_distance' in result:
                                st.info(f"🔍 EMBEDDING DISTANCE: {result['property_embedding_distance']:.4f} (0=IDENTICAL, 2=OPPOSITE)")
                        
                        # Additional reasoning
                        if 'reasoning' in result:
                            st.markdown("### 💭 ARIA'S REASONING:")
                            st.info(result['reasoning'])
                else:
                    st.warning("⚠️ CONFIGURE SYNTHESIS PARAMETERS BEFORE EXECUTION")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)
        st.markdown("## 🎯 INVERSE DESIGN PROTOCOL")
        st.markdown("*Specify target properties to generate synthesis recommendations*")
        
        # Reset configuration
        if st.button("🔄 RESET CONFIGURATION"):
            st.session_state.current_order = {
                'synthesis_conditions': {},
                'material_properties': {}
            }
            st.rerun()
        
        st.markdown("### 🎯 TARGET PROPERTY SPECIFICATION:")
        
        # Display material properties in grid
        col1, col2, col3 = st.columns(3)
        props_per_col = len(material_properties) // 3 + 1
        
        for i, prop in enumerate(material_properties):
            col_idx = i // props_per_col
            if col_idx == 0:
                with col1:
                    create_system_module(prop, 'material_properties', st.session_state.current_order)
            elif col_idx == 1:
                with col2:
                    create_system_module(prop, 'material_properties', st.session_state.current_order)
            else:
                with col3:
                    create_system_module(prop, 'material_properties', st.session_state.current_order)
        
        # Current configuration display
        display_current_configuration(st.session_state.current_order)
        
        # Execute inverse design
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎯 EXECUTE INVERSE DESIGN", type="primary", use_container_width=True):
                if st.session_state.current_order['material_properties']:
                    with st.spinner("🧠 ARIA GENERATING SYNTHESIS PROTOCOLS..."):
                        # Prepare inputs
                        property_inputs = {}
                        for category, data in st.session_state.current_order['material_properties'].items():
                            if isinstance(data, dict):
                                item_list = data.get('items', [])
                            elif isinstance(data, list):
                                item_list = data
                            else:
                                item_list = []
                            
                            for item in item_list:
                                property_inputs[item] = item       

                        # Execute inverse design
                        result = st.session_state.engine.inverse_design(property_inputs)
                        
                        # Log mission
                        mission_record = {
                            'timestamp': datetime.now().isoformat(),
                            'type': 'inverse_design',
                            'input': property_inputs,
                            'result': result
                        }
                        st.session_state.order_history.append(mission_record)
                        
                        # Display results
                        st.success("✅ SYNTHESIS PROTOCOL GENERATED")
                        
                        # Results display
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown("### 🔬 RECOMMENDED SYNTHESIS PROTOCOL:")
                            if 'suggested_synthesis_conditions' in result:
                                for condition, value in result['suggested_synthesis_conditions'].items():
                                    st.markdown(f'<span class="status-light status-online"></span>**{condition.upper()}**: {value}', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("### 📊 PROTOCOL CONFIDENCE:")
                            confidence = result.get('confidence', 0.0)
                            st.markdown(format_confidence_display(confidence), unsafe_allow_html=True)
                            
                            # Confidence gauge
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=confidence * 100,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "CONFIDENCE", 'font': {'color': '#00ffff', 'family': 'Tahoma'}},
                                number={'font': {'color': '#00ffff', 'family': 'Tahoma'}},
                                gauge={
                                    'axis': {'range': [None, 100], 'tickcolor': '#00ffff'},
                                    'bar': {'color': "#00ff88" if confidence > 0.8 else "#ffaa00" if confidence > 0.5 else "#ff4444"},
                                    'bgcolor': "rgba(0,0,0,0.3)",
                                    'borderwidth': 2,
                                    'bordercolor': "#00d4ff"
                                }
                            ))
                            fig.update_layout(
                                height=200,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font={'color': '#00ffff', 'family': 'Tahoma'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display all analysis modules (same as forward prediction)
                        if 'chain_of_thought' in result:
                            display_reasoning_chain(result['chain_of_thought'])
                        
                        if 'mechanistic_reasoning' in result:
                            display_analysis_module(result['mechanistic_reasoning'])
                        elif 'mechanistic_explanation' in result:
                            display_analysis_module(result['mechanistic_explanation'])
                        
                        if 'quantitative_estimates' in result:
                            display_quantitative_readouts(result['quantitative_estimates'])
                        
                        if 'uncertainty_analysis' in result:
                            display_system_status(result['uncertainty_analysis'])
                        
                        if 'alternative_mechanisms' in result:
                            display_alternative_protocols(result['alternative_mechanisms'])
                        
                        if 'analogous_path_used' in result:
                            st.markdown("### 🗺️ CAUSAL PATHWAY VISUALIZATION:")
                            fig = create_causal_pathway_viz(st.session_state.engine, result, "Inverse Design")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            if 'property_embedding_distance' in result:
                                st.info(f"🔍 EMBEDDING DISTANCE: {result['property_embedding_distance']:.4f}")
                        
                        if 'reasoning' in result:
                            st.markdown("### 💭 ARIA'S REASONING:")
                            st.info(result['reasoning'])
                        
                        if 'suggested_next_steps' in result:
                            st.markdown("### 📋 RECOMMENDED NEXT STEPS:")
                            st.write(result['suggested_next_steps'])
                        
                        if 'transfer_learning_analysis' in result:
                            with st.expander("🔍 DETAILED TRANSFER LEARNING ANALYSIS"):
                                analysis = result['transfer_learning_analysis']
                                for step, description in analysis.items():
                                    st.markdown(f"**{step.replace('_', ' ').upper()}:**")
                                    st.write(description)
                                    st.markdown("---")
                else:
                    st.warning("⚠️ SPECIFY TARGET PROPERTIES BEFORE EXECUTION")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)
        st.markdown("## 📊 MISSION DASHBOARD")
        
        if st.session_state.order_history:
            # Mission statistics
            col1, col2, col3, col4 = st.columns(4)
            
            total_missions = len(st.session_state.order_history)
            forward_missions = sum(1 for m in st.session_state.order_history if m['type'] == 'forward_prediction')
            inverse_missions = sum(1 for m in st.session_state.order_history if m['type'] == 'inverse_design')
            avg_confidence = sum(m['result'].get('confidence', 0) for m in st.session_state.order_history) / total_missions if total_missions > 0 else 0
            
            with col1:
                st.markdown(f'''
                <div class="metric-display">
                    <div style="font-size: 0.8rem; color: #00d4ff;">TOTAL MISSIONS</div>
                    <div style="font-size: 2rem; font-weight: bold; color: #00ffff;">{total_missions}</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'''
                <div class="metric-display">
                    <div style="font-size: 0.8rem; color: #00d4ff;">FORWARD PRED.</div>
                    <div style="font-size: 2rem; font-weight: bold; color: #00ffff;">{forward_missions}</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f'''
                <div class="metric-display">
                    <div style="font-size: 0.8rem; color: #00d4ff;">INVERSE DESIGN</div>
                    <div style="font-size: 2rem; font-weight: bold; color: #00ffff;">{inverse_missions}</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col4:
                st.markdown(f'''
                <div class="metric-display">
                    <div style="font-size: 0.8rem; color: #00d4ff;">AVG CONFIDENCE</div>
                    <div style="font-size: 2rem; font-weight: bold; color: #00ffff;">{avg_confidence:.1%}</div>
                </div>
                ''', unsafe_allow_html=True)
            
            # Mission analytics
            st.markdown("### 📈 MISSION ANALYTICS")
            
            # Prepare data
            df_data = []
            for mission in st.session_state.order_history:
                df_data.append({
                    'timestamp': pd.to_datetime(mission['timestamp']),
                    'type': mission['type'],
                    'confidence': mission['result'].get('confidence', 0)
                })
            
            df = pd.DataFrame(df_data)
            
            # Confidence over time with aerospace styling
            fig = px.line(df, x='timestamp', y='confidence', color='type',
                         title='CONFIDENCE LEVELS OVER TIME',
                         labels={'confidence': 'CONFIDENCE LEVEL', 'timestamp': 'MISSION TIME'},
                         color_discrete_map={'forward_prediction': '#00d4ff', 'inverse_design': '#ff6b35'})
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': '#00ffff', 'family': 'Tahoma'},
                title_font={'color': '#00ffff', 'family': 'Tahoma'},
                yaxis_range=[0, 1.1]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Mission distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(df, names='type', title='MISSION TYPE DISTRIBUTION',
                           color_discrete_map={'forward_prediction': '#00d4ff', 'inverse_design': '#ff6b35'})
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#00ffff', 'family': 'Tahoma'},
                    title_font={'color': '#00ffff', 'family': 'Tahoma'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.histogram(df, x='confidence', nbins=10, title='CONFIDENCE DISTRIBUTION',
                                 labels={'confidence': 'CONFIDENCE LEVEL', 'count': 'MISSION COUNT'})
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': '#00ffff', 'family': 'Tahoma'},
                    title_font={'color': '#00ffff', 'family': 'Tahoma'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Mission log
            st.markdown("### 📜 DETAILED MISSION LOG")
            
            for i, mission in enumerate(reversed(st.session_state.order_history)):
                mission_type = "🔮 FORWARD PREDICTION" if mission['type'] == 'forward_prediction' else "🎯 INVERSE DESIGN"
                confidence = mission['result'].get('confidence', 0)
                confidence_status = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                
                with st.expander(f"{mission_type} - MISSION #{len(st.session_state.order_history)-i} {confidence_status} ({confidence:.1%})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**MISSION INPUT:**")
                        st.json(mission['input'])
                    
                    with col2:
                        st.markdown("**MISSION RESULT:**")
                        st.json(mission['result'])
                    
                    st.markdown(f"**MISSION TIMESTAMP:** {mission['timestamp']}")
        else:
            st.markdown('''
            <div class="control-panel">
                <h3 style="text-align: center;">📭 NO MISSIONS LOGGED</h3>
                <p style="text-align: center;">Execute Forward Predictions or Inverse Design protocols to populate the mission dashboard.</p>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #00d4ff; font-family: Tahoma;'>
            <p>🛸 ARIA - AUTONOMOUS REASONING INTELLIGENCE FOR ATOMICS</p>
            <p style='font-style: italic; color: #e0e6ed;'>"Understanding not just what materials exist, but WHY they work"</p>
            <p style='font-size: 0.8rem; color: #666;'>Johns Hopkins University | Powered by Causal Reasoning Engine</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()