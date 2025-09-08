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
    page_title="🍔 Materials Science Design Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for McDonald's-style theme
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        background-color: #FFC72C;
        color: #DA291C;
        font-weight: bold;
        border-radius: 20px;
        border: 2px solid #DA291C;
        padding: 0.5rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #DA291C;
        color: white;
        transform: scale(1.05);
    }
    .order-card {
        background-color: #f8f9fa;
        border: 2px solid #DA291C;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .menu-item {
        background-color: white;
        border: 1px solid #FFC72C;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        cursor: pointer;
        transition: all 0.3s;
    }
    .menu-item:hover {
        background-color: #FFF3CD;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .selected-item {
        background-color: #FFC72C;
        border: 2px solid #DA291C;
    }
    h1 {
        color: #DA291C;
        text-align: center;
        font-family: Arial, sans-serif;
    }
    h2, h3 {
        color: #DA291C;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
    .chain-step {
        background-color: #FFF3CD;
        border: 1px solid #FFC72C;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        position: relative;
        padding-left: 3rem;
    }
    .chain-number {
        position: absolute;
        left: 0.5rem;
        top: 50%;
        transform: translateY(-50%);
        background-color: #DA291C;
        color: white;
        width: 2rem;
        height: 2rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
    }
    .mechanism-box {
        background-color: #e8f4f8;
        border: 2px solid #17a2b8;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .uncertainty-box {
        background-color: #f8f9fa;
        border-left: 4px solid;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
    }
    .uncertainty-high { border-color: #28a745; }
    .uncertainty-medium { border-color: #ffc107; }
    .uncertainty-low { border-color: #dc3545; }
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
        with st.spinner("🔧 Preparing the Materials Kitchen..."):
            engine = CausalReasoningEngine(json_path)
        return engine
    except Exception as e:
        st.error(f"Failed to load engine: {str(e)}")
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

def create_menu_card(item, category, selected_items):
    """Create a McDonald's style menu card for an item"""
    if category not in selected_items:
        selected_items[category] = {}

    is_selected = item in selected_items[category].get("items", [])

    col1, col2 = st.columns([4, 1])
    with col1:
        if st.button(
            f"{'✅ ' if is_selected else ''}**{item}**",
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
            st.success("Added!")

# def display_current_order(order):
#     """Display the current order in a receipt-like format"""
#     st.markdown('<div class="order-card">', unsafe_allow_html=True)
#     st.markdown("### 📋 Your Current Order")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("**🔧 Synthesis Conditions:**")
#         if order['synthesis_conditions']:
#             for key, items in order['synthesis_conditions'].items():
#                 for item in items.get('items', []):
#                     st.write(f"• {item}")
#         else:
#             st.write("*No items selected*")
    
#     with col2:
#         st.markdown("**🎯 Material Properties:**")
#         if order['material_properties']:
#             for key, items in order['material_properties'].items():
#                 for item in items.get('items', []):
#                     st.write(f"• {item}")
#         else:
#             st.write("*No items selected*")
    
#     st.markdown('</div>', unsafe_allow_html=True)

def display_current_order(order):
    """Display the current order in a receipt-like format"""
    st.markdown('<div class="order-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Your Current Order")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 Synthesis Conditions:**")
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
                    st.write(f"• {item}")
        else:
            st.write("*No items selected*")
    
    with col2:
        st.markdown("**🎯 Material Properties:**")
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
                    st.write(f"• {item}")
        else:
            st.write("*No items selected*")
    
    st.markdown('</div>', unsafe_allow_html=True)

def format_confidence_score(confidence):
    """Format confidence score with color coding"""
    if confidence >= 0.8:
        return f'<span class="confidence-high">{confidence:.2%}</span>'
    elif confidence >= 0.5:
        return f'<span class="confidence-medium">{confidence:.2%}</span>'
    else:
        return f'<span class="confidence-low">{confidence:.2%}</span>'

def visualize_chain_of_thought(chain_steps):
    """Create a visual representation of the chain of thought"""
    st.markdown("### 🧠 Chain of Thought Reasoning")
    
    for i, step in enumerate(chain_steps, 1):
        st.markdown(f'''
        <div class="chain-step">
            <div class="chain-number">{i}</div>
            {step}
        </div>
        ''', unsafe_allow_html=True)

def display_mechanistic_reasoning(reasoning):
    """Display mechanistic reasoning in a structured format"""
    st.markdown("### 🔬 Mechanistic Analysis")
    
    st.markdown('<div class="mechanism-box">', unsafe_allow_html=True)
    
    if 'similarity_analysis' in reasoning:
        st.markdown("**📊 Similarity Analysis:**")
        st.write(reasoning['similarity_analysis'])
        st.markdown("---")
    
    if 'adapted_mechanism' in reasoning:
        st.markdown("**⚙️ Adapted Mechanism:**")
        st.write(reasoning['adapted_mechanism'])
        st.markdown("---")
    
    if 'electronic_structure' in reasoning:
        st.markdown("**⚡ Electronic Structure:**")
        st.write(reasoning['electronic_structure'])
        st.markdown("---")
    
    if 'thermodynamic_analysis' in reasoning:
        st.markdown("**🌡️ Thermodynamic Analysis:**")
        st.write(reasoning['thermodynamic_analysis'])
        st.markdown("---")
    
    if 'defect_chemistry' in reasoning:
        st.markdown("**🔍 Defect Chemistry:**")
        st.write(reasoning['defect_chemistry'])
        st.markdown("---")
    
    if 'kinetics' in reasoning:
        st.markdown("**⏱️ Kinetics:**")
        st.write(reasoning['kinetics'])
    
    st.markdown('</div>', unsafe_allow_html=True)

def display_uncertainty_analysis(uncertainty):
    """Display uncertainty analysis with color coding"""
    st.markdown("### 📈 Uncertainty Analysis")
    
    if 'high_confidence' in uncertainty:
        st.markdown(f'''
        <div class="uncertainty-box uncertainty-high">
            <strong>✅ High Confidence:</strong> {uncertainty['high_confidence']}
        </div>
        ''', unsafe_allow_html=True)
    
    if 'medium_confidence' in uncertainty:
        st.markdown(f'''
        <div class="uncertainty-box uncertainty-medium">
            <strong>⚠️ Medium Confidence:</strong> {uncertainty['medium_confidence']}
        </div>
        ''', unsafe_allow_html=True)
    
    if 'low_confidence' in uncertainty:
        st.markdown(f'''
        <div class="uncertainty-box uncertainty-low">
            <strong>❗ Low Confidence:</strong> {uncertainty['low_confidence']}
        </div>
        ''', unsafe_allow_html=True)

def visualize_causal_path(engine, result, query_type):
    """Create a visualization of the causal reasoning path"""
    fig = go.Figure()
    
    # Extract nodes and edges from result
    if 'analogous_path_used' in result:
        path_str = result['analogous_path_used']
        nodes = [node.strip() for node in path_str.split(' -> ')]
        
        # Create node positions
        pos_x = list(range(len(nodes)))
        pos_y = [0] * len(nodes)
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=pos_x,
            y=pos_y,
            mode='markers+text',
            marker=dict(size=30, color='#FFC72C', line=dict(color='#DA291C', width=2)),
            text=nodes,
            textposition="top center",
            hoverinfo='text',
            hovertext=nodes
        ))
        
        # Add edges
        for i in range(len(nodes) - 1):
            fig.add_trace(go.Scatter(
                x=[pos_x[i], pos_x[i+1]],
                y=[0, 0],
                mode='lines',
                line=dict(color='#DA291C', width=2),
                hoverinfo='none',
                showlegend=False
            ))
    
    fig.update_layout(
        title=f"Causal Reasoning Path ({query_type})",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        height=200
    )
    
    return fig


def display_quantitative_estimates(estimates):
    """Display quantitative estimates in a structured format with robust error handling"""
    st.markdown("### 📏 Quantitative Estimates")
    
    # Check if estimates exists and is valid
    if not estimates:
        st.info("No quantitative estimates available for this query.")
        return
    
    if not isinstance(estimates, dict):
        st.warning("⚠️ Quantitative estimates format is invalid.")
        st.write(f"Received type: {type(estimates)}")
        return
    
    # Filter out empty, null, or invalid estimates
    valid_estimates = {}
    for k, v in estimates.items():
        if v is not None and str(v).strip() and str(v).lower() not in ['none', 'null', 'n/a', '']:
            valid_estimates[k] = v
    
    if not valid_estimates:
        st.info("No valid quantitative estimates found in the response.")
        return
    
    # Safely create columns with reasonable limits
    num_estimates = len(valid_estimates)
    max_columns = min(num_estimates, 4)  # Limit to max 4 columns for better display
    
    try:
        cols = st.columns(max_columns)
        
        for i, (key, value) in enumerate(valid_estimates.items()):
            col_idx = i % max_columns  # Wrap around if more estimates than columns
            
            with cols[col_idx]:
                # Clean up the key for display
                display_key = key.replace('_', ' ').replace('-', ' ').title()
                
                # Handle different value types gracefully
                try:
                    # If it's a number, format it nicely
                    if isinstance(value, (int, float)):
                        display_value = f"{value:,.3f}".rstrip('0').rstrip('.')
                    else:
                        display_value = str(value)
                    
                    st.metric(
                        label=display_key,
                        value=display_value
                    )
                except Exception as e:
                    # Fallback display if metric fails
                    st.write(f"**{display_key}:** {value}")
                    
    except Exception as e:
        # Ultimate fallback - just display as key-value pairs
        st.warning("⚠️ Could not display estimates in metric format. Showing as list:")
        for key, value in valid_estimates.items():
            st.write(f"• **{key.replace('_', ' ').title()}:** {value}")

def display_alternative_mechanisms(alternatives):
    """Display alternative mechanisms"""
    st.markdown("### 🔄 Alternative Mechanisms")
    
    for alt in alternatives:
        st.info(f"• {alt}")

# Main Application
def main():
    st.markdown("# 🍔 Materials Science Design Studio")
    st.markdown("### *Order Your Perfect Material Like a Happy Meal!*")
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # JSON file selection
        json_file = st.text_input(
            "Knowledge Graph JSON Path",
            value="outputs/combined_doping_data.json",
            help="Path to your causal relationships JSON file"
        )
        
        # Load engine button
        if st.button("🚀 Start Materials Kitchen", type="primary"):
            if os.path.exists(json_file):
                st.session_state.engine = load_engine(json_file)
                if st.session_state.engine:
                    st.success("✅ Materials Kitchen is ready!")
                    st.balloons()
            else:
                st.error(f"File not found: {json_file}")
        
        # Display engine status
        if st.session_state.engine:
            st.success("✅ Engine Loaded")
            graph = st.session_state.engine.graph
            st.metric("Total Nodes", graph.number_of_nodes())
            st.metric("Total Edges", graph.number_of_edges())
        
        # Order history
        st.markdown("---")
        st.markdown("## 📜 Order History")
        if st.session_state.order_history:
            for i, order in enumerate(reversed(st.session_state.order_history[-5:])):
                with st.expander(f"Order #{len(st.session_state.order_history)-i}"):
                    st.json(order)
    
    # Main content area
    if st.session_state.engine is None:
        st.info("👈 Please load the Materials Kitchen from the sidebar to start ordering!")
        return
    
    # Extract menu items
    synthesis_params, material_properties = extract_menu_items(st.session_state.engine)
    
    # Create tabs for different ordering modes
    tab1, tab2, tab3 = st.tabs(["🍟 Forward Prediction", "🍔 Inverse Design", "📊 Results Dashboard"])
    
    with tab1:
        st.markdown("## 🍟 Forward Prediction Menu")
        st.markdown("*Select synthesis conditions to predict material properties*")
        
        # Menu selection area
        st.markdown("### Choose Your Synthesis Ingredients:")
        
        # Group synthesis parameters by category (simplified for demo) 
        col1, col2, col3 = st.columns(3)
        
        params_per_col = len(synthesis_params) // 3 + 1
        
        for i, param in enumerate(synthesis_params):
            col_idx = i // params_per_col
            if col_idx == 0:
                with col1:
                    create_menu_card(param, 'synthesis_conditions', st.session_state.current_order)
            elif col_idx == 1:
                with col2:
                    create_menu_card(param, 'synthesis_conditions', st.session_state.current_order)
            else:
                with col3:
                    create_menu_card(param, 'synthesis_conditions', st.session_state.current_order)
        
        # Display current order
        display_current_order(st.session_state.current_order)
        
        # Order button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎯 Predict Properties!", type="primary", use_container_width=True):
                if st.session_state.current_order['synthesis_conditions']:
                    with st.spinner("👨‍🔬 Our scientists are working on your order..."):
                        # Prepare synthesis inputs
                        # synthesis_inputs = {}
                        # for category, data in st.session_state.current_order['synthesis_conditions'].items():
                        #     for item in data.get('items', []):
                        #         synthesis_inputs[item] = item
                        # Prepare synthesis inputs
                        synthesis_inputs = {}
                        for category, data in st.session_state.current_order['synthesis_conditions'].items():
                            # Handle both list and dict formats
                            if isinstance(data, dict):
                                item_list = data.get('items', [])
                            elif isinstance(data, list):
                                item_list = data
                            else:
                                item_list = []
                            
                            for item in item_list:
                                synthesis_inputs[item] = item

                        # Run prediction
                        result = st.session_state.engine.forward_prediction(synthesis_inputs)
                        
                        # Store in history
                        order_record = {
                            'timestamp': datetime.now().isoformat(),
                            'type': 'forward_prediction',
                            'input': synthesis_inputs,
                            'result': result
                        }
                        st.session_state.order_history.append(order_record)
                        
                        # Display results
                        st.success("✅ Order Complete!")
                        
                        # Main results section
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown("### 🎯 Predicted Properties:")
                            if 'predicted_properties' in result:
                                for prop, value in result['predicted_properties'].items():
                                    st.write(f"• **{prop}**: {value}")
                        
                        with col2:
                            st.markdown("### 📊 Confidence Score:")
                            confidence = result.get('confidence', 0.0)
                            st.markdown(format_confidence_score(confidence), unsafe_allow_html=True)
                            
                            # Confidence gauge
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=confidence * 100,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Confidence %"},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "darkgreen" if confidence > 0.8 else "orange" if confidence > 0.5 else "red"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 80], 'color': "gray"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 90
                                    }
                                }
                            ))
                            fig.update_layout(height=200)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display chain of thought if available
                        if 'chain_of_thought' in result:
                            visualize_chain_of_thought(result['chain_of_thought'])
                        
                        # Display mechanistic reasoning if available
                        if 'mechanistic_reasoning' in result:
                            display_mechanistic_reasoning(result['mechanistic_reasoning'])
                        elif 'mechanistic_explanation' in result:
                            display_mechanistic_reasoning(result['mechanistic_explanation'])
                        
                        # Display quantitative estimates if available
                        if 'quantitative_estimates' in result:
                            display_quantitative_estimates(result['quantitative_estimates'])
                        
                        # Display uncertainty analysis if available
                        if 'uncertainty_analysis' in result:
                            display_uncertainty_analysis(result['uncertainty_analysis'])
                        
                        # Display alternative mechanisms if available
                        if 'alternative_mechanisms' in result:
                            display_alternative_mechanisms(result['alternative_mechanisms'])
                        
                        # Visualize causal path if available
                        if 'analogous_path_used' in result:
                            st.markdown("### 🗺️ Causal Reasoning Path:")
                            fig = visualize_causal_path(st.session_state.engine, result, "Forward Prediction")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display embedding distance if available
                            if 'property_embedding_distance' in result:
                                st.info(f"📏 Embedding Distance: {result['property_embedding_distance']:.4f} (0=identical, 2=opposite)")
                        
                        # Display general reasoning if available
                        if 'reasoning' in result:
                            st.markdown("### 💭 Additional Reasoning:")
                            st.info(result['reasoning'])
                else:
                    st.warning("⚠️ Please select at least one synthesis condition!")
    
    with tab2:
        st.markdown("## 🍔 Inverse Design Menu") 
        st.markdown("*Select desired properties to get synthesis recommendations*")
        
        # Clear order for inverse design
        if st.button("🔄 Start New Inverse Design Order"):
            st.session_state.current_order = {
                'synthesis_conditions': {},
                'material_properties': {}
            }
            st.rerun()
        
        # Menu selection area
        st.markdown("### Choose Your Desired Properties:")
        
        # Display material properties in columns
        col1, col2, col3 = st.columns(3)
        
        props_per_col = len(material_properties) // 3 + 1
        
        for i, prop in enumerate(material_properties):
            col_idx = i // props_per_col
            if col_idx == 0:
                with col1:
                    create_menu_card(prop, 'material_properties', st.session_state.current_order)
            elif col_idx == 1:
                with col2:
                    create_menu_card(prop, 'material_properties', st.session_state.current_order)
            else:
                with col3:
                    create_menu_card(prop, 'material_properties', st.session_state.current_order)
        
        # Display current order
        display_current_order(st.session_state.current_order)
        
        # Order button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔬 Get Synthesis Recipe!", type="primary", use_container_width=True):
                if st.session_state.current_order['material_properties']:
                    with st.spinner("👨‍🍳 Our materials chefs are creating your recipe..."):
                        # Prepare property inputs
                        # property_inputs = {}
                        # for category, data in st.session_state.current_order['material_properties'].items():
                        #     for item in data.get('items', []):
                        #         property_inputs[item] = item
                        property_inputs = {}
                        for category, data in st.session_state.current_order['material_properties'].items():
                            # Handle both list and dict formats
                            if isinstance(data, dict):
                                item_list = data.get('items', [])
                            elif isinstance(data, list):
                                item_list = data
                            else:
                                item_list = []
                            
                            for item in item_list:
                                property_inputs[item] = item       

                        # Run inverse design
                        result = st.session_state.engine.inverse_design(property_inputs)
                        
                        # Store in history
                        order_record = {
                            'timestamp': datetime.now().isoformat(),
                            'type': 'inverse_design',
                            'input': property_inputs,
                            'result': result
                        }
                        st.session_state.order_history.append(order_record)
                        
                        # Display results
                        st.success("✅ Recipe Ready!")
                        
                        # Main results section
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown("### 🔬 Recommended Synthesis Recipe:")
                            if 'suggested_synthesis_conditions' in result:
                                for condition, value in result['suggested_synthesis_conditions'].items():
                                    st.write(f"• **{condition}**: {value}")
                        
                        with col2:
                            st.markdown("### 📊 Confidence Score:")
                            confidence = result.get('confidence', 0.0)
                            st.markdown(format_confidence_score(confidence), unsafe_allow_html=True)
                            
                            # Confidence gauge
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=confidence * 100,
                                domain={'x': [0, 1], 'y': [0, 1]},
                                title={'text': "Confidence %"},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "darkgreen" if confidence > 0.8 else "orange" if confidence > 0.5 else "red"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 80], 'color': "gray"}
                                    ]
                                }
                            ))
                            fig.update_layout(height=200)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display chain of thought if available
                        if 'chain_of_thought' in result:
                            visualize_chain_of_thought(result['chain_of_thought'])
                        
                        # Display mechanistic reasoning if available
                        if 'mechanistic_reasoning' in result:
                            display_mechanistic_reasoning(result['mechanistic_reasoning'])
                        elif 'mechanistic_explanation' in result:
                            display_mechanistic_reasoning(result['mechanistic_explanation'])
                        
                        # Display quantitative estimates if available
                        if 'quantitative_estimates' in result:
                            display_quantitative_estimates(result['quantitative_estimates'])
                        
                        # Display uncertainty analysis if available
                        if 'uncertainty_analysis' in result:
                            display_uncertainty_analysis(result['uncertainty_analysis'])
                        
                        # Display alternative mechanisms if available
                        if 'alternative_mechanisms' in result:
                            display_alternative_mechanisms(result['alternative_mechanisms'])
                        
                        # Visualize causal path if available
                        if 'analogous_path_used' in result:
                            st.markdown("### 🗺️ Causal Reasoning Path:")
                            fig = visualize_causal_path(st.session_state.engine, result, "Inverse Design")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display embedding distance if available
                            if 'property_embedding_distance' in result:
                                st.info(f"📏 Embedding Distance: {result['property_embedding_distance']:.4f} (0=identical, 2=opposite)")
                        
                        # Display general reasoning if available
                        if 'reasoning' in result:
                            st.markdown("### 💭 Additional Reasoning:")
                            st.info(result['reasoning'])
                        
                        # Display suggested next steps if available
                        if 'suggested_next_steps' in result:
                            st.markdown("### 📋 Suggested Next Steps:")
                            st.write(result['suggested_next_steps'])
                        
                        # Show transfer learning analysis if present
                        if 'transfer_learning_analysis' in result:
                            with st.expander("🔍 View Detailed Transfer Learning Analysis"):
                                analysis = result['transfer_learning_analysis']
                                for step, description in analysis.items():
                                    st.markdown(f"**{step.replace('_', ' ').title()}:**")
                                    st.write(description)
                                    st.markdown("---")
                else:
                    st.warning("⚠️ Please select at least one desired property!")
    
    with tab3:
        st.markdown("## 📊 Results Dashboard")
        
        if st.session_state.order_history:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            total_orders = len(st.session_state.order_history)
            forward_orders = sum(1 for o in st.session_state.order_history if o['type'] == 'forward_prediction')
            inverse_orders = sum(1 for o in st.session_state.order_history if o['type'] == 'inverse_design')
            avg_confidence = sum(o['result'].get('confidence', 0) for o in st.session_state.order_history) / total_orders if total_orders > 0 else 0
            
            with col1:
                st.metric("Total Orders", total_orders)
            with col2:
                st.metric("Forward Predictions", forward_orders)
            with col3:
                st.metric("Inverse Designs", inverse_orders)
            with col4:
                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
            
            # Order history chart
            st.markdown("### 📈 Order History")
            
            # Prepare data for visualization
            df_data = []
            for order in st.session_state.order_history:
                df_data.append({
                    'timestamp': pd.to_datetime(order['timestamp']),
                    'type': order['type'],
                    'confidence': order['result'].get('confidence', 0)
                })
            
            df = pd.DataFrame(df_data)
            
            # Confidence over time
            fig = px.line(df, x='timestamp', y='confidence', color='type',
                         title='Confidence Scores Over Time',
                         labels={'confidence': 'Confidence Score', 'timestamp': 'Time'})
            fig.update_layout(yaxis_range=[0, 1.1])
            st.plotly_chart(fig, use_container_width=True)
            
            # Order type distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(df, names='type', title='Order Type Distribution',
                           color_discrete_map={'forward_prediction': '#FFC72C', 'inverse_design': '#DA291C'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Confidence distribution
                fig = px.histogram(df, x='confidence', nbins=10, title='Confidence Distribution',
                                 labels={'confidence': 'Confidence Score', 'count': 'Number of Orders'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed order history
            st.markdown("### 📋 Detailed Order History")
            
            # Create expandable sections for each order
            for i, order in enumerate(reversed(st.session_state.order_history)):
                order_type = "🍟 Forward Prediction" if order['type'] == 'forward_prediction' else "🍔 Inverse Design"
                confidence = order['result'].get('confidence', 0)
                confidence_emoji = "🟢" if confidence > 0.8 else "🟡" if confidence > 0.5 else "🔴"
                
                with st.expander(f"{order_type} - Order #{len(st.session_state.order_history)-i} {confidence_emoji} ({confidence:.2%})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Input:**")
                        st.json(order['input'])
                    
                    with col2:
                        st.markdown("**Result:**")
                        st.json(order['result'])
                    
                    st.markdown(f"**Timestamp:** {order['timestamp']}")
        else:
            st.info("📭 No orders yet! Start by making a Forward Prediction or Inverse Design order.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>🧪 Materials Science Design Studio | Powered by Causal Reasoning Engine</p>
            <p>Made with ❤️ using Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()