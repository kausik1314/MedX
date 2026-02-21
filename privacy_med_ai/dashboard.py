import streamlit as st
import pandas as pd
import time
import os
import plotly.graph_objects as plotly_go
import yaml

st.set_page_config(page_title="MedX Privacy AI Engine", page_icon="🏥", layout="wide", initial_sidebar_state="expanded")

# Advanced Premium Glassmorphism & Animations CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Inter:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6, .metric-value, .header-title {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b 0%, #020617 100%);
        color: #f8fafc;
    }
    
    /* Hide the default Streamlit main menu and footer for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .glass-card {
        background: rgba(15, 23, 42, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(167, 139, 250, 0.3), transparent);
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(167, 139, 250, 0.4);
        box-shadow: 0 20px 40px -10px rgba(139, 92, 246, 0.2);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.1;
        background: -webkit-linear-gradient(135deg, #a78bfa, #2dd4bf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    
    .header-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #ffffff, #a78bfa, #2dd4bf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        letter-spacing: -1px;
        padding-top: 20px;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 1.25rem;
        font-weight: 400;
        margin-top: 10px;
        margin-bottom: 40px;
        max-width: 600px;
    }
    
    /* Sleek Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .pulse-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 10px #10b981;
        margin-right: 10px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
</style>
""", unsafe_allow_html=True)

# Main Header Area
st.markdown('''
    <div>
        <h1 class="header-title">🏥 MedX Core Interface</h1>
        <p class="header-subtitle">Real-time telemetry and cryptographic monitoring for decentralized differential privacy learning.</p>
    </div>
''', unsafe_allow_html=True)

METRICS_FILE = "metrics.csv"
if not os.path.exists(METRICS_FILE):
    pd.DataFrame(columns=["Round", "Accuracy", "Privacy_Epsilon", "Loss"]).to_csv(METRICS_FILE, index=False)

def load_data():
    if os.path.exists(METRICS_FILE):
        try:
            return pd.read_csv(METRICS_FILE)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=["Round", "Accuracy", "Privacy_Epsilon", "Loss"])
    return pd.DataFrame(columns=["Round", "Accuracy", "Privacy_Epsilon", "Loss"])

def load_config():
    try:
        with open("config.yaml", "r") as f:
            return yaml.safe_load(f)
    except:
        return {}

config = load_config()

# Sidebar Control Panel
with st.sidebar:
    st.markdown("### 🎛️ Simulation Parameters")
    st.markdown("<hr style='border-color: rgba(255,255,255,0.1); margin: 10px 0;'>", unsafe_allow_html=True)
    
    st.markdown(f"**📡 Active Dataset:** `{config.get('experiment', {}).get('dataset', 'Unknown')}`")
    st.markdown(f"**🔗 Connected Nodes:** `{config.get('experiment', {}).get('num_clients', 0)}` Nodes")
    st.markdown(f"**🔒 Target Privacy (ε):** `{config.get('privacy', {}).get('target_epsilon', 10.0)}`")
    st.markdown(f"**🧠 Neural Arch:** `{config.get('model', {}).get('architecture', 'Unknown').upper()}`")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    if st.button("🔄 Purge Telemetry History", use_container_width=True):
        if os.path.exists(METRICS_FILE):
            os.remove(METRICS_FILE)
            st.rerun()
            
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center; color:#64748b; font-size:0.8rem;'>MedX DP-AI Engine v2.0-PRO<br>System Nominal</div>", unsafe_allow_html=True)

metrics_container = st.empty()
charts_container = st.empty()

while True:
    df = load_data()
    
    if not df.empty:
        last_round = df.iloc[-1]
        
        # Upper Metrics Row
        with metrics_container.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'''
                <div class="glass-card">
                    <div class="metric-label">Global Model Accuracy</div>
                    <div class="metric-value">{last_round["Accuracy"]:.2%}</div>
                    <div style="color: #34d399; font-size: 0.9rem;">
                        <span class="pulse-indicator"></span> Sync Successful
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            with col2:
                loss_val = last_round.get("Loss", 0.0)
                st.markdown(f'''
                <div class="glass-card">
                    <div class="metric-label">Aggregated Inference Loss</div>
                    <div class="metric-value" style="background: -webkit-linear-gradient(135deg, #fbbf24, #f87171); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{loss_val:.4f}</div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">MSE Global Evaluator</div>
                </div>
                ''', unsafe_allow_html=True)
            with col3:
                eps = last_round["Privacy_Epsilon"]
                target_eps = config.get('privacy', {}).get('target_epsilon', 10.0)
                eps_color = "#34d399" if eps < target_eps else "#f87171"
                
                st.markdown(f'''
                <div class="glass-card">
                    <div class="metric-label">Privacy Budget Expended (ε)</div>
                    <div class="metric-value" style="background: -webkit-linear-gradient(135deg, {eps_color}, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{eps:.2f}</div>
                    <div style="color: #94a3b8; font-size: 0.9rem;">Target boundary limit: {target_eps}</div>
                </div>
                ''', unsafe_allow_html=True)

        with charts_container.container():
            st.markdown("<br><br>", unsafe_allow_html=True)
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown('<h3 style="font-family: Outfit; font-weight: 600; color: #e2e8f0; margin-bottom: 20px;">📈 Convergence Trajectory</h3>', unsafe_allow_html=True)
                fig1 = plotly_go.Figure()
                
                # Glowing shadow trace
                fig1.add_trace(plotly_go.Scatter(x=df['Round'], y=df['Accuracy'], mode='lines', 
                                        line=dict(color='rgba(45, 212, 191, 0.3)', width=8), hoverinfo='skip'))
                # Main line
                fig1.add_trace(plotly_go.Scatter(x=df['Round'], y=df['Accuracy'], mode='lines+markers', name='Accuracy',
                                        marker=dict(size=8, color='#f8fafc', line=dict(width=2, color='#2dd4bf')),
                                        line=dict(color='#2dd4bf', width=3), 
                                        fill='tozeroy', fillcolor='rgba(45, 212, 191, 0.1)'))
                
                fig1.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                                   font=dict(color='#cbd5e1', family='Inter'), margin=dict(l=0, r=0, t=10, b=0),
                                   xaxis=dict(showgrid=False, title="Federation Round"), 
                                   yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', gridwidth=1, zeroline=False))
                st.plotly_chart(fig1, use_container_width=True, key=f"acc_chart_{time.time()}")
                
            with chart_col2:
                st.markdown('<h3 style="font-family: Outfit; font-weight: 600; color: #e2e8f0; margin-bottom: 20px;">🔒 Differential Privacy Audit</h3>', unsafe_allow_html=True)
                fig2 = plotly_go.Figure()
                
                # Glowing shadow trace
                fig2.add_trace(plotly_go.Scatter(x=df['Round'], y=df['Privacy_Epsilon'], mode='lines', 
                                        line=dict(color='rgba(167, 139, 250, 0.3)', width=8), hoverinfo='skip'))
                # Main line
                fig2.add_trace(plotly_go.Scatter(x=df['Round'], y=df['Privacy_Epsilon'], mode='lines+markers', name='Epsilon',
                                        marker=dict(size=8, color='#f8fafc', line=dict(width=2, color='#a78bfa')),
                                        line=dict(color='#a78bfa', width=3), fill='tozeroy', fillcolor='rgba(167, 139, 250, 0.1)'))
                
                target_eps = config.get('privacy', {}).get('target_epsilon', 10.0)
                fig2.add_hline(y=target_eps, line_dash="dash", line_color="#ef4444", line_width=2, 
                               annotation_text=f"MAX SECURITY THRESHOLD: ε={target_eps}", 
                               annotation_position="bottom right", annotation_font_color="#ef4444")
                
                fig2.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                                   font=dict(color='#cbd5e1', family='Inter'), margin=dict(l=0, r=0, t=10, b=0),
                                   xaxis=dict(showgrid=False, title="Federation Round"), 
                                   yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', gridwidth=1, zeroline=False))
                st.plotly_chart(fig2, use_container_width=True, key=f"priv_chart_{time.time()}")

    else:
        metrics_container.empty()
        charts_container.markdown('''
        <div style="text-align: center; padding: 100px; background: rgba(15, 23, 42, 0.4); border-radius: 20px; border: 1px dashed rgba(255, 255, 255, 0.2);">
            <div class="pulse-indicator" style="margin-bottom: 20px; width: 20px; height: 20px;"></div>
            <h2 style="font-family: Outfit; font-weight: 300; color: #94a3b8;">Awaiting Telemetry Feed...</h2>
            <p style="color: #64748b;">Execute <code style="color: #2dd4bf; background: rgba(0,0,0,0.5); padding: 5px; border-radius: 5px;">python main.py</code> in the terminal to initiate the secure decentralized learning cluster.</p>
        </div>
        ''', unsafe_allow_html=True)

    time.sleep(1.5) # Refresh rate
