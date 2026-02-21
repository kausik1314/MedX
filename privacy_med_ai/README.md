<div align="center">
  <img src="medx_logo.png" alt="MedX Premium Logo" width="200" />
  <h1>🏥 MedX DP-AI Framework</h1>
  <p><strong>Maximum-Level Privacy-Preserving Federated Medical Image Classification</strong></p>
  
  [![Python version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
  [![Status](https://img.shields.io/badge/status-active-brightgreen.svg)]()
  [![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
</div>

---

## ✨ Overview

**MedX** is a state-of-the-art secure federated learning system engineered for medical image classification using **Differential Privacy (DP)**. Designed for scale and maximum security, it seamlessly orchestrates multiple hospital nodes collaborating to train a global deep learning model—without ever exposing raw, sensitive patient data.

## 🚀 Key Features

- **Decentralized AI**: Powered by [Flower](https://flower.ai/), seamlessly managing high-performance client-server communication.
- **Cryptographic Privacy**: Integrates [Opacus](https://opacus.ai/) for rigorous Differential Privacy (DP-SGD), providing provable protections against model inversion attacks.
- **Secure Aggregation**: Implements enterprise-grade encryption simulation for model updates.
- **Real-Time Analytics**: Built-in glassmorphism web dashboard for tracking accuracy, global loss, and exactly tracking the privacy budget ($\epsilon$).

## 🏗️ Architecture

The MedX framework guarantees that local data never leaves the hospital environment. Updates are noise-injected, clipped, and encrypted before aggregation.

```mermaid
graph TD
    classDef server fill:#0f172a,stroke:#818cf8,stroke-width:2px,color:#fff;
    classDef client fill:#1e1b4b,stroke:#c084fc,stroke-width:2px,color:#fff;
    classDef data fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#fff;
    
    Server[🌐 Global Aggregation Server]:::server
    
    subgraph "🏥 Hospital Node Alpha (Secure Enclave)"
        DataA[(Private Medical Data)]:::data
        ModelA[Local AI Engine]:::client
        DP_A[Privacy Injector (DP-SGD)]:::client
        Enc_A[Encryption Layer]:::client
        
        DataA -->|Train| ModelA
        ModelA -->|Compute Gradients| DP_A
        DP_A -->|Noisy Updates| Enc_A
    end

    subgraph "🏥 Hospital Node Beta (Secure Enclave)"
        DataB[(Private Medical Data)]:::data
        ModelB[Local AI Engine]:::client
        DP_B[Privacy Injector (DP-SGD)]:::client
        Enc_B[Encryption Layer]:::client
        
        DataB -->|Train| ModelB
        ModelB -->|Compute Gradients| DP_B
        DP_B -->|Noisy Updates| Enc_B
    end

    Server -->|Sync Weights| ModelA
    Server -->|Sync Weights| ModelB
    
    Enc_A -->|Encrypted Transmission| Server
    Enc_B -->|Encrypted Transmission| Server
```

## 🛠️ Installation & Quick Start

### 1. Environment Setup

Create an isolated virtual environment to protect dependencies:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launching the Simulation

Our high-fidelity premium CLI features real-time progress indicators:
```bash
python main.py --config config.yaml
```

### 3. Analytics Command Center

Monitor your decentralized network live through our modern, responsive dashboard:
```bash
streamlit run dashboard.py
```

## 📊 Analytics & Reporting

- **Dashboard**: Run `dashboard.py` for real-time, interactive insights.
- **Research**: Read our detailed academic write-up at `research_report.md`.
- **Static Assets**: Automatically generated plots are saved to `evaluation/`.

## ⚙️ Configuration
Fine-tune the simulation by adjusting the hyperparameters in `config.yaml`:
- **Rounds**: `experiment.rounds`
- **Clients**: `experiment.num_clients`
- **Privacy Budget**: `privacy.target_epsilon`

---
> **Note**: This is a simulated environment created for research and demonstration purposes.
