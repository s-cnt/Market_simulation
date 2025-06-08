# Market Simulation – Real-Time Trade Simulator

A high-performance Streamlit application that calculates the true cost of executing large cryptocurrency orders in real time (slippage, fees, market impact) using full depth-of-book data from OKX.

## Features
- Live WebSocket ingestion of level-2 order-book updates  
- Modular models: **Slippage**, **Maker/Taker**, **Fee**, **Almgren-Chriss impact**  
- Performance dashboard (latency, CPU, memory, throughput)  
- Detailed DEBUG-level logging and PDF documentation generator  

## Quick Start
```bash
git clone [https://github.com/s-cnt/Market_simulation.git](https://github.com/s-cnt/Market_simulation.git)
cd Market_simulation
pip install -r requirements.txt
streamlit run streamlit_app.py

**PROJECT LAYOUT--**

myproject/
├─ streamlit_app.py            ← main UI
├─ trade_simulator/            ← core package
│   ├─ websocket_client.py
│   └─ models/
├─ models/                     ← standalone ML / regression models
├─ utils/                      ← logging & performance helpers
├─ advanced_documentation_pdf.py
└─ requirements.txt





