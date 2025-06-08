"""Generate Trade Simulator documentation PDF.
Run: python generate_documentation_pdf.py
Requires: reportlab (pip install reportlab)
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from textwrap import wrap

doc_text = """
Trade Simulator Documentation

Overview
The Trade Simulator is a high-performance application designed to estimate transaction costs and market impact for cryptocurrency trading using real-time market data. The system connects to WebSocket endpoints that stream full L2 orderbook data for cryptocurrency exchanges, particularly OKX SPOT exchange.

System Architecture
The application is built on the following components:
1. UI Layer: Implemented using Streamlit for a responsive and interactive user interface
2. WebSocket Client: Connects to market data streams and processes real-time orderbook data
3. Simulation Models: Components for estimating slippage, market impact, and fee calculations
4. Performance Monitoring: Tools for tracking processing latency and system resource usage

User Interface
The UI provides:
- Left panel for input parameters (exchange, pair, quantity, volatility, fee tier)
- Right panel for output parameters (slippage, fees, market impact, net cost, maker/taker proportion, latency)
- Performance metrics visualization

Models and Algorithms
SlippageModel, MakerTakerModel, FeeCalculator, and Almgren-Chriss Market Impact model are implemented to provide accurate estimates.

Error Handling and Logging
Comprehensive logging with DEBUG level, file and console handlers, and a utility function to capture exceptions with full tracebacks.

Performance Optimization
Memory bounded histories, thread-safe locks, and efficient data structures ensure high throughput.

Usage Guide
1. streamlit run streamlit_app.py
2. Enter parameters and run simulation.

Monitoring and Diagnostics
Logs are written to simulator_web.log; performance metrics are visualized in the UI.
"""

def generate_pdf(output_path: str):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    text_obj = c.beginText(0.75 * inch, height - inch)
    text_obj.setFont("Helvetica", 11)

    # Wrap text to fit page width
    for line in doc_text.split("\n"):
        wrapped = wrap(line, 95)
        for wrap_line in wrapped:
            text_obj.textLine(wrap_line)
        if not wrapped:
            text_obj.textLine("")  # empty line for paragraph spacing

    c.drawText(text_obj)
    c.showPage()
    c.save()

if __name__ == "__main__":
    output_file = os.path.join(os.path.dirname(__file__), "trade_simulator_documentation.pdf")
    generate_pdf(output_file)
    print(f"PDF generated at {output_file}")
