import tkinter as tk
from tkinter import ttk
import logging

class OutputPanel:
    """
    Panel for displaying output parameters and results.
    """
    
    def __init__(self, parent):
        self.logger = logging.getLogger("output_panel")
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="Output Parameters", padding=10)
        self.frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Create widgets
        self.create_widgets()
        
    def create_widgets(self):
        """Create all output widgets."""
        # Current Price
        ttk.Label(self.frame, text="Current Price (USD):").grid(row=0, column=0, sticky=tk.W, pady=(5, 0))
        self.price_var = tk.StringVar(value="0.00")
        ttk.Entry(self.frame, textvariable=self.price_var, state="readonly").grid(
            row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Expected Slippage
        ttk.Label(self.frame, text="Expected Slippage:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.slippage_var = tk.StringVar(value="0.00%")
        ttk.Entry(self.frame, textvariable=self.slippage_var, state="readonly").grid(
            row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Expected Fees
        ttk.Label(self.frame, text="Expected Fees:").grid(row=4, column=0, sticky=tk.W, pady=(5, 0))
        self.fees_var = tk.StringVar(value="$0.00")
        ttk.Entry(self.frame, textvariable=self.fees_var, state="readonly").grid(
            row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Expected Market Impact
        ttk.Label(self.frame, text="Expected Market Impact:").grid(row=6, column=0, sticky=tk.W, pady=(5, 0))
        self.impact_var = tk.StringVar(value="0.00%")
        ttk.Entry(self.frame, textvariable=self.impact_var, state="readonly").grid(
            row=7, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Net Cost
        ttk.Label(self.frame, text="Net Cost:").grid(row=8, column=0, sticky=tk.W, pady=(5, 0))
        self.cost_var = tk.StringVar(value="$0.00")
        ttk.Entry(self.frame, textvariable=self.cost_var, state="readonly").grid(
            row=9, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Maker/Taker Proportion
        ttk.Label(self.frame, text="Maker/Taker Proportion:").grid(row=10, column=0, sticky=tk.W, pady=(5, 0))
        self.maker_taker_var = tk.StringVar(value="0% / 100%")
        ttk.Entry(self.frame, textvariable=self.maker_taker_var, state="readonly").grid(
            row=11, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Internal Latency
        ttk.Label(self.frame, text="Internal Latency:").grid(row=12, column=0, sticky=tk.W, pady=(5, 0))
        self.latency_var = tk.StringVar(value="0.00ms")
        ttk.Entry(self.frame, textvariable=self.latency_var, state="readonly").grid(
            row=13, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Configure grid
        self.frame.columnconfigure(0, weight=1)
        
    def update_values(self, slippage_pct, fees_usd, impact_pct, net_cost_usd, 
                      maker_pct, taker_pct, latency_ms, price):
        """Update all output values."""
        try:
            # Format values for display
            self.price_var.set(f"{price:.2f}")
            self.slippage_var.set(f"{slippage_pct:.4f}%")
            self.fees_var.set(f"${fees_usd:.4f}")
            self.impact_var.set(f"{impact_pct:.4f}%")
            self.cost_var.set(f"${net_cost_usd:.4f}")
            self.maker_taker_var.set(f"{maker_pct:.1f}% / {taker_pct:.1f}%")
            self.latency_var.set(f"{latency_ms:.2f}ms")
            
        except Exception as e:
            self.logger.error(f"Error updating output values: {str(e)}")