
## 9. `ui/input_panel.py` - Input Panel


import tkinter as tk
from tkinter import ttk
import logging

class InputPanel:
    """
    Panel for displaying and managing input parameters.
    """
    
    def __init__(self, parent, start_callback, stop_callback):
        self.logger = logging.getLogger("input_panel")
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="Input Parameters", padding=10)
        self.frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # Store callbacks
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        
        # Create widgets
        self.create_widgets()
        
    def create_widgets(self):
        """Create all input widgets."""
        # Exchange (fixed to OKX for this assignment)
        ttk.Label(self.frame, text="Exchange:").grid(row=0, column=0, sticky=tk.W, pady=(5, 0))
        self.exchange_var = tk.StringVar(value="OKX")
        ttk.Entry(self.frame, textvariable=self.exchange_var, state="readonly").grid(
            row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Spot Asset
        ttk.Label(self.frame, text="Spot Asset:").grid(row=2, column=0, sticky=tk.W, pady=(5, 0))
        self.asset_var = tk.StringVar(value="BTC-USDT")
        assets = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BTC-USDT-SWAP"]
        ttk.Combobox(self.frame, textvariable=self.asset_var, values=assets).grid(
            row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Order Type (fixed to market for this assignment)
        ttk.Label(self.frame, text="Order Type:").grid(row=4, column=0, sticky=tk.W, pady=(5, 0))
        self.order_type_var = tk.StringVar(value="market")
        ttk.Entry(self.frame, textvariable=self.order_type_var, state="readonly").grid(
            row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Quantity (USD)
        ttk.Label(self.frame, text="Quantity (USD):").grid(row=6, column=0, sticky=tk.W, pady=(5, 0))
        self.quantity_var = tk.DoubleVar(value=100.0)
        ttk.Spinbox(self.frame, from_=10, to=1000, textvariable=self.quantity_var, 
                    increment=10).grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Volatility
        ttk.Label(self.frame, text="Volatility:").grid(row=8, column=0, sticky=tk.W, pady=(5, 0))
        self.volatility_var = tk.DoubleVar(value=0.02)
        ttk.Spinbox(self.frame, from_=0.001, to=0.1, textvariable=self.volatility_var, 
                    increment=0.005, format="%.3f").grid(
            row=9, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Fee Tier
        ttk.Label(self.frame, text="Fee Tier:").grid(row=10, column=0, sticky=tk.W, pady=(5, 0))
        self.fee_tier_var = tk.StringVar(value="Level 1")
        fee_tiers = ["Level 1", "Level 2", "Level 3"]
        ttk.Combobox(self.frame, textvariable=self.fee_tier_var, values=fee_tiers).grid(
            row=11, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Buttons
        self.button_frame = ttk.Frame(self.frame)
        self.button_frame.grid(row=12, column=0, sticky=(tk.W, tk.E), pady=20)
        
        self.start_button = ttk.Button(self.button_frame, text="Start Simulation", 
                                      command=self.start_callback)
        self.start_button.pack(side=tk.LEFT, padx=(0, 5))
        
        self.stop_button = ttk.Button(self.button_frame, text="Stop Simulation", 
                                     command=self.stop_callback, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)
        
        # Configure grid
        self.frame.columnconfigure(0, weight=1)
        
    def get_parameters(self):
        """Get all current parameter values."""
        return {
            'exchange': self.exchange_var.get(),
            'asset': self.asset_var.get(),
            'order_type': self.order_type_var.get(),
            'quantity': self.quantity_var.get(),
            'volatility': self.volatility_var.get(),
            'fee_tier': self.fee_tier_var.get()
        }
        
    def set_running(self, is_running):
        """Update the UI based on running state."""
        if is_running:
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
        else:
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)