import asyncio
import logging
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox

from trade_simulator.websocket_client import OrderbookWebsocketClient
from ui.input_panel import InputPanel
from ui.output_panel import OutputPanel
from models.market_impact import AlmgrenChrissModel as MarketImpactModel
from models.slippage import SlippageModel  # No "as" needed
from models.maker_taker import MakerTakerModel
from models.fee_calculator import FeeCalculator
from utils.performance import PerformanceMonitor


class TradeSimulatorApp:
    """
    Main application class for the Trade Simulator.
    """
    
    def __init__(self, root):
        self.logger = logging.getLogger("simulator_app")
        
        # Set up main window
        self.root = root
        self.root.title("Trade Cost Simulator")
        self.root.geometry("900x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initialize models
        self.market_impact_model = MarketImpactModel()
        self.slippage_model = SlippageModel()
        self.maker_taker_model = MakerTakerModel()
        self.fee_calculator = FeeCalculator()
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize variables
        self.running = False
        self.websocket_thread = None
        self.websocket_loop = None
        self.websocket_client = None
        self.last_orderbook = None
        self.last_mid_price = 0
        
        # Create main frames
        self.create_ui()
        
        self.logger.info("Application initialized")
        
    def create_ui(self):
        """Create and configure the user interface."""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                              anchor=tk.W, relief=tk.SUNKEN, padding=(5, 2))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Create input and output panels
        self.input_panel = InputPanel(main_frame, 
                                     self.on_start_simulation, 
                                     self.on_stop_simulation)
        self.output_panel = OutputPanel(main_frame)
        
    def on_start_simulation(self):
        """Start the simulation process."""
        if self.running:
            return
            
        # Get parameters
        params = self.input_panel.get_parameters()
        exchange = params['exchange']
        asset = params['asset']
        
        # Set up WebSocket URL based on exchange
        if exchange == "OKX":
            url = "wss://ws.okx.com:8443/ws/v5/public"
            channel = f"books-l2-tbt"
            symbol = asset  # e.g., "BTC-USDT"
        else:
            messagebox.showerror("Error", f"Unsupported exchange: {exchange}")
            return
        
        # Update UI
        self.status_var.set("Starting simulation...")
        self.input_panel.set_running(True)
        self.running = True
        
        # Reset performance monitor
        self.performance_monitor.reset()
        
        # Start WebSocket thread
        self.websocket_thread = threading.Thread(
            target=self.run_websocket,
            args=(url, symbol),
            daemon=True
        )
        self.websocket_thread.start()
        
        self.logger.info(f"Simulation started for {exchange}:{asset}")
        
    def on_stop_simulation(self):
        """Stop the simulation process."""
        if not self.running:
            return
            
        self.status_var.set("Stopping simulation...")
        
        # Stop the WebSocket client
        if self.websocket_loop and self.websocket_client:
            asyncio.run_coroutine_threadsafe(
                self.websocket_client.disconnect(),
                self.websocket_loop
            )
            
        self.running = False
        self.input_panel.set_running(False)
        self.status_var.set("Simulation stopped")
        
        self.logger.info("Simulation stopped")
        
    def run_websocket(self, url, symbol):
        """
        Run the WebSocket client in a separate thread.
        This creates an event loop for asyncio operations.
        """
        try:
            # Create new event loop for this thread
            self.websocket_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.websocket_loop)
            
            # Create WebSocket client
            self.websocket_client = OrderbookWebsocketClient(
                url=url,
                symbol=symbol,
                callback=self.process_orderbook
            )
            
            # Start listening for messages
            self.websocket_loop.run_until_complete(self.websocket_client.listen())
            
        except Exception as e:
            self.logger.error(f"WebSocket thread error: {str(e)}")
            
        finally:
            self.websocket_loop.close()
            self.logger.info("WebSocket thread terminated")
            
    async def process_orderbook(self, data, start_time):
        """
        Process incoming orderbook data and update the UI.
        
        This is called by the WebSocket client when new data arrives.
        """
        try:
            # Record tick for performance monitoring
            self.performance_monitor.record_tick()
            
            # Save orderbook for later use
            self.last_orderbook = data
            
            # Calculate mid price
            best_bid = float(data["bids"][0][0]) if data["bids"] else 0
            best_ask = float(data["asks"][0][0]) if data["asks"] else 0
            mid_price = (best_bid + best_ask) / 2
            self.last_mid_price = mid_price
            
            # Get current parameters
            params = self.input_panel.get_parameters()
            
            # Process end time for this stage
            process_end_time = time.time()
            processing_latency = (process_end_time - start_time) * 1000  # ms
            self.performance_monitor.record_processing_latency(processing_latency)
            
            # Calculate all outputs
            ui_start_time = time.time()
            
            # Only update UI from the main thread
            # We'll use tkinter's after method to schedule the update
            self.root.after(0, lambda: self.update_ui(params, data, mid_price, ui_start_time, start_time))
            
        except Exception as e:
            self.logger.error(f"Error processing orderbook: {str(e)}")
            
    def update_ui(self, params, orderbook, price, ui_start_time, start_time):
        """Update the UI with calculated values."""
        try:
            # Calculate all metrics
            quantity_usd = params['quantity']
            volatility = params['volatility']
            fee_tier = params['fee_tier']
            
            # 1. Maker/Taker proportion
            maker_pct, taker_pct = self.maker_taker_model.predict_maker_taker(
                orderbook, quantity_usd, price, volatility
            )
            
            # 2. Calculate slippage
            slippage_pct = self.slippage_model.predict_slippage(
                orderbook, quantity_usd, price, volatility
            )
            
            # 3. Calculate market impact
            impact_pct = self.market_impact_model.calculate_market_impact(
                orderbook, quantity_usd, price, side="buy"
            )
            
            # 4. Calculate fees
            fees_usd = self.fee_calculator.calculate_fees(
                fee_tier, quantity_usd, maker_pct, taker_pct
            )
            
            # 5. Calculate net cost
            slippage_usd = (slippage_pct / 100) * quantity_usd
            impact_usd = (impact_pct / 100) * quantity_usd
            net_cost_usd = slippage_usd + fees_usd + impact_usd
            
            # 6. Get latency
            latency_ms = self.performance_monitor.get_current_latency()
            
            # Update the UI
            self.output_panel.update_values(
                slippage_pct=slippage_pct,
                fees_usd=fees_usd,
                impact_pct=impact_pct,
                net_cost_usd=net_cost_usd,
                maker_pct=maker_pct,
                taker_pct=taker_pct,
                latency_ms=latency_ms,
                price=price
            )
            
            # Record UI update latency
            ui_end_time = time.time()
            ui_latency = (ui_end_time - ui_start_time) * 1000  # ms
            self.performance_monitor.record_ui_update_latency(ui_latency)
            
            # Record end-to-end latency
            end_to_end_latency = (ui_end_time - start_time) * 1000  # ms
            self.performance_monitor.record_end_to_end_latency(end_to_end_latency)
            
            # Update status
            tick_rate = self.performance_monitor.get_tick_rate()
            timestamp = orderbook.get("timestamp", datetime.now().isoformat())
            self.status_var.set(f"Processing data | Rate: {tick_rate:.1f} ticks/s | Last update: {timestamp}")
            
        except Exception as e:
            self.logger.error(f"Error updating UI: {str(e)}")
            
    def on_closing(self):
        """Handle window closing event."""
        if self.running:
            self.on_stop_simulation()
            
        self.root.destroy()