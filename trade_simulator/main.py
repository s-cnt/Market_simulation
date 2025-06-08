import asyncio
import logging
import tkinter as tk
from ui.app import TradeSimulatorApp
import os
import sys
import argparse
import csv
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simulator.log"),
        logging.StreamHandler()
    ]
)

# Import our Almgren-Chriss model
from trade_simulator.models.almgren_chriss import calculate_market_impact

# Check if running in headless environment
HEADLESS = os.environ.get('DISPLAY', '') == ''

def process_orderbook_update(orderbook_data, quantity, risk_aversion=0.01):
    """Process orderbook update and calculate market impact"""
    logging.info(f"Processing orderbook update for quantity {quantity}")
    
    # Calculate market impact using the Almgren-Chriss model
    impact_results = calculate_market_impact(orderbook_data, quantity, risk_aversion)
    
    logging.info(f"Market impact calculated: {impact_results}")
    return impact_results

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Trade Cost Simulator")
    parser.add_argument("--symbol", default="BTCUSD", help="Trading pair symbol")
    parser.add_argument("--quantity", type=float, default=1.0, help="Order quantity")
    parser.add_argument("--side", choices=["buy", "sell"], default="buy", help="Order side")
    parser.add_argument("--output", default="results.csv", help="Output file for results")
    return parser.parse_args()

def run_headless_simulation(args):
    """Run simulation without GUI and save results to file"""
    logging.info(f"Starting headless simulation for {args.symbol}")
    
    # Initialize models
    from models.slippage import SlippageModel
    from models.market_impact import AlmgrenChrissModel as MarketImpactModel
    from models.maker_taker import MakerTakerModel
    from models.fee_calculator import FeeCalculator
    
    market_impact_model = MarketImpactModel()
    slippage_model = SlippageModel()
    maker_taker_model = MakerTakerModel()
    fee_calculator = FeeCalculator()
    
    # Load sample orderbook data - in real app, this would come from WebSocket
    sample_orderbook = {
        "bids": [["35000.5", "2.5"], ["34999.0", "1.2"]],
        "asks": [["35001.0", "1.8"], ["35002.5", "3.0"]]
    }
    
    # Calculate market impact using our new Almgren-Chriss model
    market_impact_results = process_orderbook_update(sample_orderbook, args.quantity)
    market_impact = market_impact_results["total_impact_usd"]
    
    # Calculate other costs (using existing models)
    # Using predict_slippage instead of calculate_slippage with proper parameters
    # The method requires orderbook, quantity_usd, price, volatility, side
    price = float(sample_orderbook['bids'][0][0])  # Use best bid price as reference
    volatility = 0.02  # Default volatility value
    slippage = slippage_model.predict_slippage(sample_orderbook, args.quantity, price, volatility, args.side)
    
    # Get maker/taker percentages from the model
    maker_pct, taker_pct = maker_taker_model.predict_maker_taker(sample_orderbook, args.quantity, price, volatility)
    
    # Calculate fees using the correct parameters: fee_tier, quantity_usd, maker_pct, taker_pct
    # Using 'Level 1' as the default fee tier
    fee_tier = "Level 1"
    fees = fee_calculator.calculate_fees(fee_tier, args.quantity, maker_pct, taker_pct)
    total_cost = market_impact + slippage + fees
    
    # Example simulation result
    results = {
        "symbol": args.symbol,
        "quantity": args.quantity,
        "side": args.side,
        "market_impact": market_impact,
        "slippage": slippage,
        "fees": fees,
        "total_cost": total_cost,
        "impact_details": json.dumps(market_impact_results)
    }
    
    # Save results
    with open(args.output, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results.keys())
        writer.writeheader()
        writer.writerow(results)
    
    logging.info(f"Simulation complete. Results saved to {args.output}")
    return results

class OrderbookProcessor:
    """Process orderbook updates and update UI"""
    def __init__(self, app=None):
        self.app = app
    
    def update_from_orderbook(self, orderbook_data):
        # Get parameters from UI or use defaults
        quantity = 1.0
        risk_aversion = 0.01
        
        if self.app and hasattr(self.app, 'input_panel'):
            quantity = self.app.input_panel.get_quantity()
            # Could add UI control for risk aversion
        
        # Calculate market impact
        impact_results = process_orderbook_update(orderbook_data, quantity, risk_aversion)
        
        # Update UI if available
        if self.app and hasattr(self.app, 'output_panel'):
            self.app.output_panel.update_values({
                "Market Impact": f"{impact_results['total_impact_usd']:.2f} USD",
                "Impact %": f"{impact_results['impact_percentage']:.4f}%",
                "Permanent Impact": f"{impact_results['permanent_impact']:.4f}",
                "Temporary Impact": f"{impact_results['temporary_impact']:.4f}"
            })
        
        return impact_results

if __name__ == "__main__":
    # Create parser
    parser = argparse.ArgumentParser(description="Trade Simulator CLI")
    parser.add_argument('--symbol', default='BTCUSD', help='Trading symbol')
    parser.add_argument('--quantity', type=float, default=1.0, help='Order quantity')
    parser.add_argument('--side', choices=['buy', 'sell'], default='buy', help='Order side')
    parser.add_argument('--output', default='results.csv', help='Output file')
    parser.add_argument('--gui', action='store_true', help='Run in GUI mode instead of headless')
    
    # Parse arguments
    args = parser.parse_args()
    
    headless = not args.gui
    
    if headless:
        print("Running in headless mode - GUI disabled")
        run_headless_simulation(args)
    else:
        print("Starting GUI mode")
        # Initialize main application window
        root = tk.Tk()
        app = TradeSimulatorApp(root)
        
        # Create orderbook processor
        orderbook_processor = OrderbookProcessor(app)
        
        # Register orderbook processor with WebSocket client
        app.register_orderbook_callback(orderbook_processor.update_from_orderbook)
        
        # Apply command line arguments if provided
        if hasattr(app, 'input_panel') and hasattr(app.input_panel, 'set_values'):
            app.input_panel.set_values(args.symbol, args.quantity, args.side)
        
        # Start the Tkinter event loop
        root.mainloop()