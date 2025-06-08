import numpy as np
import pandas as pd
import logging

class AlmgrenChrissModel:
    """
    Implementation of the Almgren-Chriss market impact model.
    
    The model estimates market impact based on order size, market volatility,
    and order book depth.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("market_impact_model")
        # Model parameters (can be calibrated based on historical data)
        self.temporary_impact_factor = 0.1
        self.permanent_impact_factor = 0.05
        # Volatility will be set from input
        self.volatility = None
        
    def update_params(self, volatility):
        """Update model parameters with the latest values."""
        self.volatility = volatility
        self.logger.debug(f"Updated volatility to {volatility}")
        
    def calculate_market_impact(self, orderbook, quantity_usd, price, side="buy"):
        """
        Calculate the expected market impact using the Almgren-Chriss model.
        
        Args:
            orderbook: Dictionary containing bids and asks
            quantity_usd: Order size in USD
            price: Current mid price
            side: Trading side ('buy' or 'sell')
            
        Returns:
            Market impact as a percentage of the price
        """
        try:
            # Convert USD quantity to asset quantity
            quantity = quantity_usd / price
            
            # Extract orderbook data
            if side.lower() == "buy":
                book_side = np.array(orderbook["asks"], dtype=float)
            else:
                book_side = np.array(orderbook["bids"], dtype=float)
                
            # Extract prices and volumes
            prices = book_side[:, 0]
            volumes = book_side[:, 1]
            
            # Calculate liquidity in the book (market depth)
            depth = np.sum(volumes)
            if depth == 0:
                self.logger.warning("Order book depth is zero, using default impact estimate")
                return 0.001  # Default 0.1% impact
                
            # Calculate market impact using Almgren-Chriss formula
            # I = temporary_impact * (Q/V) + permanent_impact * Q * σ
            # where Q is quantity, V is market depth, σ is volatility
            
            # Temporary impact - immediate price movement due to order execution
            temp_impact = self.temporary_impact_factor * (quantity / depth)
            
            # Permanent impact - lasting price change due to information signaling
            perm_impact = 0
            if self.volatility is not None:
                perm_impact = self.permanent_impact_factor * quantity * self.volatility
                
            total_impact = temp_impact + perm_impact
            
            # Convert impact to percentage
            impact_percentage = total_impact * 100
            
            self.logger.debug(f"Calculated market impact: {impact_percentage:.4f}%")
            return impact_percentage
            
        except Exception as e:
            self.logger.error(f"Error calculating market impact: {str(e)}")
            return 0.002  # Default 0.2% impact as fallback