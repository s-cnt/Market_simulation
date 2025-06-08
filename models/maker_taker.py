import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import logging

class MakerTakerModel:
    """
    Model for predicting the maker/taker proportion of an order.
    
    Uses logistic regression to estimate the probability of an order
    being filled as a maker vs taker.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("maker_taker_model")
        self.model = LogisticRegression()
        
        # Historical data for model training
        self.features_history = []
        self.min_data_points = 15  # Minimum data points before modeling
        
    def extract_features(self, orderbook, quantity_usd, price, volatility):
        """Extract features for maker/taker prediction."""
        try:
            # Calculate price levels needed to fill the order
            side_key = "asks"  # For buy orders
            side_data = np.array(orderbook[side_key], dtype=float)
            
            # Convert USD quantity to asset quantity
            quantity = quantity_usd / price
            
            # Calculate cumulative volume
            prices = side_data[:, 0]
            volumes = side_data[:, 1]
            cumulative_volume = np.cumsum(volumes)
            
            # How many levels needed to fill the order
            levels_needed = np.searchsorted(cumulative_volume, quantity)
            if levels_needed >= len(cumulative_volume):
                levels_needed = len(cumulative_volume) - 1
                
            levels_ratio = levels_needed / len(cumulative_volume)
            
            # Calculate volume at best price
            best_price_volume = volumes[0] if len(volumes) > 0 else 0
            volume_at_best_ratio = best_price_volume / quantity if quantity > 0 else 0
            
            # Calculate spread
            best_bid = float(orderbook["bids"][0][0]) if len(orderbook["bids"]) > 0 else 0
            best_ask = float(orderbook["asks"][0][0]) if len(orderbook["asks"]) > 0 else 0
            spread = (best_ask - best_bid) / price if price > 0 else 0
            
            # Create feature vector
            features = {
                'levels_ratio': levels_ratio,
                'volume_at_best_ratio': volume_at_best_ratio,
                'spread': spread,
                'volatility': volatility if volatility is not None else 0.02,
                'order_size_normalized': min(1.0, quantity_usd / 1000)  # Normalize by $1000
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting maker/taker features: {str(e)}")
            # Return default features as fallback
            return {
                'levels_ratio': 0.2,
                'volume_at_best_ratio': 0.3,
                'spread': 0.001,
                'volatility': volatility if volatility is not None else 0.02,
                'order_size_normalized': 0.1
            }
    
    def predict_maker_taker(self, orderbook, quantity_usd, price, volatility):
        """
        Predict the maker/taker proportion for the given order.
        
        Returns:
            Tuple (maker_pct, taker_pct) as percentages
        """
        try:
            # Extract features
            features = self.extract_features(orderbook, quantity_usd, price, volatility)
            
            # Store features for future training
            self.features_history.append(features)
            if len(self.features_history) > 100:  # Keep history limited
                self.features_history = self.features_history[-100:]
            
            # If we don't have enough data points, use heuristic
            if len(self.features_history) < self.min_data_points:
                # Simple heuristic based on order size vs. best price volume
                volume_ratio = features['volume_at_best_ratio']
                
                # Higher ratio means more likely to be filled at best price (as taker)
                taker_pct = min(100, volume_ratio * 100)
                maker_pct = 100 - taker_pct
                
                # Adjust for volatility - higher volatility means more likely to be taker
                vol_adjustment = features['volatility'] * 100
                taker_pct = min(100, taker_pct + vol_adjustment)
                maker_pct = 100 - taker_pct
                
                return (maker_pct, taker_pct)
            
            # Otherwise use logistic regression
            # For simulation, we'll generate synthetic targets
            X = pd.DataFrame(self.features_history)
            
            # Higher volatility, spread, and order size -> more likely to be taker
            # More volume at best price -> more likely to be taker
            synthetic_taker_prob = (
                0.3 * X['volatility'] + 
                0.2 * X['spread'] + 
                0.4 * X['order_size_normalized'] + 
                0.1 * X['volume_at_best_ratio']
            )
            
            # Ensure probabilities are between 0 and 1
            synthetic_taker_prob = np.clip(synthetic_taker_prob, 0.1, 0.9)
            
            # Generate binary outcomes for training
            y = (np.random.random(len(synthetic_taker_prob)) < synthetic_taker_prob).astype(int)
            
            # Fit model
            feature_cols = ['levels_ratio', 'volume_at_best_ratio', 'spread', 
                           'volatility', 'order_size_normalized']
            self.model.fit(X[feature_cols], y)
            
            # Predict for current features
            current_X = pd.DataFrame([features])[feature_cols]
            taker_prob = self.model.predict_proba(current_X)[0, 1]
            
            taker_pct = taker_prob * 100
            maker_pct = 100 - taker_pct
            
            return (maker_pct, taker_pct)
            
        except Exception as e:
            self.logger.error(f"Error predicting maker/taker proportion: {str(e)}")
            return (20, 80)  # Default 20% maker, 80% taker as fallback