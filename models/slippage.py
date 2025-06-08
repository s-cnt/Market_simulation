import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor
import logging

class SlippageModel:
    """
    Model for predicting execution slippage based on orderbook data.
    
    Uses either linear regression or quantile regression to estimate slippage.
    """
    
    def __init__(self, model_type="linear"):
        self.logger = logging.getLogger("slippage_model")
        self.model_type = model_type
        
        # Initialize model based on type
        if model_type == "linear":
            self.model = LinearRegression()
        elif model_type == "quantile":
            self.model = QuantileRegressor(quantile=0.75, alpha=0.5)
        else:
            self.logger.warning(f"Unknown model type: {model_type}, defaulting to linear")
            self.model = LinearRegression()
            
        # Historical data for online learning
        self.features_history = []
        self.min_data_points = 10  # Minimum data points before starting to predict
        
        # Features to use for prediction
        self.feature_names = [
            'spread_pct', 'depth_imbalance', 'volatility', 
            'order_size_relative', 'order_count_ratio'
        ]
        
    def extract_features(self, orderbook, quantity_usd, price, volatility):
        """Extract features from orderbook data for slippage prediction."""
        try:
            # Calculate bid-ask spread
            best_bid = float(orderbook["bids"][0][0])
            best_ask = float(orderbook["asks"][0][0])
            spread = best_ask - best_bid
            spread_pct = spread / price
            
            # Calculate depth imbalance (ask depth vs bid depth)
            bid_depth = sum(float(level[1]) for level in orderbook["bids"])
            ask_depth = sum(float(level[1]) for level in orderbook["asks"])
            total_depth = bid_depth + ask_depth
            depth_imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
            
            # Calculate order size relative to depth
            order_size_relative = quantity_usd / (price * total_depth) if total_depth > 0 else 0
            
            # Calculate order count ratio (number of orders on each side)
            bid_orders = len(orderbook["bids"])
            ask_orders = len(orderbook["asks"])
            order_count_ratio = bid_orders / ask_orders if ask_orders > 0 else 1
            
            # Create feature vector
            features = {
                'spread_pct': spread_pct,
                'depth_imbalance': depth_imbalance,
                'volatility': volatility,
                'order_size_relative': order_size_relative,
                'order_count_ratio': order_count_ratio
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            # Return default features as fallback
            return {
                'spread_pct': 0.001,
                'depth_imbalance': 0,
                'volatility': volatility if volatility is not None else 0.02,
                'order_size_relative': 0.01,
                'order_count_ratio': 1.0
            }
    
    def predict_slippage(self, orderbook, quantity_usd, price, volatility, side="buy"):
        """
        Predict execution slippage based on orderbook data and order parameters.
        
        In the initial phase (before having enough data for model training),
        uses a simple rule-based approach.
        """
        try:
            # Extract features from current orderbook
            features = self.extract_features(orderbook, quantity_usd, price, volatility)
            
            # Store features for future model training
            self.features_history.append(features)
            if len(self.features_history) > 100:  # Keep only recent data
                self.features_history = self.features_history[-100:]
            
            # If we don't have enough data for modeling, use rule-based approach
            if len(self.features_history) < self.min_data_points:
                # Simple rule-based slippage estimate
                base_slippage = features['spread_pct'] / 2  # Half the spread
                size_factor = 1 + 5 * features['order_size_relative']  # Size impact
                
                # Adjust for side and imbalance
                if side.lower() == "buy":
                    imbalance_factor = 1 + (0.5 * -features['depth_imbalance'])
                else:  # sell
                    imbalance_factor = 1 + (0.5 * features['depth_imbalance'])
                    
                slippage = base_slippage * size_factor * imbalance_factor
                return max(0, slippage * 100)  # Return as percentage
            
            # Otherwise use our regression model
            # For simplicity, we'll use a synthetic target for training
            # In production, you'd use actual observed slippage
            X = pd.DataFrame(self.features_history)
            
            # Synthetic target: combination of spread and size impact
            y = X['spread_pct'] / 2 + X['order_size_relative'] * X['spread_pct'] * 5
            y += 0.0001 * np.random.randn(len(y))  # Add some noise
            
            # Fit the model
            self.model.fit(X[self.feature_names], y)
            
            # Predict slippage for current features
            current_X = pd.DataFrame([features])[self.feature_names]
            predicted_slippage = self.model.predict(current_X)[0]
            
            return max(0, predicted_slippage * 100)  # Return as percentage
            
        except Exception as e:
            self.logger.error(f"Error predicting slippage: {str(e)}")
            return 0.05  # Default 0.05% slippage as fallback