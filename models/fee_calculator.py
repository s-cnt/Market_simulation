import logging

class FeeCalculator:
    """
    Calculate trading fees based on exchange fee structure and order parameters.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("fee_calculator")
        
        # OKX fee tiers (simplified)
        # Based on: https://www.okx.com/fees
        self.fee_tiers = {
            "Level 1": {"maker": 0.0008, "taker": 0.0010},  # 0.08% maker, 0.10% taker
            "Level 2": {"maker": 0.0006, "taker": 0.0008},  # 0.06% maker, 0.08% taker
            "Level 3": {"maker": 0.0004, "taker": 0.0006},  # 0.04% maker, 0.06% taker
        }
    
    def calculate_fees(self, fee_tier, quantity_usd, maker_pct, taker_pct):
        """
        Calculate expected fees for a given order.
        
        Args:
            fee_tier: The fee tier (Level 1, Level 2, Level 3)
            quantity_usd: Order size in USD
            maker_pct: Percentage of order expected to be filled as maker
            taker_pct: Percentage of order expected to be filled as taker
            
        Returns:
            Total fees in USD
        """
        try:
            # Get fee rates for the tier
            if fee_tier not in self.fee_tiers:
                self.logger.warning(f"Unknown fee tier: {fee_tier}, using Level 1")
                fee_tier = "Level 1"
                
            fee_rates = self.fee_tiers[fee_tier]
            
            # Calculate maker and taker portions
            maker_portion = quantity_usd * (maker_pct / 100)
            taker_portion = quantity_usd * (taker_pct / 100)
            
            # Calculate fees
            maker_fee = maker_portion * fee_rates["maker"]
            taker_fee = taker_portion * fee_rates["taker"]
            
            total_fee = maker_fee + taker_fee
            
            self.logger.debug(f"Calculated fees: ${total_fee:.4f} (maker: ${maker_fee:.4f}, taker: ${taker_fee:.4f})")
            return total_fee
            
        except Exception as e:
            self.logger.error(f"Error calculating fees: {str(e)}")
            return quantity_usd * 0.001  # Default 0.1% fee as fallback