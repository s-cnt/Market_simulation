# Edit the file to ensure the function exists
#at > /home/sushant/myproject/trade_simulator/models/almgren_chriss.py << 'EOF'
import numpy as np
import matplotlib.pyplot as plt

# Utility functions for market impact
def temporary_impact(volume, alpha, eta):
    return eta * volume ** alpha

def permanent_impact(volume, beta, gamma):
    return gamma * volume ** beta

def hamiltonian(inventory, sell_amount, risk_aversion, alpha, beta, gamma, eta, volatility=0.3, time_step=0.5):
    """
    Hamiltonian equation. To be minimized through dynamic programming.
    """
    temp_impact = risk_aversion * sell_amount * permanent_impact(sell_amount / time_step, beta, gamma)
    perm_impact = risk_aversion * (inventory - sell_amount) * time_step * temporary_impact(sell_amount / time_step, alpha, eta)
    exec_risk = 0.5 * (risk_aversion ** 2) * (volatility ** 2) * time_step * ((inventory - sell_amount) ** 2)
    return temp_impact + perm_impact + exec_risk

# Dynamic programming function
def optimal_execution(time_steps, total_shares, risk_aversion, alpha, beta, gamma, eta, plot=False):
    """
    Bellman equation and value iteration for solving the Markov Decision Process of the Almgren-Chriss model.
    
    Parameters:
    - time_steps: Number of time intervals
    - total_shares: Total number of shares to be liquidated
    - risk_aversion: Risk aversion parameter
    """
    
    # Initialization
    value_function = np.zeros((time_steps, total_shares + 1), dtype="float64")
    best_moves = np.zeros((time_steps, total_shares + 1), dtype="int")
    inventory_path = np.zeros((time_steps, 1), dtype="int")
    inventory_path[0] = total_shares
    optimal_trajectory = []
    time_step_size = 0.5
    
    # Terminal condition
    for shares in range(total_shares + 1):
        value_function[time_steps - 1, shares] = np.exp(shares * temporary_impact(shares / time_step_size, alpha, eta))
        best_moves[time_steps - 1, shares] = shares
    
    # Backward induction
    for t in range(time_steps - 2, -1, -1):
        for shares in range(total_shares + 1):
            best_value = value_function[t + 1, 0] * np.exp(hamiltonian(shares, shares, risk_aversion, alpha, beta, gamma, eta))
            best_share_amount = shares
            for n in range(shares):
                current_value = value_function[t + 1, shares - n] * np.exp(hamiltonian(shares, n, risk_aversion, alpha, beta, gamma, eta))
                if current_value < best_value:
                    best_value = current_value
                    best_share_amount = n
            value_function[t, shares] = best_value
            best_moves[t, shares] = best_share_amount
    
    # Optimal trajectory
    for t in range(1, time_steps):
        inventory_path[t] = inventory_path[t - 1] - best_moves[t, inventory_path[t - 1]]
        optimal_trajectory.append(best_moves[t, inventory_path[t - 1]])
    
    optimal_trajectory = np.asarray(optimal_trajectory)
    
    # Plot results
    if plot:
        plt.figure(figsize=(7, 5))
        plt.plot(inventory_path, color='blue', lw=1.5)
        plt.xlabel('Trading periods')
        plt.ylabel('Number of shares')
        plt.grid(True)
        plt.show()
    
    return value_function, best_moves, inventory_path, optimal_trajectory

# THIS IS THE FUNCTION BEING IMPORTED IN main.py
def calculate_market_impact(orderbook_data, quantity, risk_aversion=0.01):
    """
    Calculate market impact using the Almgren-Chriss model
    
    Args:
        orderbook_data: Current orderbook state (with bids and asks)
        quantity: Order size to execute (in base currency units)
        risk_aversion: Risk aversion parameter
        
    Returns:
        Dictionary with market impact metrics
    """
    # Model parameters
    temp_impact_alpha = 1
    perm_impact_beta = 1
    perm_impact_gamma = 0.05
    temp_impact_eta = 0.05
    
    # Get current price from orderbook
    price = float(orderbook_data.get('bids', [])[0][0]) if orderbook_data.get('bids') else 0
    
    # Convert quantity to shares (position units)
    shares = int(quantity)
    
    # Calculate optimal execution
    _, _, inventory_path, _ = optimal_execution(
        51, shares, risk_aversion, 
        temp_impact_alpha, perm_impact_beta, 
        perm_impact_gamma, temp_impact_eta,
        plot=False
    )
    
    # Return impact metrics
    return {
        "permanent_impact": perm_impact_gamma * shares,
        "temporary_impact": temp_impact_eta * shares,
        "total_impact_usd": (perm_impact_gamma + temp_impact_eta) * shares * price,
        "impact_percentage": (perm_impact_gamma + temp_impact_eta) * 100  # as percentage
    }
