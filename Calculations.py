import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime, timezone
from sklearn.ensemble import RandomForestRegressor
from Fetch_data import Fetching_data
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

fetch_data = Fetching_data()

def black_scholes(underlying_price, strike_price, time_to_expiration, risk_free_rate, implied_volatility, option_type):
    try:
        # Validate inputs
        if time_to_expiration <= 0:
            raise ValueError("Time to expiration must be positive.")
        if implied_volatility <= 0:
            raise ValueError("Implied volatility must be positive.")
        if underlying_price <= 0 or strike_price <= 0:
            raise ValueError("Underlying and strike prices must be positive.")
        if option_type not in ['call', 'put']:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")

        sqrt_time = np.sqrt(time_to_expiration)
        d1 = (np.log(underlying_price / strike_price) + (risk_free_rate + 0.5 * implied_volatility**2) * time_to_expiration) / (implied_volatility * sqrt_time)
        d2 = d1 - implied_volatility * sqrt_time

        if option_type == 'call':
            return underlying_price * norm.cdf(d1) - strike_price * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(d2)
        else:  # option_type == 'put'
            return strike_price * np.exp(-risk_free_rate * time_to_expiration) * norm.cdf(-d2) - underlying_price * norm.cdf(-d1)

    except (ValueError, ZeroDivisionError, FloatingPointError) as e:
        # Log or handle the error as needed
        print(f"Black-Scholes error: {e}")
        return np.nan  # or return 0.0, or raise, depending on your use case

def calculate_profit(current_price, option_price, strike_price, option_type, quantity, is_buy):
    intrinsic_value = max(current_price - strike_price, 0) if option_type == 'call' else max(strike_price - current_price, 0)
    profit = intrinsic_value - option_price if is_buy else option_price - intrinsic_value
    return profit * quantity

def simulate_option_greek(initial_option_price, delta, gamma, theta, vega,
                          underlying_price_change, iv_change, time_change):
    """
    Simulate the change in an option's price using a first- and second-order Taylor expansion
    incorporating the Greeks.

    Parameters:
        initial_option_price (float): The starting option price.
        delta (float): Option delta (price sensitivity to changes in the underlying).
        gamma (float): Option gamma (rate of change of delta).
        theta (float): Option theta (time decay per day).
        vega (float): Option vega (price sensitivity to 1% change in IV).
        underlying_price_change (float): Change in the underlying's price (dollars).
        iv_change (float): Change in implied volatility in percentage points (e.g., 1 for a 1% change).
        time_change (float): Time passed in days.

    Returns:
        float: The estimated new option price.
    """
    # Delta contribution: change in option price due to change in underlying price.
    delta_effect = delta * underlying_price_change

    # Gamma contribution: second-order effect of the underlying's price change.
    gamma_effect = 0.5 * gamma * underlying_price_change**2

    # Theta contribution: time decay (usually negative for long options).
    theta_effect = theta * time_change

    # Vega contribution: change in option price due to a change in implied volatility.
    vega_effect = vega * iv_change

    extrinsic_option_value = initial_option_price + delta_effect + gamma_effect + theta_effect + vega_effect
    return extrinsic_option_value 


def calculate_deribit_option_fee(option_price, underlying_price, contract_size, is_maker):
    """
    Calculate the trading fee for an options contract on Deribit.

    Parameters:
    - option_price (float): The price of the option in BTC or ETH.
    - underlying_price (float): The current price of the underlying asset in USD.
    - contract_size (float): The size of the contract (e.g., 1 BTC or 1 ETH).
    - is_maker (bool): True if the order is a maker order, False if it's a taker order.

    Returns:
    - fee (float): The calculated fee in BTC or ETH.
    """

    # Define the fee rates
    maker_fee_rate = 0.0003  # 0.03%
    taker_fee_rate = 0.0003  # 0.03%
    fee_cap_percentage = 0.125  # 12.5%

    # Determine the applicable fee rate
    fee_rate = maker_fee_rate if is_maker else taker_fee_rate

    # Calculate the initial fee
    initial_fee = fee_rate * underlying_price * contract_size

    # Calculate the fee cap
    fee_cap = fee_cap_percentage * option_price

    # The final fee is the lesser of the initial fee and the fee cap
    final_fee = min(initial_fee, fee_cap)

    return final_fee

def calculate_option_profit(options_data_df,days_ahead_slider, quantity, risk_free_rate,change_in_iv , specific_symbol=None):
    """
    Calculate the profit for all in-the-money (ITM) options using the Black-Scholes model.

    Parameters:
        itm_options_df (pd.DataFrame): DataFrame containing ITM options with columns:
                                        ['symbol', 'strike_price', 'option_type', 'bid_price_usd', 'ask_price_usd', 'bid_iv', 'ask_iv', 'expiration_date']
        index_price_range (np.ndarray): Array of underlying prices to evaluate.
        days_ahead_slider (int): Number of days ahead for the evaluation.
        quantity (float): The quantity of options being considered for profit calculations.
        risk_free_rate (float): The risk-free interest rate to be used in Black-Scholes calculations.
        specific_symbol (str, optional): If provided, filter results by this specific option symbol.

    Returns:
        pd.DataFrame: DataFrame containing option symbols and their corresponding profits.
    """

     # Determine the range starting from the minimum of BTC price and strike price
    #btc_price = get_btcusd_price()
    #start_price = min(btc_price, strike_price)
   # end_price = max(btc_price, strike_price)
    #index_price_range = np.arange(start_price - 20000, end_price + 20000, 1000)  # Adjust based on your requirement

    index_price_range = np.arange(40000, 141000, 1000)
    now_utc = datetime.now(timezone.utc).date()
    results = []

    # Loop through each ITM option
    for _, option in options_data_df.iterrows():
        option_symbol = option['Instrument']
        
        # If a specific symbol is provided, filter before processing
        if specific_symbol is not None and option_symbol != specific_symbol:
            continue

        strike_price = option['Strike Price']
        option_type = option['Option Type']
        bid_price = option['Bid Price (USD)']
        ask_price = option['Ask Price (USD)']
        bid_iv = option['Bid IV'] 
        ask_iv = option['Ask IV'] 


        future_ask_iv = ask_iv / 100 + (change_in_iv  / 100.0)
        future_bid_iv = bid_iv / 100 + (change_in_iv  / 100.0)
        expiration_date_str = option['Expiration Date']
        expiration_date = pd.to_datetime(expiration_date_str).date()  # Ensure it's converted to date

        # Compute total days to expiration (at least 1 to avoid zero)
        time_to_expiration_days = max((expiration_date - now_utc).days, 1)
        remaining_days = time_to_expiration_days - days_ahead_slider
        time_to_expiration_future = max(remaining_days / 365.0, 0.0001)

        for u_price in index_price_range:
            # Calculate the option price using the Black-Scholes model
            mtm_price_buy = black_scholes(u_price, strike_price, time_to_expiration_future, risk_free_rate, future_ask_iv, option_type)
            mtm_price_sell = black_scholes(u_price, strike_price, time_to_expiration_future, risk_free_rate, future_bid_iv, option_type)
            
            # Calculate mark-to-market profits
            mtm_profits_buy = (mtm_price_buy - ask_price) * quantity
            mtm_profits_sell = (bid_price - mtm_price_sell) * quantity

            # ------------------------------------------------------------------------------ 
            # Expiration Profit (Intrinsic-based) 
            # ------------------------------------------------------------------------------ 
            expiration_profits_buy = calculate_profit(u_price, ask_price, strike_price, option_type, quantity, is_buy=True)
            expiration_profits_sell = calculate_profit(u_price, bid_price, strike_price, option_type, quantity, is_buy=False)

            # Store results for this option
            results.append({
                'Underlying Price': u_price,
                'Expiration Profit (BUY)': expiration_profits_buy,
                f'Day {days_ahead_slider} Profit (BUY)': mtm_profits_buy,
                'Expiration Profit (SELL)': expiration_profits_sell,
                f'Day {days_ahead_slider} Profit (SELL)': mtm_profits_sell,
            })

    # Create a DataFrame from results
    results_df = pd.DataFrame(results)

    # Return the results DataFrame without further grouping if specific symbol is provided
    if specific_symbol is not None:
        return results_df 


    return results_df


def calculate_totals_for_options(df):
    """
    Calculate total options, total size, and total entry values from the given DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing options data with 'Size' and 'Entry Value' columns.

    Returns:
        tuple: A tuple containing:
            - total_options (int): Total number of rows in the DataFrame.
            - total_amount (float): Total size of options from the 'Size' column.
            - total_entry_values (float): Total entry values from the 'Entry Value' column.
    """
    total_options = df.shape[0]  # Total number of rows

    total_amount = df['Size'].sum() if 'Size' in df.columns else 0  # Total size from 'Size' column
    
    if 'Entry_Value' in df.columns:
        df['Entry_Value'] = pd.to_numeric(df['Entry_Value'], errors='coerce').fillna(0)
        total_entry_values = df['Entry_Value'].sum()
    else:
        total_entry_values = 0
    return (total_options, 
            round(total_amount, 2), 
            round(total_entry_values, 2))


def get_most_traded_instruments(df):
    """
    Returns the ten most traded instruments, their total sizes, and counts of BUY and SELL orders
    as a combined DataFrame.
    """
    # Group by 'Instrument' and count the number of trades
    trade_counts = df.groupby('Instrument').size().reset_index(name='Trade Count')
    
    # Sort the DataFrame by trade count in descending order and select the top 10
    most_traded = trade_counts.sort_values(by='Trade Count', ascending=False).head(10)
    
    # Count the BUY and SELL occurrences for the most traded instruments
    buy_sell_counts = df[df['Instrument'].isin(most_traded['Instrument'])].groupby(['Instrument', 'Side']).size().unstack(fill_value=0)

    # Ensure that both BUY and SELL columns exist, defaulting to 0 if they don't
    buy_sell_counts = buy_sell_counts.reindex(columns=['BUY', 'SELL'], fill_value=0)

    # Combine the most traded DataFrame with the buy/sell counts
    combined_results = most_traded.merge(buy_sell_counts, on='Instrument', how='left')

    # Prepare for fetching option data
    option_list = combined_results['Instrument'].astype(str).tolist()  # Ensure it is a list of strings
    top_options_chains = fetch_data.fetch_option_chain(option_list)
    
    # Cleanup options data (if you want to keep this logic)
    top_options_chains = top_options_chains.drop(columns=['Option_Type', 'Strike_Price', 'Gamma', 
                                                           'Expiration_Date', 'Last_Price_USD', 
                                                           'open_interest', 'Total_Traded_Volume', 
                                                           'Monetary_Volume'], errors='ignore')
    new_order = ['Instrument', 'Ask_Price_USD', 'Bid_Price_USD', 'Delta', 'Theta', 'Vega', 'Ask_IV', 'Bid_IV' , 'Probability_Percent']
    if top_options_chains is not None:
        top_options_chains = top_options_chains[new_order]

    return combined_results, top_options_chains

def calculate_sums_of_public_trades_profit(results_df):
    """
    Calculate the sum of profits against underlying price and find
    the underlying price where the sum of profits is negative.
    
    Returns:
        summed_df: DataFrame with underlying prices and sums of profits.
        negative_price: Underlying price where the profit sum is negative, or None if it never becomes negative.
    """
    # Calculate the sum of profits for each underlying price
    sums = results_df.drop(columns='Underlying_Price').sum(axis=1)
    
    # Create a DataFrame for the results
    summed_df = pd.DataFrame({
        'Underlying Price': results_df['Underlying_Price'],
        'Sum of Profits': sums
    })
    
    # Find the first underlying price where the sum of profits becomes negative
    negative_index = summed_df[summed_df['Sum of Profits'] < 0].index
    negative_price = None
    
    if not negative_index.empty:
        negative_price = summed_df['Underlying_Price'].iloc[negative_index[0]]
    
    return summed_df, negative_price







