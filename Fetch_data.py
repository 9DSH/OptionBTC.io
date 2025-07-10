import requests
import logging
from datetime import datetime, date, timezone , timedelta
import pandas as pd
import numpy as np
import os
import time
from fetch_btc_price import get_btcusd_price
from sqlalchemy import select

from db import SessionLocal, OptionChain, PublicTrade , init_db



# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Fetching_data:
    def __init__(self):
        init_db() 
        self._chains_df = None
        self._trades_df = None
        self._load_caches()

    def _load_caches(self):
        """
        Load full tables into pandas DataFrames and cache them.
        """
        session = SessionLocal()
        try:
            # Load option chains
            chains = session.execute(select(OptionChain)).scalars().all()
            chains_data = [{
                'Instrument': c.Instrument,
                'Option_Type': c.Option_Type,
                'Strike_Price': c.Strike_Price,
                'Expiration_Date': c.Expiration_Date,
                'Last_Price_USD': c.Last_Price_USD,
                'Bid_Price_USD': c.Bid_Price_USD,
                'Ask_Price_USD': c.Ask_Price_USD,
                'Bid_IV': c.Bid_IV,
                'Ask_IV': c.Ask_IV,
                'Delta': c.Delta,
                'Gamma': c.Gamma,
                'Theta': c.Theta,
                'Vega': c.Vega,
                'Open_Interest': c.Open_Interest,
                'Total_Traded_Volume': c.Total_Traded_Volume,
                'Monetary_Volume': c.Monetary_Volume,
                'Probability_Percent': c.Probability_Percent,
                'Timestamp': c.Timestamp
            } for c in chains]
            self._chains_df = pd.DataFrame(chains_data)

            # Load public trades
            trades = session.execute(select(PublicTrade)).scalars().all()
            trades_data = [{
                'Trade_ID': t.Trade_ID,
                'Side': t.Side.upper(),
                'Instrument': t.Instrument,
                'Price_BTC': t.Price_BTC,
                'Price_USD': t.Price_USD,
                'IV_Percent': t.IV_Percent,
                'Size': t.Size,
                'Entry_Value': t.Entry_Value,
                'Underlying_Price': t.Underlying_Price,
                'Expiration_Date': t.Expiration_Date,
                'Strike_Price': t.Strike_Price,
                'Option_Type': t.Option_Type,
                'Entry_Date': t.Entry_Date,
                'BlockTrade_IDs': t.BlockTrade_IDs,
                'BlockTrade_Count': t.BlockTrade_Count,
                'Combo_ID': t.Combo_ID,
                'ComboTrade_IDs': t.ComboTrade_IDs
            } for t in trades]
            self._trades_df = pd.DataFrame(trades_data)

        finally:
            session.close()

    def get_available_currencies(self):
        """Fetch available currencies for options."""
        return ['BTC', 'ETH']
    
       
    def fetch_available_dates(self):
        """
        Returns a sorted list of unique Expiration_Date values.
        """
        if self._chains_df is None or self._chains_df.empty:
            return []
        dates = self._chains_df['Expiration_Date'].dropna().unique()
        return sorted(dates)

    
    def get_options_for_date(self, expiration_date):
        """
        Returns DataFrame of instruments where Expiration_Date matches.
        """
        if self._chains_df is None:
            return pd.DataFrame()

        mask = self._chains_df['Expiration_Date'] == expiration_date
        
        return self._chains_df[mask].copy()
    
    def fetch_option_chain(self, option_symbol=None):
        """
        Returns DataFrame of detail for a given Instrument or list of Instruments.
        If option_symbol is None, returns full chain DataFrame.
        """
        df = self._chains_df.copy() if self._chains_df is not None else pd.DataFrame()
        if option_symbol is None:
            return df
        if isinstance(option_symbol, list):
            return df[df['Instrument'].isin(option_symbol)].copy()
        else:
            return df[df['Instrument'] == option_symbol].copy()
        
    def get_instrument_probabilities(self):
        """
        Returns a DataFrame with columns ['Instrument', 'Probability (%)'] 
        and the top instrument based on probability within a price range.

        Parameters:
        options_data (pd.DataFrame): DataFrame containing options data with Probability (%).
        current_price (float): The current price around which to filter instruments.
        price_range (float): The range around the current price to filter instruments.

        Returns:
        filtered_df (pd.DataFrame): DataFrame with 'Instrument' and 'Probability (%)'.
        top_instrument (str): The instrument with the highest probability in the given range.
        """
        current_price, highest, lowest = get_btcusd_price()
        if current_price is None or current_price == 0:
            current_price = 100000
        
        # Calculate lower and upper bounds for filtering
        price_range=50000
        lower_bound = current_price - price_range
        upper_bound = current_price + price_range
        

        # Filter the DataFrame based on the specified range
        df = self._chains_df.copy() if self._chains_df is not None else pd.DataFrame()

        options_data = df   
        all_probabilities_df = options_data[
            (options_data['Strike_Price'] >= lower_bound) & 
            (options_data['Strike_Price'] <= upper_bound)
        ][['Instrument', 'Probability_Percent']]
        
        # Sort filtered DataFrame by Probability (%) in descending order
        all_probabilities_df.sort_values(by='Probability_Percent', ascending=False, inplace=True)
       
        # Get the top instrument
        top_probability_instrument = all_probabilities_df.iloc[0]['Instrument'] if not all_probabilities_df.empty else None

        return all_probabilities_df, top_probability_instrument


    
    def get_all_options_for_strike(self, option_strike=None, option_type=None):
        """
        Returns DataFrame filtered by Strike_Price and/or Option_Type.
        """
        df = self._chains_df.copy() if self._chains_df is not None else pd.DataFrame()
        if option_strike is not None:
            df = df[df['Strike_Price'] == option_strike]
        if option_type is not None:
            # Option_Type values are like 'C' or 'P'
            df = df[df['Option_Type'] == option_type]

        return df
    
    
    
    def load_public_trades(self, symbol_filter=None, show_24h_public_trades=False) -> pd.DataFrame:
        """
        Returns a DataFrame of public trades.
        
        - Filters by Instrument if symbol_filter is provided.
        - Limits to the last 24 hours if show_24h_public_trades is True.
        - Sorts by Entry_Date if available.
        """
        df = self._trades_df.copy() if self._trades_df is not None else pd.DataFrame()
        if df.empty:
            logger.warning("load_public_trades: No trade data available.")
            return df

        if symbol_filter:
            if 'Instrument' in df.columns:
                df = df[df['Instrument'] == symbol_filter]
            else:
                logger.warning("'Instrument' column missing from public trades DataFrame.")
                return pd.DataFrame()  # Return empty if essential column is missing

        if 'Entry_Date' in df.columns:
            if show_24h_public_trades:
                cutoff = datetime.utcnow() - timedelta(hours=24)
                df = df[df['Entry_Date'] >= cutoff]

            df = df.sort_values(by='Entry_Date', ascending=False)
        else:
            logger.warning("'Entry_Date' column missing from public trades DataFrame. Skipping date filtering and sorting.")

        return df.reset_index(drop=True)
        
    
    def extract_instrument_info(self, column_name):
        """Extracts instrument name and option side from the column name."""
        parts = column_name.split('-')
        instrument_name = '-'.join(parts[:-1])  # All parts except the last part (option side)
        option_side = parts[-1]                  # Last part is the option side
        return instrument_name, option_side

    def filter_best_options_combo(self, loss_threshold, premium_threshold, quantity, show_buy, show_sell):
        """
        Filters the combined results to identify options that meet the loss threshold and premium criteria.

        Parameters:
            combined_results (pd.DataFrame): DataFrame containing profit results.
            loss_threshold (float): Maximum loss to consider an option as acceptable.
            premium_threshold (float): Minimum premium to consider an option as profitable.
            quantity (int): Quantity of options for calculating premium.

        Returns:
            pd.DataFrame: A DataFrame where the first row contains premiums,
                        followed by the filtered profit results.
        """

        self.load_from_csv(data_type="analytics_data")
         
        combined_results = self.analytic_data.copy()
        # Ensure combined_results has the expected structure
        if combined_results.empty or 'Underlying Price' not in combined_results.columns:
            logger.error("No valid data available in the combined results.")
            return pd.DataFrame()  # Return an empty DataFrame on error

        # Create a copy of combined_results to modify
        filtered_results = combined_results.copy()

        # This will hold the premiums for valid options
        premium_row = [np.nan] * (len(filtered_results.columns) - 1)  # Initialize with NaNs for all valid columns

        # Loop through each column (ignoring the first column which is 'Underlying Price')
        for i, column in enumerate(combined_results.columns[1:]):
            try:
                # Check if any value in the column is below the loss threshold
                if (combined_results[column] < loss_threshold).any():
                    filtered_results = filtered_results.drop(columns=[column], errors='ignore')  # Drop if loss threshold not met
                    continue  # Skip further processing for this column

                # Extract instrument name and option side
                instrument_name, option_side = self.extract_instrument_info(column)

                # Fetch option data from your data source
                instrument_detail, _ = self.fetch_option_data(instrument_name)

                if instrument_detail is None or instrument_detail.empty:
                    logger.warning(f"No details found for instrument: {instrument_name}.")
                    continue  # Skip to the next column if no details are found

                # Validate that the fetched instrument details correspond to the instrument name
                if 'Instrument' in instrument_detail.columns and instrument_detail['Instrument'].values[0] != instrument_name:
                    logger.error(f"Incompatible instrument data for {instrument_name}. Expected name not found.")
                    continue  # Skip processing for this column

                # Calculate premium based on the option side
                if option_side == 'BUY':
                    if 'Ask Price (USD)' not in instrument_detail.columns:
                        logger.error(f"No ask price data available for instrument: {instrument_name}.")
                        continue
                    instrument_premium = instrument_detail['Ask Pirce (USD)'].values[0] * quantity
                else:
                    if 'Bid Price (USD)' not in instrument_detail.columns:
                        logger.error(f"No bid price data available for instrument: {instrument_name}.")
                        continue
                    instrument_premium = instrument_detail['Bid Price (USD)'].values[0] * quantity

                # Check if the premium meets the threshold
                if instrument_premium <= premium_threshold and instrument_premium > 0:
                    # Valid premium, update the premium row at the correct index
                    premium_row[i] = instrument_premium
                else:
                    # Drop the column in filtered_results since the premium is below the threshold
                    filtered_results = filtered_results.drop(columns=[column], errors='ignore')

            except Exception as e:
                logger.error(f"Error processing column {column}: {e}")
                continue  # Continue processing other columns

        # If there are valid premiums, insert them as the first row
        valid_premiums = [p for p in premium_row if pd.notna(p)]
        if valid_premiums:
            premium_row_df = pd.DataFrame([valid_premiums], columns=filtered_results.columns[1:len(valid_premiums) + 1])
            
            # Insert the new row into the top of the filtered results
            filtered_results = pd.concat([premium_row_df, filtered_results], ignore_index=True)

            # Add a new row index name for the premium row
            filtered_results.index = ['Premium'] + list(range(len(filtered_results) - 1))

        filtered_results_copy = filtered_results[['Underlying Price']].copy()
        # Applying filters based on flags
        if show_buy and show_sell:
            # If both flags are true, do nothing since we want all relevant columns
            filtered_results_copy = filtered_results 
        else:
            if show_buy:
                buy_columns = filtered_results.filter(like="BUY").columns
                filtered_results_copy = filtered_results [['Underlying Price']].join(filtered_results[buy_columns])
            if show_sell:
                sell_columns = filtered_results.filter(like="SELL").columns
                filtered_results_copy = filtered_results [['Underlying Price']].join(filtered_results[sell_columns])

        return filtered_results_copy
    

    def load_public_trades_profit(self, filter=None):
        """Load options screener data and return it after filtering.

        Parameters:
            filter (str, optional): The instrument name to filter by. If None, return the full DataFrame.

        Returns:
            pd.DataFrame: The filtered or unfiltered options screener DataFrame.
        """
        # Load the complete options screener data from the CSV
        self.load_from_csv(data_type="public_profits")

        # Create a copy of the DataFrame to avoid SettingWithCopyWarning
        options_screener_copy = self.public_profits.copy()

        # Check if 'entry date' exists before sorting
        if 'Entry Date' in options_screener_copy.columns:
            options_screener_copy['Entry Date'] = pd.to_datetime(options_screener_copy['Entry Date'], errors='coerce')
            # Sort the DataFrame by 'entry date'
            options_screener_copy.sort_values(by='Entry Date', ascending=False, inplace=True)

        # Drop unnecessary columns
        options_screener_copy = options_screener_copy.drop(columns=['block_trade_id', 'combo_id', 'block_trade_leg_count', 'combo_trade_id'], errors='ignore')

        # If filter is provided and not None, filter the DataFrame
        if filter:
            options_screener_copy = options_screener_copy[options_screener_copy['Instrument'] == filter]

        return options_screener_copy
    
    # def validate_datatable(self, df, data_type):
    #  # Define the desired data types for each column based on data_type
    #     if data_type == "Public":
    #         desired_dtypes = {
    #             'Side': 'object',
    #             'Instrument': 'object',
    #             'Price (BTC)': 'float64',
    #             'Price (USD)': 'float64',
    #             'Mark Price (BTC)': 'float64',
    #             'IV (%)': 'float64',
    #             'Size': 'float64',
    #             'Entry Value': 'float64',
    #             'Underlying Price': 'float64',
    #             'Expiration Date': 'object',
    #             'Strike Price': 'float64',
    #             'Option Type': 'object',
    #             'Entry Date': 'datetime64[ns]',
    #             'BlockTrade IDs': 'object',
    #             'BlockTrade Count': 'float64',
    #             'Combo ID': 'object',
    #             'ComboTrade IDs': 'float64',
    #             'Trade ID': 'float64'
    #         }
    #     elif data_type == "Trade":
    #         desired_dtypes = {
    #             'Instrument': 'object',
    #             'Option Type': 'object',
    #             'Strike Price': 'float64',
    #             'Expiration Date': 'object',
    #             'Last Price (USD)': 'float64',
    #             'Bid Price (USD)': 'float64',
    #             'Ask Price (USD)': 'float64',
    #             'Bid IV': 'float64',
    #             'Ask IV': 'float64',
    #             'Delta': 'float64',
    #             'Gamma': 'float64',
    #             'Theta': 'float64',
    #             'Vega': 'float64',
    #             'open_interest': 'float64',
    #             'total traded volume': 'float64',
    #             'monetary volume': 'float64'
    #         }
    #     else:
    #         raise ValueError("Invalid data_type. Expected 'Public' or 'Trade'.")

    #     # Iterate over the columns and convert data types
    #     for column, dtype in desired_dtypes.items():
    #         if column in df.columns:
    #             # Convert the column to the desired data type using .loc to avoid SettingWithCopyWarning
    #             try:
    #                 df.loc[:, column] = df[column].astype(dtype)
    #             except ValueError:
    #                 print(f"Warning: Could not convert column {column} to {dtype}.")
            
        
    #     # Drop rows with incorrect data types
    #     for column, dtype in desired_dtypes.items():
    #         if column in df.columns:
    #             # Check each value in the column
    #             for index, value in df[column].items():
    #                 try:
    #                     # Attempt to convert the value to the desired type
    #                     if dtype == 'object':
    #                         str(value)  # Ensure it can be converted to string
    #                     elif dtype == 'datetime64[ns]':
    #                         pd.to_datetime(value)  # Ensure it can be converted to datetime
    #                     else:
    #                         float(value)  # Ensure it can be converted to float
    #                 except (ValueError, TypeError):
    #                     # Drop the row if conversion fails
    #                     df.drop(index, inplace=True)
    #                     print(f"Dropped row {index} due to invalid data type in column {column}.")
        
    #     return df
                    


        