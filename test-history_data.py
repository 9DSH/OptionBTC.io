import yfinance as yf
import pandas as pd

def fetch_historical_data():
        """
        Fetches the latest price of a specified asset from Yahoo Finance.

        Returns:
            A dataframe of the asset's open, high, low, close, and volume for the most recent timeframe.
        """ 
        # Set the start date for the historical data 
        start_date = (pd.Timestamp.now() - pd.DateOffset(years=1)).strftime('%Y-%m-%d')

        # Retrieve historical price data
        df = yf.download("BTC-USD", start=start_date, auto_adjust=True, interval="1d")

        # If the dataframe is empty, return an empty dataframe
        if df.empty:
            print("No data retrieved. Please check the asset symbol or the API response.")
            return df

        # Trim and rename columns to match the desired output structure
        df = df.reset_index()
        df = df.rename(columns={'Date': 'Datetime'})  # Rename to differentiate
        df.columns = df.columns.map(lambda x: x[0] if isinstance(x, tuple) else x)

        # Rearranging columns
        df = df[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.rename(columns={'Datetime': 'Date'}, inplace=True)

        # Round the values of relevant columns
        numerical_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df[numerical_columns] = df[numerical_columns].apply(lambda x: round(x, 3))

        df.columns = df.columns.str.lower()
        # Return the DataFrame
        return df


data =  fetch_historical_data()
print(data)