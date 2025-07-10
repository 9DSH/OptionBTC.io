import yfinance as yf
import pandas as pd
import numpy as np
import os
import logging
from datetime import timezone


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    def __init__(
                    self,
                    symbol,
                    interval=str,  # 1d, 4h, 1h
                    technical_analysis_data_csv=str
                ):
        
        self.symbol = symbol
        self.interval = interval
        self.technical_analysis_data_csv = technical_analysis_data_csv
        self.technical_analysis_data = pd.DataFrame()
    
    def get_technical_data(self):
        """Load historical data from CSV, fetching it if not present."""
        logger.info("Checking if daily data exists and loading if available")
        if self.interval == "1d":
            if os.path.exists(self.technical_analysis_data_csv):
                df = pd.read_csv(self.technical_analysis_data_csv)
                logger.info(f"Loaded historical data for {self.symbol} on {self.interval} interval")
                
                df['date'] = pd.to_datetime(df['date'])
                latest_date_in_data = df['date'].max().date()
                current_date = pd.Timestamp.now().date()

                if current_date > latest_date_in_data:
                    logger.info(f"Fetching new analatical for {self.interval}...")
                    self.technical_analysis_data = self.prepare_technical_data()
                    trend_insights = self.analyze_market_trend(self.technical_analysis_data)
                    return trend_insights
                else:
                    trend_insights = self.analyze_market_trend(df)
                    return trend_insights
            else:
                self.technical_analysis_data = self.prepare_technical_data()
                trend_insights = self.analyze_market_trend(self.technical_analysis_data)
                logger.info(f"No existing data found for {self.symbol} on {self.interval} interval. Fetching new data.")

                return trend_insights
         
                 
        elif self.interval == "4h":
            if os.path.exists(self.technical_analysis_data_csv):
                df = pd.read_csv(self.technical_analysis_data_csv)
                logger.info(f"Loaded historical data for {self.symbol} on {self.interval} interval")

                df['date'] = pd.to_datetime(df['date'])
                latest_datetime_in_data = df['date'].max()

                latest_hour_in_data = latest_datetime_in_data.hour
                adjusted_hour = (latest_hour_in_data + 4) % 24
                current_hour = pd.Timestamp.now(tz=timezone.utc).hour
                current_date = pd.Timestamp.now().date()
                latest_date_only = pd.Timestamp(latest_datetime_in_data).date()

                if adjusted_hour > current_hour or current_date > latest_date_only:
                    logger.info(f"Fetching new analatical for {self.interval}...")
                    self.technical_analysis_data = self.prepare_technical_data()
                    trend_insights = self.analyze_market_trend(self.technical_analysis_data)
                    return trend_insights
                else:
                    trend_insights = self.analyze_market_trend(df)
                    return trend_insights
            else:
                self.technical_analysis_data = self.prepare_technical_data()
                trend_insights = self.analyze_market_trend(self.technical_analysis_data)
                logger.info(f"No existing data found for {self.symbol} on {self.interval} interval. Fetching new data.")

                return trend_insights
        
        
    def analyze_market_trend(self, df: pd.DataFrame) -> dict:
        # Ensure we're working on a fresh copy after dropping NAs to avoid SettingWithCopyWarning
        if df is None or df.empty:
            return {}  # ✅ Return empty dict if df is None or empty
        
        try:
            df = (
                df.dropna(subset=['price_action', 'predicted_trend', 'close'])
                .copy()
            )

            # Convert columns to numeric safely
            df.loc[:, 'price_action'] = pd.to_numeric(df['price_action'], errors='coerce')
            df.loc[:, 'close'] = pd.to_numeric(df['close'], errors='coerce')

            # Drop any rows that became NaN after conversion
            df = df.dropna(subset=['price_action', 'close'])

            # Unique sorted price actions
            unique_price_actions = (
                df['price_action']
                .drop_duplicates()
                .sort_values()
                .reset_index(drop=True)
            )

            # Merge close values within 100-point ranges, keeping the larger
            if not unique_price_actions.empty:
                filtered = [unique_price_actions.iloc[0]]
                for val in unique_price_actions.iloc[1:]:
                    if val - filtered[-1] >= 100:
                        filtered.append(val)
                    else:
                        filtered[-1] = max(filtered[-1], val)
                unique_price_actions = pd.Series(filtered)

            # Latest predictions and current price
            last_predicted_trend = df['predicted_trend'].iat[-1]
            current_price = df['close'].iat[-1]

            # Determine support and resistance levels around current price
            lower_bound = current_price - 3000
            upper_bound = current_price + 3000
            nearby = unique_price_actions[(unique_price_actions >= lower_bound) & (unique_price_actions <= upper_bound)]

            resistance = sorted([p for p in unique_price_actions if p > current_price])[:3]
            support = sorted([p for p in unique_price_actions if p < current_price])[-3:]

            # Get the latest price action
            active_price_action = df['price_action'].iat[-1]

            return {
                'unique_price_actions': unique_price_actions.tolist(),
                'nearby_price_actions': nearby.tolist(),
                'last_predicted_trend': last_predicted_trend,
                'current_price': current_price,
                'resistance': resistance,
                'support': support,
                'active_price_action': active_price_action 
            }
        except Exception as e:
            print(f"An error occurred: {e}")
    
    def fetch_historical_data(self):
        """
        Fetches the latest price of a specified asset from Yahoo Finance.

        Returns:
            A dataframe of the asset's open, high, low, close, and volume for the most recent timeframe.
        """ 
        # Set the start date for the historical data 
        start_date = (pd.Timestamp.now() - pd.DateOffset(years=1)).strftime('%Y-%m-%d')

        # Retrieve historical price data
        df = yf.download(self.symbol, start=start_date, auto_adjust=True, interval=self.interval)

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

    def prepare_technical_data(self) -> pd.DataFrame:
        """Prepare technical analysis data with Ichimoku indicators."""
        data = self.fetch_historical_data()
        if data.empty:
            return pd.DataFrame()

        # Calculate Ichimoku components
        ichimoku_df = self.calculate_ichimoku(data)
        data = pd.concat([data, ichimoku_df], axis=1)

        # Trend analysis
        data["switch_komu"] = self.switch_komu(data[["senkou_span_a", "senkou_span_b"]])
        data[["price_action", "previous_price_action"]] = self.find_price_action(data)
        data["predicted_trend"] = self.predict_trend(data)
 
        # Save to CSV
        data.to_csv(self.technical_analysis_data_csv, index=False)
        logger.info(f"Technical analysis data saved to {self.technical_analysis_data_csv}")
        return data

    
    @staticmethod    
    def calculate_ichimoku(df: pd.DataFrame, shift=False):
        # Calculate the Tenkan-sen (Conversion Line)
        data = df.copy()
        data["tenkan_sen"] = (
            data["high"].rolling(window=9).max() + data["low"].rolling(window=9).min()
        ) / 2

        # Calculate the Kijun-sen (Base Line)
        data["kijun_sen"] = (
            data["high"].rolling(window=26).max() + data["low"].rolling(window=26).min()
        ) / 2

        if shift:
            # Calculate the Senkou Span A (Leading Span A)
            data["senkou_span_a"] = ((data["tenkan_sen"] + data["kijun_sen"]) / 2).shift(26)

            # Calculate the Senkou Span B (Leading Span B)
            data["senkou_span_b"] = (
                (
                    data["high"].rolling(window=52).max() +
                    data["low"].rolling(window=52).min()
                ) / 2
            ).shift(26)

            # Calculate the Chikou Span (Lagging Span)
            data["chikou_span"] = data["close"].shift(-26)
        else:
            # Calculate the Senkou Span A (Leading Span A)
            data["senkou_span_a"] = (data["tenkan_sen"] + data["kijun_sen"]) / 2

            # Calculate the Senkou Span B (Leading Span B)
            data["senkou_span_b"] = (
                data["high"].rolling(window=52).max() + data["low"].rolling(window=52).min()
            ) / 2

            # Calculate the Chikou Span (Lagging Span)
            data["chikou_span"] = data["close"]

        # Select relevant columns and round the values
        columns_to_round = ["tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b", "chikou_span"]
        
        # Return the DataFrame with rounded values
        return data[columns_to_round].round(3)

    @staticmethod
    def switch_komu(df: pd.DataFrame):
        """Gives a series of indexes and their corresponding switch komu where
        1 denotes a bullish switch (Senkou Span A goes above Senkou Span B), -1 denotes a bearish switch (Senkou Span B goes above Senkou Span A),
        and 0 denotes no switch
        Parameters:
            df: dataframe that contains at least two columns named "senkou_span_a" and "senkou_span_b"
        Return:
            Series of indexes taken from df and values corresponding to Komu switches.
        """
        df_spans = df[["senkou_span_a", "senkou_span_b"]].copy()
        df_spans["a_b_diff"] = df_spans["senkou_span_a"] - df_spans["senkou_span_b"]
        df_spans["a_b_diff_previous"] = df_spans["a_b_diff"].shift(1)
        df_spans["switch_komu"] = df_spans.apply(
            lambda row: 1
            if (row["a_b_diff"] > 0) and (row["a_b_diff_previous"] <= 0)
            else -1
            if (row["a_b_diff"] < 0) and (row["a_b_diff_previous"] > 0)
            else 0,
            axis=1,
        )
        return df_spans["switch_komu"]

    @staticmethod
    def find_price_action(df: pd.DataFrame):
        """Outputs price action which is the maximum of senkou_span_a between a bull and bear switch and the minimum of
        senkou_span_a between a bear and a bull switch.
        Parameters:
            df: dataframe in which the columns senkou_span_a and senkou_span_b exist.
        Returns:
            Series denoting price action.
        """
        df_switches = df[["senkou_span_a", "senkou_span_b", "switch_komu"]].copy()
        df_switches["mark"] = (
            (df_switches["switch_komu"] == 1) | (df_switches["switch_komu"] == -1)
        ) & (df_switches["switch_komu"] != df_switches["switch_komu"].shift())
        df_switches["group"] = df_switches["mark"].cumsum()

        first_switch_index = (df_switches["switch_komu"] != 0).idxmax()
        first_switch = df_switches.loc[first_switch_index, "switch_komu"]

        if first_switch == 1:
            df_min = df_switches[df_switches.group % 2 == 0]
            df_max = df_switches[df_switches.group % 2 != 0]
        else:
            df_max = df_switches[df_switches.group % 2 == 0]
            df_min = df_switches[df_switches.group % 2 != 0]

        result_min = df_min.groupby("group")["senkou_span_a"].agg("min").reset_index()
        result_max = df_max.groupby("group")["senkou_span_a"].agg("max").reset_index()

        result_min.rename(columns={"senkou_span_a": "price_action"}, inplace=True)
        result_max.rename(columns={"senkou_span_a": "price_action"}, inplace=True)

        df_switches = df_switches.merge(result_min, how="left", on="group")
        df_switches = df_switches.merge(result_max, how="left", on="group")

        # Replace fillna using assignment instead of inplace=True
        df_switches["price_action_x"] = df_switches["price_action_x"].fillna(df_switches["price_action_y"])

        df_switches.drop("price_action_y", inplace=True, axis=1)
        df_switches.rename(columns={"price_action_x": "price_action"}, inplace=True)

        # So far price_action has been set between each two switch komus. Here we change it to be set between
        # each two max and mins

        df_switches["mark"] = df_switches["senkou_span_a"] == df_switches["price_action"]
        df_switches["group"] = df_switches["mark"].cumsum()
        result = (
            df_switches[df_switches.group != 0]
            .groupby("group")["price_action"]
            .agg("first")
            .reset_index()
        )

        df_switches = df_switches.merge(result, how="left", on="group")
        df_switches.rename(columns={"price_action_y": "price_action"}, inplace=True)

        # Using .loc to create a new column and update existing ones
        df_switches['previous_price_action'] = df_switches['price_action'].shift(1).fillna(0)
        current_price_action = df_switches['price_action']

        # Update previous price action based on current price action changes
        last_group = df_switches["group"].iloc[-1]
        if df_switches["switch_komu"].iloc[-1] == 0:
            # If the last switch is neutral, carry forward the last known price action
            df_switches.loc[df_switches["group"] == last_group, "price_action"] = df_switches["price_action"].iloc[-2]

        # Update previous price action based on current price action changes
        for i in range(1, len(df_switches)):
            if current_price_action.iloc[i] != current_price_action.iloc[i - 1]:
                df_switches.loc[i, 'previous_price_action'] = current_price_action.iloc[i - 1]
            else:
                df_switches.loc[i, 'previous_price_action'] = df_switches.loc[i - 1, 'previous_price_action']

        df_switches['previous_price_action'].fillna(0, inplace=True)

        return df_switches[["price_action", "previous_price_action"]]


        
    @staticmethod
    def extract_trends(a: pd.Series, b: pd.Series, window=10):
        """
        Specifies whether we have a "bull_trend", "side_trend", or "bear_trend" for two specific series a and b
        where they respectively represent Senkou Span A and Senkou Span B.

        If in all the previous window periods a is above b then it is a bull trend. Otherwise, if there is a
        cross in the previous window periods and in all the window periods before that cross a is below b then
        it is a bear trend. Otherwise it is a side trend. In all the opposite conditions instead of bull we have bear and
        instead of bear we have a bull trend.

        Parameter:
            a: the first series denoting Senkou Span A
            b: the second series denoting Senkou Span B
            window: the number of periods for consideration to decide the trend

        Returns:
            "bull_trend", "side_trend" or "bear_trend" depending on the trend.
        """
        if len(a) != len(b):
            raise Exception("Objects are not of the same length!")
        lastIndex = len(a) - 1
        if a[lastIndex] >= b[lastIndex]:
            findBullTrend = 1
        else:
            findBullTrend = 0
        startIndex = max(0, len(a) - window)
        if findBullTrend:
            comparisonResult = a[startIndex:] >= b[startIndex:]
            if len(comparisonResult[comparisonResult == 1]) == window:
                return "bull_trend"
            else:
                beforeCross = comparisonResult.index[comparisonResult == 0]
                lastBeforeCross = beforeCross[-1]
                starthAgainIndex = max(0, lastBeforeCross - window)
                comparisonAgainResult = (
                    a[starthAgainIndex : lastBeforeCross + 1]
                    <= b[starthAgainIndex : lastBeforeCross + 1]
                )
                if len(comparisonAgainResult[comparisonAgainResult == 1]) == window:
                    return "bear_trend"
                else:
                    return "side_trend"
        else:
            comparisonResult = a[startIndex:] <= b[startIndex:]
            if len(comparisonResult[comparisonResult == 1]) == window:
                return "bear_trend"
            else:
                beforeCross = comparisonResult.index[comparisonResult == 0]
                lastBeforeCross = beforeCross[-1]
                starthAgainIndex = max(0, lastBeforeCross - window)
                comparisonAgainResult = (
                    a[starthAgainIndex : lastBeforeCross + 1]
                    >= b[starthAgainIndex : lastBeforeCross + 1]
                )
                if len(comparisonAgainResult[comparisonAgainResult == 1]) == window:
                    return "bull_trend"
                else:
                    return "side_trend"
    @staticmethod
    def predict_trend(df: pd.DataFrame) -> pd.Series:
        """Predict the upcoming trend based on price actions and Ichimoku indicators.

        Parameters:
            df: dataframe consisting of at least four columns 'tenkan_sen', 'kijun_sen', 'price_action', and 'previous_price_action'.
        
        Returns:
            A Series denoting the predicted trend at each row ("bullish", "bearish", or "neutral").
        """
        conditions_bullish = (
            (df["previous_price_action"] > df["price_action"]) &
            (df["tenkan_sen"] > df["previous_price_action"]) &
            (df["kijun_sen"] > df["previous_price_action"])
        )

        conditions_bearish = (
            (df["previous_price_action"] < df["price_action"]) &
            (df["tenkan_sen"] < df["previous_price_action"]) &
            (df["kijun_sen"] < df["previous_price_action"])
        )
        
        # Initialize a trend Series with "neutral"
        trend_series = pd.Series("Neutral", index=df.index)

        # Apply conditions
        trend_series[conditions_bullish] = "Bullish"
        trend_series[conditions_bearish] = "Bearish"

        return trend_series
    
    