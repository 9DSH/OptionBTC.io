import streamlit as st
import pandas as pd
from openai import OpenAI 
from Fetch_data import Fetching_data
from Calculations import get_most_traded_instruments
from Analytics import Analytic_processing
from fetch_btc_price import get_btcusd_price


class Chatbar:
    def __init__(self, openai_api_key, model="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.fetch_data = Fetching_data()
        self.analytics = Analytic_processing()
        self.current_price = get_btcusd_price()
        self.init_session_state()
        
    def init_session_state(self):
        """Initializes session state variables."""
        if "market_trades_df" not in st.session_state:
            trades_df, strategies_insight , strategy_analysis_df  = self.load_market_insights()
            st.session_state.market_trades_df = trades_df
            st.session_state.strategy_df = strategies_insight  # Store insights separately
            st.session_state.strategy_analytics = strategy_analysis_df  # Store insights separately

        
        if "last_user_message" not in st.session_state:
            st.session_state.last_user_message = ""
        
        if "last_ai_message" not in st.session_state:
            st.session_state.last_ai_message = "Ask me about Deribit options analytics or public trades."

    def load_market_insights(self):
        """Loads market trades data, ensuring it always returns a DataFrame for insights."""
        try:
            # Load the public trades data
            public_trades_df = self.fetch_data.load_market_trades()
            
            if not isinstance(public_trades_df, pd.DataFrame):
                raise ValueError("Expected a DataFrame but got a non-DataFrame object.")
            # Process the DataFrame for insights
            top_options, _ = get_most_traded_instruments(public_trades_df)
            
            target_columns = ['BlockTrade IDs', 'BlockTrade Count', 'Combo ID', 'ComboTrade IDs']
            strategy_trades_df = public_trades_df[~public_trades_df[target_columns].isna().all(axis=1)]

            strategy_groups = strategy_trades_df.groupby(['BlockTrade IDs', 'Combo ID'])
            strategy_df = self.analytics.Identify_combo_strategies(strategy_groups)

            # Analyze block trades and get insights
            strategy_insights = self.analytics.analyze_block_trades(strategy_df)

            
            # Create DataFrames for Summary Stats and Strategy Analysis
            insights_summary_stats = strategy_insights.get('summary_stats', {})
            insights_summary_df = pd.DataFrame([insights_summary_stats])

            # To handle strategy_analysis for conversion
            strategy_analysis_df = strategy_insights.get('strategy_analysis', {})
            if isinstance(strategy_analysis_df, dict):
                # Assuming strategy_analysis contains a DataFrame-like structure
                strategy_analysis_df = pd.DataFrame(strategy_analysis_df.get('strategy_distribution').reset_index())

            return top_options, insights_summary_df, strategy_analysis_df  # Return as needed
        except Exception as e:
            st.warning(f"Error loading market trades: {str(e)}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()  # Adjust return structure to match expectations
                
    def df_to_json(self, df):
        """Convert DataFrame to JSON format."""
        return df.to_json(orient='records')  # Convert to JSON records format

    def call_openai(self, messages):
        """Calls the OpenAI API."""
        # Initial instruction to the AI
        system_message = {
            "role": "system",
            "content": ("You are a sophisticated trading assistant specializing in options trading on Deribit. "
                        "Your responses should be concise, focused, and directly related to the query. "
                        "Provide insights based on the current market data and avoid irrelevant information.")
        }

        # Include the system message in the messages to OpenAI
        messages.insert(0, system_message)

        params = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 200,  # Limit token count to encourage brevity
        }
        
        response = self.client.chat.completions.create(**params)
        return response

    def display_chat(self):
        """Handles the chat display and messaging logic."""
        with st.sidebar:
            st.subheader("Chat with AI")
            prompt = st.chat_input("Type your question..")
            chat_message = st.empty()

            with chat_message:
                st.chat_message("assistant").write(st.session_state.last_ai_message)

            if prompt:
                st.session_state.last_user_message = prompt
                
                # Prepare the messages for OpenAI API
                trades_df = st.session_state.market_trades_df
                trategies_insight = st.session_state.strategy_df
                strategy_analysis_df =  st.session_state.strategy_analytics  # Store insights separately
                if trades_df.empty:
                    response_message = "Market trades data is not available."
                    st.session_state.last_ai_message = response_message
                    chat_message.empty()
                    with chat_message:
                        st.chat_message("assistant").write(response_message)
                    return
                
                # Convert the DataFrame to JSON format
                trades_json = self.df_to_json(trades_df)
                strategies_json = self.df_to_json( trategies_insight)

                strategy_analysis_json =self.df_to_json(strategy_analysis_df )
                
                btc_price , highest , lowest = self.current_price
                # Consolidate data into a single JSON structure
                combined_json = {
                                "market_trades": trades_json,
                                "strategy_insights": strategies_json,
                                "btc_price": btc_price,
                                "highest_price": highest,
                                "lowest_price": lowest
                            }

                # Add context for OpenAI API
                messages = [
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nHere is the market trades data in JSON format:\n{trades_json}\n\nHere are the strategy insights:\n{strategies_json}\n\nHere are the strategy analytics:\n{strategy_analysis_json}"
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nHere is the BTC price :\n{btc_price}\n\nHere ais the highest price of BTC for today:\n{highest}\n\nHere is the lowest BTC Price for today:\n{ lowest}"
                    }
                ]

                # Call OpenAI API
                response = self.call_openai(messages)

                if not response.choices:
                    response_message = "No response from OpenAI."
                else:
                    ai_message = response.choices[0].message.content
                    response_message = ai_message

                # Update last AI message and display it
                st.session_state.last_ai_message = response_message
                chat_message.empty()
                with chat_message:
                    st.chat_message("assistant").write(response_message)