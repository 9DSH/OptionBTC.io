import streamlit as st
import numpy as np
import pandas as pd
import sys
import os
import asyncio
import subprocess
import threading
from datetime import date, datetime, timedelta , timezone
import logging
import warnings
from Calculations import *
from Charts import *
import plotly.graph_objects as go 
from AI import Chatbar
from Fetch_data import Fetching_data
from Analytics import Analytic_processing
from main_data_stream import main as fetch_main  # async function
from fetch_btc_price import get_btcusd_price

from Technical_Analysis import TechnicalAnalysis

warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.set_page_config(page_title='Trading Dashboard', layout='wide')




fetch_data = Fetching_data()
analytics = Analytic_processing()

technical_4h = TechnicalAnalysis("BTC-USD", "4h" ,'technical_analysis_4h.csv') 
technical_daily = TechnicalAnalysis("BTC-USD", "1d" ,'technical_analysis_daily.csv') 
technical_w = TechnicalAnalysis("BTC-USD", "1w" ,'technical_analysis_w.csv') 



# Chat sidebar 

#OpenAI_KEY = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
#chat = Chatbar(openai_api_key=OpenAI_KEY )


# Initialize the thread reference globally
data_refresh_thread = None
public_trades_thread = None




def app():

    initializing_states()
    start_data_refresh_thread()
    update_public_trades()
    technical_analysis()
    #chat.display_chat()
    
    currency = 'BTC'
    premium_buy = 0
    premium_sell = 0

   
    #-----------------------------------------------------------------
    #-------------------------- Technical Bar ------------------------
    #-----------------------------------------------------------------
     # Fetch and display the current price
    btc_price , highest, lowest = get_btcusd_price()
    upper_strike_filter = int(btc_price) + 30000
    lower_strike_filter = abs(int(btc_price) - 30000)

    technical_row = st.container()
    with technical_row:
        
        padding_colomn , Technical_bar_column, col3 = st.columns([0.5, 5, 0.5])  # Adjust ratio for centering
        with Technical_bar_column:
            technical_daily_row1 = st.container()
            technical_4h_row2 = st.container()
            
            if st.session_state.technical_daily is not None:
                daily_support_list= st.session_state.technical_daily.get("support") or []
                daily_resistance_list = st.session_state.technical_daily.get("resistance") or []
            else:
                daily_support_list = None  
                daily_resistance_list = None    
                print('technical_daily is None')

            if st.session_state.technical_4h is not None:
                _4h_support_list = st.session_state.technical_4h.get("support") or []
                _4h_resistance_list = st.session_state.technical_4h.get("resistance") or []
                _4h_trend = st.session_state.technical_4h.get("last_predicted_trend") or []    
            else:
                _4h_support_list = None  
                _4h_resistance_list  = None    
                print('technical_daily is None')

            daily_support_1, daily_support_2 , daily_support_3 = (daily_support_list + [None, None, None])[:3]
            daily_resistance_1, daily_resistance_2, daily_resistance_3 = (daily_resistance_list + [None, None, None])[:3]

            support_4h_1, support_4h_2, support_4h_3 = (_4h_support_list + [None, None, None])[:3]
            resistance_4h_1, resistance_4h_2, resistance_4h_3 = (_4h_resistance_list + [None, None, None])[:3]
            
            
            support_title_col , support_col1, support_col2,  support_col3,  lowest_col , price_col, highest_col, resistance_col1, resistance_col2,  resistance_col3 , res_title_col= st.columns([1,1,1,1,1,1,1,1,1,1,1])

            with technical_daily_row1 : 
                with support_title_col : 
                    st.markdown(f"<div style='font-size: 12px; color: gray;text-align: center;'>Supports (D)</div>", unsafe_allow_html=True)
                    st.markdown("<hr style='margin: 10px;'>", unsafe_allow_html=True) 
                with support_col1:
                    st.markdown(f"<div style='font-size: 12px; color: white; text-align: center;'>{daily_support_1:.0f}</div>" if isinstance(daily_support_1, float) else "", unsafe_allow_html=True)
                    st.markdown("<hr style='margin: 10px;'>", unsafe_allow_html=True) 
                with support_col2:
                    st.markdown(f"<div style='font-size: 12px; color: white;text-align: center;'>{daily_support_2:.0f}</div>" if isinstance(daily_support_2, float) else "", unsafe_allow_html=True)
                    st.markdown("<hr style='margin: 10px;'>", unsafe_allow_html=True) 
                with support_col3:    
                    st.markdown(f"<div style='font-size: 12px; color: white;text-align: center;'>{daily_support_3:.0f}</div>" if isinstance(daily_support_3, float) else "", unsafe_allow_html=True)
                    st.markdown("<hr style='margin:10px;'>", unsafe_allow_html=True) 

                with resistance_col1:
                    st.markdown(f"<div style='font-size: 12px; color: white;text-align: center;'>{daily_resistance_1:.0f}</div>" if isinstance(daily_resistance_1, float) else "", unsafe_allow_html=True)
                    st.markdown("<hr style='margin: 10px;'>", unsafe_allow_html=True) 
                with resistance_col2:
                    st.markdown(f"<div style='font-size: 12px; color: white;text-align: center;'>{daily_resistance_2:.0f}</div>" if isinstance(daily_resistance_2, float) else "", unsafe_allow_html=True)
                    st.markdown("<hr style='margin: 10px;'>", unsafe_allow_html=True) 
                with resistance_col3:
                    st.markdown(f"<div style='font-size: 12px; color: white;text-align: center;'>{daily_resistance_3:.0f}</div>" if isinstance(daily_resistance_3, float) else "", unsafe_allow_html=True)
                    st.markdown("<hr style='margin: 10px;'>", unsafe_allow_html=True) 
                with res_title_col:
                    
                    st.markdown(f"<div style='font-size: 12px; color: gray;text-align: center;'>Resistances (D)</div>", unsafe_allow_html=True)
                    st.markdown("<hr style='margin: 10px;'>", unsafe_allow_html=True) 


            with  technical_4h_row2 :
                with support_title_col : 
                    st.markdown(f"<div style='font-size: 12px; color: gray;text-align: center;'>Supports (4H)</div>", unsafe_allow_html=True)
                with support_col1:
                    st.markdown(f"<div style='font-size: 12px; color: white;text-align: center;'>{support_4h_1:.0f}</div>" if isinstance(support_4h_1, float) else "", unsafe_allow_html=True)
                with support_col2:
                    st.markdown(f"<div style='font-size: 12px; color: white;text-align: center;'>{support_4h_2:.0f}</div>" if isinstance(support_4h_2, float) else "", unsafe_allow_html=True)
                with support_col3:    
                    st.markdown(f"<div style='font-size: 12px; color: white;text-align: center;'>{support_4h_3:.0f}</div>" if isinstance(support_4h_3, float) else "", unsafe_allow_html=True)
                
                with resistance_col1:
                    st.markdown(f"<div style='font-size: 12px; color: white;text-align: center;'>{resistance_4h_1:.0f}</div>" if isinstance(resistance_4h_1, float) else "", unsafe_allow_html=True)
                with resistance_col2:
                    st.markdown(f"<div style='font-size: 12px; color: white;text-align: center;'>{resistance_4h_2:.0f}</div>" if isinstance(resistance_4h_2, float) else "", unsafe_allow_html=True)
                with resistance_col3:
                    st.markdown(f"<div style='font-size: 12px; color: white;text-align: center;'>{resistance_4h_3:.0f}</div>" if isinstance(resistance_4h_3, float) else "", unsafe_allow_html=True)
                with res_title_col:
                    st.markdown(f"<div style='font-size: 12px; color: gray;text-align: center;'>Resistances (4H)</div>", unsafe_allow_html=True)

            with lowest_col:
                lowest_row_1= st.container()
                lowest_row_2 = st.container()
                lowest_row_3  = st.container()
                with lowest_row_1:
                    st.write("")
                with lowest_row_2:
                    st.markdown(f"<div style='font-size: 12px; color: gray;text-align: center;'>Lowest</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 14px;color: #f54b4b;text-align: center; '>{lowest}</div>", unsafe_allow_html=True)
                with lowest_row_3:
                    st.write("")

            with price_col : 
                price_row_1 = st.container()
                price_row_2 = st.container()
                price_row_3 = st.container()
                
                with price_row_1:
                    if _4h_trend == "Neutral":
                        textcolor = "white"
                    elif _4h_trend == "Bullish":
                        textcolor = "#90EE90"
                    elif _4h_trend == "Bearish":
                        textcolor = "#f54b4b"
                    else:
                        textcolor = "white"  # Default color if _4h_trend is None or any other value
                    st.markdown(f"<div style='font-size: 14px; color:{textcolor}; text-align: center; letter-spacing: 2px;'>{_4h_trend}</div>", unsafe_allow_html=True)
                with price_row_2: 
                
                    btc_display_price = f"{btc_price:.0f}" if btc_price is not None else "Loading..."
                    st.markdown(f"<div style='font-size: 25px;text-align: center;'>{btc_display_price}</div>", unsafe_allow_html=True)
                with price_row_3:
                    st.write("")
            
            with highest_col:
                highest_row_1 = st.container()
                highest_row_2 = st.container()
                highest_row_3 = st.container()
                with highest_row_1:
                    st.write("")
                with highest_row_2:
                    st.markdown(f"<div style='font-size: 12px; color: gray;text-align: center;'>Highest</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 14px; color: #90EE90;text-align: center;'>{highest}</div>", unsafe_allow_html=True)
            
                with highest_row_3 :
                    st.write("")

                
        with col3:
            current_utc_time = datetime.now(timezone.utc).strftime("%H:%M:%S")
            st.markdown(f"<div style='font-size: 12px; text-align: center;'>UTC Time</div><div style='font-size: 16px; color: gray;text-align: center;'>{current_utc_time}</div>", unsafe_allow_html=True)
            
   
       
        #current_date_initials = pd.to_datetime(datetime.now()).date()

        
        # Initialize session state for inputs if they don't exist
    

    ##---------------------------------------------------------------
    ##------------------------- MAIN TABS ---------------------------
    #-----------------------------------------------------------------
        
    
    main_tabs = st.tabs(["Market Watch",  "Live Trade Option", "Simulation", "Technical Analysis" ])
#------------------------------------------------------------
#-----------------------Market Watch ------------------
#------------------------------------------------------------
    with main_tabs[0]: 
             # Initialize trades variable outside of any if conditions
            market_screener_df = st.session_state.public_trades_df
            #market_screener_df =   fetch_data.validate_datatable(market_screener_df_raw, "Public")
            total_number_public_trade = market_screener_df.shape[0]
                        
            # Force Entry Date to be timezone-naive datetime64[ns]
            # ✅ Force 'Entry_Date' to timezone-naive and remove microseconds
            if not market_screener_df.empty:
                market_screener_df['Entry_Date'] = pd.to_datetime(market_screener_df['Entry_Date'], errors='coerce')
                market_screener_df['Entry_Date'] = market_screener_df['Entry_Date'].dt.tz_localize(None)
                market_screener_df['Entry_Date'] = market_screener_df['Entry_Date'].dt.floor('s')
            # here we can have the function that simulates the public trades profit
            
            if not market_screener_df.empty:
                market_screener_df.dropna(subset=['Entry_Date', 'Underlying_Price'], inplace=True)
                
                
                filter_row = st.container()
                with filter_row:
                    col_date,col_vertical_1, col_strike_size_range, col_vertical_2,  col_expiration, col_vertical_3, col_side_type = st.columns([0.3, 0.01, 0.2, 0.01, 0.3, 0.01, 0.1])
                    #with col_refresh: 
                        #apply_market_filter = st.button(label="Apply", key="apply_market_filter")

                    with col_date:
                        date_row1 = st.container()
                        date_row2 = st.container()

                        with date_row1:
                            cc1, cc2, cc3 = st.columns([0.04, 0.02, 0.02])
                            oldest_entry_date = st.session_state.public_trades_df['Entry_Date'].min()
                            oldest_entry_date_date = oldest_entry_date.date()
                            oldest_entry_date_hour = oldest_entry_date.hour
                            oldest_entry_date_minute = oldest_entry_date.minute
                            with cc1:
                                start_date = st.date_input("Start Entry Date", value=oldest_entry_date_date, key= "start_date_input")
                            with cc2:
                                start_hour = st.number_input("Hour", min_value=0, max_value=23, value=oldest_entry_date_hour, key = "start_hours_input")
                            with cc3:
                                start_minute = st.number_input("Minute", min_value=0, max_value=59, value=oldest_entry_date_minute, key = "end_hours_input")

                        with date_row2:
                            ca1, ca2, ca3 = st.columns([0.04, 0.02, 0.02])
                            with ca1:
                                current_utc_date = datetime.now(timezone.utc).date()
                                end_date = st.date_input("End Entry Date", value=current_utc_date, key="Entry_end_input")

                                with ca2:
                                    end_hour = st.number_input("Hour", min_value=0, max_value=23, value=23, key = "start_hours_input_2")
                                with ca3:
                                    end_minute = st.number_input("Minute", min_value=0, max_value=59, value=59, key = "end_hours_input_2")

                            # Combine date and time into a single datetime object
                        start_datetime = datetime.combine(start_date, datetime.min.time().replace(hour=start_hour, minute=start_minute))
                        end_datetime = datetime.combine(end_date, datetime.min.time().replace(hour=end_hour, minute=end_minute))
                                                # Make both naive (remove timezone info)
                        start_datetime = start_datetime.replace(tzinfo=None)
                        end_datetime = end_datetime.replace(tzinfo=None)
                    with col_vertical_1:
                            st.markdown("<div style='height: 150px; width: 1px; background-color: gray; margin: auto;'></div>", unsafe_allow_html=True)  # Vertical line

                    with col_strike_size_range:
                        row_one = st.container()
                        row_two = st.container()
                        with row_one:
                            strike_col1, strike_col2 = st.columns(2)
                            with strike_col1:
                                min_strike = st.number_input("Minimum strike", min_value=0, max_value=400000, value=lower_strike_filter, key = "start_strike_input")
                            with strike_col2:
                                max_strike = st.number_input("Maximum strike", min_value=0, max_value=400000, value=upper_strike_filter, key = "end_strike_input")
                        with row_two:
                            size_col1, size_col2 = st.columns(2)
                            with size_col1:
                                min_size= st.number_input("Minimum size", min_value=0.1, max_value=500.0, value=0.1, key = "min_size_input")
                            with size_col2:
                                max_size = st.number_input("Maximum size", min_value=0.1, max_value=500.0, value=500.0, key = "max_size_input")
                            size_range = (min_size, max_size)

                            if 'Size' in market_screener_df.columns:
                                market_screener_df['Size'] = pd.to_numeric(market_screener_df['Size'], errors='coerce')


                    with col_vertical_2:
                        st.markdown("<div style='height: 150px; width: 1px; background-color: gray; margin: auto;'></div>", unsafe_allow_html=True)  # Vertical line
                    with col_expiration:
                        row_expiration = st.container()
                        row_strike = st.container()
                        with row_expiration:
                            market_available_dates = market_screener_df['Expiration_Date'].dropna().unique().tolist()

                            # Convert to datetime to sort
                            market_available_dates = pd.to_datetime(market_available_dates, format='%d-%b-%y', errors='coerce')
                            
                            # Filter out NaT values
                            market_available_dates = market_available_dates.dropna()
                            # Sort the dates
                            sorted_market_available_dates = sorted(market_available_dates)

                            # Optionally convert back to desired string format for display purposes
                            sorted_market_available_dates = [date.strftime("%#d-%b-%y") for date in sorted_market_available_dates]

                            selected_expiration_filter = st.multiselect("Filter by Expiration Date", sorted_market_available_dates, key="watch_exp_filter")

                        with row_strike:
                            
                            strike_range = (min_strike, max_strike)
                            if 'Strike_Price' in market_screener_df.columns and not market_screener_df.empty:
                                # Filter the DataFrame for strikes within the selected range
                                filtered_strikes_df = market_screener_df[
                                    (market_screener_df['Strike_Price'] >= strike_range[0]) &
                                    (market_screener_df['Strike_Price'] <= strike_range[1])
                                ]
                                unique_strikes = filtered_strikes_df['Strike_Price'].unique()
                                sorted_strikes = sorted(unique_strikes, reverse=True)  # Sort in descending order

                                # Create the multiselect for the filtered strike prices
                                multi_strike_filter = st.multiselect("Filter by Strike Price", options=sorted_strikes, key="multisselect_strike")
                            else:
                                # Handle case where no strikes are available
                                multi_strike_filter = st.multiselect("Filter by Strike Price", options=[], default=[], help="No available strikes to select.", key="default_multiselect")
                    
                    with col_vertical_3:
                            st.markdown("<div style='height: 150px; width: 1px; background-color: gray; margin: auto;'></div>", unsafe_allow_html=True)  # Vertical line

                    with col_side_type:
                            
                            side_row = st.container()
                            type_row = st.container()
                            with side_row:
                                show_sides_buy = st.checkbox("BUY", value=True, key='show_buys')
                                show_sides_sell = st.checkbox("SELL", value=True, key='show_sells')
                            with type_row:
                                show_type_call = st.checkbox("Call", value=True, key='show_calss')
                                show_type_put = st.checkbox("Put", value=True, key='show_puts')

                    start_strike, end_strike = strike_range  # Unpack the tuple to get start and end values
                    start_size , end_size = size_range

                    filtered_df = market_screener_df[
                        (market_screener_df['Size'] >= start_size) &
                        (market_screener_df['Size'] <= end_size) &
                        (market_screener_df['Strike_Price'] >= start_strike) &
                        (market_screener_df['Strike_Price'] <= end_strike) &
                        (market_screener_df['Entry_Date'] >= start_datetime) &
                        (market_screener_df['Entry_Date'] <= end_datetime)
                    ]
                    if selected_expiration_filter:
                        filtered_df = filtered_df[(filtered_df['Expiration_Date'].isin(selected_expiration_filter))]

                    # Filter by selected strikes
                    if multi_strike_filter:
                        filtered_df = filtered_df[filtered_df['Strike_Price'].isin(multi_strike_filter)]

                    # Apply filtering for buy/sell sides
                    sides_to_filter = []
                    if show_sides_buy:
                        sides_to_filter.append('BUY')  # append the actual value as per your column data
                    if show_sides_sell:
                        sides_to_filter.append('SELL')  # append the actual value as per your column data

                    if sides_to_filter:
                        filtered_df = filtered_df[filtered_df['Side'].isin(sides_to_filter)]

                    # Apply filtering for call/put types
                    types_to_filter = []
                    if show_type_call:
                        types_to_filter.append('Call')  # append the actual value as per your column data
                    if show_type_put:
                        types_to_filter.append('Put')  # append the actual value as per your column data

                    if types_to_filter:
                        filtered_df = filtered_df[filtered_df['Option_Type'].isin(types_to_filter)]
                        
            if not market_screener_df.empty:
                # Ensure 'Entry Date' is in datetime format
                market_screener_df['Entry_Date'] = pd.to_datetime(market_screener_df['Entry_Date'], errors='coerce')

                # Check for any NaT values
                if market_screener_df['Entry_Date'].isna().any():
                    st.warning("Some entries in the 'Entry Date' column were invalid and have been set to NaT.")
                
                
                tabs = st.tabs(["Insights",  "Top Options", "Public Trade Strategies", "Whales" , "Data table"])
                
                if not filtered_df.empty:
                    with tabs[0]:
                        details_row =st.container()
                        with details_row : 
                            col1,col2,col3,col4,col5 = st.columns([0.2,0.1,0.1,0.1,0.2])
                            with col2:
                                total_options, total_amount, total_entry_values = calculate_totals_for_options(filtered_df)
                                total_trades_percentage = ( total_options/ total_number_public_trade) * 100

                                row_count_title = st.container()
                                row_count = st.container()
                                with row_count_title:
                                    st.markdown(f"<p style='font-size: 12px; color: gray;'> Total Trades:</p>", unsafe_allow_html=True)
                                with row_count:
                                    st.markdown(f"<p style='font-size: 17px; font-weight: bold;'> {total_options:,}</p>", unsafe_allow_html=True)
                                
                            with col3:
                                
                                row_count_title = st.container()
                                row_count = st.container()
                                with row_count_title:
                                    st.markdown(f"<p style='font-size: 12px; color: gray;'> Percentage of Total Trades:</p>", unsafe_allow_html=True)
                                with row_count:
                                    st.markdown(f"<p style='font-size: 17px; font-weight: bold;'> {total_trades_percentage:.1f}%</p>", unsafe_allow_html=True)

                            with col4:
                                
                                row_size_title = st.container()
                                row_size = st.container()
                                with row_size_title:
                                    st.markdown(f"<p style='font-size: 12px; color: gray;'> Total Values:</p>", unsafe_allow_html=True)
                                with row_size:
                                    st.markdown(f"<p style='font-size: 17px;font-weight: bold;'> {total_entry_values:,.0f}</p>", unsafe_allow_html=True)
                            with col5:
                                st.checkbox(
                                             "Show 24h Public Trades",
                                               key='show_24_public_trades',
                                               on_change=update_public_trades
                                                                        )

                        detail_column_2, detail_column_3 = st.columns(2)                       

                        with detail_column_2:
                            fig_2 = plot_strike_price_vs_entry_value(filtered_df)
                            st.plotly_chart(fig_2)
                        with detail_column_3:
                            fig_3 = plot_stacked_calls_puts(filtered_df)
                            st.plotly_chart(fig_3)
                        st.markdown("---") 

                    with tabs[1]:  
                        padding, cal1, cal2,cal3 = st.columns([0.02, 0.7, 0.01 ,0.6])

                        with padding: 
                            st.write("")

                        with cal1: 

                            most_traded_options , top_options_chains = get_most_traded_instruments(filtered_df)
                            fig_pie = plot_most_traded_instruments(most_traded_options)
                            st.plotly_chart(fig_pie)

                        with cal3:
                            
                            st.markdown(f"<p style='font-size: 14px; margin-top: 28px;'></p>", unsafe_allow_html=True) 
                            st.dataframe(top_options_chains, use_container_width=True, hide_index=True)
                        
                        

                #------------------------------------------
                #       public trades insights
                #-----------------------------------------------

                    with tabs[2]:
                        target_columns = ['BlockTrade_IDs', 'BlockTrade_Count', 'Combo_ID', 'ComboTrade_IDs']
                        filtered_df = filtered_df.drop('hover_text', axis=1)
                        
                        filtered_startegy = filtered_df.copy()
                        # Separate strategy trades
                        strategy_trades_df = filtered_startegy[~filtered_startegy[target_columns].isna().all(axis=1)]
                        print(f'Strategy found : {strategy_trades_df.shape[0]}')

                        if not strategy_trades_df.empty:

                            # Group by BlockTrade IDs and Combo ID to identify unique strategies
                            strategy_groups = strategy_trades_df.groupby(['BlockTrade_IDs', 'Combo_ID'])
                            
                            # Create subtabs for different views
                            strategy_subtabs = st.tabs(["Strategy Overview", "Strategy Details"])
                            with strategy_subtabs[0]:
                                # Summary statistics for each strategy
                                strategy_df = analytics.Identify_combo_strategies(strategy_groups)
                                strategy_df_copy = strategy_df.copy()
                                if not strategy_df_copy.empty:
                                        
                                    insights = analytics.analyze_block_trades(strategy_df_copy)
                                    
                                    st.caption(f"Analyzed {len(strategy_df_copy)} trades from {insights['summary_stats']['time_range_start']} to {insights['summary_stats']['time_range_end']}")

                                    # 1. Key Metrics
                                    padding , col1, col2, col3, col4, padding2 = st.columns([1,1,1,1,2,0.5])
                                    col1.metric("Total Strategies", f"{len(strategy_df_copy)}")
                                    col2.metric("Total Volume (BTC)", f"{insights['summary_stats']['total_size_btc']:,.1f}")
                                    col3.metric("Average Trade Size", f"{insights['summary_stats']['avg_trade_size']:,.1f} BTC")
                                    most_active_strategy = insights['strategy_analysis']['top_strategies'].index[0]
                                    col4.metric("Most Active Strategy", str(most_active_strategy))
                                    st.markdown("---")  # Horizontal line

                                    # 2. Strategy Distribution
                                    
                                    colu1, colu2 = st.columns(2)
                                    strategy_df_copy = insights['strategy_analysis']['strategy_distribution']
                                    with colu1 : 
                                        most_strag_fig = plot_most_strategy_bar_chart(strategy_df_copy)
                                        st.plotly_chart( most_strag_fig)
                                    with colu2 : 
                                        # 3. Top Strikes
                                        top_strikes = insights['strike_analysis']['top_strikes']
                                        fig_startegy_top_strikes = plot_top_strikes_pie_chart(top_strikes)
                                        st.plotly_chart(fig_startegy_top_strikes)

                                    # 4. Time Analysis
                                    hourly = insights['time_analysis']['hourly_activity']
                                    fig_hourly = plot_hourly_activity(hourly)
                                    st.plotly_chart(fig_hourly)
                                
                                    # 5. Recommendations
                                    st.subheader("💡 Trader Insights")
                                    for rec in insights['recommendations']:
                                        st.info(rec)

                                    # Raw data expander
                                    with st.expander("📝 View Raw Analysis Data"):
                                        st.dataframe(strategy_df, use_container_width=True, hide_index=True)
                                else : st.warning("No Strategy found for this data set")                               
                            
                            with strategy_subtabs[1]:
                                # Detailed view of each strategy
                                # Convert strategy groups to list and sort by total premium
                                sorted_strategies = []
                                for (block_id, combo_id), group in strategy_groups:
                                    total_premium = group['Entry_Value'].sum()
                                    strategy_type = strategy_df[strategy_df['Strategy ID'] == combo_id]['Strategy Type'].iloc[0]
                                    sorted_strategies.append((block_id, combo_id, group, total_premium, strategy_type))
                                
                                # Sort by total premium in descending order
                                sorted_strategies.sort(key=lambda x: x[3], reverse=True)

                                # Create a list of strategy labels for the selectbox
                                def format_value(value):
                                    if value > 1000000:
                                        return f"{value/1000000:.1f}M"
                                    elif value > 1000:
                                        return f"{value/1000:.1f}k"
                                    else:
                                        return f"{value:,.1f}"
                                    
                                def format_date(expiration_date_str):
                                    # If it's already a datetime object, just format it
                                    if isinstance(expiration_date_str, datetime):
                                        return expiration_date_str.strftime('%d%b')
                                    # Try parsing as '%d-%b-%y'
                                    try:
                                        return datetime.strptime(expiration_date_str, '%d-%b-%y').strftime('%d%b')
                                    except Exception:
                                        pass
                                    # Try parsing as '%Y-%m-%d'
                                    try:
                                        return datetime.strptime(expiration_date_str, '%Y-%m-%d').strftime('%d%b')
                                    except Exception:
                                        pass
                                    # If all fails, return as is or handle as needed
                                    return str(expiration_date_str)

                                strategy_labels = []
                                for _, _, group, total_premium, strategy_type in sorted_strategies:
                                    
                                    option_details = " ** ".join(
                                        f"{format_date(row['Expiration_Date'])}-{row['Side']}-{row['Option_Type']}-{int(row['Strike_Price'])}-{format_value(row['Entry_Value'])}"
                                        for _, row in group.iterrows()
                                    )
                                    strategy_labels.append(f"{format_value(total_premium)} | {option_details}")

                                # Use a selectbox to choose a strategy
                                selected_strategy_label = st.selectbox("Select a Strategy", strategy_labels, key=f'show_{strategy_labels}')
                                
                                # Find the selected strategy details
                                selected_strategy = next((s for s in sorted_strategies if f"{format_value(s[3])} | " + " ** ".join(
                                    f"{format_date(row['Expiration_Date'])}-{row['Side']}-{row['Option_Type']}-{int(row['Strike_Price'])}-{format_value(row['Entry_Value'])}"
                                    for _, row in s[2].iterrows()
                                ) == selected_strategy_label), None)
                                
                                if selected_strategy:
                                    block_id, combo_id, group, total_premium, strategy_type = selected_strategy
                                    
                                    # Calculate strategy metrics
                                    total_size = group['Size'].sum()
                                    # Display strategy metrics
                                    col1, col2, col3, col4, col5  = st.columns([0.3, 0.2, 0.4, 0.2,0.2])
                                    with col1:
                                        st.metric("Total Premium", f"${format_value(total_premium)}")
                                    with col2:
                                        st.metric("Total Size", f"{total_size:,.0f}")
                                    with col3:
                                        strategy_type_from_summary = strategy_df[strategy_df['Strategy ID'] == combo_id]['Strategy Type'].iloc[0]
                                        st.metric("Strategy Type", strategy_type_from_summary)
                                    with col4:
                                        st.metric("Number of Legs", len(group))
                                    with col5:
                                        # Create a dictionary with 'Instrument' as key and 'Side' as value for each row in the group
                                        startegy_dict = {row['Instrument']: row['Side'] for _, row in group.iterrows()}
                                        pick_strategy_button = st.button(label="Pick this Strategy", key="Pick_strategy")
                                        if pick_strategy_button:
                                            st.session_state['stored_strategy_dict'] = startegy_dict

                                    
                                    # Display strategy components
                                    
                                    
                                    # Display the group DataFrame
                                    st.dataframe(group, use_container_width=True, hide_index=True)
                                    
                                    # Create visualization for this strategy
                                    chart_col1, chart_col2 = st.columns(2)
                                    strategy_data = group.copy()
                                    
                                    # Calculate profits using multithreading
                                    
                                    #strategy_data_df =   fetch_data.validate_datatable(strategy_data , "Public")
                                    fig_profit, fig_strategy = plot_public_profits(strategy_data, "Public", trade_option_details=None)
                                    
                                    with chart_col1:
                                        st.plotly_chart(fig_strategy, use_container_width=True, key=f"strategy_plot_{block_id}_{combo_id}")

                                    with chart_col2:
                                        st.plotly_chart(fig_profit, use_container_width=True, key=f"alldays_plot_{block_id}_{combo_id}")
                                
                        else :               
                            st.warning("No strategy trades found in the current selection.")

                    

                    with tabs[3]:
                        whale_cal1,whale_cal2, whale_cal3 = st.columns([0.5,0.5,1])
                        with whale_cal1:
                            whale_filter_type = st.selectbox("Analyze values by:", options=['Size', 'Entry Value'], index=1)

                        with whale_cal2:
                            if whale_filter_type == "Entry Value" :
                                entry_filter = st.number_input("Set Entry Filter Value", min_value=0, value=10000, step=100, key = "entry_filter_whales_input")
                            else : entry_filter = st.number_input("Set Size Filter Value", min_value=0.1, value=2.0, step=0.1 , key = "size_filter_whales_input")
                            

                        outliers , whales_fig = plot_identified_whale_trades(filtered_df,
                                                                              min_size=8,
                                                                              max_size=35,
                                                                              min_opacity=0.2,
                                                                              max_opacity=0.8, 
                                                                              entry_value_threshold = entry_filter , 
                                                                              filter_type = whale_filter_type )
                        st.plotly_chart(whales_fig)

                        #st.markdown("---")

                        with st.expander("View the Whales Data Table", expanded=False): 
                            st.dataframe(outliers , use_container_width=True, hide_index=True)  # Show index


                    
                    with tabs[4]:
                        datatable = st.tabs(["Processed Data" , "Raw Data"])
                        with datatable[0]:
                            processed_df = filtered_df[filtered_df[target_columns].isna().all(axis=1)]
                            processed_df = processed_df.iloc[:, :-4]
                            if 'Entry_Date' in processed_df.columns:
                                processed_df = processed_df.sort_values(by='Entry_Date', ascending=False)
                            else:
                                st.warning("Processed data missing 'Entry_Date'. Cannot sort chronologically.")
                            st.dataframe(processed_df, use_container_width=True, hide_index=False)  # Show index

                            
                            st.markdown("---")  # Horizontal line
                            analyze_row = st.container()
                            with analyze_row:
                                analyze_col1, analyze_col2 = st.columns([0.2,1])
                                with analyze_col1:
                                    selected_index = st.selectbox("Select an Index to Analyze", options=processed_df.index, key="analyze_profit_select")
                                with analyze_col2:
                                    if selected_index is not None:
                                            # Filter the raw data for the selected index
                                            selected_option_data = filtered_df.loc[[selected_index]]
                                            
                                            #selected_option_data =   fetch_data.validate_datatable(selected_option_data_raw , "Public")
                                            # Calculate profits using the existing function
                                            fig_public_profit, fig_public_strategy = plot_public_profits(selected_option_data, "Public", trade_option_details=None)

                                            instrument = selected_option_data['Instrument'].values[0]
                                            entry_value = selected_option_data['Entry_Value'].values[0]
                                            expiration_date = selected_option_data['Expiration_Date'].values[0]
                                            side = selected_option_data['Side'].values[0]
                                            size = selected_option_data['Size'].values[0]
                                            underlying_price = selected_option_data['Underlying_Price'].values[0]
                                            
                                            entry_date = pd.to_datetime(selected_option_data['Entry_Date'].values[0]).strftime('%d-%b-%y %H:%M')
                                            st.markdown(
                                                f"""
                                                <div style='display: flex; justify-content: flex-start; gap: 10px; padding-top: 10px;'>
                                                    <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                        <div style='font-size: smaller; color: gray;'>Instrument</div>
                                                        {instrument}
                                                    </div>
                                                    <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                        <div style='font-size: smaller; color: gray;'>Side</div>
                                                        {side}
                                                    </div>
                                                    <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                        <div style='font-size: smaller; color: gray;'>Size</div>
                                                        {size}
                                                    </div>
                                                    <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                        <div style='font-size: smaller; color: gray;'>Entry Value</div>
                                                        {entry_value}
                                                    </div>
                                                    <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                        <div style='font-size: smaller; color: gray;'>Entry Date</div>
                                                        {entry_date}
                                                    </div>
                                                    <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                        <div style='font-size: smaller; color: gray;'>Expiration Date</div>
                                                        {expiration_date}
                                                    </div>
                                                    <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                        <div style='font-size: smaller; color: gray;'>Underlying Price</div>
                                                        {underlying_price}
                                                    </div>
                                                </div>
                                                """, 
                                                unsafe_allow_html=True
                                            )

                            
                            st.markdown("---")  # Horizontal line
                            analyze_col1, analyze_col2 = st.columns(2)
                            # Create a selectbox for the user to choose an index from the processed_df
                        
                                    # Display the profit chart
                            with analyze_col1:
                                st.plotly_chart( fig_public_strategy)
                            
                                
                            with analyze_col2:
                                st.plotly_chart(  fig_public_profit)


                        with datatable[1]:
                            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
                else:
                    st.warning("No Public Trades found due to the weak connection, Please refresh and try after 5 minutes..")

                        
          
   
            else:
                st.warning("No trades available for the selected options, wait while app is fetching data...")

#------------------------------------------------------------------------------------
#----------------------------------------Trade Live Option -------------------------------------
#--------------------------------------------------------------------------------------

    with main_tabs[1]:
            #--------------------------------------------------------------------------------
            #-------------------------------- Poltting Results -----------------------------
            #--------------------------------------------------------------------------------
            available_dates = fetch_data.fetch_available_dates()
            options_row = st.container()
            trade_option_detail = []
            with options_row:
                option_col1, option_col2, vertical_line , option_details= st.columns([1,0.5,0.1,4])
                with option_col1:
                    row_1 = st.container()
                    row_2 = st.container()
                    with row_1 : 
                    
                        selected_date = st.selectbox("Select Expiration Date", available_dates, 
                                                            index=available_dates.index(st.session_state.selected_date))
                        st.session_state.selected_date = selected_date  # Update session state

                    with row_2 : 

                        if selected_date:
                                # Fetch options available for the selected date
                            options_for_date = fetch_data.get_options_for_date(selected_date)
                            
                            if hasattr(options_for_date, "empty") and not isinstance(options_for_date, list):
                                options_for_date = options_for_date['Instrument'].tolist()
                                # Allow user to select an option
                            if options_for_date:
                                    # Check if the currently selected option_symbol is in the available options
                                if st.session_state.option_symbol not in options_for_date:
                                        # Reset option_symbol to None if it's not available
                                    st.session_state.option_symbol = None

                                option_symbol_index = 0 if st.session_state.option_symbol is None else options_for_date.index(st.session_state.option_symbol)
                                option_symbol = st.selectbox("Select Option", options_for_date, index=option_symbol_index)

                with option_col2:
                    row_1 = st.container()
                    row_2 = st.container()
                    # Use st.session_state.quantity directly in the number_input
                    with row_1:
                        quantity = st.number_input('Quantity',
                                                    min_value=0.1,
                                                    step=0.1,
                                                    value=st.session_state.quantity ,
                                                    key = "qty_input")  # Current value from session state
                        # Update session state only if the value changes
                        if quantity != st.session_state.quantity:
                            st.session_state.quantity = quantity  # Update the session state after the widget is used
                    with row_2:

                        option_side = st.selectbox("Select Side", options=['BUY', 'SELL'], 
                                                    index=['BUY', 'SELL'].index(st.session_state.option_side))
                        st.session_state.option_side = option_side  # Store input in session state

                    trade_option_detail.extend([option_side, quantity])

                #----------------------Vertical Line -----------------------
                with vertical_line:
                    st.markdown("<div style='height: 170px; width: 1px; background-color: gray; margin: auto;'></div>", unsafe_allow_html=True)  # Vertical line
                with option_details:
                    row_1 = st.container()
                    row_2 = st.container()
                    if option_symbol:
                            # Get and display the details of the selected option
                        option_details = fetch_data.fetch_option_chain(option_symbol)
                        recent_public_trades_df = fetch_data.load_public_trades(symbol_filter= option_symbol , show_24h_public_trades = st.session_state.show_24_public_trades)
                        
                        #recent_public_trades_df =   fetch_data.validate_datatable(recent_public_trades_df_raw, "Public")
                        if not option_details.empty:
                                # Extracting details safely
                                
                            expiration_date_str = option_details['Expiration_Date'].values[0]
                            expiration_date = pd.to_datetime(expiration_date_str).date()  # Ensure it's converted to date

                            option_type = option_details['Option_Type'].values[0]
                            bid_iv = option_details['Bid_IV'].values[0]
                            ask_iv = option_details['Ask_IV'].values[0]
                            bid_price = option_details['Bid_Price_USD'].values[0]
                            ask_price = option_details['Ask_Price_USD'].values[0]
                            strike_price = option_details['Strike_Price'].values[0]
                            probability = option_details['Probability_Percent'].values[0]
                            premium_buy = ask_price * quantity
                            premium_sell = bid_price * quantity

                            breakeven_call_buy = premium_buy + strike_price
                            breakeven_call_sell = premium_sell+ strike_price
                            breakeven_put_buy = strike_price - premium_buy
                            breakeven_put_sell = strike_price - premium_sell
      
                            breakeven_buy = breakeven_call_buy if option_type == 'call' else breakeven_put_buy
                            breakeven_sell = breakeven_put_sell if option_type == 'put' else breakeven_call_sell   

                            if option_side == "BUY":
                                breakeven_sell = None
                                premium = premium_buy
                                breakeven_price = breakeven_buy 
                                            
                            if option_side == "SELL":
                                breakeven_buy = None
                                premium = premium_sell
                                breakeven_price = breakeven_sell


                            now_utc = datetime.now(timezone.utc).date()

                                # Compute total days to expiration (at least 1 to avoid zero)
                            time_to_expiration_days = max((expiration_date - now_utc).days, 1)

                            fee_cap = 0.125 * premium
                            initial_fee = 0.0003 * btc_price * quantity
                            final_fee_selected_option = min(initial_fee, fee_cap)

                            with row_1: 
                                option_details_copy = option_details.copy().iloc[:, 3:]
                                st.dataframe(option_details_copy, use_container_width=True, hide_index=True)
                            
                            with row_2: 
                                st.markdown(
                                                f"""
                                                <div style='display: flex; justify-content: flex-start; gap: 10px; padding-top: 10px;'>
                                                    <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                        <div style='font-size: smaller; color: gray;'>Instrument</div>
                                                        {option_symbol}
                                                    </div>
                                                    <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                        <div style='font-size: smaller; color: gray;'>Entry Value</div>
                                                        {premium:.1f}
                                                    </div>
                                                    <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                        <div style='font-size: smaller; color: gray;'>Fee (USD)</div>
                                                        {final_fee_selected_option:.1f}
                                                    </div>
                                                    <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                        <div style='font-size: smaller; color: gray;'>Breakeven Price</div>
                                                        {breakeven_price:.0f}
                                                    </div>
                                                    <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                        <div style='font-size: smaller; color: gray;'>Time To Expiration</div>
                                                        {time_to_expiration_days}
                                                    </div>
                                                    <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                        <div style='font-size: smaller; color: gray;'>Probability (%)</div>
                                                        {probability:.0f}
                                                    </div>
                                                </div>
                                                </div>
                                                """, 
                                                unsafe_allow_html=True
                                            )


                        else:
                            st.write("Error fetching option details.")     

            df_options_for_strike = fetch_data.get_all_options_for_strike(strike_price, option_type)
            Other_tab = f'All Other Options for {strike_price:.0f} {option_type.upper()}'
            analytic_tabs = st.tabs(["Analytics", "Recent trades", Other_tab])
            with analytic_tabs[0]:
                if recent_public_trades_df is None or recent_public_trades_df.empty:
                    st.warning("There are no recent trades for the selected option.")
                
                if not recent_public_trades_df.empty:
                    st.subheader(f'Analytics for Option {strike_price:.0f} {option_type.upper()} ')
                    padding, chart_1, chart_2, chart_3 = st.columns([0.1, 2, 2, 2 ])
                    with padding: 
                        st.write("")
                    with chart_1:
                        
                        
                        fig_selected_symbol = plot_underlying_price_vs_entry_value(recent_public_trades_df, btc_price, premium)
                        st.plotly_chart(fig_selected_symbol)
                    with chart_2:
                        # whales_in_option_fig = plot_whales(recent_public_trades_df, min_count=2, min_avg_size=5, max_marker_size=30, showlegend=False)
                        # st.plotly_chart(whales_in_option_fig)
                        price_vs_date = plot_price_vs_entry_date(recent_public_trades_df)
                        st.plotly_chart(price_vs_date)
                    with chart_3:
                        open_by_expiration_radar = plot_radar_chart(df_options_for_strike)
                        st.plotly_chart(open_by_expiration_radar)
                    
                    st.markdown("---")

                #option_details =   fetch_data.validate_datatable(option_details, "Trade")
                profit_fig , expiration_profit = plot_public_profits(option_details , "Trade", trade_option_detail)
                
                chart_col_1, chart_col_2 = st.columns(2)
                with chart_col_1:
                            
                    st.plotly_chart(expiration_profit)
                with chart_col_2:
                    st.plotly_chart(profit_fig)

            with analytic_tabs[1]:
                 
                 if not recent_public_trades_df.empty:
                    st.subheader(f'Recent Trades for {strike_price:.0f} {option_type.upper()}')
                    st.dataframe(recent_public_trades_df, use_container_width=True, hide_index=False)
                 else :
                     st.warning(f'No Trade History for {strike_price:.0f} {option_type.upper()}')

            with analytic_tabs[2]:
                st.subheader(f'Available Expirations for {strike_price:.0f} {option_type.upper()}')
                df_options_for_strike = df_options_for_strike.drop(columns=['Strike_Price', 'Option Type'], errors='ignore')
                st.dataframe(df_options_for_strike, use_container_width=True, hide_index=True)
 
 #--------------------------------------------------------------
#-----------------------Simulation ----------------------------
#-------------------------------------------------------------   

    with main_tabs[2]: 
        simulation_tabs = st.tabs(["Startegy",  "Manual"])
        with simulation_tabs[0]:
            row_option_details = st.container()
            sim_strategy_list , sim_verticalline , sim_buttons= st.columns([0.8 ,0.01, 0.1 ])
            with sim_strategy_list:
                option_details_list = []
                if 'stored_strategy_dict' in st.session_state:
                        startegy_dict = st.session_state['stored_strategy_dict']
                        if startegy_dict:  # Check if dictionary is not empty
                            for instrument, side in startegy_dict.items():
                                # Fetch option data for each instrument
                                strategy_option_details = fetch_data.fetch_option_chain(option_symbol=instrument)
                                print(strategy_option_details)
                                if not strategy_option_details.empty:
                                    # Extract details safely
                                    sim_option_details = strategy_option_details.copy()  # Make a copy to avoid the warning
                                    sim_last_price = sim_option_details['Last_Price_USD'].values[0] 
                                    sim_bid_price = sim_option_details['Bid_Price_USD'].values[0] 
                                    sim_ask_price = sim_option_details['Ask_Price_USD'].values[0] 
                                    sim_delta = sim_option_details['Delta'].values[0] 
                                    sim_gamma = sim_option_details['Gamma'].values[0]
                                    sim_theta = sim_option_details['Theta'].values[0]
                                    sim_vega = sim_option_details['Vega'].values[0]
                                    sim_bid_iv = sim_option_details['Bid_IV'].values[0]
                                    sim_ask_iv = sim_option_details['Ask_IV'].values[0]
                                    sim_probability = sim_option_details['Probability_Percent'].values[0]
                                    sim_side = side
                                    sim_option_details.loc[:, 'Side'] = side

                                    c1, c2 = st.columns([0.9,0.1])
                                    with c1 : 
                                        st.markdown(
                                            f"""
                                            <div style='display: flex; justify-content: flex-start; gap: 10px; padding-top: 10px;'>
                                                <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                    <div style='font-size: x-small; color: gray;'>Instrument</div>
                                                    {instrument}
                                                </div>
                                                <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                    <div style='font-size: x-small; color: gray;'>Last Price (USD)</div>
                                                    {sim_last_price:.0f}
                                                </div>
                                                <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                    <div style='font-size: x-small; color: gray;'>Bid Price (USD)</div>
                                                    {sim_bid_price:.0f}
                                                </div>
                                                <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                    <div style='font-size: x-small; color: gray;'>Ask Price (USD)</div>
                                                    { sim_ask_price:.0f}
                                                </div>
                                                <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                    <div style='font-size: x-small; color: gray;'>Bid IV %</div>
                                                    {  sim_bid_iv:.0f}
                                                </div>
                                                <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                    <div style='font-size: x-small; color: gray;'>Ask IV %</div>
                                                    {  sim_ask_iv:.0f}
                                                </div>
                                                <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                    <div style='font-size: x-small; color: gray;'>Delta</div>
                                                    {sim_delta:.2f}
                                                </div>
                                                <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                    <div style='font-size: x-small; color: gray;'>Gamma</div>
                                                    {sim_gamma:.2f}
                                                </div>
                                                <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                    <div style='font-size: x-small; color: gray;'>Theta</div>
                                                    {sim_theta:.2f}
                                                </div>
                                                <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                    <div style='font-size: x-small; color: gray;'>Vega</div>
                                                    {sim_vega:.2f}
                                                </div>
                                                <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                    <div style='font-size: x-small; color: gray;'>Side</div>
                                                    {sim_side}
                                                </div>
                                                <div style='border:1px solid gray;padding:10px;border-radius:5px; text-align: center;'>
                                                    <div style='font-size: x-small; color: gray;'>Probability</div>
                                                    {sim_probability }
                                                </div>
                                            </div>
                                            """, 
                                            unsafe_allow_html=True
                                        )
                                    with c2: 
                                        sim_size = st.number_input(label="Set Size", min_value=0.1, value=0.1, key=f"sim_size_startegy{instrument}")
                                        sim_option_details.loc[:, 'Size'] = sim_size
                                        option_details_list.append(sim_option_details)
                                    
                                else:
                                    st.warning(f"No details found for instrument: {instrument}")
                        else:
                            st.warning("Picked strategy is empty.")
                else:
                    st.warning("No strategy has been picked yet.")
            if 'stored_strategy_dict' in st.session_state and st.session_state['stored_strategy_dict']:
                with sim_verticalline: 
                    st.markdown("<div style='height: 150px; width: 1px; background-color: gray; margin: auto;'></div>", unsafe_allow_html=True)  # Vertical line
                with sim_buttons:
                        apply_simlutaion = st.button(label="Apply Strategy", key="apply_simulation")
                        remove_strategy = st.button(label="Remove Strategy", key="Remove strategy")
                        if remove_strategy:
                            st.session_state['stored_strategy_dict'] = {}
                        

            st.markdown("---")

            if option_details_list:
                        if apply_simlutaion:
                            sim_strategy_data = pd.concat(option_details_list, ignore_index=True)
                            # Now you can use strategy_data with plot_public_profits
                            #sim_strategy_data =  fetch_data.validate_datatable(sim_strategy_data_raw, "Trade")
                            sim_fig_profit, sim_fig_strategy = plot_public_profits(sim_strategy_data, "Trade", trade_option_details=None)
                            sim_col1,sim_col2 = st.columns(2)
                            with sim_col1:
                                st.plotly_chart(sim_fig_strategy )
                            with sim_col2:
                                st.plotly_chart(sim_fig_profit)
                        else:
                            st.warning("Set values then Press apply button to see the profit chart")
            else:
                        st.warning("No valid option details found for the picked strategy.")


        with simulation_tabs[1]:
            row_inputs = st.container()
            with row_inputs:
                cool1,cool2,cool3,cool4,cool5, cool6, cool7, button  = st.columns(8)
                with cool1: 
                    sim_expiration_date = st.date_input("Expiration Date")
                with cool2:
                    sim_strike_price = st.number_input("Strike Price", min_value=1000, step=500, value=90000, key = "strike_number_input")
                with cool3:
                    
                    sim_type = st.selectbox("Type", options=["Put", "Call"], key="type_select")
                with cool4:
                    
                    sim_side = st.selectbox("Side", options=["BUY", "SELL"], key= "side_select")
                    
                with cool5:
                    
                    sim_size = st.number_input("Size", min_value=0.1, step=0.1, key = "sim_size_input")
                with cool6:
                    
                    sim_price_usd = st.number_input("Price (USD)", min_value=0.1, step=0.1, value=10.0, key = "sim_price_input")
                with cool7:
                
                    iv_percent = st.number_input("IV (%)", min_value=0.1, step=0.1 , value=45.5, key= "sim_iv_input")
                with button:
                    apply_button = st.button("Simulate")


            # Create a DataFrame with the collected inputs
            if apply_button:
                simulation_details = pd.DataFrame({
                    'Strike_Price': [sim_strike_price],
                    'Expiration_Date': [sim_expiration_date ],
                    'Option_Type' : [sim_type],
                    'Size': [sim_size],
                    'Side': [sim_side],
                    'IV_Percent': [iv_percent],
                    'Price_USD': [sim_price_usd]
                })
                #simulation_details =  fetch_data.validate_datatable(simulation_details_raw, "Trade")
                profit_fig_sim , expiration_profit_sim = plot_public_profits(simulation_details , "Public", trade_option_detail)

                
                chart_col_1, chart_col_2 = st.columns(2)
                with chart_col_1:
                                
                    st.plotly_chart(expiration_profit_sim )
                with chart_col_2:
                    st.plotly_chart( profit_fig_sim )
            else: 
                st.warning("Set values then press simulate button")
    
 #--------------------------------------------------------------
#-----------------------Technical analysis ----------------------------
#-------------------------------------------------------------   
    with main_tabs[3]: 
        #-------------------------- Trend -----------------------------
        trend_row = st.container()
        with trend_row:
            trend_col1 , trend_col2, trend_col3 = st.columns([0.2,0.4,0.2])
            with trend_col2:
                padding , title, trend_column , price_action_column= st.columns([0.1, 0.1,0.2,0.2])

                last_4h_trend = st.session_state.technical_4h.get("last_predicted_trend", 'Key not found') 
                last_daily_trend = st.session_state.technical_daily.get("last_predicted_trend", 'Key not found')
                last_4h_price_action = st.session_state.technical_4h.get("active_price_action", 'Key not found')
                last_daily_price_action = st.session_state.technical_daily.get("active_price_action", 'Key not found')
                if last_4h_trend == "Neutral":
                        textcolor_4h_trend = "white"
                elif last_4h_trend == "Bullish":
                        textcolor_4h_trend = "#90EE90" 
                elif last_4h_trend == "Bearish":
                        textcolor_4h_trend = "#f54b4b"
                else:
                        textcolor_4h_trend = "white"  # Default color if _4h_trend is None or any other value

                if last_daily_trend == "Neutral":
                        textcolor_daily_trend = "white"
                elif last_daily_trend == "Bullish":
                        textcolor_daily_trend = "#90EE90"
                elif last_daily_trend == "Bearish":
                        textcolor_daily_trend = "#f54b4b"
                else:
                        textcolor_daily_trend = "white"  # Default color if _4h_trend is None or any other value

                with title: 
                    st.markdown(f"<div style='font-size: 14px; color:gray; text-align: right; margin-top:20px;'>Timeframe</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 14px; color:gray; text-align: right;margin-top:15px; '>4 Hours</div>", unsafe_allow_html=True)

                with trend_column:
                    
                    st.markdown(f"<div style='font-size: 14px; color:gray; text-align: center; margin-top:20px;'>Trend</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 14px; color:{textcolor_4h_trend}; text-align: center; letter-spacing: 2px;margin-top:15px;'>{last_4h_trend}</div>", unsafe_allow_html=True)

                with title:
                    st.markdown(f"<div style='font-size: 14px; color:gray; text-align: right; margin-top:5px;'>Daily</div>", unsafe_allow_html=True)  
                with trend_column:  
                    st.markdown(f"<div style='font-size: 14px; color:{ textcolor_daily_trend}; text-align: center; letter-spacing: 2px;margin-top:5px;'>{last_daily_trend}</div>", unsafe_allow_html=True)
                with price_action_column:
                    st.markdown(f"<div style='font-size: 14px; color:gray; text-align: left; margin-top:20px;'>Active Price Action</div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size: 14px; color:white; text-align: left; letter-spacing: 2px;margin-top:15px;'>{last_4h_price_action:.0f}</div>", unsafe_allow_html=True)
                    
                    st.markdown(f"<div style='font-size: 14px; color:white; text-align: left; letter-spacing: 2px;margin-top:5px;'>{last_daily_price_action:.0f}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ------------------------Probability --------------------------------

        all_probabilities_df, top_probability_instrument = fetch_data.get_instrument_probabilities()
        
        tch_leftpadding, tch_col2, prob_padding,  tch_col3 ,tch_rightpadding = st.columns([0.2, 0.3,0.05, 0.4, 0.2])
        with tch_col2:
            prob_filter_column_1 , prob_filter_column_2  = st.columns(2)

            with prob_filter_column_1:
                prob_expiration = st.multiselect(label="Expiration Date", options=sorted_market_available_dates, key="prob_expiration")
            
            with prob_filter_column_2:
                prob_selected_strike_prices = st.multiselect(label="Strike Price", options=sorted_strikes , key="prob_strike_price")

            
            # Check if any expiration dates are selected
            if prob_expiration:
                # Formatting selected expiration dates correctly to match the required format in instrument names
                prob_expiration_format = [date.strftime("%#d%b%y").upper()  for date in pd.to_datetime(prob_expiration)]
                all_probabilities_df = all_probabilities_df[all_probabilities_df['Instrument'].str.contains('|'.join(prob_expiration_format))]
            else:
                # If no expiration date is selected, keep the DataFrame as is or handle it accordingly
                pass  # (or set all_probabilities_df to an empty DataFrame, if needed)

            if prob_selected_strike_prices:
                    strike_price_strs = [str(int(price)) for price in prob_selected_strike_prices] 
                    all_probabilities_df = all_probabilities_df[all_probabilities_df['Instrument'].str.contains('|'.join(strike_price_strs))]

            all_probabilities_df = all_probabilities_df.dropna()
            st.dataframe(all_probabilities_df, use_container_width=True, hide_index=True)

        # ---------------- Cumulative Trend ----------------------       
        with tch_col3:
            prob_fig =  plot_probability_heatmap(all_probabilities_df)
            st.plotly_chart(prob_fig)  
            
            
        st.markdown("---")
        left_padding, tch_col1, right_padding = st.columns([0.1,0.4 ,0.1])
        with tch_col1:
            timeframe = ["4h", "Daily"]
            fig_trend = plot_predicted_trend(timeframes=timeframe)
            st.plotly_chart(fig_trend )    

#--------------------------------------------------------------
#-----------------------Combinations ----------------------------
#-------------------------------------------------------------
    # with main_tabs[4]:
    #
    #         if st.session_state.most_profitable_df.shape[1] > 1:  # Assuming at least 'Underlying Price' and one profit column exists
    #                 most_profitable_df = st.session_state.most_profitable_df
    #                 num_combos = most_profitable_df.shape[1]
    #                 num_combo_to_show = 10  # Number of columns to show per tab
    #                 num_tabs = (num_combos // num_combo_to_show) + (num_combos % num_combo_to_show > 0)
    #
    #                         # Create tab names based on the number of combinations
    #                 tab_names = [f"Combination {i + 1}" for i in range(num_tabs)]
    #
    #                         # Create the tabs dynamically
    #                 combo_tabs = st.tabs(tab_names)
    #
    #                         # Loop through each tab and display the corresponding data
    #                 for i, tab in enumerate(combo_tabs):
    #                     with tab:
    #                                 # Copy the main DataFrame to avoid modifying original
    #                         display_df = most_profitable_df.copy()
    #                                 
    #                                 # Isolate and remove the 'Underlying Price' column
    #                         underlying_price = display_df.pop('Underlying Price')
    #
    #                                 # Determine the columns to display in this tab
    #                         start_col = i * num_combo_to_show
    #                         end_col = start_col + num_combo_to_show
    #
    #                                 # Slice the DataFrame for the current tab.
    #                         columns_to_display = display_df.columns[start_col:end_col]
    #                                 
    #                                 # Create a new DataFrame for display without the 'Underlying Price'
    #                         sliced_df = display_df[list(columns_to_display)]
    #                                 
    #                                 # Insert the 'Underlying Price' column at the front
    #                         sliced_df.insert(0, 'Underlying Price', underlying_price)
    #
    #                                 # Render the DataFrame with Streamlit
    #                         styled_results = style_combined_results(sliced_df)  # Pass the full DataFrame
    #
    #                                 # Render the styled DataFrame in the tab using markdown
    #                         st.markdown(styled_results.to_html(escape=False), unsafe_allow_html=True)
    #             
    #         else:
    #             st.warning("No combinations meet the criteria, Press Analyze then Filter button.")
    #  


def style_combined_results(combined_results):
    """
    Apply conditional formatting to the combined results DataFrame with richer color distinctions.

    Parameters:
        combined_results (pd.DataFrame): The DataFrame to apply styles on.

    Returns:
        pd.io.formats.style.Styler: A styled DataFrame for better insights.
    """
    def color_profits(value):
        """
        Color profits based on their values using a gradient:
        - Green for positive profits
        - Yellow for values transitioning around zero
        - Red for negative profits
        """
        if value > 200:
            r = 0  # No red
            g = 255  # Full green
            b = 0  # No blue
            return f'background-color: rgba({r}, {g}, {b}, 1.0); color: black'  # Bright green

        elif 0 < value <= 200:
            # Gradient from green (at 200) to yellow (at 0)
            r = int((255 * (200 - value)) / 200)  # More red as value decreases
            g = 255  # Green stays full
            b = 0  # No blue
            return f'background-color: rgba({r}, {g}, {b}, 1.0); color: black'  # Green to yellow gradient

        elif value == 0:
            # Pure yellow for profit of zero
            r = 255  # Full red
            g = 255  # Full green
            b = 0  # No blue
            return f'background-color: rgba({r}, {g}, {b}, 1.0); color: black'  # Opaque yellow

        elif -100 < value < 0:
            # Gradient from yellow (at 0) to red (at -100)
            r = 255  # Full red
            g = int((255 * (100 + value)) / 100)  # Green decreases as it goes more negative
            b = 0  # No blue
            return f'background-color: rgba({r}, {g}, {b}, 1.0); color: black'  # Yellow to red gradient

        else:
            # Solid red for values lower than -100
            r = 255  # Full red
            g = 0  # No green
            b = 0  # No blue
            return f'background-color: rgba({r}, {g}, {b}, 1.0); color: white'  # Solid red

    # Create a masking function to skip the first row and the "Premium" row
    def apply_color(row):
        # Check if the row name is "Premium", and skip coloring if it is
        if row.name == 'Premium':
            return [''] * len(row)  # No styling
        else:
            return [color_profits(value) for value in row]

    # Apply the styling using Styler.apply() on all columns except the first one
    styled_df = combined_results.style.apply(apply_color, axis=1, subset=combined_results.columns[1:])  # Skip the 'Underlying Price' column

    # Format numeric values with specific precision
    formatted_dict = {
        'Underlying_Price': '{:.0f}',  # 0 decimal for underlying price
    }

    # Apply formatting for the 'Underlying Price' column and 1 decimal for other profit columns
    for col in combined_results.columns[1:]:  # Assuming first two are not profit columns
        formatted_dict[col] = '{:.1f}'  # 1 decimal for profit columns

    # Format the styled DataFrame
    styled_df = styled_df.format(formatted_dict)


    return styled_df


def initializing_states():
    if 'data_refresh_thread' not in st.session_state:
        st.session_state.data_refresh_thread = None
    

    if 'technical_4h' not in st.session_state:
        st.session_state.technical_4h = None

    if 'technical_daily' not in st.session_state:
        st.session_state.technical_daily = None

    if 'show_24_public_trades' not in st.session_state:
        st.session_state.show_24_public_trades = True

    if 'most_profitable_df' not in st.session_state:
        st.session_state.most_profitable_df = pd.DataFrame()  # Initialize with an empty DataFrame

    if 'public_trades_df' not in st.session_state:
        st.session_state.public_trades_df = None
    
    
    if 'selected_date' not in st.session_state:
        default_date = (datetime.now() + timedelta(days=1)).date()
        st.session_state.selected_date = default_date
    if 'option_symbol' not in st.session_state:
            st.session_state.option_symbol = None  # Initialize this as None or an empty value
    if 'quantity' not in st.session_state:
            st.session_state.quantity = 0.1  # Default quantity
    if 'option_side' not in st.session_state:
            st.session_state.option_side = "BUY"  # Default side

def technical_analysis():


    if st.session_state.technical_4h is None:
        analytics_insight_4h = technical_4h.get_technical_data()
        st.session_state.technical_4h  = analytics_insight_4h

    if st.session_state.technical_daily is None:
        analytics_insight_daily = technical_daily.get_technical_data()
        st.session_state.technical_daily = analytics_insight_daily 


def start_data_refresh_thread():
    if st.session_state.data_refresh_thread is None or not st.session_state.data_refresh_thread.is_alive():
        st.session_state.data_refresh_thread = threading.Thread(target=run_async_loop_in_thread, args=(fetch_main,), daemon=True)
        st.session_state.data_refresh_thread.start()
        logger.info("Started background data fetch thread.")

def update_public_trades(filter=None):
    df = fetch_data.load_public_trades(
        symbol_filter=filter,
        show_24h_public_trades=st.session_state.show_24_public_trades
    )
    # Only assign to session state from main thread
    st.session_state.public_trades_df = df

    
def run_async_loop_in_thread(coroutine_func):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(coroutine_func())     

if __name__ == "__main__":
    
    # Check for our custom environment flag.
    if os.environ.get("STREAMLIT_RUN") != "1":
       os.environ["STREAMLIT_RUN"] = "1"

      
        # Launch the Streamlit app using the current Python interpreter.
       subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
    else:
        # We are already running under "streamlit run" – proceed with your app.
        # Start the background thread once (if not already started).
        # Now call your app() function to render the Streamlit interface.
        app()
        
    