import plotly.graph_objects as go
import os
import pandas as pd
from datetime import datetime , timedelta , timezone
import numpy as np
from Analytics import Analytic_processing
from Calculations import calculate_profit
from concurrent.futures import ThreadPoolExecutor

analytics = Analytic_processing()

currency = 'BTC'


# Helper function to format large numbers
def format_large_number(value):
    if abs(value) >= 1e6:
        return f"{value/1e6:.1f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.1f}k"
    else:
        return f"{value:.1f}"
    
def plot_stacked_calls_puts(df):
    """
    Plot a stacked column chart of total Calls and total Puts against Strike Price,
    separating Buy and Sell transactions.

    Parameters:
        df (pd.DataFrame): DataFrame containing options data with 'Strike Price', 
                           'Option Type', and 'Side' columns.
    """
    # Create a DataFrame to hold counts of Calls and Puts by Strike Price
    plot_data = df.copy()

    # Initialize the structure for grouped data
    grouped_data = plot_data.groupby(['Strike_Price', 'Option_Type']).agg(
        Buy_Total=('Side', lambda x: (x == 'BUY').sum()),   # Total Buys
        Sell_Total=('Side', lambda x: (x == 'SELL').sum())  # Total Sells
    ).unstack(fill_value=0)  # Unstack the DataFrame for clarity

    # Flatten the MultiIndex in the columns
    grouped_data.columns = ['_'.join(col).strip() for col in grouped_data.columns.values]

    # Prepare total counts for Calls and Puts based on available data
    grouped_data['Total_Calls'] = grouped_data.get('Buy_Total_Call', 0) + grouped_data.get('Sell_Total_Call', 0)
    grouped_data['Total_Puts'] = grouped_data.get('Buy_Total_Put', 0) + grouped_data.get('Sell_Total_Put', 0)

    # Calculate total trades for percentage calculations
    total_trades = len(df)

    # Create custom data for hover display, calculating distributions
    customdata = []
    for index, row in grouped_data.iterrows():
        buy_calls_total = row.get('Buy_Total_Call', 0)
        sell_calls_total = row.get('Sell_Total_Call', 0)
        buy_puts_total = row.get('Buy_Total_Put', 0)
        sell_puts_total = row.get('Sell_Total_Put', 0)

        customdata.append([ 
            buy_calls_total,
            sell_calls_total,
            buy_puts_total,
            sell_puts_total,
            (buy_calls_total / total_trades * 100),
            (sell_calls_total / total_trades * 100),
            (buy_puts_total / total_trades * 100),
            (sell_puts_total / total_trades * 100),
        ])

    customdata = pd.DataFrame(customdata).values

    # Create an interactive stacked bar chart
    fig = go.Figure()

    # Define the order for stacking (reversed order)
    stack_order = [
        ('Sell', 'Call'),
        ('Buy', 'Call'),
        ('Sell', 'Put'),
        ('Buy', 'Put')
    ]

    # Add traces based on availability of data
    for opt_type in stack_order:
        action, option = opt_type
        y_value = grouped_data.get(f'{action}_Total_{option}', pd.Series([0] * len(grouped_data)))

        # Only add traces if the option type has data
        if any(y_value):
            color = 'green' if action == 'Buy' and option == 'Put' \
                    else 'darkred' if action == 'Sell' and option == 'Put' \
                    else 'teal' if action == 'Buy' and option == 'Call' \
                    else 'darkorange'
            textcolor = 'black' if action == 'Sell' and  option == 'Call' \
                    else 'white'
            fig.add_trace(go.Bar(
                x=grouped_data.index,
                y=y_value,
                name=f'{action} {option}s',
                marker=dict(
                    color=color,
                    line=dict(color='black', width=1) 
                ),
                hovertemplate=(
                    "Strike Price: %{x}<br>" +
                    "Buy Calls: %{customdata[0]} (%{customdata[4]:.2f}%)<br>" +
                    "Sell Calls: %{customdata[1]} (%{customdata[5]:.2f}%)<br>" +
                    "Buy Puts: %{customdata[2]} (%{customdata[6]:.2f}%)<br>" +
                    "Sell Puts: %{customdata[3]} (%{customdata[7]:.2f}%)<br>" 
                ),
                hoverlabel=dict(bgcolor=color,  font=dict(color=textcolor)),  # Set hover background color to match the bar color
                customdata=customdata
            ))

    # Update layout settings
    fig.update_layout(
        xaxis_title='Strike Price',
        yaxis_title='Total Number',
        barmode='stack',
        template="plotly_white"  # Clean background
    )

    return fig

def plot_strike_price_vs_entry_value(filtered_df):
    fig = go.Figure()

    # Create hover text in a vectorized manner
    filtered_df['hover_text'] = (
        "Strike Price: " + filtered_df['Strike_Price'].apply(lambda x: f"{int(x/1000)}k" if x >= 1000 else f"{int(x):,}").astype(str) + "<br>" +
        "Expiry Date: " + filtered_df['Expiration_Date'].astype(str) + "<br>" +
        "Entry Value: " + filtered_df['Entry_Value'].apply(lambda x: f"{int(x):,}").astype(str) + "<br>" +
        "Type: " + filtered_df['Option_Type'] +"<br>" +
        "Size: " + filtered_df['Size'].astype(str) + "<br>" +
        "Side: " + filtered_df['Side'] + "<br>" +
        "Underlying Price: " + filtered_df['Underlying_Price'].apply(lambda x: f"{int(x):,}").astype(str) + "<br>"  +
        "Entry Date: " + filtered_df['Entry_Date'].astype(str) + "<br>" 
    )

    # Add traces for Buy Put options (downward green triangle)
    fig.add_trace(go.Scatter(
        x=filtered_df.loc[(filtered_df['Option_Type'] == 'Put') & (filtered_df['Side'] == 'BUY'), 'Strike_Price'],
        y=filtered_df.loc[(filtered_df['Option_Type'] == 'Put') & (filtered_df['Side'] == 'BUY'), 'Entry_Value'],
        mode='markers',
        marker=dict(symbol='triangle-down', color='green', size=10, opacity=1, line=dict(color='black', width=0.5)),  # Downward green triangle for BUY with black border
        name='Buy Puts',
        hoverinfo='text', 
        hovertext=filtered_df.loc[(filtered_df['Option_Type'] == 'Put') & (filtered_df['Side'] == 'BUY'), 'hover_text'],
        hoverlabel=dict(bgcolor='green')
    ))

    # Add traces for Sell Put options (downward red triangle)
    fig.add_trace(go.Scatter(
        x=filtered_df.loc[(filtered_df['Option_Type'] == 'Put') & (filtered_df['Side'] == 'SELL'), 'Strike_Price'],
        y=filtered_df.loc[(filtered_df['Option_Type'] == 'Put') & (filtered_df['Side'] == 'SELL'), 'Entry_Value'],
        mode='markers',
        marker=dict(symbol='triangle-up', color='darkred', size=10, opacity=1, line=dict(color='black', width=0.5)),  # Downward red triangle for SELL with black border
        name='Sell Puts',
        hoverinfo='text', 
        hovertext=filtered_df.loc[(filtered_df['Option_Type'] == 'Put') & (filtered_df['Side'] == 'SELL'), 'hover_text'],
        hoverlabel=dict(bgcolor='darkred')
    ))

    # Add traces for Buy Call options (upward green triangle)
    fig.add_trace(go.Scatter(
        x=filtered_df.loc[(filtered_df['Option_Type'] == 'Call') & (filtered_df['Side'] == 'BUY'), 'Strike_Price'],
        y=filtered_df.loc[(filtered_df['Option_Type'] == 'Call') & (filtered_df['Side'] == 'BUY'), 'Entry_Value'],
        mode='markers',
        marker=dict(symbol='triangle-up', color='teal', size=10, opacity=1, line=dict(color='black', width=0.5)),  # Upward teal triangle for BUY with black border
        name='Buy Calls',
        hoverinfo='text', 
        hovertext=filtered_df.loc[(filtered_df['Option_Type'] == 'Call') & (filtered_df['Side'] == 'BUY'), 'hover_text'],
        hoverlabel=dict(bgcolor='teal')
    ))

    # Add traces for Sell Call options (upward red triangle)
    fig.add_trace(go.Scatter(
        x=filtered_df.loc[(filtered_df['Option_Type'] == 'Call') & (filtered_df['Side'] == 'SELL'), 'Strike_Price'],
        y=filtered_df.loc[(filtered_df['Option_Type'] == 'Call') & (filtered_df['Side'] == 'SELL'), 'Entry_Value'],
        mode='markers',
        marker=dict(symbol='triangle-down', color='darkorange', size=10, opacity=1, line=dict(color='black', width=0.5)),  # Upward dark orange triangle for SELL with black border
        name='Sell Calls',
        hoverinfo='text', 
        hovertext=filtered_df.loc[(filtered_df['Option_Type'] == 'Call') & (filtered_df['Side'] == 'SELL'), 'hover_text'],
        hoverlabel=dict(bgcolor='darkorange', font=dict(color='black'))
    ))

    # Update layout
    fig.update_layout(
        xaxis_title='Strike Price',
        yaxis_title='Entry Value',
        template="plotly_white",  # Use a clean white template
        showlegend=True
    )

    return fig

def plot_radar_chart(df_options_for_strike):
    # Ensure required columns exist
    if 'Expiration_Date' not in df_options_for_strike.columns or 'Open_Interest' not in df_options_for_strike.columns:
        print("DataFrame must contain 'Expiration_Date' and 'Open_Interest' columns.")
        return go.Figure()

    # Sort the DataFrame by Expiration_Date in ascending order
    df_sorted = df_options_for_strike.sort_values(by='Expiration_Date')

    # Convert to datetime
    exp_dates = pd.to_datetime(df_sorted['Expiration_Date'])

    # Format for axis and display
    categories = exp_dates.dt.strftime('%m/%d/%Y').tolist()
    formatted_categories = exp_dates.dt.strftime('%#d %B').tolist()

    # Extract values
    values = df_sorted['Open_Interest'].tolist()

    # Close the loop for radar chart
    values += values[:1]
    categories += categories[:1]
    formatted_categories += formatted_categories[:1]

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Open Interest',
        line=dict(color='red', width=2),
        hovertemplate='Open Interest: %{r}<br>Expiration Date: %{theta}<extra></extra>'
    ))

    # Layout
    fig.update_layout(
        title='Open Interest by Expiration Date',
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True,
                range=[0, max(values) + 10],
            ),
            angularaxis=dict(
                direction='clockwise',  # Ensure clockwise
                tickcolor='white',
                tickfont=dict(color='white'),
                tickvals=categories,
                ticktext=formatted_categories
            )
        ),
        showlegend=True
    )

    return fig

def plot_public_profit_sums(summed_df):
    """
    Plot the Underlying Price against the Sum of Profits using Plotly.
    
    Parameters:
        summed_df (pd.DataFrame): DataFrame containing 'Underlying Price' and 'Sum of Profits'.
    """
    fig = go.Figure()

    # Add a line trace
    fig.add_trace(go.Scatter(
        x=summed_df['Underlying_Price'],
        y=summed_df['Sum of Profits'],
        mode='lines+markers',
        name='Sum of Profits',
        line=dict(shape='linear'),  # Change line shape to linear
        marker=dict(size=8)
    ))

    fig.update_layout(
        title='Underlying Price vs. Sum of Profits',
        xaxis_title='Underlying Price',
        yaxis_title='Sum of Profits',
        template='plotly_white'
    )

    return fig

def plot_most_traded_instruments(most_traded):
    """
    Plots a pie chart of the most traded instruments with hover info showing
    total trade counts, buy, and sell counts, each clearly displayed.
    """
    # Create hover text to display buy and sell portions clearly
    most_traded['hover_text'] = (
        "Instrument: " + most_traded['Instrument'] + "<br>" +
        "Total Trades: " + most_traded['Trade Count'].astype(int).astype(str) + "<br>" +  # Total trade count
        "Buy Contracts: " + most_traded['BUY'].astype(str) + " - " + ((most_traded['BUY'] / (most_traded['SELL'] + most_traded['BUY'])) * 100).round(2).astype(str) + "%" + "<br>" +  # Buy percentage
        "Sell Contracts: " + most_traded['SELL'].astype(str) + " - " + ((most_traded['SELL'] / (most_traded['SELL'] + most_traded['BUY'])) * 100).round(2).astype(str) + "%"  # Sell percentage
    )

    # Create a pie chart using Plotly
    fig = go.Figure(data=[go.Pie(
        labels=most_traded['Instrument'],
        values=most_traded['Trade Count'],  # Use the count of trades as the values for the pie chart
        hole=0.6,
        hoverinfo='text',
        textinfo='value+percent+label',
        text=most_traded['hover_text']
    )])

    # Update the layout of the chart
    fig.update_layout(
        title_text='Top 10 Most Traded Instruments by Trade Count',
        title_font_size=24
    )

    return fig

def plot_underlying_price_vs_entry_value(df, custom_price=None, custom_entry_value=None):
    # Helper function to format date with error handling
    def format_entry_date(date_obj):
        if isinstance(date_obj, datetime):
            return date_obj.strftime('%Y-%m-%d %H:%M:%S')
        return 'Invalid Date'

    # Create a scatter plot of Entry Value against Underlying Price
    fig = go.Figure()

    # Separate data for BUY and SELL with corresponding colors
    df_sell = df[df['Side'] == 'SELL']
    df_buy = df[df['Side'] == 'BUY']
    # Avoid SettingWithCopyWarning by assigning to a new DataFrame
    df_sell = df_sell.assign(Formatted_Entry_Date=df_sell['Entry_Date'].apply(format_entry_date))
    df_buy = df_buy.assign(Formatted_Entry_Date=df_buy['Entry_Date'].apply(format_entry_date))

    # Add SELL data points (red)
    fig.add_trace(go.Scatter(
        x=df_sell['Entry_Value'],
        y=df_sell['Underlying_Price'],
        mode='markers',
        marker=dict(size=10, 
                    color='red', 
                    opacity=0.6,       
                    line=dict(        
                        color='black', # Color of the border
                        width=1         # Width of the border
                )),
        name='SELL',
        hovertemplate=(  
            '<b>Underlying Price:</b> %{y:.1f}<br>'  
            '<b>Entry Value:</b> %{customdata[0]}<br>' 
            '<b>Size:</b> %{customdata[2]}<br>'   
            '<b>Entry Date:</b> %{customdata[1]}<br>' 
            '<extra></extra>'  
        ),
        customdata=np.array([
            [format_large_number(x), date, size]
            for x, date, size in zip(df_sell['Entry_Value'], df_sell['Formatted_Entry_Date'], df_sell['Size'])
        ])
    ))

    # Add BUY data points (white)
    fig.add_trace(go.Scatter(
        x=df_buy['Entry_Value'],
        y=df_buy['Underlying_Price'],
        mode='markers',
        marker=dict(
            size=10, 
            color='white', 
            opacity=0.6,       
            line=dict(        
                color='black', # Color of the border
                width=1         # Width of the border
        )),
        name='BUY',
        hovertemplate=(  
            '<b>Underlying Price:</b> %{y:.1f}<br>'  
            '<b>Entry Value:</b> %{customdata[0]}<br>'
            '<b>Size:</b> %{customdata[2]}<br>'  
            '<b>Entry Date:</b> %{customdata[1]}<br>'  
            '<extra></extra>'  
        ),
        customdata=np.array([
            [format_large_number(x), date, size]
            for x, date, size in zip(df_buy['Entry_Value'], df_buy['Formatted_Entry_Date'], df_buy['Size'])
        ])
    ))

    # Custom point if provided
    if custom_price is not None and custom_entry_value is not None:
        fig.add_trace(go.Scatter(
            x=[custom_entry_value],
            y=[custom_price],
            mode='markers',
            marker=dict(size=15, 
                        color='green', 
                        symbol='circle',       
                        line=dict(        
                            color='black', # Color of the border
                            width=1         # Width of the border
                    )),
            name='Your Option',
            hovertemplate='<b>BTC Price:</b> ' + format_large_number(custom_price)  + '<br>' +
                          '<b>Your Premium:</b> ' + format_large_number(custom_entry_value) + '<br>' +
                          '<extra></extra>'
        ))

    # Update layout
    fig.update_layout(
        title='Your Premium vs Others',
        xaxis_title='Premium',    
        yaxis_title=f'{currency} Price',       
        template='plotly_dark',
        hoverlabel=dict(bgcolor='black', font_color='white')
    )

    return fig

def plot_option_price_vs_entry_date(df):
    # Ensure 'Entry Date' is in datetime format
    df['Entry_Date'] = pd.to_datetime(df['Entry_Date'])


    # Create traces for BUY and SELL
    buy_df = df[df['Side'] == 'BUY']
    sell_df = df[df['Side'] == 'SELL']

    fig = go.Figure()

    # Add BUY trace with transparency
    fig.add_trace(go.Scatter(
        x=buy_df['Entry_Date'],
        y=buy_df['Price_USD'],  # Use 'Price (USD)' here
        mode='lines',  # Use lines
        name='BUY',
        line=dict(color='white', width=2),  # White line
        marker=dict(size=8),
        hovertext=(
            'Time: ' + buy_df['Entry_Date'].dt.strftime('%H:%M') + '<br>' +
            'Price (USD): ' + buy_df['Price_USD'].apply(format_large_number) + '<br>' +  
            'Underlying Price: ' + buy_df['Underlying_Price'].apply(format_large_number) + '<br>' +
            'Size: ' + buy_df['Size'].astype(str) + '<br>' +
            'Side: ' + buy_df['Side'] + '<br>' 
        ),
        hoverinfo='text',
        hoverlabel=dict(bgcolor='black', font=dict(color='white')),  # Set hover background to black
        opacity=0.6  # Set trace transparency here
    ))

    # Add SELL trace with transparency
    fig.add_trace(go.Scatter(
        x=sell_df['Entry_Date'],
        y=sell_df['Price_USD'],  # Use 'Price (USD)' here
        mode='lines',  # Use lines
        name='SELL',
        line=dict(color='red', width=2),  # Red line
        marker=dict(size=8),
        hovertext=(
            'Time: ' + buy_df['Entry_Date'].dt.strftime('%H:%M') + '<br>' +
            'Price (USD): ' + buy_df['Price_USD'].map('{:.1f}'.format) + '<br>' +  
            'Underlying Price: ' + buy_df['Underlying_Price'].map('{:.1f}'.format).astype(str) + '<br>' +
            'Size: ' + sell_df['Size'].astype(str) + '<br>' +
            'Side: ' + sell_df['Side'] + '<br>' 
        ),
        hoverinfo='text',
        hoverlabel=dict(bgcolor='black', font=dict(color='white')),  # Set hover background to black
        opacity=0.6  # Set trace transparency here
    ))

    # Update layout with transparent background
    fig.update_layout(
        title='Option Price in Time',
        xaxis_title='Entry Date',
        yaxis_title='Price (USD)',  # Updated y-axis label
        plot_bgcolor='rgba(0, 0, 0, 0)',  # Transparent background for the entire figure
        font=dict(color='white'),  # White font for labels
        xaxis=dict(
            showgrid=False,
            title_standoff=10,
            zerolinecolor='gray'  # Gray zero line
        ),
        yaxis=dict(
            showgrid=False,
            title_standoff=10,
            zerolinecolor='gray'  # Gray zero line
        ),
        hovermode='closest'
    )

    # Show the plot
    return fig

def plot_identified_whale_trades(df, 
                                 min_size=5, 
                                 max_size=50, 
                                 min_opacity=0.3, 
                                 max_opacity=1.0, 
                                 entry_value_threshold=None, 
                                 filter_type=None):
    if filter_type == "Entry Value":
        filter_type_str = 'Entry_Value'
    else:
        filter_type_str = 'Size'

    def scale_marker_size_and_opacity(entry_values):
        if entry_values.max() == entry_values.min():
            # All values are the same, avoid division by zero
            normalized_values = np.ones_like(entry_values)
        else:
            normalized_values = (entry_values - entry_values.min()) / (entry_values.max() - entry_values.min())
        scaled_sizes = normalized_values * (max_size - min_size) + min_size
        scaled_opacities = normalized_values * (max_opacity - min_opacity) + min_opacity
        return scaled_sizes, scaled_opacities

    def format_value(value):
        """Format the value to show 'k' for thousands and 'M' for millions."""
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            return f"{value / 1_000:.0f}k"
        else:
            return f"{value:,}"
        
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

    # Ensure 'Entry Date' is in datetime format
    df['Entry_Date'] = pd.to_datetime(df['Entry_Date'])

    # Calculate IQR for 'Entry Value' or 'Size'
    Q1 = df[filter_type_str].quantile(0.25)
    Q3 = df[filter_type_str].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate the upper bound for outliers
    upper_bound = Q3 + 1.5 * IQR

    # Use the manual threshold if provided, otherwise use the calculated upper bound
    threshold = entry_value_threshold if entry_value_threshold is not None else upper_bound

    # Filter out the outliers based on the threshold
    outliers = df[df[filter_type_str] > threshold].copy()
    outliers = outliers.reset_index(drop=True)  # Reset index for positional access

    # Scale marker sizes and opacities
    marker_sizes, marker_opacities = scale_marker_size_and_opacity(outliers[filter_type_str])

    # Create a Plotly figure
    fig = go.Figure()

    # Group by 'Entry Date' and plot each group
    grouped = outliers.groupby('Entry_Date')

    for entry_date, group in grouped:
        # Prepare hover text for all markers with the same entry date
        hover_text = "<b>Connected Trades:</b><br>"
        for _, row in group.iterrows():
                hover_text += (
                    f"{format_value(row['Strike_Price'])} | {format_value(row[filter_type_str])} | {row['Side']} | {row['Option_Type']} | {format_date(row['Expiration_Date'])}<br>"
                )
        

        # Add markers for each outlier
        for _, row in group.iterrows():
            # Find the position of this row in outliers (positional index)
            outlier_pos = row.name  # After reset_index, .name is the position
            marker_color = 'red' if pd.notnull(row['BlockTrade_IDs']) else 'yellow'
            
            fig.add_trace(go.Scatter(
                x=[row['Entry_Date']],
                y=[row['Strike_Price']],
                mode='markers',
                marker=dict(
                    size=marker_sizes[outlier_pos],  # Use scaled marker size
                    color=marker_color,
                    opacity=marker_opacities[outlier_pos],  # Use scaled opacity
                    line=dict(color='black', width=1)  # Remove stroke around markers
                ),
                name=f"Strike: {row['Strike_Price']}",
                hovertemplate=(
                    f"Entry Date:<br> {row['Entry_Date']}<br><br>"
                    f"{hover_text}<extra></extra>"
                )
            ))

    # Update layout of the figure
    fig.update_layout(
        xaxis_title='Entry Date',
        yaxis_title='Strike Price',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(color='white'),  # Set font color to white for contrast
        xaxis=dict(showgrid=False, zerolinecolor='gray'),
        yaxis=dict(showgrid=False, zerolinecolor='gray'),
        hovermode='closest'
    )

    # Return the filtered DataFrame and the figure
    return outliers, fig

def plot_expiration_profit(strategy_data, chart_type, trade_option_details):
    
    # Determine the minimum and maximum strike prices
    min_strike_price = strategy_data['Strike_Price'].min()
    max_strike_price = strategy_data['Strike_Price'].max()

    # Generate index price range based on strike prices
    index_price_range = np.arange(min_strike_price - 35000, max_strike_price + 35000, 1000)
    profit_matrix = []
    position_labels = []

    for i, position in strategy_data.iterrows():
        # Extract necessary data for profit calculation
        if chart_type == "Trade":

            if 'Side' in strategy_data.columns:
                trade_direction = strategy_data['Side'].iloc[i]
            else:
                trade_direction = trade_option_details[0]
                
            if 'Size' in strategy_data.columns:
                trade_qty = strategy_data['Size'].iloc[i]
            else:
                trade_qty = trade_option_details[1]
            if trade_direction == "Buy":
                position_value = position['Ask_Price_USD']
            else:
                position_value = position['Bid_Price_USD']
            position_size = trade_qty
            position_side = trade_direction

        if chart_type == "Public":
            position_size = position['Size']
            position_side = position['Side']
            position_value = position['Price_USD']

        strike_price = position['Strike_Price']
        position_type = position['Option_Type'].lower()

        position_label = f"{int(strike_price)} - {position_type.upper()} - {position_side.upper()}"
        position_labels.append(position_label)  # Add to the list of labels

        premium_value = position_size * position_value
        if position_type == "put":
            breakeven = strike_price - premium_value
        else:
            breakeven = premium_value + strike_price

        profits = [
            calculate_profit(
                current_price=index_price,
                option_price=position_value,
                strike_price=strike_price,
                option_type=position_type,
                quantity=position_size,
                is_buy=(position_side.lower() == 'buy')
            )
            for index_price in index_price_range
        ]
        
        # Append profits to the matrix
        profit_matrix.append(profits)
    
    # Convert profit matrix to a numpy array for plotting
    profit_matrix = np.array(profit_matrix)
    
    # Calculate the sum of profits for each underlying price
    total_profit_row = np.sum(profit_matrix, axis=0)
    
    # Append the total profit row to the profit matrix
    profit_matrix = np.vstack([profit_matrix, total_profit_row])
    
    # Determine if all profit values are negative
    all_negative = np.all(total_profit_row < 0)
    
    # Choose colorscale based on profit values
    colorscale = 'Reds' if all_negative else 'RdYlGn'
    
    # Create a Plotly figure
    fig = go.Figure()

    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=profit_matrix,
        x=index_price_range,
        y=np.arange(len(strategy_data) + 1),  # Adjust for the new total profit row
        colorscale=colorscale,  # Use conditional colorscale
        colorbar=dict(title='Profit'),
        hovertemplate=(
            "Expiration Profit<br>"
            "Underlying Price: %{x}<br>"  # Use custom data for x-axis
            "Profit: %{customdata}<br>"  # Use custom data for z-axis
            "<extra></extra>"  # Suppress default hover info
        ),
        customdata=[
            [f"{int(val/1e6)}M" if abs(val) >= 1e6 else f"{int(val/1e3)}k" if abs(val) >= 1e3 else f"{int(val):,}" for val in row]
            for row in profit_matrix
        ],
        showscale=True,
        zsmooth=False,  # Disable smoothing to make grid lines visible
        xgap=0.5,  # Add gap between x values
        ygap=5   # Add gap between y values
    ))

    # Update layout
    fig.update_layout(
        xaxis_title='Underlying Price',
        yaxis=dict(tickvals=np.arange(len(strategy_data) + 1), ticktext=[*position_labels, 'Total Profit']),
        showlegend=False
    )

    return fig

def plot_all_days_profits(group , chart_type , trade_option_details):
    # Calculate the minimum and maximum time to expiration in days for existing positions


    # Determine the minimum and maximum strike prices
    min_strike_price = group['Strike_Price'].min()
    max_strike_price = group['Strike_Price'].max()

    # Generate index price range based on strike prices
    index_price_range = np.arange(min_strike_price - 35000, max_strike_price + 35000, 1000)

    expiration_dates = [
        (pd.to_datetime(position['Expiration_Date'], utc=True) - datetime.now(timezone.utc)).days 
        for i, position in group.iterrows()
    ]
    max_time_to_expiration_days = max(expiration_dates)
    if max_time_to_expiration_days <= 1:
        max_time_to_expiration_days = 1
    
    # Calculate profits for each day until expiration
    days_to_expiration = np.arange(1, max_time_to_expiration_days + 1)
    profit_over_days = {day: [] for day in days_to_expiration}
    
    for day in days_to_expiration:
        daily_profits = []
        for i, position in group.iterrows():
            # Calculate profits for the current day
            expiration_date = pd.to_datetime(position['Expiration_Date'], utc=True)
            now_utc = datetime.now(timezone.utc)
            time_to_expiration_days = expiration_date - now_utc - timedelta(days=int(day))  # Convert to int
            
            time_to_expiration_years = max(time_to_expiration_days.total_seconds() / (365 * 24 * 3600), 0.0001)

            if chart_type == "Trade" : 
                
                if 'Side' in group.columns:
                    trade_direction = group['Side'].iloc[i]
                else:
                    trade_direction = trade_option_details[0]
                
                if 'Size' in group.columns:
                    trade_qty = group['Size'].iloc[i]
                else:
                    trade_qty = trade_option_details[1]

                if trade_direction == "Buy" : 
                    IV_str = 'Ask_IV' 
                    position_price = position['Ask_Price_USD'] 
                else:
                    IV_str ='Bid_IV'
                    position_price = position['Bid_Price_USD'] 
                position_size = trade_qty
            
            if chart_type == "Public" : 
                IV_str = 'IV_Percent'
                position_size = position['Size']
                trade_direction = position['Side']
                position_price = position['Price_USD']
                
            group = group.copy()        
            group[IV_str] = pd.to_numeric(group[IV_str], errors='coerce').fillna(0)   

            future_iv = position[IV_str] / 100
            risk_free_rate = 0.0  # Example risk-free rate
            
            profits = analytics.calculate_public_profits(
                (index_price_range, trade_direction, position['Strike_Price'], position_price, 
                 position_size , time_to_expiration_years, risk_free_rate, future_iv, position['Option_Type'].lower())
            )
            
            daily_profits.append(profits)
        
        # Sum profits for the current day
        total_daily_profit = np.sum(daily_profits, axis=0)
        profit_over_days[day] = total_daily_profit

    # Create a Plotly line chart for profits over all days
    fig_profit = go.Figure()

    for day in days_to_expiration:
        # Calculate breakeven prices for the current day
        breakeven_prices = []
        for i in range(1, len(index_price_range)):
            if (profit_over_days[day][i-1] < 0 and profit_over_days[day][i] > 0) or \
               (profit_over_days[day][i-1] > 0 and profit_over_days[day][i] < 0):
                breakeven_prices.append(index_price_range[i])

        # Format breakeven prices for hover
        breakeven_prices_str = ', '.join(
            f"{price/1000000:.1f}M" if abs(price) >= 1000000 else (f"{price/1000:.0f}k" if abs(price) >= 1000 else f"{price:,.0f}")
            for price in breakeven_prices
        ) if breakeven_prices else "N/A"

        fig_profit.add_trace(go.Scatter(
            x=index_price_range,
            y=profit_over_days[day],
            mode='lines',
            name=f"Day {day}",
            showlegend=True,
            hovertemplate=(
                "%{text}<br>"  # Add number of days to expiration
                "Underlying Price: %{customdata[0]}<br>"  # Use custom data for formatted x-axis value
                "Breakeven Prices: %{customdata[2]}<br>"  # Add breakeven prices to hover
                "Profit: %{customdata[1]}<br>"  # Use custom data for formatted y-axis value
                "<extra></extra>"  # Suppress default hover info
            ),
            text=[f"Day {day}" for _ in range(len(profit_over_days[day]))],  # Add day information for hover
            customdata=[
                [
                    f"{x/1000000:.1f}M" if abs(x) >= 1000000 else (f"{x/1000:.0f}k" if abs(x) >= 1000 else f"{x:,.0f}"),
                    f"{y/1000000:.1f}M" if abs(y) >= 1000000 else (f"{y/1000:.0f}k" if abs(y) >= 1000 else f"{y:,.0f}"),
                    breakeven_prices_str
                ]
                for x, y in zip(index_price_range, profit_over_days[day])
            ]  # Format values greater than 1000000 with 'M' and greater than 1000 with 'k', considering negative values
        ))

    # Update layout for the profit chart
    fig_profit.update_layout(
        xaxis_title='Underlying Price',
        yaxis_title='Profit',
        showlegend=True
    )

    return fig_profit

def plot_public_profits(strategy_data , chart_type , trade_option_details):
                                # Calculate profits using multithreading
                        
    with ThreadPoolExecutor() as executor:
        future_all_days_profits = executor.submit(plot_all_days_profits, strategy_data , chart_type , trade_option_details)
        future_strategy_profit = executor.submit(plot_expiration_profit, strategy_data, chart_type , trade_option_details)
                                    
                                    # Retrieve results
    fig_profit = future_all_days_profits.result()
    fig_strategy = future_strategy_profit.result()
                                
    return fig_profit, fig_strategy

def plot_top_strikes_pie_chart(top_strikes):
    """
    Plots a pie chart of the top strike prices with hover info showing
    total size and trade count.
    """
    # Ensure the DataFrame is not empty
    if top_strikes.empty:
        raise ValueError("The top_strikes DataFrame is empty. Cannot plot pie chart.")

    # Create hover text to display total size and trade count
    top_strikes['hover_text'] = (
        "Strike Price: " + top_strikes.index.astype(str) + "<br>" +
        "Total Size: " + top_strikes[('Total Size', 'sum')].astype(int).astype(str) + " BTC<br>" +
        "Count: " + top_strikes[('Total Size', 'count')].astype(int).astype(str) + " Strategies"
    )

    # Create a pie chart using Plotly
    fig = go.Figure(data=[go.Pie(
        labels=top_strikes.index.astype(str),
        values=top_strikes[('Total Size', 'sum')],
        hole=0.6,
        hoverinfo='text',
        textinfo='value+percent+label',
        text=top_strikes['hover_text'],
        textposition='outside'
    )])

    # Update the layout of the chart
    
    return fig

def plot_hourly_activity_radar(hourly_activity):
    """
    Plots a radar chart of hourly activities.
    
    Parameters:
        hourly_activity (pd.DataFrame): DataFrame containing 'Total Size' by hour.
    """
    # Ensure the DataFrame is not empty
    if hourly_activity.empty:
        raise ValueError("The hourly_activity DataFrame is empty. Cannot plot radar chart.")

    # Extract hours and total sizes
    hours = hourly_activity.index.astype(str)  # Convert hours to string for plotting
    total_sizes = hourly_activity['Total Size']['sum'].tolist()

    # Close the loop for the radar chart by repeating the first value
    hours = list(hours) + [hours[0]]
    total_sizes = total_sizes + [total_sizes[0]]

    # Create radar chart
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=total_sizes,
        theta=hours,
        fill='toself',
        name='Hourly Activity',
        line=dict(color='blue', width=2)  # Change color of the line to blue
    ))

    # Update the layout of the radar chart
    fig.update_layout(
        title='Hourly Activity by Total Size',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(total_sizes) + 10],  # Adjust the range as necessary
            ),
            angularaxis=dict(
                tickvals=hours[:-1],  # Exclude the repeated first value for ticks
                ticktext=hours[:-1]   # Exclude the repeated first value for labels
            )
        ),
        showlegend=True
    )

    return fig

def plot_most_strategy_bar_chart(strategy_df_copy):
    """
    Plots a horizontal bar chart of the total size from the strategy DataFrame using Plotly's graph_objects.

    Parameters:
    - strategy_df_copy (pd.DataFrame): DataFrame containing the 'Total Size' data to plot.
    """
    if 'Total Size' in strategy_df_copy and 'sum' in strategy_df_copy['Total Size']:
        # Create a horizontal bar chart
        fig = go.Figure(
            data=[
                go.Bar(
                    y=strategy_df_copy.index,  # Use the index as the y-axis
                    x=strategy_df_copy['Total Size']['sum'],  # Use 'Total Size' as the x-axis
                    orientation='h'  # Horizontal orientation
                )
            ]
        )
        fig.update_layout(
            xaxis_title='Total Size',
            yaxis_title='Top Strategies'
        )
    
    return fig

def plot_hourly_activity(hourly_activity):
    """
    Plots a bar chart of hourly activity using Plotly.

    Parameters:
        hourly_activity (pd.DataFrame): DataFrame containing 'hour' and 'activity' columns.
    """
    # Ensure the DataFrame is not empty
    if hourly_activity.empty:
        raise ValueError("The hourly_activity DataFrame is empty. Cannot plot.")

    # Create a bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=hourly_activity.index,  # Assuming the index contains the hours
        y=hourly_activity['Total Size']['sum'],  
        marker=dict(color='skyblue'),  # Set the color of the bars to sky blue
        hoverinfo='y',  
      ))

    # Update layout
    fig.update_layout(
        title='Hourly Activity',
        xaxis_title='Hour',
        yaxis_title='Activity (Volume)',
        template='plotly_white',  # Use a clean white template
        xaxis=dict(tickmode='linear'),  # Ensure all hours are shown
        height=300 # Decrease the height of the chart
    )

    return fig

def plot_predicted_trend(timeframes):
    """
    Plot cumulative trend for each timeframe in the list.
    For each timeframe, looks for a CSV file named 'technical_analysis_<timeframe>.csv' in the current directory.
    Adds a trace for each existing file.

    Args:
        timeframes (str or list): A single timeframe string (e.g., "4h") or a list of timeframes (e.g., ["1h", "4h", "1d"]).

    Returns:
        fig (plotly.graph_objs.Figure): The plotly figure with traces for each timeframe found.
    """
    if isinstance(timeframes, str):
        timeframes = [timeframes]

    fig = go.Figure()
    for tf in timeframes:
        # Normalize common names
        tf_str = str(tf).lower()
        if tf_str in ["1d", "daily"]:
            tf_file = "technical_analysis_daily.csv"
            tf_label = "Daily"
        elif tf_str in ["4h"]:
            tf_file = "technical_analysis_4h.csv"
            tf_label = "4H"
        elif tf_str in ["1h"]:
            tf_file = "technical_analysis_1h.csv"
            tf_label = "1H"
        else:
            # fallback: use the string directly
            tf_file = f"technical_analysis_{tf_str}.csv"
            tf_label = tf_str.upper()

        file_path = os.path.join(os.getcwd(), tf_file)
        if not os.path.exists(file_path):
            continue  # skip if file does not exist

        df = pd.read_csv(file_path)
        if "date" not in df.columns or "predicted_trend" not in df.columns:
            continue  # skip if required columns are missing

        df['date'] = pd.to_datetime(df['date'])
        df['trend_value'] = 0
        df.loc[df['predicted_trend'] == 'Bullish', 'trend_value'] = 1
        df.loc[df['predicted_trend'] == 'Bearish', 'trend_value'] = -1
        df.loc[df['predicted_trend'] == 'Neutral', 'trend_value'] = 0
        df['cumulative_trend'] = df['trend_value'].cumsum()

        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['cumulative_trend'],
            mode='lines+markers',
            name=f'Trend - {tf_label}',
            line=dict(width=2),  # Narrower line
            marker=dict(size=2)  # Smaller markers
        ))

    fig.update_layout(
        title='Cumulative Trend Over Time',
        xaxis_title='Date',
        yaxis_title='Cumulative Trend Value',
        xaxis=dict(title='Date', tickformat='%Y-%m-%d'),
        yaxis=dict(title='Cumulative Trend Value'),
        legend=dict(x=0, y=1)
    )

    return fig

def plot_probability_heatmap(df):
    """
    Generate a heatmap based on the grouped probabilities of expiration dates and strike prices.

    Parameters:
    filtered_df (pd.DataFrame): The DataFrame containing instruments, expiration dates, strike prices, and probabilities.

    Returns:
    fig: A Plotly figure object containing the heatmap.
    """

    filtered_df = df.copy()
    
    def prepare_probability_data(df):
        """
        Prepare the DataFrame by grouping by expiration date and strike price,
        averaging the probability values.

        Parameters:
        df (pd.DataFrame): The DataFrame containing instrument, expiration date, strike price, and probability data.

        Returns:
        pd.DataFrame: Grouped DataFrame with average probabilities for each expiration date and strike price.
        """
        # Extracting expiration date and strike price from the 'Instrument' column
        df['Expiration_Date'] = df['Instrument'].str.extract(r'-(\d{1,2}[A-Z]{3}\d{2})-')[0]
        df['Strike_Price'] = df['Instrument'].str.extract(r'-(\d{5,6})-')[0]
        
        # Convert 'Strike Price' to numeric
        df['Strike_Price'] = pd.to_numeric(df['Strike_Price'], errors='coerce')

        # Convert to datetime format: e.g., '24MAY26' to '2026-05-24'
        df['Expiration_Date'] = pd.to_datetime(df['Expiration_Date'], format='%d%b%y', errors='coerce')

        # Drop rows with NaT in 'Expiration Date' or NaN in 'Probability (%)'
        df.dropna(subset=['Expiration_Date', 'Probability_Percent'], inplace=True)

        # Group by 'Expiration Date' and 'Strike Price', taking the mean of probabilities
        grouped_df = df.groupby(['Expiration_Date', 'Strike_Price']).agg(
            Probability=('Probability_Percent', 'mean')
        ).reset_index()
        
        # Sort grouped_df by 'Expiration Date' in ascending order
        grouped_df.sort_values('Expiration_Date', inplace=True)

        return grouped_df
    
    # Prepare the data
    grouped_df = prepare_probability_data(filtered_df)

    # Create the pivot table for the heatmap
    probability_matrix = grouped_df.pivot(index='Expiration_Date', columns='Strike_Price', values='Probability')

    # Create the heatmap figure keeping the dates in ascending order
    fig = go.Figure(data=go.Heatmap(
        z=probability_matrix.values,
        x=probability_matrix.columns,
        y=probability_matrix.index.strftime('%d %B %Y'),  # Format for the display
        colorscale='Cividis',
        zmin=0,  # Set minimum for probabilities
        zmax=100,  # Set maximum for probabilities
        text=probability_matrix.values,
        hoverinfo='text',
        showscale=False  # Disable the color scale (removes the legend)
    ))

    # Update layout for better presentation
    fig.update_layout(
        title='Probability Heatmap for Options',
        xaxis=dict(title='Strike Prices'),
        yaxis=dict(title='Expiration Dates'),
    )

    # Add hover text formatting
    fig.data[0].hovertemplate = (
        "Expiration Date: %{y}<br>"  # Expiration date label
        "Strike Price: %{x}<br>"      # Strike price label
        "Probability: %{z:.2f}%"       # Probability label
    )

    return fig
