import streamlit as st
import pandas as pd
import io
import base64
import re
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
from collections import defaultdict

st.set_page_config(page_title="Crypto Tax Report", layout="wide")

# Define stablecoin groups
USD_STABLES = ['BUSD', 'FDUSD', 'USD', 'USDT', 'USDC', 'TUSD'] # Added TUSD based on sample data
EUR_STABLES = ['EUR', 'EURI']

def extract_crypto_amount(text):
    # Extract the numeric part (amount) before the currency code
    match = re.match(r"([0-9\.]+)([A-Z]+)", str(text))
    if match:
        return float(match.group(1))
    return 0.0

def extract_currency(text):
    # Extract the currency code part
    match = re.match(r"([0-9\.]+)([A-Z]+)", str(text))
    if match:
        return match.group(2)
    return ''

def get_unified_pair(pair_str):
    """Convert a trading pair string to its unified version based on stablecoin groups."""
    for stable in USD_STABLES:
        if pair_str.endswith(stable):
            base = pair_str[:-len(stable)]
            return f"{base}USD" # Changed from USD_stable
    for stable in EUR_STABLES:
        if pair_str.endswith(stable):
            base = pair_str[:-len(stable)]
            return f"{base}EUR" # Changed from EUR_stable
    return pair_str # Return original if no stablecoin match

def process_binance_csv(df):
    """Process Binance CSV files to extract trading information"""
    # Filter completed transactions only
    df = df[df['Status'] == 'FILLED'].copy()

    # Convert date string to datetime and ensure chronological order
    df['Date'] = pd.to_datetime(df['Date(UTC)'])
    df = df.sort_values('Date')  # Ensure chronological order
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Extract crypto amount and currency
    df['Amount'] = df['Executed'].apply(extract_crypto_amount)
    df['Currency'] = df['Executed'].apply(extract_currency)

    # Extract pair information (keep the full pair) and create unified pair
    df['Trading Pair'] = df['Pair']
    df['Unified Pair'] = df['Trading Pair'].apply(get_unified_pair) # Add Unified Pair column

    # Calculate total in BUSD (or equivalent quote currency)
    df['Total Amount'] = df['Trading total'].apply(extract_crypto_amount)
    df['Quote Currency'] = df['Trading total'].apply(extract_currency)
    df['Total BUSD'] = df['Total Amount'] # Keep original total for calculations

    # Mark BUY/SELL transactions
    df['Operation'] = df['Side']

    return df

def calculate_pair_holdings_and_pl(processed_df):
    """Calculate holdings and profit/loss for each UNIFIED trading pair in chronological order"""
    # Get unique UNIFIED trading pairs
    unified_pairs = processed_df['Unified Pair'].unique()

    pair_results = []

    for unified_pair in unified_pairs:
        # Get transactions for this unified pair in chronological order
        pair_df = processed_df[processed_df['Unified Pair'] == unified_pair].sort_values('Date')

        # Initialize tracking variables
        running_base_amount = 0  # Amount of the base cryptocurrency
        running_cost_basis = 0   # Total cost basis in Quote Currency equivalent
        running_proceeds = 0     # Total proceeds from sales in Quote Currency equivalent
        realized_pl = 0          # Realized profit/loss in Quote Currency equivalent

        # Keep track of all transactions for this pair
        pair_transactions = []
        original_pairs_in_group = set(pair_df['Trading Pair'].unique()) # Keep track of original pairs

        # Process each transaction chronologically
        for idx, row in pair_df.iterrows():
            operation = row['Operation']
            amount = row['Amount']
            total_quote_value = row['Total BUSD'] # Use the Total BUSD column as the common value unit
            currency = row['Currency']
            date = row['Date']
            year = row['Year']
            original_pair = row['Trading Pair'] # Store original pair for reference if needed

            # Record transaction
            transaction = {
                'Date': date,
                'Year': year,
                'Unified Pair': unified_pair, # Store unified pair
                'Original Pair': original_pair, # Store original pair
                'Operation': operation,
                'Amount': amount,
                'Currency': currency,
                'Total Quote Value': total_quote_value # Changed name for clarity
            }

            # Update running totals
            if operation == 'BUY':
                # Buying increases our holdings and cost basis
                running_base_amount += amount
                running_cost_basis += total_quote_value
                transaction['Running Holdings'] = running_base_amount
                transaction['Cost Basis'] = running_cost_basis
                transaction['Avg Cost'] = running_cost_basis / running_base_amount if running_base_amount > 0 else 0
                transaction['Realized P/L'] = 0

            elif operation == 'SELL':
                # Calculate profit/loss for this sale using FIFO logic on the unified pool
                if running_base_amount > 0:
                    # Calculate average cost per unit for the current holdings
                    avg_cost_per_unit = running_cost_basis / running_base_amount

                    # Determine the cost of the units being sold
                    # Handle case where selling more than currently held (should not happen with correct data, but safer)
                    sell_amount_adjusted = min(amount, running_base_amount)
                    cost_of_units_sold = sell_amount_adjusted * avg_cost_per_unit

                    # Update running totals
                    running_base_amount -= sell_amount_adjusted

                    # Proportionally reduce the cost basis
                    # Avoid division by zero if sell_amount_adjusted was 0 or amount was 0
                    if (running_base_amount + sell_amount_adjusted) > 0:
                         running_cost_basis = running_cost_basis * (running_base_amount / (running_base_amount + sell_amount_adjusted))
                    else:
                         running_cost_basis = 0 # Reset if holdings are now zero

                    # Calculate profit/loss for this transaction
                    transaction_pl = total_quote_value - cost_of_units_sold
                    realized_pl += transaction_pl
                    running_proceeds += total_quote_value

                    transaction['Running Holdings'] = running_base_amount
                    transaction['Cost Basis'] = running_cost_basis # Updated cost basis
                    transaction['Avg Cost'] = running_cost_basis / running_base_amount if running_base_amount > 0 else 0
                    transaction['Realized P/L'] = transaction_pl
                else:
                    # Selling without holdings - treat as 0 P/L for cost basis calculation
                    transaction['Running Holdings'] = 0
                    transaction['Cost Basis'] = 0
                    transaction['Avg Cost'] = 0
                    transaction['Realized P/L'] = 0 # P/L might be negative if fees > proceeds
                    running_proceeds += total_quote_value

            pair_transactions.append(transaction)

        # Calculate final unrealized P/L based on current holdings (requires market price - leave as 0)
        unrealized_pl = 0

        # Calculate summary metrics for this UNIFIED pair
        total_invested = sum(t['Total Quote Value'] for t in pair_transactions if t['Operation'] == 'BUY')
        total_proceeds = sum(t['Total Quote Value'] for t in pair_transactions if t['Operation'] == 'SELL')
        avg_cost = running_cost_basis / running_base_amount if running_base_amount > 0 else 0

        pair_summary = {
            'Unified Pair': unified_pair, # Use Unified Pair name
            'Original Pairs': ', '.join(sorted(list(original_pairs_in_group))), # Show original pairs involved
            'Current Holdings': running_base_amount,
            'Cost Basis': running_cost_basis,
            'Average Cost': avg_cost,
            'Total Invested': total_invested,
            'Total Proceeds': total_proceeds,
            'Realized P/L': realized_pl,
            'Unrealized P/L': unrealized_pl, # Placeholder
            'Total P/L': realized_pl + unrealized_pl,
            'Transactions': pair_transactions # Keep detailed transactions
        }

        pair_results.append(pair_summary)

    return pair_results

def generate_yearly_summary(pair_results):
    """Generate a summary of transactions by year"""
    yearly_data = {}

    # Extract all transactions from all pairs
    all_transactions = []
    for pair_result in pair_results:
        all_transactions.extend(pair_result['Transactions'])

    if not all_transactions:
        return pd.DataFrame()

    # Convert to DataFrame for easier grouping
    transactions_df = pd.DataFrame(all_transactions)
    transactions_df['Year'] = pd.to_datetime(transactions_df['Date']).dt.year

    # Group by year
    for year, year_df in transactions_df.groupby('Year'):
        buy_df = year_df[year_df['Operation'] == 'BUY']
        sell_df = year_df[year_df['Operation'] == 'SELL']

        total_buy = buy_df['Total Quote Value'].sum() # Use updated column name
        total_sell = sell_df['Total Quote Value'].sum() # Use updated column name
        realized_pl = sell_df['Realized P/L'].sum()

        yearly_data[year] = {
            'Year': year,
            'Total Buy Value': total_buy,
            'Total Sell Value': total_sell,
            'Realized P/L': realized_pl
        }

    yearly_summary = pd.DataFrame(list(yearly_data.values()))
    return yearly_summary

def generate_yearly_pair_summary(pair_results):
    """Generate a summary of transactions by year and UNIFIED trading pair"""
    all_transactions = []
    for pair_result in pair_results:
        unified_pair = pair_result['Unified Pair']
        for transaction in pair_result['Transactions']:
            transaction_copy = transaction.copy()
            # Ensure 'Unified Pair' is correctly propagated; it should already be there
            # transaction_copy['Unified Pair'] = unified_pair # This line might be redundant
            all_transactions.append(transaction_copy)

    if not all_transactions:
        return pd.DataFrame()

    # Convert to DataFrame for easier grouping
    transactions_df = pd.DataFrame(all_transactions)

    # Make sure Year is present
    if 'Year' not in transactions_df.columns:
        transactions_df['Year'] = pd.to_datetime(transactions_df['Date']).dt.year

    # Group by year and UNIFIED trading pair
    yearly_pair_data = []

    # Use 'Unified Pair' for grouping
    for (year, unified_pair), group_df in transactions_df.groupby(['Year', 'Unified Pair']):
        buy_df = group_df[group_df['Operation'] == 'BUY']
        sell_df = group_df[group_df['Operation'] == 'SELL']

        # Calculate yearly metrics for this unified pair
        total_buy_amount = buy_df['Amount'].sum()
        total_sell_amount = sell_df['Amount'].sum()
        total_buy_value = buy_df['Total Quote Value'].sum() # Use updated column name
        total_sell_value = sell_df['Total Quote Value'].sum() # Use updated column name
        realized_pl = sell_df['Realized P/L'].sum()

        # Calculate average buy price for the year
        avg_buy_price = total_buy_value / total_buy_amount if total_buy_amount > 0 else 0

        # Get end-of-year holdings and average cost from the last transaction of the year for this unified pair
        latest_transaction = group_df.sort_values('Date').iloc[-1] if not group_df.empty else None

        eoy_holdings = latest_transaction['Running Holdings'] if latest_transaction is not None else 0
        eoy_avg_cost = latest_transaction['Avg Cost'] if latest_transaction is not None else 0

        # Find the base currency (should be consistent within a unified pair's base)
        base_currency = group_df['Currency'].iloc[0] if not group_df.empty else ''

        yearly_pair_data.append({
            'Year': year,
            'Unified Pair': unified_pair, # Use Unified Pair
            'Total Buy Amount': total_buy_amount,
            'Total Sell Amount': total_sell_amount,
            'Total Buy Value': total_buy_value,
            'Total Sell Value': total_sell_value,
            'Realized P/L': realized_pl,
            'Avg Buy Price': avg_buy_price,
            'EOY Holdings': eoy_holdings,
            'EOY Avg Cost': eoy_avg_cost,
            'Base Currency': base_currency,
        })

    yearly_pair_df = pd.DataFrame(yearly_pair_data)
    return yearly_pair_df

def get_download_link(df, filename):
    """Create a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def plot_portfolio_composition(pair_summary_df):
    """Create a pie chart showing the relative value of cryptocurrencies in the portfolio based on Unified Pairs"""
    if pair_summary_df.empty or 'Current Holdings' not in pair_summary_df.columns:
        return None

    # Calculate current value (cost basis is a proxy for value since we don't have current prices)
    df = pair_summary_df.copy()
    df = df[df['Current Holdings'] > 0].copy()  # Only include assets we currently hold

    if df.empty:
        return None

    # Calculate value of each asset (using cost basis as proxy)
    df['Value'] = df['Cost Basis']
    df['% of Portfolio'] = df['Value'] / df['Value'].sum() * 100

    # Determine profitability for color-coding
    df['Is Profitable'] = df['Realized P/L'] > 0

    # Create a color map
    color_map = {True: '#45de85', False: '#ff6b6b'}  # Green for profitable, Red for unprofitable

    # Create a pie chart using Unified Pair
    fig = px.pie(
        df,
        values='Value',
        names='Unified Pair', # Use Unified Pair
        title='Portfolio Composition by Cost Basis (Green: Profitable, Red: Unprofitable)',
        hover_data=['Current Holdings', '% of Portfolio', 'Average Cost', 'Realized P/L', 'Original Pairs'], # Add Original Pairs to hover
        color='Is Profitable',
        color_discrete_map=color_map
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        legend_title_text='Unified Trading Pairs', # Update legend title
        height=600
    )

    # Add a treemap as an alternative view using Unified Pair
    treemap_fig = px.treemap(
        df,
        path=['Unified Pair'], # Use Unified Pair
        values='Value',
        color='Is Profitable',
        color_discrete_map=color_map,
        hover_data=['Current Holdings', 'Average Cost', 'Realized P/L', 'Original Pairs'], # Add Original Pairs to hover
        title='Portfolio Composition Treemap (Green: Profitable, Red: Unprofitable)'
    )

    treemap_fig.update_layout(height=600)

    return fig, treemap_fig

def plot_cumulative_pl_waterfall(pair_summary_df):
    """Create a waterfall chart showing how each UNIFIED trading pair contributes to overall P/L"""
    if pair_summary_df.empty:
        return None

    # Sort by P/L impact
    df = pair_summary_df.sort_values('Realized P/L').copy()

    # Filter to only include pairs with non-zero P/L
    df = df[df['Realized P/L'] != 0]

    if df.empty:
        return None

    # Create lists for the waterfall chart using Unified Pair
    unified_pairs = df['Unified Pair'].tolist() # Use Unified Pair
    values = df['Realized P/L'].tolist()

    # Add total at the end
    unified_pairs.append('TOTAL')
    values.append(sum(values))

    # Create measure list
    measure = ['relative'] * (len(unified_pairs) - 1) + ['total']

    # Create the waterfall chart
    fig = go.Figure(go.Waterfall(
        name="P/L Contribution",
        orientation="v",
        measure=measure,
        x=unified_pairs, # Use Unified Pair
        y=values,
        text=[f'{v:,.2f}' for v in values], # Format text
        textposition='outside',
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#45de85"}},
        decreasing={"marker": {"color": "#ff6b6b"}},
        totals={"marker": {"color": "#3366ff"}}
    ))

    fig.update_layout(
        title="Cumulative Profit/Loss by Unified Trading Pair", # Update title
        xaxis_title="Unified Trading Pair", # Update axis title
        yaxis_title="Contribution to P/L (Quote Currency)", # Update axis title
        showlegend=False,
        height=600
    )

    return fig

def plot_yearly_pair_performance(yearly_pair_summary):
    """Create a bar chart comparing P/L for each UNIFIED trading pair across years"""
    if yearly_pair_summary.empty:
        return None

    # Only include pairs with some P/L
    df = yearly_pair_summary[yearly_pair_summary['Realized P/L'] != 0].copy()

    if df.empty:
        return None

    # Sort by absolute P/L to highlight most impactful UNIFIED pairs
    # Use 'Unified Pair' for grouping and sorting
    total_pl_by_pair = df.groupby('Unified Pair')['Realized P/L'].sum().abs().sort_values(ascending=False)
    top_pairs = total_pl_by_pair.index.tolist()

    # Limit to top N pairs if there are many
    MAX_PAIRS_TO_SHOW = 15
    if len(top_pairs) > MAX_PAIRS_TO_SHOW:
        top_pairs = top_pairs[:MAX_PAIRS_TO_SHOW]
        df = df[df['Unified Pair'].isin(top_pairs)]

    # Convert Year to string for better display
    df['Year'] = df['Year'].astype(str)

    # Create the bar chart using Unified Pair
    fig = px.bar(
        df,
        x='Unified Pair', # Use Unified Pair
        y='Realized P/L',
        color='Year',
        barmode='group',
        title='Yearly Profit/Loss by Unified Trading Pair', # Update title
        labels={'Realized P/L': 'Realized P/L (Quote Currency)', 'Unified Pair': 'Unified Trading Pair'}, # Update labels
        color_discrete_sequence=px.colors.qualitative.Bold,
        hover_data=['Base Currency', 'Total Buy Value', 'Total Sell Value'] # Add hover data
    )

    fig.update_layout(
        xaxis_title='Unified Trading Pair', # Update axis title
        yaxis_title='Realized P/L (Quote Currency)', # Update axis title
        legend_title='Year',
        height=600
    )

    return fig

def plot_monthly_trading_heatmap(all_transactions_df):
    """Create a calendar heatmap showing trading intensity by month/year"""
    if all_transactions_df.empty:
        return None

    # Add month/year columns for grouping
    df = all_transactions_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['MonthName'] = df['Date'].dt.month.apply(lambda x: calendar.month_abbr[x])

    # Count transactions by month/year
    monthly_counts = df.groupby(['Year', 'Month', 'MonthName']).size().reset_index(name='Transaction Count')

    # Pivot for heatmap
    heatmap_data = monthly_counts.pivot(index='MonthName', columns='Year', values='Transaction Count').fillna(0)

    # Ensure correct month order
    month_order = [calendar.month_abbr[i] for i in range(1, 13)]
    heatmap_data = heatmap_data.reindex(month_order)

    # Create the heatmap
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Year", y="Month", color="Transaction Count"),
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale=px.colors.sequential.Viridis,
        title='Monthly Trading Activity (Transaction Count)'
    )

    fig.update_xaxes(side="top")
    fig.update_layout(height=500)

    return fig

def plot_pl_vs_volume_scatter(yearly_pair_summary):
    """Create a scatter plot showing P/L vs trading volume per pair per year."""
    if yearly_pair_summary.empty or 'Total Buy Value' not in yearly_pair_summary.columns or 'Total Sell Value' not in yearly_pair_summary.columns:
        return None

    df = yearly_pair_summary.copy()

    # Calculate total trading volume for the year/pair
    df['Total Trading Volume'] = df['Total Buy Value'] + df['Total Sell Value']

    # Filter out entries with zero volume or zero P/L? Optional, but might clean the plot.
    df = df[(df['Total Trading Volume'] > 0) & (df['Realized P/L'] != 0)]

    if df.empty:
        return None

    # Create the scatter plot using Unified Pair
    fig = px.scatter(
        df,
        x='Total Trading Volume',
        y='Realized P/L',
        color='Unified Pair', # Color by Unified Pair
        size='Total Trading Volume', # Size dots by volume
        hover_data=['Year', 'Base Currency', 'Avg Buy Price'],
        title='P/L vs Trading Volume by Unified Pair and Year',
        labels={
            'Total Trading Volume': 'Total Trading Volume (Quote Currency)',
            'Realized P/L': 'Realized P/L (Quote Currency)'
        },
        color_discrete_sequence=px.colors.qualitative.Plotly # Use a qualitative color scale
    )

    # Add a horizontal line at y=0 to separate profit/loss
    fig.add_hline(y=0, line_dash="dash", line_color="grey")

    # Add annotations for quadrants if desired (e.g., High Volume/High Profit)
    # Example: Add text annotation
    # fig.add_annotation(
    #     x=df['Total Trading Volume'].quantile(0.75), y=df['Realized P/L'].quantile(0.75),
    #     text="High Volume / High Profit", showarrow=False,
    #     xanchor="left", yanchor="bottom"
    # )

    fig.update_layout(
        height=600,
        xaxis_title='Total Trading Volume (Quote Currency)', # Update axis title
        yaxis_title='Realized P/L (Quote Currency)', # Update axis title
        legend_title='Unified Trading Pair' # Update legend title
    )

    return fig

def plot_holding_period_histogram(all_transactions_df, pair_results):
    """Create a histogram showing distribution of holding periods before selling for Unified Pairs"""
    # NOTE: This function uses the raw 'pair_results' which now contains unified pairs.
    # The internal logic matches buys/sells within each result item.
    # We might need to adjust how buys/sells are matched if FIFO should consider
    # buys across different original pairs within the same unified group.
    # For now, let's assume FIFO within the unified group's transaction list is sufficient.

    holding_periods = []

    for pair_result in pair_results:
        unified_pair = pair_result['Unified Pair'] # Get unified pair name
        # Use the transactions list which is already ordered chronologically within the unified pair
        transactions = pd.DataFrame(pair_result['Transactions'])

        if transactions.empty or 'Operation' not in transactions.columns:
            continue

        # Separate buys and sells within the unified group
        buy_transactions = transactions[transactions['Operation'] == 'BUY'].copy()
        sell_transactions = transactions[transactions['Operation'] == 'SELL'].copy()

        if buy_transactions.empty or sell_transactions.empty:
            continue

        # Convert dates if needed (should already be datetime)
        buy_transactions['Date'] = pd.to_datetime(buy_transactions['Date'])
        sell_transactions['Date'] = pd.to_datetime(sell_transactions['Date'])

        # Simple FIFO matching for holding period calculation within the unified group
        # This matches *any* buy before a sell in the group, regardless of original pair
        buy_queue = buy_transactions.sort_values('Date').to_dict('records')

        for _, sell in sell_transactions.sort_values('Date').iterrows():
            sell_date = sell['Date']
            sell_amount_remaining = sell['Amount']

            temp_buy_queue = []
            processed_buy_indices = [] # Keep track of indices to remove later

            for i, buy in enumerate(buy_queue):
                 # Ensure buy happened before sell
                if buy['Date'] >= sell_date:
                    temp_buy_queue.append(buy) # Keep buy for later sells
                    continue

                if sell_amount_remaining <= 0:
                    temp_buy_queue.append(buy) # Keep buy for later sells
                    continue

                buy_amount_available = buy.get('Remaining Amount', buy['Amount']) # Use remaining if partially used

                if buy_amount_available <= 0:
                     temp_buy_queue.append(buy) # Keep buy for later sells
                     continue


                amount_to_match = min(sell_amount_remaining, buy_amount_available)

                if amount_to_match > 0:
                    buy_date = buy['Date']
                    holding_period_days = (sell_date - buy_date).days

                    holding_periods.append({
                        'Unified Pair': unified_pair, # Use Unified Pair
                        'Buy Date': buy_date,
                        'Sell Date': sell_date,
                        'Amount Matched': amount_to_match,
                        'Holding Period (Days)': holding_period_days
                    })

                    # Update remaining amounts
                    sell_amount_remaining -= amount_to_match
                    buy['Remaining Amount'] = buy_amount_available - amount_to_match

                    # Keep the buy in the queue if partially used
                    if buy['Remaining Amount'] > 1e-9: # Use tolerance for float comparison
                         temp_buy_queue.append(buy)
                    # else: # Mark for removal if fully used (or handle below)
                    #      processed_buy_indices.append(i)


            # Update buy_queue for next sell iteration (only keep unused or partially used buys)
            # A simpler approach might be to just consume from the front of the queue
            # Let's stick to the slightly more complex remaining amount for now
            buy_queue = [b for b in temp_buy_queue if b.get('Remaining Amount', b['Amount']) > 1e-9]


    if not holding_periods:
        return None

    # Convert to DataFrame
    holding_df = pd.DataFrame(holding_periods)

    if holding_df.empty:
        return None

    # Create histograms using Unified Pair
    fig = px.histogram(
        holding_df,
        x='Holding Period (Days)',
        color='Unified Pair', # Color by Unified Pair
        title='Distribution of Holding Periods Before Selling (Unified Pairs)', # Update title
        labels={'Holding Period (Days)': 'Holding Period (Days)'},
        opacity=0.7,
        barmode='overlay', # Use overlay or stack? Overlay might be better for comparison
        nbins=50, # Adjust number of bins as needed
        marginal='box', # Add box plot marginal
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    fig.update_layout(
        xaxis_title='Holding Period (Days)',
        yaxis_title='Count of Sell Portions Matched', # Y-axis represents portions of sells matched
        legend_title='Unified Trading Pair', # Update legend title
        height=600
    )

    return fig

def main():
    st.title("Crypto Tax Report Generator")

    st.markdown("""
    This app generates tax reports from your Binance transaction history CSV files.
    Upload one or more files to get started.
    *Note: Calculations assume FIFO cost basis method.*
    """)

    # File uploader or use sample data
    data_option = st.radio(
        "Choose data source",
        ["Use sample data from data folder", "Upload your own CSV file(s)"],
        index=0,  # Default to sample data
        horizontal=True
    )

    all_data = []

    if data_option == "Upload your own CSV file(s)":
        uploaded_files = st.file_uploader("Upload Binance CSV report file(s)", type="csv", accept_multiple_files=True)

        if uploaded_files:
            for file in uploaded_files:
                try:
                    df = pd.read_csv(file)
                    all_data.append(df)
                except Exception as e:
                    st.error(f"Error reading file {file.name}: {e}")
    else:
        # Load sample data from data folder
        data_folder = "data"
        if os.path.exists(data_folder):
            try:
                csv_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')]) # Sort files
                if not csv_files:
                     st.warning(f"No CSV files found in the '{data_folder}' directory.")
                else:
                    selected_files = st.multiselect(
                        "Select sample files to process",
                        csv_files,
                        default=csv_files # Default to all sample files
                    )

                    if selected_files:
                        for file in selected_files:
                            file_path = os.path.join(data_folder, file)
                            try:
                                df = pd.read_csv(file_path)
                                all_data.append(df)
                            except Exception as e:
                                st.error(f"Error reading sample file {file}: {e}")
                        st.success(f"Loaded {len(selected_files)} sample files.")
                    else:
                         st.info("No sample files selected.")

            except FileNotFoundError:
                 st.error(f"The sample data folder '{data_folder}' was not found.")
            except Exception as e:
                 st.error(f"An error occurred while accessing the data folder: {e}")

        else:
            st.warning(f"The sample data folder '{data_folder}' does not exist.")


    if all_data:
        # Combine all data
        try:
            combined_df = pd.concat(all_data, ignore_index=True)

            # Basic validation
            required_columns = ['Date(UTC)', 'Pair', 'Side', 'Executed', 'Trading total', 'Status']
            if not all(col in combined_df.columns for col in required_columns):
                st.error(f"One or more uploaded/selected files are missing required columns: {required_columns}")
                st.stop() # Stop execution if columns are missing

            # Process data
            with st.spinner("Processing data... This may take a moment for large datasets."):
                processed_df = process_binance_csv(combined_df)

                # Calculate pair holdings and profit/loss using UNIFIED pairs
                pair_results = calculate_pair_holdings_and_pl(processed_df)

                # Create summary dataframe for all UNIFIED pairs
                pair_summary_data = []
                for result in pair_results:
                    # Exclude the detailed transactions list from the summary table
                    summary_row = {k: v for k, v in result.items() if k != 'Transactions'}
                    pair_summary_data.append(summary_row)

                if not pair_summary_data:
                     st.warning("No valid trading pairs found after processing.")
                     pair_summary_df = pd.DataFrame()
                else:
                     pair_summary_df = pd.DataFrame(pair_summary_data)
                     # Ensure specific column order for better readability
                     summary_cols = [
                         'Unified Pair', 'Original Pairs', 'Current Holdings', 'Average Cost',
                         'Cost Basis', 'Total Invested', 'Total Proceeds', 'Realized P/L'
                         # Add 'Unrealized P/L', 'Total P/L' if needed later
                     ]
                     # Filter for existing columns only to avoid errors if some are missing
                     pair_summary_df = pair_summary_df[[col for col in summary_cols if col in pair_summary_df.columns]]


                # Generate yearly summary
                yearly_summary = generate_yearly_summary(pair_results)

                # Generate yearly UNIFIED pair summary
                yearly_pair_summary = generate_yearly_pair_summary(pair_results)

                # Create transactions dataframe FOR ALL original transactions (for detailed view/debugging)
                all_transactions_list = []
                for pair_result in pair_results:
                    all_transactions_list.extend(pair_result['Transactions'])

                if not all_transactions_list:
                     st.warning("No transactions found after processing.")
                     all_transactions_df = pd.DataFrame()
                else:
                     all_transactions_df = pd.DataFrame(all_transactions_list)
                     # Convert Date back to datetime if needed (should be correct)
                     all_transactions_df['Date'] = pd.to_datetime(all_transactions_df['Date'])


            st.success("Data processing complete!")

            # Create navigation tabs for better organization
            tabs = st.tabs([
                "📊 Summary Tables",
                "📈 Portfolio Analysis",
                "📉 Trading Activity",
                "🔍 Detailed View",
                "💾 Download Reports"
            ])

            # =========== SUMMARY TABLES TAB ===========
            with tabs[0]:
                st.header("Summary Tables")

                # Overall Summary by UNIFIED Trading Pair
                st.subheader("Summary by Unified Trading Pair")
                if not pair_summary_df.empty:
                    # Apply specific formatting including 4 decimal places for Holdings and Avg Cost
                    st.dataframe(pair_summary_df.sort_values('Unified Pair').style.format({
                        'Current Holdings': '{:,.4f}',  # Format to 4 decimal places
                        'Average Cost': '{:,.4f}',    # Format to 4 decimal places
                        'Cost Basis': '{:,.2f}',
                        'Total Invested': '{:,.2f}',
                        'Total Proceeds': '{:,.2f}',
                        'Realized P/L': '{:,.2f}'
                    }))

                    # Add an expander for column explanations
                    with st.expander("Explanation of Summary Columns"):
                        st.markdown("""
                        *   **Unified Pair**: The trading pair, with common stablecoins grouped (e.g., BTCBUSD, BTCUSDT become BTCUSD).
                        *   **Original Pairs**: The specific pairs from your Binance data included in this unified group.
                        *   **Current Holdings**: The amount of the base asset you currently hold for this unified pair after processing all transactions.
                        *   **Average Cost**: The average price paid per unit for the assets currently held, calculated using the FIFO (First-In, First-Out) cost basis across all trades within the unified pair (`Cost Basis / Current Holdings`).
                        *   **Cost Basis**: The total cost (in the quote currency, e.g., USD) of the assets currently held, determined by the FIFO method.
                        *   **Total Invested**: The total quote currency value spent on BUY transactions for this unified pair.
                        *   **Total Proceeds**: The total quote currency value received from SELL transactions for this unified pair.
                        *   **Realized P/L**: The total profit or loss realized from completed sell transactions for this unified pair (`Total Proceeds - Cost of Goods Sold calculated via FIFO`).
                        """)
                else:
                    st.info("No pair summary data to display.")


                # Yearly Summary
                st.subheader("Overall Yearly Summary")
                if not yearly_summary.empty:
                    st.dataframe(yearly_summary.sort_values('Year').style.format({
                         'Total Buy Value': '{:,.2f}',
                         'Total Sell Value': '{:,.2f}',
                         'Realized P/L': '{:,.2f}'
                    }))
                else:
                     st.info("No yearly summary data to display.")

                # UNIFIED Trading Pair Performance by Year
                st.subheader("Unified Trading Pair Performance by Year") # Update title
                if not yearly_pair_summary.empty:
                    # Add year filter
                    available_years = sorted(yearly_pair_summary['Year'].unique())
                    selected_year = st.selectbox(
                        "Filter by Year",
                        ["All Years"] + [str(year) for year in available_years]
                    )

                    # Filter by selected year
                    if selected_year != "All Years":
                        filtered_summary = yearly_pair_summary[yearly_pair_summary['Year'] == int(selected_year)]
                    else:
                        filtered_summary = yearly_pair_summary

                    # Display the filtered table
                    st.dataframe(filtered_summary.sort_values(['Year', 'Unified Pair']).style.format({ # Use updated name
                        'Total Buy Amount': '{:,.8f}',
                        'Total Sell Amount': '{:,.8f}',
                        'Total Buy Value': '{:,.2f}',
                        'Total Sell Value': '{:,.2f}',
                        'Realized P/L': '{:,.2f}',
                        'Avg Buy Price': '{:,.8f}',
                        'EOY Holdings': '{:,.8f}',
                        'EOY Avg Cost': '{:,.8f}'
                     }))
                else:
                     st.info("No yearly pair summary data to display.")

                # Show raw data sample (optional, maybe move to detailed?)
                # st.subheader("Processed Transaction Data Sample")
                # st.dataframe(processed_df.head())


            # =========== PORTFOLIO ANALYSIS TAB ===========
            with tabs[1]:
                st.header("Portfolio Analysis")

                # Portfolio Composition Visualization using UNIFIED Pairs
                st.subheader("Portfolio Composition (Unified Pairs)") # Update title

                if not pair_summary_df.empty:
                    portfolio_results = plot_portfolio_composition(pair_summary_df)
                    if portfolio_results:
                        pie_fig, treemap_fig = portfolio_results

                        # Add a selectbox to choose between pie chart and treemap
                        chart_type = st.radio(
                            "Select visualization type",
                            ["Pie Chart", "Treemap"],
                            horizontal=True,
                            key="portfolio_chart_type" # Add key for unique widget
                        )

                        if chart_type == "Pie Chart":
                            st.plotly_chart(pie_fig, use_container_width=True)
                        else:
                            st.plotly_chart(treemap_fig, use_container_width=True)

                        # Explain the color coding
                        st.info("Colors indicate profitability based on realized P/L: 🟢 Green = Positive Realized P/L, 🔴 Red = Negative or Zero Realized P/L for the unified pair.")
                    else:
                        st.info("No current holdings with cost basis data to display in the portfolio composition.")
                else:
                    st.info("No pair summary data available for portfolio composition.")


                # P/L Analysis using UNIFIED Pairs
                st.subheader("Profit/Loss Analysis (Unified Pairs)") # Update title

                col1, col2 = st.columns(2)

                with col1:
                    # Cumulative P/L Waterfall Chart using UNIFIED Pairs
                    st.subheader("P/L Contribution by Unified Pair") # Update title
                    if not pair_summary_df.empty:
                        pl_waterfall_fig = plot_cumulative_pl_waterfall(pair_summary_df)
                        if pl_waterfall_fig:
                            st.plotly_chart(pl_waterfall_fig, use_container_width=True)
                        else:
                            st.info("No profit/loss data to display in waterfall chart.")
                    else:
                         st.info("No pair summary data available for P/L waterfall.")

                with col2:
                    # Year-over-Year Performance Comparison using UNIFIED Pairs
                    st.subheader("Year-over-Year P/L Comparison (Top Unified Pairs)") # Update title
                    if not yearly_pair_summary.empty:
                        yearly_performance_fig = plot_yearly_pair_performance(yearly_pair_summary)
                        if yearly_performance_fig:
                            st.plotly_chart(yearly_performance_fig, use_container_width=True)
                        else:
                            st.info("Insufficient data or no P/L for year-over-year comparison chart.")
                    else:
                        st.info("No yearly pair summary data available for Y-o-Y comparison.")

                # P/L vs Trading Volume Scatter Plot using UNIFIED Pairs
                st.subheader("P/L vs Trading Volume Analysis (Unified Pairs)") # Update title
                if not yearly_pair_summary.empty:
                    pl_volume_fig = plot_pl_vs_volume_scatter(yearly_pair_summary)
                    if pl_volume_fig:
                        st.plotly_chart(pl_volume_fig, use_container_width=True)
                    else:
                        st.info("Insufficient trading volume or P/L data to display scatter plot.")
                else:
                    st.info("No yearly pair summary data available for P/L vs Volume analysis.")

            # =========== TRADING ACTIVITY TAB ===========
            with tabs[2]:
                st.header("Trading Activity Analysis")

                # Monthly Trading Volume Heatmap (based on all transactions)
                st.subheader("Monthly Trading Activity Heatmap")
                if not all_transactions_df.empty:
                    monthly_heatmap_fig = plot_monthly_trading_heatmap(all_transactions_df)
                    if monthly_heatmap_fig:
                        st.plotly_chart(monthly_heatmap_fig, use_container_width=True)
                    else:
                        st.info("Insufficient data for monthly activity heatmap.")
                else:
                    st.info("No transaction data available for heatmap.")


                # Holding Period Analysis using UNIFIED Pairs
                st.subheader("Holding Period Analysis (Unified Pairs)") # Update title
                if not all_transactions_df.empty and pair_results:
                    # Pass all_transactions_df for calculating periods, pair_results for grouping info?
                    # Let's stick to passing pair_results as it contains the grouped transactions needed
                    holding_period_fig = plot_holding_period_histogram(all_transactions_df, pair_results) # Pass both for now
                    if holding_period_fig:
                        st.plotly_chart(holding_period_fig, use_container_width=True)
                        st.caption("Note: Holding periods are calculated using FIFO matching within each unified pair group.")
                    else:
                        st.info("Insufficient data or no sell transactions found for holding period analysis.")
                else:
                    st.info("No transaction data available for holding period analysis.")

            # =========== DETAILED VIEW TAB ===========
            with tabs[3]:
                st.header("Detailed Unified Pair Analysis") # Update title

                # Select UNIFIED Trading Pair for detailed view
                if not pair_summary_df.empty:
                    available_unified_pairs = sorted(pair_summary_df['Unified Pair'].unique()) # Use Unified Pair

                    selected_unified_pair = st.selectbox(
                        "Select a Unified Trading Pair to view details", # Update label
                        available_unified_pairs,
                        key="detailed_pair_select" # Add key
                    )

                    if selected_unified_pair:
                        # Find the selected UNIFIED pair data
                        pair_data = next((p for p in pair_results if p['Unified Pair'] == selected_unified_pair), None) # Use Unified Pair

                        if pair_data:
                            st.subheader(f"Detailed Analysis for {selected_unified_pair}")
                            st.write(f"(Includes original pairs: {pair_data.get('Original Pairs', 'N/A')})")

                            col1, col2 = st.columns([1, 2]) # Adjust column widths if needed

                            with col1:
                                # Show pair metrics
                                st.subheader(f"Summary Metrics")
                                metrics_data = {k: v for k, v in pair_data.items() if k not in ['Transactions', 'Unified Pair']} # Exclude transactions list

                                # Format the metrics for better display
                                formatted_metrics = {}
                                for k, v in metrics_data.items():
                                    if isinstance(v, float):
                                        # Use more precision for Average Cost, less for totals
                                        if k == 'Average Cost':
                                             formatted_metrics[k] = f"{v:,.8f}"
                                        elif k in ['Cost Basis', 'Total Invested', 'Total Proceeds', 'Realized P/L', 'Total P/L']:
                                             formatted_metrics[k] = f"{v:,.2f}"
                                        elif k == 'Current Holdings':
                                             formatted_metrics[k] = f"{v:,.8f}"
                                        else:
                                             formatted_metrics[k] = round(v, 4) # Default rounding for other floats
                                    else:
                                        formatted_metrics[k] = v # Keep strings/other types as is

                                st.json(formatted_metrics) # Display metrics as JSON

                            with col2:
                                # Create enhanced charts for selected UNIFIED pair
                                st.subheader("Transaction History & P/L")
                                transactions_df = pd.DataFrame(pair_data['Transactions'])
                                transactions_df['Date'] = pd.to_datetime(transactions_df['Date'])

                                if not transactions_df.empty:
                                    # Calculate cumulative metrics needed for plotting (might be redundant if already in df)
                                    transactions_df = transactions_df.sort_values('Date')
                                    transactions_df['Cumulative P/L'] = transactions_df['Realized P/L'].cumsum()

                                    # Create a detailed chart with buy/sell annotations
                                    fig = make_subplots(
                                        rows=2,
                                        cols=1,
                                        shared_xaxes=True,
                                        vertical_spacing=0.1,
                                        subplot_titles=(
                                            f"Holdings & Trades for {selected_unified_pair}",
                                            f"Cumulative Realized P/L for {selected_unified_pair}"
                                        ),
                                        row_heights=[0.6, 0.4]
                                    )

                                    # Add holdings line
                                    fig.add_trace(
                                        go.Scatter(
                                            x=transactions_df['Date'],
                                            y=transactions_df['Running Holdings'],
                                            mode='lines',
                                            name='Holdings',
                                            line=dict(width=2, color='royalblue')
                                        ),
                                        row=1, col=1
                                    )

                                    # Add buy markers
                                    buy_df = transactions_df[transactions_df['Operation'] == 'BUY']
                                    if not buy_df.empty:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=buy_df['Date'],
                                                y=buy_df['Running Holdings'],
                                                mode='markers',
                                                name='Buy',
                                                marker=dict(color='green', size=8, symbol='triangle-up'),
                                                hovertemplate=(
                                                    '<b>BUY</b><br>' +
                                                    'Date: %{x|%Y-%m-%d %H:%M}<br>' +
                                                    'Amount: %{customdata[0]:.8f} %{customdata[1]}<br>' +
                                                    'Quote Value: %{customdata[2]:.2f}<br>' +
                                                    'Original Pair: %{customdata[3]}<br>' +
                                                    'Holdings After: %{y:.8f}<extra></extra>'
                                                ),
                                                customdata=buy_df[['Amount', 'Currency', 'Total Quote Value', 'Original Pair']].values
                                            ),
                                            row=1, col=1
                                        )

                                    # Add sell markers
                                    sell_df = transactions_df[transactions_df['Operation'] == 'SELL']
                                    if not sell_df.empty:
                                        fig.add_trace(
                                            go.Scatter(
                                                x=sell_df['Date'],
                                                y=sell_df['Running Holdings'],
                                                mode='markers',
                                                name='Sell',
                                                marker=dict(color='red', size=8, symbol='triangle-down'),
                                                 hovertemplate=(
                                                    '<b>SELL</b><br>' +
                                                    'Date: %{x|%Y-%m-%d %H:%M}<br>' +
                                                    'Amount: %{customdata[0]:.8f} %{customdata[1]}<br>' +
                                                    'Quote Value: %{customdata[2]:.2f}<br>' +
                                                    'P/L: %{customdata[3]:.2f}<br>' +
                                                    'Original Pair: %{customdata[4]}<br>' +
                                                    'Holdings After: %{y:.8f}<extra></extra>'
                                                ),
                                                customdata=sell_df[['Amount', 'Currency', 'Total Quote Value', 'Realized P/L', 'Original Pair']].values
                                            ),
                                            row=1, col=1
                                        )

                                    # Add P/L line in second subplot
                                    fig.add_trace(
                                        go.Scatter(
                                            x=transactions_df['Date'],
                                            y=transactions_df['Cumulative P/L'],
                                            mode='lines',
                                            name='Cumulative P/L',
                                            line=dict(width=2, color='gold')
                                        ),
                                        row=2, col=1
                                    )

                                    # Update layout
                                    fig.update_layout(
                                        height=700, # Increase height for better readability
                                        hovermode='x unified',
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                    )
                                    fig.update_yaxes(title_text="Amount (Base Currency)", row=1, col=1)
                                    fig.update_yaxes(title_text="Cumulative P/L (Quote)", row=2, col=1)

                                    st.plotly_chart(fig, use_container_width=True)

                                    # Display detailed transactions table for the selected unified pair
                                    st.subheader("Transactions List")
                                    # Select and order columns for display
                                    display_cols = [
                                        'Date', 'Operation', 'Amount', 'Currency', 'Total Quote Value',
                                        'Realized P/L', 'Avg Cost', 'Running Holdings', 'Original Pair'
                                    ]
                                    st.dataframe(transactions_df[display_cols].style.format({
                                        'Amount': '{:,.8f}',
                                        'Total Quote Value': '{:,.2f}',
                                        'Realized P/L': '{:,.2f}',
                                        'Avg Cost': '{:,.8f}',
                                        'Running Holdings': '{:,.8f}'
                                     }))
                                else:
                                     st.info("No transactions found for this unified pair.")

                        else:
                            st.warning(f"Could not find detailed data for {selected_unified_pair}.")
                else:
                     st.info("No data available to select a detailed pair.")


            # =========== DOWNLOAD REPORTS TAB ===========
            with tabs[4]:
                st.header("Download Reports")
                st.info("Download summary data and transaction lists as CSV files.")

                col1, col2 = st.columns(2)

                with col1:
                    if not pair_summary_df.empty:
                        st.markdown(get_download_link(pair_summary_df, "unified_pair_summary.csv"), unsafe_allow_html=True)
                    else:
                        st.caption("No Unified Pair Summary to download.")

                    if not yearly_summary.empty:
                        st.markdown(get_download_link(yearly_summary, "yearly_summary.csv"), unsafe_allow_html=True)
                    else:
                        st.caption("No Yearly Summary to download.")

                with col2:
                     if not yearly_pair_summary.empty:
                        st.markdown(get_download_link(yearly_pair_summary, "yearly_unified_pair_summary.csv"), unsafe_allow_html=True)
                     else:
                        st.caption("No Yearly Unified Pair Summary to download.")

                     if not all_transactions_df.empty:
                        # Sort transactions by date before downloading
                        all_transactions_df_sorted = all_transactions_df.sort_values('Date')
                        st.markdown(get_download_link(all_transactions_df_sorted, "all_processed_transactions.csv"), unsafe_allow_html=True)
                     else:
                        st.caption("No Processed Transactions to download.")


        except Exception as e:
            st.exception(e) # Display the full error and traceback
            st.error("An error occurred during data processing or display. Please check the input files and data format.")

    else:
        st.info("Please upload CSV file(s) or select sample data to begin.")


if __name__ == "__main__":
    main()
