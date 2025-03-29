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

st.set_page_config(page_title="Crypto Tax Report", layout="wide")

def extract_crypto_amount(amount_str):
    """Extract numeric amount and currency from strings like '10.5BTC' or '500ADA'"""
    if pd.isna(amount_str) or amount_str == '0' or amount_str == '':
        return 0
    match = re.match(r'([\d.]+)([A-Za-z]+)', str(amount_str))
    if match:
        return float(match.group(1))
    return 0

def extract_currency(amount_str):
    """Extract currency from strings like '10.5BTC' or '500ADA'"""
    if pd.isna(amount_str) or amount_str == '0' or amount_str == '':
        return ''
    match = re.match(r'([\d.]+)([A-Za-z]+)', str(amount_str))
    if match:
        return match.group(2)
    return ''

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

    # Extract pair information (keep the full pair)
    df['Trading Pair'] = df['Pair']

    # Calculate total in BUSD
    df['Total Amount'] = df['Trading total'].apply(extract_crypto_amount)
    df['Quote Currency'] = df['Trading total'].apply(extract_currency)
    df['Total BUSD'] = df['Total Amount']

    # Mark BUY/SELL transactions
    df['Operation'] = df['Side']

    return df

def calculate_pair_holdings_and_pl(processed_df):
    """Calculate holdings and profit/loss for each trading pair in chronological order"""
    # Get unique trading pairs
    pairs = processed_df['Trading Pair'].unique()

    pair_results = []

    for pair in pairs:
        # Get transactions for this pair in chronological order
        pair_df = processed_df[processed_df['Trading Pair'] == pair].sort_values('Date')

        # Initialize tracking variables
        running_base_amount = 0  # Amount of the base cryptocurrency
        running_cost_basis = 0   # Total cost basis in BUSD
        running_proceeds = 0     # Total proceeds from sales in BUSD
        realized_pl = 0          # Realized profit/loss

        # Keep track of all transactions for this pair
        pair_transactions = []

        # Process each transaction chronologically
        for idx, row in pair_df.iterrows():
            operation = row['Operation']
            amount = row['Amount']
            total_busd = row['Total BUSD']
            currency = row['Currency']
            date = row['Date']
            year = row['Year']

            # Record transaction
            transaction = {
                'Date': date,
                'Year': year,
                'Pair': pair,
                'Operation': operation,
                'Amount': amount,
                'Currency': currency,
                'Total BUSD': total_busd
            }

            # Update running totals
            if operation == 'BUY':
                # Buying increases our holdings and cost basis
                running_base_amount += amount
                running_cost_basis += total_busd
                transaction['Running Holdings'] = running_base_amount
                transaction['Cost Basis'] = running_cost_basis
                transaction['Avg Cost'] = running_cost_basis / running_base_amount if running_base_amount > 0 else 0
                transaction['Realized P/L'] = 0

            elif operation == 'SELL':
                # Calculate profit/loss for this sale
                if running_base_amount > 0:
                    # Calculate average cost for the amount being sold
                    avg_cost_per_unit = running_cost_basis / running_base_amount
                    cost_of_units_sold = amount * avg_cost_per_unit

                    # Update running totals
                    running_base_amount -= amount

                    # If we're selling more than we have, cap it at zero
                    if running_base_amount < 0:
                        running_base_amount = 0

                    # Proportionally reduce the cost basis
                    if amount > 0:
                        running_cost_basis = running_cost_basis * (running_base_amount / (running_base_amount + amount))

                    # Calculate profit/loss
                    transaction_pl = total_busd - cost_of_units_sold
                    realized_pl += transaction_pl
                    running_proceeds += total_busd

                    transaction['Running Holdings'] = running_base_amount
                    transaction['Cost Basis'] = running_cost_basis
                    transaction['Avg Cost'] = running_cost_basis / running_base_amount if running_base_amount > 0 else 0
                    transaction['Realized P/L'] = transaction_pl
                else:
                    # We're selling but don't have any holdings (possible in some cases)
                    transaction['Running Holdings'] = 0
                    transaction['Cost Basis'] = 0
                    transaction['Avg Cost'] = 0
                    transaction['Realized P/L'] = 0
                    running_proceeds += total_busd

            pair_transactions.append(transaction)

        # Calculate final unrealized P/L based on current holdings
        unrealized_pl = 0  # We would need current market prices to calculate this

        # Calculate metrics for this pair
        pair_summary = {
            'Trading Pair': pair,
            'Current Holdings': running_base_amount,
            'Cost Basis': running_cost_basis,
            'Average Cost': running_cost_basis / running_base_amount if running_base_amount > 0 else 0,
            'Total Invested': sum(t['Total BUSD'] for t in pair_transactions if t['Operation'] == 'BUY'),
            'Total Proceeds': sum(t['Total BUSD'] for t in pair_transactions if t['Operation'] == 'SELL'),
            'Realized P/L': realized_pl,
            'Unrealized P/L': unrealized_pl,
            'Total P/L': realized_pl + unrealized_pl,
            'Transactions': pair_transactions
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

    # Convert to DataFrame for easier grouping
    transactions_df = pd.DataFrame(all_transactions)
    transactions_df['Year'] = pd.to_datetime(transactions_df['Date']).dt.year

    # Group by year
    for year, year_df in transactions_df.groupby('Year'):
        buy_df = year_df[year_df['Operation'] == 'BUY']
        sell_df = year_df[year_df['Operation'] == 'SELL']

        total_buy = buy_df['Total BUSD'].sum()
        total_sell = sell_df['Total BUSD'].sum()
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
    """Generate a summary of transactions by year and trading pair"""
    # Extract all transactions from all pairs
    all_transactions = []
    for pair_result in pair_results:
        pair = pair_result['Trading Pair']
        for transaction in pair_result['Transactions']:
            transaction_copy = transaction.copy()
            transaction_copy['Trading Pair'] = pair
            all_transactions.append(transaction_copy)

    # Convert to DataFrame for easier grouping
    transactions_df = pd.DataFrame(all_transactions)

    # Make sure Year is present (should be added during transaction processing)
    if 'Year' not in transactions_df.columns:
        transactions_df['Year'] = pd.to_datetime(transactions_df['Date']).dt.year

    # Group by year and trading pair
    yearly_pair_data = []

    for (year, pair), group_df in transactions_df.groupby(['Year', 'Trading Pair']):
        buy_df = group_df[group_df['Operation'] == 'BUY']
        sell_df = group_df[group_df['Operation'] == 'SELL']

        # Calculate yearly metrics for this pair
        total_buy_amount = buy_df['Amount'].sum()
        total_sell_amount = sell_df['Amount'].sum()
        total_buy_value = buy_df['Total BUSD'].sum()
        total_sell_value = sell_df['Total BUSD'].sum()
        realized_pl = sell_df['Realized P/L'].sum()

        # Calculate average buy price for the year
        avg_buy_price = total_buy_value / total_buy_amount if total_buy_amount > 0 else 0

        # Get end-of-year holdings and average cost
        # We'll use the latest transaction for this pair in this year
        latest_transaction = group_df.sort_values('Date').iloc[-1] if not group_df.empty else None

        eoy_holdings = latest_transaction['Running Holdings'] if latest_transaction is not None else 0
        eoy_avg_cost = latest_transaction['Avg Cost'] if latest_transaction is not None else 0

        yearly_pair_data.append({
            'Year': year,
            'Trading Pair': pair,
            'Total Buy Amount': total_buy_amount,
            'Total Sell Amount': total_sell_amount,
            'Total Buy Value': total_buy_value,
            'Total Sell Value': total_sell_value,
            'Realized P/L': realized_pl,
            'Avg Buy Price': avg_buy_price,
            'EOY Holdings': eoy_holdings,
            'EOY Avg Cost': eoy_avg_cost,
            'Base Currency': group_df['Currency'].iloc[0] if not group_df.empty else '',
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
    """Create a pie chart showing the relative value of cryptocurrencies in the portfolio"""
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
    # If Realized P/L is positive, the position is considered profitable
    df['Is Profitable'] = df['Realized P/L'] > 0

    # Create a color map
    color_map = {True: '#45de85', False: '#ff6b6b'}  # Green for profitable, Red for unprofitable

    # Create a pie chart
    fig = px.pie(
        df,
        values='Value',
        names='Trading Pair',
        title='Portfolio Composition by Cost Basis (Green: Profitable, Red: Unprofitable)',
        hover_data=['Current Holdings', '% of Portfolio', 'Average Cost', 'Realized P/L'],
        color='Is Profitable',
        color_discrete_map=color_map
    )

    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        legend_title_text='Trading Pairs',
        height=600
    )

    # Add a treemap as an alternative view
    treemap_fig = px.treemap(
        df,
        path=['Trading Pair'],
        values='Value',
        color='Is Profitable',
        color_discrete_map=color_map,
        hover_data=['Current Holdings', 'Average Cost', 'Realized P/L'],
        title='Portfolio Composition Treemap (Green: Profitable, Red: Unprofitable)'
    )

    treemap_fig.update_layout(height=600)

    return fig, treemap_fig

def plot_cumulative_pl_waterfall(pair_summary_df):
    """Create a waterfall chart showing how each trading pair contributes to overall P/L"""
    if pair_summary_df.empty:
        return None

    # Sort by P/L impact
    df = pair_summary_df.sort_values('Realized P/L').copy()

    # Filter to only include pairs with non-zero P/L
    df = df[df['Realized P/L'] != 0]

    if df.empty:
        return None

    # Create lists for the waterfall chart
    trading_pairs = df['Trading Pair'].tolist()
    values = df['Realized P/L'].tolist()

    # Add total at the end
    trading_pairs.append('TOTAL')
    values.append(sum(values))

    # Create measure list (relative for individual pairs, total for the sum)
    measure = ['relative'] * (len(trading_pairs) - 1) + ['total']

    # Create the waterfall chart
    fig = go.Figure(go.Waterfall(
        name="Profit/Loss Contribution",
        orientation="v",
        measure=measure,
        x=trading_pairs,
        y=values,
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#45de85"}},
        decreasing={"marker": {"color": "#ff6b6b"}},
        totals={"marker": {"color": "#3366ff"}}
    ))

    fig.update_layout(
        title="Cumulative Profit/Loss by Trading Pair",
        xaxis_title="Trading Pair",
        yaxis_title="Contribution to P/L (BUSD)",
        showlegend=False,
        height=600
    )

    return fig

def plot_yearly_pair_performance(yearly_pair_summary):
    """Create a bar chart comparing P/L for each trading pair across years"""
    if yearly_pair_summary.empty:
        return None

    # Only include pairs with some P/L
    df = yearly_pair_summary[yearly_pair_summary['Realized P/L'] != 0].copy()

    if df.empty:
        return None

    # Sort by absolute P/L to highlight most impactful pairs
    total_pl_by_pair = df.groupby('Trading Pair')['Realized P/L'].sum().abs().sort_values(ascending=False)
    top_pairs = total_pl_by_pair.index.tolist()

    # Limit to top 10 pairs if there are more
    if len(top_pairs) > 10:
        top_pairs = top_pairs[:10]
        df = df[df['Trading Pair'].isin(top_pairs)]

    # Convert Year to string for better display
    df['Year'] = df['Year'].astype(str)

    # Create the bar chart
    fig = px.bar(
        df,
        x='Trading Pair',
        y='Realized P/L',
        color='Year',
        barmode='group',
        title='Yearly Profit/Loss by Trading Pair',
        labels={'Realized P/L': 'Realized Profit/Loss (BUSD)', 'Trading Pair': 'Trading Pair'},
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    fig.update_layout(
        xaxis_title='Trading Pair',
        yaxis_title='Realized Profit/Loss (BUSD)',
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
    pivot_df = monthly_counts.pivot(index='Month', columns='Year', values='Transaction Count')

    # Create a custom month ordering
    month_names = [calendar.month_abbr[i] for i in range(1, 13)]

    # Create the heatmap
    fig = px.imshow(
        pivot_df,
        labels=dict(x="Year", y="Month", color="Transaction Count"),
        y=[calendar.month_abbr[m] for m in pivot_df.index],
        x=pivot_df.columns,
        color_continuous_scale='Viridis',
        title='Monthly Trading Activity'
    )

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Month',
        coloraxis_colorbar_title='Transaction Count',
        height=500
    )

    return fig

def plot_pl_vs_volume_scatter(yearly_pair_summary):
    """Create a scatter plot of P/L vs trading volume for each pair by year"""
    if yearly_pair_summary.empty:
        return None

    # Create a copy of the data
    df = yearly_pair_summary.copy()

    # Calculate the total trading volume
    df['Total Volume'] = df['Total Buy Value'] + df['Total Sell Value']

    # Only include entries with some volume
    df = df[df['Total Volume'] > 0]

    if df.empty:
        return None

    # Convert Year to string for better display
    df['Year'] = df['Year'].astype(str)

    # Create the scatter plot
    fig = px.scatter(
        df,
        x='Total Volume',
        y='Realized P/L',
        color='Year',
        size='Total Volume',
        hover_name='Trading Pair',
        hover_data=['Total Buy Value', 'Total Sell Value'],
        title='Profit/Loss vs Trading Volume by Year',
        labels={
            'Total Volume': 'Total Trading Volume (BUSD)',
            'Realized P/L': 'Realized Profit/Loss (BUSD)'
        },
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    # Add a horizontal line at y=0 to distinguish profit from loss
    fig.add_shape(
        type='line',
        x0=0,
        y0=0,
        x1=1,
        y1=0,
        line=dict(color='white', width=1, dash='dash'),
        xref='paper',
        yref='y'
    )

    fig.update_layout(
        height=600,
        xaxis_title='Total Trading Volume (BUSD)',
        yaxis_title='Realized Profit/Loss (BUSD)',
        legend_title='Year'
    )

    return fig

def plot_holding_period_histogram(all_transactions_df, pair_results):
    """Create a histogram showing distribution of holding periods before selling"""
    # Need to calculate holding periods for each sell transaction
    holding_periods = []

    for pair_result in pair_results:
        pair = pair_result['Trading Pair']
        transactions = pd.DataFrame(pair_result['Transactions'])

        if transactions.empty or 'Operation' not in transactions.columns:
            continue

        # Add buy dates to sell transactions using FIFO
        buy_transactions = transactions[transactions['Operation'] == 'BUY'].sort_values('Date').copy()
        sell_transactions = transactions[transactions['Operation'] == 'SELL'].sort_values('Date').copy()

        if buy_transactions.empty or sell_transactions.empty:
            continue

        # Convert dates to datetime if they're not already
        buy_transactions['Date'] = pd.to_datetime(buy_transactions['Date'])
        sell_transactions['Date'] = pd.to_datetime(sell_transactions['Date'])

        # For each sell, match with corresponding buys based on FIFO
        for _, sell in sell_transactions.iterrows():
            sell_date = sell['Date']
            sell_amount = sell['Amount']

            # Get buy transactions before this sell
            available_buys = buy_transactions[buy_transactions['Date'] < sell_date].copy()

            if available_buys.empty:
                continue

            # Calculate holding period in days
            for _, buy in available_buys.iterrows():
                buy_date = buy['Date']
                holding_period_days = (sell_date - buy_date).days

                holding_periods.append({
                    'Trading Pair': pair,
                    'Buy Date': buy_date,
                    'Sell Date': sell_date,
                    'Holding Period (Days)': holding_period_days
                })

    if not holding_periods:
        return None

    # Convert to DataFrame
    holding_df = pd.DataFrame(holding_periods)

    # Create histograms
    fig = px.histogram(
        holding_df,
        x='Holding Period (Days)',
        color='Trading Pair',
        title='Distribution of Holding Periods Before Selling',
        labels={'Holding Period (Days)': 'Holding Period (Days)'},
        opacity=0.7,
        barmode='overlay',
        nbins=50,
        marginal='box',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )

    fig.update_layout(
        xaxis_title='Holding Period (Days)',
        yaxis_title='Count',
        height=600
    )

    return fig

def main():
    st.title("Crypto Tax Report Generator")

    st.markdown("""
    This app generates tax reports from your Binance transaction history CSV files.
    Upload one or more files to get started.
    """)

    # File uploader or use sample data
    data_option = st.radio(
        "Choose data source",
        ["Use sample data from data folder", "Upload your own CSV file(s)"],
        index=0  # Set the first option (sample data) as default
    )

    all_data = []

    if data_option == "Upload your own CSV file(s)":
        uploaded_files = st.file_uploader("Upload Binance CSV report file(s)", type="csv", accept_multiple_files=True)

        if uploaded_files:
            for file in uploaded_files:
                df = pd.read_csv(file)
                all_data.append(df)
    else:
        # Load sample data from data folder
        data_folder = "data"
        if os.path.exists(data_folder):
            csv_files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
            selected_files = st.multiselect(
                "Select files to process",
                csv_files,
                default=csv_files
            )

            if selected_files:
                for file in selected_files:
                    file_path = os.path.join(data_folder, file)
                    df = pd.read_csv(file_path)
                    all_data.append(df)

                st.success(f"Loaded {len(selected_files)} files from data folder")

    if all_data:
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Process data
        with st.spinner("Processing data..."):
            processed_df = process_binance_csv(combined_df)

            # Calculate pair holdings and profit/loss
            pair_results = calculate_pair_holdings_and_pl(processed_df)

            # Create summary dataframe for all pairs
            pair_summary_data = []
            for result in pair_results:
                summary_row = {k: v for k, v in result.items() if k != 'Transactions'}
                pair_summary_data.append(summary_row)

            pair_summary_df = pd.DataFrame(pair_summary_data)

            # Generate yearly summary
            yearly_summary = generate_yearly_summary(pair_results)

            # Generate yearly pair summary
            yearly_pair_summary = generate_yearly_pair_summary(pair_results)

            # Create transactions dataframe for all transactions
            all_transactions = []
            for pair_result in pair_results:
                for transaction in pair_result['Transactions']:
                    transaction_copy = transaction.copy()
                    transaction_copy['Trading Pair'] = pair_result['Trading Pair']
                    all_transactions.append(transaction_copy)

            all_transactions_df = pd.DataFrame(all_transactions)

            # Create navigation tabs for better organization
            tabs = st.tabs([
                "Summary Tables",
                "Portfolio Analysis",
                "Trading Activity",
                "Detailed View",
                "Download Reports"
            ])

            # =========== SUMMARY TABLES TAB ===========
            with tabs[0]:
                # Show raw data sample
                st.subheader("Raw Transaction Data Sample")
                st.dataframe(processed_df.head())

                # Overall Summary by Trading Pair
                st.subheader("Summary by Trading Pair")
                st.dataframe(pair_summary_df.sort_values('Trading Pair'))

                # Yearly Summary
                st.subheader("Yearly Summary")
                st.dataframe(yearly_summary.sort_values('Year'))

                # Trading Pair Performance by Year
                st.subheader("Trading Pair Performance by Year")

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
                st.dataframe(filtered_summary.sort_values(['Year', 'Trading Pair']))

            # =========== PORTFOLIO ANALYSIS TAB ===========
            with tabs[1]:
                st.header("Portfolio Analysis")

                # Portfolio Composition Visualization
                st.subheader("Portfolio Composition")

                portfolio_results = plot_portfolio_composition(pair_summary_df)
                if portfolio_results:
                    pie_fig, treemap_fig = portfolio_results

                    # Add a selectbox to choose between pie chart and treemap
                    chart_type = st.radio(
                        "Select visualization type",
                        ["Pie Chart", "Treemap"],
                        horizontal=True
                    )

                    if chart_type == "Pie Chart":
                        st.plotly_chart(pie_fig, use_container_width=True)
                    else:
                        st.plotly_chart(treemap_fig, use_container_width=True)

                    # Explain the color coding
                    st.info("Colors indicate profitability: 🟢 Green = Profitable positions, 🔴 Red = Unprofitable positions")
                else:
                    st.info("No current holdings to display in the portfolio composition.")

                # P/L Analysis
                st.subheader("Profit/Loss Analysis")

                col1, col2 = st.columns(2)

                with col1:
                    # Cumulative P/L Waterfall Chart
                    st.subheader("P/L Contribution by Trading Pair")
                    pl_waterfall_fig = plot_cumulative_pl_waterfall(pair_summary_df)
                    if pl_waterfall_fig:
                        st.plotly_chart(pl_waterfall_fig, use_container_width=True)
                    else:
                        st.info("No profit/loss data to display.")

                with col2:
                    # Year-over-Year Performance Comparison
                    st.subheader("Year-over-Year P/L Comparison")
                    yearly_performance_fig = plot_yearly_pair_performance(yearly_pair_summary)
                    if yearly_performance_fig:
                        st.plotly_chart(yearly_performance_fig, use_container_width=True)
                    else:
                        st.info("Insufficient data for year-over-year comparison.")

                # P/L vs Trading Volume Scatter Plot
                st.subheader("Profit/Loss vs Trading Volume Analysis")
                pl_volume_fig = plot_pl_vs_volume_scatter(yearly_pair_summary)
                if pl_volume_fig:
                    st.plotly_chart(pl_volume_fig, use_container_width=True)
                else:
                    st.info("Insufficient trading volume data to display.")

            # =========== TRADING ACTIVITY TAB ===========
            with tabs[2]:
                st.header("Trading Activity Analysis")

                # Monthly Trading Volume Heatmap
                st.subheader("Monthly Trading Activity")
                monthly_heatmap_fig = plot_monthly_trading_heatmap(all_transactions_df)
                if monthly_heatmap_fig:
                    st.plotly_chart(monthly_heatmap_fig, use_container_width=True)
                else:
                    st.info("Insufficient data for monthly activity heatmap.")

                # Holding Period Analysis
                st.subheader("Holding Period Analysis")
                holding_period_fig = plot_holding_period_histogram(all_transactions_df, pair_results)
                if holding_period_fig:
                    st.plotly_chart(holding_period_fig, use_container_width=True)
                else:
                    st.info("Insufficient data for holding period analysis.")

            # =========== DETAILED VIEW TAB ===========
            with tabs[3]:
                st.header("Detailed Trading Pair Analysis")

                # Trading Pair selection for detailed view
                available_pairs = sorted(processed_df['Trading Pair'].unique())

                selected_pair = st.selectbox(
                    "Select a trading pair to view details",
                    available_pairs
                )

                if selected_pair:
                    # Find the selected pair data
                    pair_data = next((p for p in pair_results if p['Trading Pair'] == selected_pair), None)

                    if pair_data:
                        col1, col2 = st.columns([1, 2])

                        with col1:
                            # Show pair metrics
                            st.subheader(f"Metrics for {selected_pair}")
                            metrics_data = {k: v for k, v in pair_data.items() if k != 'Transactions'}

                            # Format the metrics for better display
                            formatted_metrics = {}
                            for k, v in metrics_data.items():
                                if isinstance(v, float):
                                    formatted_metrics[k] = round(v, 4)
                                else:
                                    formatted_metrics[k] = v

                            st.json(formatted_metrics)

                        with col2:
                            # Create enhanced charts for selected pair
                            transactions_df = pd.DataFrame(pair_data['Transactions'])
                            transactions_df['Date'] = pd.to_datetime(transactions_df['Date'])

                            # Calculate additional metrics
                            transactions_df['Cumulative Buy'] = transactions_df[transactions_df['Operation'] == 'BUY']['Amount'].cumsum()
                            transactions_df['Cumulative Sell'] = transactions_df[transactions_df['Operation'] == 'SELL']['Amount'].cumsum()
                            transactions_df['Cumulative P/L'] = transactions_df['Realized P/L'].cumsum()

                            # Create a detailed chart with buy/sell annotations
                            fig = make_subplots(
                                rows=2,
                                cols=1,
                                shared_xaxes=True,
                                vertical_spacing=0.1,
                                subplot_titles=(
                                    f"Holdings History for {selected_pair}",
                                    f"Cumulative P/L for {selected_pair}"
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
                                        hovertemplate='Date: %{x}<br>Amount: %{text}<br>Price: %{customdata}<extra></extra>',
                                        text=buy_df['Amount'].round(4).astype(str) + ' ' + buy_df['Currency'],
                                        customdata=buy_df['Total BUSD'] / buy_df['Amount']
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
                                        hovertemplate='Date: %{x}<br>Amount: %{text}<br>Price: %{customdata}<br>P/L: %{customdata2}<extra></extra>',
                                        text=sell_df['Amount'].round(4).astype(str) + ' ' + sell_df['Currency'],
                                        customdata=sell_df['Total BUSD'] / sell_df['Amount'],
                                        customdata2=sell_df['Realized P/L'].round(2)
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

                            # Add zero line for P/L
                            fig.add_shape(
                                type='line',
                                x0=transactions_df['Date'].min(),
                                y0=0,
                                x1=transactions_df['Date'].max(),
                                y1=0,
                                line=dict(color='white', width=1, dash='dash'),
                                row=2, col=1
                            )

                            # Update layout
                            fig.update_layout(
                                height=700,
                                hovermode='closest',
                                showlegend=True,
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                                xaxis2_title='Date',
                                yaxis_title=f'Holdings ({transactions_df["Currency"].iloc[0]})',
                                yaxis2_title='Profit/Loss (BUSD)'
                            )

                            st.plotly_chart(fig, use_container_width=True)

                        # Show transactions
                        st.subheader(f"Transactions for {selected_pair}")
                        st.dataframe(transactions_df.sort_values('Date', ascending=False))

            # =========== DOWNLOAD REPORTS TAB ===========
            with tabs[4]:
                st.header("Download Reports")
                st.write("Download your tax and trading reports in CSV format for further analysis.")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Summary Reports")
                    st.markdown(get_download_link(pair_summary_df, "crypto_pair_summary.csv"), unsafe_allow_html=True)
                    st.markdown(get_download_link(yearly_summary, "crypto_tax_yearly.csv"), unsafe_allow_html=True)
                    st.markdown(get_download_link(yearly_pair_summary, "crypto_yearly_pair_summary.csv"), unsafe_allow_html=True)

                with col2:
                    st.subheader("Detailed Reports")
                    st.markdown(get_download_link(all_transactions_df, "crypto_tax_detailed.csv"), unsafe_allow_html=True)

                    # Create a pivot table by month and year for monthly reports
                    monthly_data = all_transactions_df.copy()
                    monthly_data['Date'] = pd.to_datetime(monthly_data['Date'])
                    monthly_data['Month'] = monthly_data['Date'].dt.month
                    monthly_data['Year'] = monthly_data['Date'].dt.year

                    # Create monthly P/L report
                    monthly_pl = monthly_data.groupby(['Year', 'Month', 'Trading Pair'])['Realized P/L'].sum().reset_index()
                    st.markdown(get_download_link(monthly_pl, "crypto_monthly_pl.csv"), unsafe_allow_html=True)
    else:
        st.info("Please upload Binance CSV files or select files from the data folder to generate a report.")

    # Display info about the app
    with st.expander("About this app"):
        st.markdown("""
        This app helps you generate tax reports from your Binance transaction history CSV files.

        ## How to use:
        1. Export your transaction history from Binance as CSV
        2. Upload the CSV file(s) here or use the sample data
        3. View the summary and detailed reports by trading pair
        4. Explore the interactive visualizations to gain insights into your trading patterns
        5. Download the reports for your tax filing

        ## Features:
        - Analyzes trading pairs instead of individual cryptocurrencies
        - Calculates accurate profit/loss by processing transactions chronologically
        - Tracks running holdings for each pair
        - Shows only filled orders (ignores canceled orders)
        - Generates yearly summaries for tax reporting
        - Provides yearly performance breakdown by trading pair
        - Tracks holdings across years for accurate profit/loss calculations
        - Interactive visualizations for deeper analysis of your trading patterns
        - Portfolio composition and performance visualization
        - Trading activity and holding period analysis

        Note: This app assumes that your CSV files follow the Binance export format.
        """)

if __name__ == "__main__":
    main()
