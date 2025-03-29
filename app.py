import streamlit as st
import pandas as pd
import io
import base64
import re
import os
from datetime import datetime

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

            # Record transaction
            transaction = {
                'Date': date,
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

def get_download_link(df, filename):
    """Create a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def main():
    st.title("Crypto Tax Report Generator")

    st.markdown("""
    This app generates tax reports from your Binance transaction history CSV files.
    Upload one or more files to get started.
    """)

    # File uploader or use sample data
    data_option = st.radio(
        "Choose data source",
        ["Upload your own CSV file(s)", "Use sample data from data folder"]
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

            # Show raw data sample
            st.subheader("Raw Transaction Data Sample")
            st.dataframe(processed_df.head())

            # Overall Summary by Trading Pair
            st.subheader("Summary by Trading Pair")
            st.dataframe(pair_summary_df.sort_values('Trading Pair'))

            # Yearly Summary
            yearly_summary = generate_yearly_summary(pair_results)

            st.subheader("Yearly Summary")
            st.dataframe(yearly_summary.sort_values('Year'))

            # Trading Pair selection for detailed view
            st.subheader("Detailed View by Trading Pair")
            available_pairs = sorted(processed_df['Trading Pair'].unique())

            selected_pair = st.selectbox(
                "Select a trading pair to view details",
                available_pairs
            )

            if selected_pair:
                # Find the selected pair data
                pair_data = next((p for p in pair_results if p['Trading Pair'] == selected_pair), None)

                if pair_data:
                    # Show pair metrics
                    st.subheader(f"Metrics for {selected_pair}")
                    metrics_data = {k: v for k, v in pair_data.items() if k != 'Transactions'}
                    st.json(metrics_data)

                    # Show transactions
                    st.subheader(f"Transactions for {selected_pair}")
                    transactions_df = pd.DataFrame(pair_data['Transactions'])
                    st.dataframe(transactions_df.sort_values('Date', ascending=False))

                    # Create a chart of running holdings
                    st.subheader(f"Holdings History for {selected_pair}")
                    holdings_data = transactions_df[['Date', 'Running Holdings']]
                    holdings_data = holdings_data.set_index('Date')
                    st.line_chart(holdings_data)

                    # Create a chart of P/L
                    st.subheader(f"Cumulative P/L for {selected_pair}")
                    transactions_df['Cumulative P/L'] = transactions_df['Realized P/L'].cumsum()
                    pl_data = transactions_df[['Date', 'Cumulative P/L']]
                    pl_data = pl_data.set_index('Date')
                    st.line_chart(pl_data)

            # Download options
            st.subheader("Download Reports")

            col1, col2, col3 = st.columns(3)

            # Convert pair_summary_df for download
            with col1:
                st.markdown(get_download_link(pair_summary_df, "crypto_pair_summary.csv"), unsafe_allow_html=True)
            with col2:
                st.markdown(get_download_link(yearly_summary, "crypto_tax_yearly.csv"), unsafe_allow_html=True)

            # Create transactions dataframe for download
            all_transactions = []
            for pair_result in pair_results:
                for transaction in pair_result['Transactions']:
                    transaction_copy = transaction.copy()
                    transaction_copy['Trading Pair'] = pair_result['Trading Pair']
                    all_transactions.append(transaction_copy)

            all_transactions_df = pd.DataFrame(all_transactions)

            with col3:
                st.markdown(get_download_link(all_transactions_df, "crypto_tax_detailed.csv"), unsafe_allow_html=True)
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
        4. Download the reports for your tax filing

        ## Features:
        - Analyzes trading pairs instead of individual cryptocurrencies
        - Calculates accurate profit/loss by processing transactions chronologically
        - Tracks running holdings for each pair
        - Shows only filled orders (ignores canceled orders)
        - Generates yearly summaries for tax reporting

        Note: This app assumes that your CSV files follow the Binance export format.
        """)

if __name__ == "__main__":
    main()
