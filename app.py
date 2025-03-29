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
    # Filter completed transactions and clean up data
    df = df[df['Status'] == 'FILLED'].copy()

    # Convert date string to datetime
    df['Date'] = pd.to_datetime(df['Date(UTC)'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month

    # Extract crypto amount and currency
    df['Amount'] = df['Executed'].apply(extract_crypto_amount)
    df['Currency'] = df['Executed'].apply(extract_currency)

    # Extract pair information
    df['Base Currency'] = df['Pair'].str.replace('BUSD', '').str.replace('EUR', '')

    # Calculate total in BUSD
    # Extract amount and quote currency from Trading total
    df['Total Amount'] = df['Trading total'].apply(extract_crypto_amount)
    df['Quote Currency'] = df['Trading total'].apply(extract_currency)

    # For consistency, we'll work with BUSD as our base currency
    # In a more advanced version, we could add currency conversion
    df['Total BUSD'] = df['Total Amount']

    # Mark BUY/SELL transactions
    df['Operation'] = df['Side']

    return df

def generate_summary(processed_df):
    """Generate a summary of all transactions by coin"""
    # Group by currency and operation
    buy_df = processed_df[processed_df['Operation'] == 'BUY']
    sell_df = processed_df[processed_df['Operation'] == 'SELL']

    # Calculate totals for each coin
    buy_totals = buy_df.groupby('Currency').agg({
        'Amount': 'sum',
        'Total BUSD': 'sum'
    })

    sell_totals = sell_df.groupby('Currency').agg({
        'Amount': 'sum',
        'Total BUSD': 'sum'
    })

    # Combine into a summary dataframe
    all_currencies = set(buy_totals.index) | set(sell_totals.index)
    summary_data = []

    for currency in all_currencies:
        buy_amount = buy_totals.loc[currency, 'Amount'] if currency in buy_totals.index else 0
        buy_value = buy_totals.loc[currency, 'Total BUSD'] if currency in buy_totals.index else 0
        sell_amount = sell_totals.loc[currency, 'Amount'] if currency in sell_totals.index else 0
        sell_value = sell_totals.loc[currency, 'Total BUSD'] if currency in sell_totals.index else 0

        # Calculate profit/loss
        net_amount = buy_amount - sell_amount
        profit_loss = sell_value - buy_value

        avg_buy_price = buy_value / buy_amount if buy_amount > 0 else 0
        avg_sell_price = sell_value / sell_amount if sell_amount > 0 else 0

        summary_data.append({
            'Currency': currency,
            'Total Bought': buy_amount,
            'Avg Buy Price': avg_buy_price,
            'Total Buy Value': buy_value,
            'Total Sold': sell_amount,
            'Avg Sell Price': avg_sell_price,
            'Total Sell Value': sell_value,
            'Net Position': net_amount,
            'Profit/Loss': profit_loss
        })

    summary_df = pd.DataFrame(summary_data)
    return summary_df

def generate_yearly_summary(processed_df):
    """Generate a summary of transactions by year"""
    yearly_data = []

    for year, year_df in processed_df.groupby('Year'):
        buy_df = year_df[year_df['Operation'] == 'BUY']
        sell_df = year_df[year_df['Operation'] == 'SELL']

        total_buy = buy_df['Total BUSD'].sum()
        total_sell = sell_df['Total BUSD'].sum()
        profit_loss = total_sell - total_buy

        yearly_data.append({
            'Year': year,
            'Total Buy Value': total_buy,
            'Total Sell Value': total_sell,
            'Profit/Loss': profit_loss
        })

    yearly_df = pd.DataFrame(yearly_data)
    return yearly_df

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

            # Show raw data sample
            st.subheader("Raw Transaction Data Sample")
            st.dataframe(processed_df.head())

            # Overall Summary
            summary_df = generate_summary(processed_df)

            st.subheader("Summary by Currency")
            st.dataframe(summary_df.sort_values('Currency'))

            # Yearly Summary
            yearly_summary = generate_yearly_summary(processed_df)

            st.subheader("Yearly Summary")
            st.dataframe(yearly_summary.sort_values('Year'))

            # Currency selection for detailed view
            st.subheader("Detailed View by Currency")
            available_currencies = sorted(processed_df['Currency'].unique())

            selected_currency = st.selectbox(
                "Select a currency to view details",
                available_currencies
            )

            if selected_currency:
                currency_data = processed_df[processed_df['Currency'] == selected_currency]
                st.dataframe(currency_data.sort_values('Date', ascending=False))

                # Simple chart
                st.subheader(f"Buy/Sell History for {selected_currency}")

                # Group by month and operation
                chart_data = currency_data.groupby(['Year', 'Month', 'Operation']).agg({
                    'Total BUSD': 'sum'
                }).reset_index()

                # Create date field for chart
                chart_data['Date'] = chart_data.apply(
                    lambda row: datetime(int(row['Year']), int(row['Month']), 1),
                    axis=1
                )

                # Pivot for chart
                pivot_data = chart_data.pivot(
                    index='Date',
                    columns='Operation',
                    values='Total BUSD'
                ).fillna(0)

                st.line_chart(pivot_data)

            # Download options
            st.subheader("Download Reports")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(get_download_link(summary_df, "crypto_tax_summary.csv"), unsafe_allow_html=True)
            with col2:
                st.markdown(get_download_link(yearly_summary, "crypto_tax_yearly.csv"), unsafe_allow_html=True)
            with col3:
                st.markdown(get_download_link(processed_df, "crypto_tax_detailed.csv"), unsafe_allow_html=True)
    else:
        st.info("Please upload Binance CSV files or select files from the data folder to generate a report.")

    # Display info about the app
    with st.expander("About this app"):
        st.markdown("""
        This app helps you generate tax reports from your Binance transaction history CSV files.

        ## How to use:
        1. Export your transaction history from Binance as CSV
        2. Upload the CSV file(s) here or use the sample data
        3. View the summary and detailed reports
        4. Download the reports for your tax filing

        Note: This app assumes that your CSV files follow the Binance export format.
        """)

if __name__ == "__main__":
    main()
