import pandas as pd
import os
import re
from datetime import datetime

def extract_crypto_amount(amount_str):
    """Extract numeric amount from strings like '10.5BTC' or '500ADA' or numbers"""
    if pd.isna(amount_str):
        return 0
    # Handle cases where it might already be a number
    if isinstance(amount_str, (int, float)):
        return float(amount_str)
    # Handle '0' string or empty string
    if amount_str == '0' or amount_str == '':
        return 0
    # Extract number from string like '123.45XYZ'
    match = re.match(r'([\-]?[\d.]+)([A-Za-z]*)', str(amount_str))
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0
    # Try converting directly if it's just a number string
    try:
        return float(amount_str)
    except ValueError:
        return 0

def extract_currency(amount_str):
    """Extract currency from strings like '10.5BTC' or '500ADA'"""
    if pd.isna(amount_str) or isinstance(amount_str, (int, float)):
        return ''
    if amount_str == '0' or amount_str == '':
        return ''
    match = re.match(r'([\-]?[\d.]+)([A-Za-z]+)', str(amount_str))
    if match and match.group(2):
        return match.group(2)
    return '' # Return empty if no currency symbol found

# Load all data files
data_folder = 'data'
all_data = []
print("Loading data files...")

for file in os.listdir(data_folder):
    if file.endswith('.csv'):
        file_path = os.path.join(data_folder, file)
        print(f"Processing {file_path}")
        try:
            # Specify dtype to avoid mixed type warnings, treat potential numbers as strings first
            df = pd.read_csv(file_path, dtype=str, keep_default_na=False)
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

if not all_data:
    print("No data files loaded. Exiting.")
    exit()

# Combine all data
combined_df = pd.concat(all_data, ignore_index=True)

# Filter only FILLED transactions
combined_df = combined_df[combined_df['Status'].str.upper() == 'FILLED'].copy()

# Convert Date and Time
combined_df['Date'] = pd.to_datetime(combined_df['Date(UTC)'], errors='coerce')
# Drop rows where date conversion failed
combined_df.dropna(subset=['Date'], inplace=True)

# Extract Base Amount and Currency from Executed
combined_df['Amount'] = combined_df['Executed'].apply(extract_crypto_amount)
combined_df['Currency'] = combined_df['Executed'].apply(extract_currency)

# Extract Total Cost and Quote Currency from Trading total
combined_df['Total Amount'] = combined_df['Trading total'].apply(extract_crypto_amount)
combined_df['Quote Currency'] = combined_df['Trading total'].apply(extract_currency)

# Extract Operation
combined_df['Operation'] = combined_df['Side'].str.upper()

# Extract Average Price (ensure it's numeric)
combined_df['Average Price'] = pd.to_numeric(combined_df['Average Price'], errors='coerce').fillna(0)

# Filter CAKE transactions
cake_transactions = combined_df[combined_df['Pair'].str.contains('CAKE', na=False)].copy()
cake_transactions = cake_transactions.sort_values('Date')

# Separate buy and sell transactions
cake_buys = cake_transactions[cake_transactions['Operation'] == 'BUY']
cake_sells = cake_transactions[cake_transactions['Operation'] == 'SELL']

# Calculate weighted average buy price
total_cake_bought = cake_buys['Amount'].sum()
total_cost = cake_buys['Total Amount'].sum()
average_buy_price = total_cost / total_cake_bought if total_cake_bought > 0 else 0

# Calculate weighted average sell price
total_cake_sold = cake_sells['Amount'].sum()
total_proceeds = cake_sells['Total Amount'].sum()
average_sell_price = total_proceeds / total_cake_sold if total_cake_sold > 0 else 0

print('\n===== CAKE TRADING ANALYSIS =====')
print(f'\nTotal CAKE transactions: {len(cake_transactions)}')
print(f'Total CAKE bought: {total_cake_bought} CAKE for {total_cost} BUSD')
print(f'Total CAKE sold: {total_cake_sold} CAKE for {total_proceeds} BUSD')
print(f'\nWeighted average buy price: {average_buy_price:.4f} BUSD')
print(f'Weighted average sell price: {average_sell_price:.4f} BUSD')

print('\n===== DETAILED CAKE BUY TRANSACTIONS =====')
print(cake_buys[['Date', 'Pair', 'Amount', 'Average Price', 'Total Amount', 'Quote Currency']].to_string(index=False))

print('\n===== MANUAL VERIFICATION =====')
running_cake_amount = 0
running_cost_basis = 0

# Process transactions in chronological order
for idx, row in cake_buys.iterrows():
    date = row['Date']
    amount = row['Amount']
    total = row['Total Amount']
    price = total / amount if amount > 0 else 0

    # Update running totals
    running_cake_amount += amount
    running_cost_basis += total

    # Calculate average cost
    avg_cost = running_cost_basis / running_cake_amount if running_cake_amount > 0 else 0

    print(f'Date: {date.date()} | Bought: {amount:.2f} CAKE @ {price:.4f} | Running Total: {running_cake_amount:.2f} CAKE | Avg Cost: {avg_cost:.4f} BUSD')

print(f'\nFinal verification - Avg buy price: {running_cost_basis / running_cake_amount if running_cake_amount > 0 else 0:.4f} BUSD')

# Let's also calculate how our method in the app does it
# This is similar to how the app calculates it
print('\n===== APP CALCULATION METHOD =====')
running_base_amount = 0  # Amount of CAKE
running_cost_basis = 0   # Total cost basis in BUSD

for idx, row in cake_transactions.sort_values('Date').iterrows():
    operation = row['Operation']
    amount = row['Amount']
    total_busd = row['Total Amount']
    avg_price_row = row['Average Price']

    # Update running totals
    if operation == 'BUY':
        # Buying increases our holdings and cost basis
        running_base_amount += amount
        running_cost_basis += total_busd
        avg_cost = running_cost_basis / running_base_amount if running_base_amount > 0 else 0
        print(f"BUY: {amount:.2f} CAKE @ {avg_price_row:.4f} | Running: {running_base_amount:.2f} CAKE | Avg Cost: {avg_cost:.4f} BUSD")

    elif operation == 'SELL':
        # Calculate profit/loss for this sale
        if running_base_amount > 0:
            # Calculate average cost for the amount being sold
            avg_cost_per_unit_before_sale = running_cost_basis / running_base_amount
            cost_of_units_sold = amount * avg_cost_per_unit_before_sale
            transaction_pl = total_busd - cost_of_units_sold

            # Update running totals
            running_base_amount -= amount

            # If we're selling more than we have, cap it at zero
            if running_base_amount < 0:
                running_base_amount = 0
                running_cost_basis = 0
            else:
                # Reduce cost basis proportionally
                running_cost_basis = running_cost_basis * (running_base_amount / (running_base_amount + amount))

            # Calculate profit/loss
            transaction_pl = total_busd - cost_of_units_sold

            print(f"SELL: {amount:.2f} CAKE @ {avg_price_row:.4f} | Running: {running_base_amount:.2f} CAKE | Avg Cost: {running_cost_basis / running_base_amount if running_base_amount > 0 else 0:.4f} BUSD | P/L: {transaction_pl:.2f} BUSD")

print(f'\nFinal running average cost: {running_cost_basis / running_base_amount if running_base_amount > 0 else 0:.4f} BUSD')
print(f'Current holdings: {running_base_amount:.2f} CAKE')
print(f'Current cost basis: {running_cost_basis:.2f} BUSD')
