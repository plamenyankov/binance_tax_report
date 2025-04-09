import pandas as pd
import os
import numpy as np

def convert_excel_to_csv(excel_file, csv_file):
    """
    Convert an Excel file to CSV format while maintaining the same structure as binance_2024.csv
    """
    try:
        # Read the Excel file
        df = pd.read_excel(excel_file)

        # --- Filtering Logic ---
        # 1. Remove rows that look like headers repeated in the data
        df = df[~df['Date(UTC)'].astype(str).str.contains('Date', na=False)]

        # 2. Keep only rows that have a valid 'Order No.' (not NaN/empty)
        #    We need to check the original column name 'Order No.'
        df = df[pd.notna(df['Order No.'])]
        df = df[df['Order No.'].astype(str).str.strip() != '']

        # 3. Keep only rows where 'Pair' contains a '/' (initial check for valid pairs)
        df = df[df['Pair'].astype(str).str.contains('/', na=False)]
        # --- End Filtering Logic ---

        # --- Data Transformation ---
        # Extract Base and Quote Currencies before modifying Pair
        df[['BaseCurrency', 'QuoteCurrency']] = df['Pair'].astype(str).str.split('/', expand=True)

        # Correct the Pair format by removing the slash *after* extracting currencies
        df['Pair_Formatted'] = df['Pair'].astype(str).str.replace('/', '', regex=False)
        # --- End Data Transformation ---

        # Create a mapping between the Excel columns and the desired CSV columns
        column_mapping = {
            'Date(UTC)': 'Date(UTC)',
            'Order No.': 'OrderNo',
            'Type': 'Type', # This will be mapped, but the Side column will determine Buy/Sell
            'Order Price': 'Order Price',
            'Order Amount': 'Order Amount',
            'AvgTrading Price': 'Average Price',
            'Status': 'Status'
            # 'Side' column will be derived if not present or needs correction
        }

        # Create a new DataFrame with the desired structure
        new_df = pd.DataFrame()

        # Map the columns that exist in both formats
        for excel_col, csv_col in column_mapping.items():
            if excel_col in df.columns:
                 # Use the potentially transformed column from df
                new_df[csv_col] = df[excel_col]

        # Assign the correctly formatted Pair column
        new_df['Pair'] = df['Pair_Formatted']

        # Construct Executed column (Amount + BaseCurrency)
        df['Filled'] = pd.to_numeric(df['Filled'], errors='coerce').fillna(0)
        new_df['Executed'] = df['Filled'].astype(str) + df['BaseCurrency'].astype(str)

        # Construct Trading total column (Amount + QuoteCurrency)
        df['Total'] = pd.to_numeric(df['Total'], errors='coerce').fillna(0)
        new_df['Trading total'] = df['Total'].astype(str) + df['QuoteCurrency'].astype(str)

        # Handle Side column explicitly based on Type column (more reliable)
        if 'Type' in df.columns:
             # Ensure Type column is string and handle potential NaNs before applying upper()
            df['Type'] = df['Type'].fillna('UNKNOWN').astype(str)
            new_df['Side'] = df['Type'].apply(lambda x: x.upper() if isinstance(x, str) else 'UNKNOWN')
            # Overwrite the 'Type' column in new_df with the Excel's 'Type' if needed for consistency,
            # but the logic above primarily uses it to derive 'Side'.
            if 'Type' in column_mapping: # Check if Type needs to be in the final CSV per mapping
                 new_df['Type'] = df['Type'] # Use the cleaned string version
        else:
            new_df['Side'] = 'UNKNOWN' # Default if Type column is missing

        # Add Time column (using Date(UTC) as source)
        if 'Date(UTC)' in df.columns:
            new_df['Time'] = df['Date(UTC)']
        else:
             new_df['Time'] = ''

        # Convert *only specific* numeric columns to numeric format,
        # leaving Executed and Trading total as strings from the source.
        numeric_columns_to_convert = ['Order Price', 'Order Amount', 'Average Price']
        for col in numeric_columns_to_convert:
            if col in new_df.columns:
                new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                new_df[col] = new_df[col].fillna(0)

        # Define the exact columns expected in the final CSV
        expected_columns = [
            "Date(UTC)", "OrderNo", "Pair", "Type", "Side",
            "Order Price", "Order Amount", "Time", "Executed",
            "Average Price", "Trading total", "Status"
        ]

        # Add any missing expected columns and fill with empty strings
        for col in expected_columns:
            if col not in new_df.columns:
                new_df[col] = ''

        # Reorder columns to match the exact expected format
        new_df = new_df[expected_columns]

        # Replace any remaining NaN values just before saving
        new_df = new_df.fillna('')

        # Convert columns intended to be strings explicitly to string type
        # This helps prevent pandas from inferring them as numeric if they look like numbers
        string_columns = ['Executed', 'Trading total', 'OrderNo', 'Status', 'Type', 'Side', 'Pair']
        for col in string_columns:
             if col in new_df.columns:
                 new_df[col] = new_df[col].astype(str)

        # Save to CSV, ensuring quoting matches the example file
        new_df.to_csv(csv_file, index=False, quoting=1) # quoting=1 corresponds to csv.QUOTE_ALL
        print(f"Successfully converted {excel_file} to {csv_file} with filtering and pair formatting.")
        return True

    except Exception as e:
        print(f"Error converting file: {str(e)}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        return False

if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define input and output file paths
    excel_file = os.path.join(script_dir, "data", "binanace_2025.xlsx") # Corrected typo
    csv_file = os.path.join(script_dir, "data", "binance_2025.csv")

    # Convert the file
    success = convert_excel_to_csv(excel_file, csv_file)

    if success:
        print("Conversion completed successfully!")
    else:
        print("Conversion failed. Please check the error messages above.")
