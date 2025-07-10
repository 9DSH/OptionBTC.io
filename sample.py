from Fetch_data import Fetching_data
import pandas as pd

# Fetch the financial data
fetch = Fetching_data()
options_ = fetch.fetch_option_data()

# Read the CSV file, allowing for bad lines to be skipped
raw_df = pd.read_csv("options_screener.csv", on_bad_lines='skip')  # Updated to skip bad lines
print(raw_df.shape)

def validate_df_mismatch(df):
    print("Starting validate_df_mismatch...")
    expected_fields = 18

    # Check for valid rows by comparing the number of columns
    valid_rows = df[df.apply(lambda row: len(row) == expected_fields, axis=1)].reset_index(drop=True)

    print(f"Number of valid rows: {valid_rows.shape[0]} out of {df.shape[0]}")
    print("validate_df_mismatch complete.")
    return valid_rows

# Validate the DataFrame for mismatched rows
filter_df = validate_df_mismatch(raw_df)

print(filter_df.shape)
print(filter_df)