import pandas as pd
import numpy as np

def analyze():
    df = pd.read_csv("data/raw/retail.csv")

    # Normalize columns for easier access
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    print("Shape:", df.shape)
    print("\nColumns and Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nCardinality:")
    for c in df.columns:
        if df[c].dtype == 'object' or df[c].dtype == 'bool':
            print(f"{c}: {df[c].nunique()}")
            if df[c].nunique() < 20:
                print(f"  Values: {df[c].unique()}")

    # Check for leakage
    # Clean up numeric columns
    df['Price_Per_Unit'] = pd.to_numeric(df['Price_Per_Unit'], errors='coerce')
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    df['Total_Spent'] = pd.to_numeric(df['Total_Spent'], errors='coerce')

    df_clean = df.dropna(subset=['Price_Per_Unit', 'Quantity', 'Total_Spent'])
    calculated_total = df_clean['Price_Per_Unit'] * df_clean['Quantity']
    correlation = calculated_total.corr(df_clean['Total_Spent'])
    print(f"\nCorrelation between (Price * Quantity) and Total_Spent: {correlation}")

    print("\nTarget Analysis (Total_Spent):")
    print(df['Total_Spent'].describe())

    print("\nTarget Analysis (Discount_Applied):")
    print(df['Discount_Applied'].value_counts(dropna=False))

    print("\nProtected Attribute Candidates:")
    print("Location:", df['Location'].unique())
    print("Payment_Method:", df['Payment_Method'].unique() if 'Payment_Method' in df.columns else "N/A")

if __name__ == "__main__":
    analyze()
