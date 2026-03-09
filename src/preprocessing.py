import pandas as pd


def load_data(path):
    """
    Load dataset from CSV file
    """
    df = pd.read_csv(path)
    return df



def handle_missing_values(df):
    """
    Replace missing values with 0 for telecom usage columns
    """
    df = df.fillna(0)
    return df

def create_churn_label(df):
    """
    Create churn column using month 9 activity
    """
    df['churn'] = (
        (df['total_ic_mou_9'] == 0) &
        (df['total_og_mou_9'] == 0) &
        (df['vol_2g_mb_9'] == 0) &
        (df['vol_3g_mb_9'] == 0)
    ).astype(int)

    return df


def filter_high_value_customers(df):
    """
    Keep only high-value customers based on recharge behavior
    """
    df['avg_rech_6_7'] = (
        df['total_rech_amt_6'] + df['total_rech_amt_7']
    ) / 2

    threshold = df['avg_rech_6_7'].quantile(0.7)

    high_value_df = df[df['avg_rech_6_7'] >= threshold]

    return high_value_df




def drop_month9_columns(df):
    """
    Remove all columns related to month 9 to prevent data leakage
    """
    cols_9 = [col for col in df.columns if '_9' in col]
    df = df.drop(cols_9, axis=1)
    return df
