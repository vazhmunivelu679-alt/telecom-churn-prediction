import pandas as pd


def create_recharge_decline(df):

    df["rech_amt_decline"] = df["total_rech_amt_8"] - df["total_rech_amt_6"]

    return df


def create_outgoing_call_decline(df):

    df["og_mou_decline"] = df["total_og_mou_8"] - df["total_og_mou_6"]

    return df


def create_incoming_call_decline(df):
   
    df["ic_mou_decline"] = df["total_ic_mou_8"] - df["total_ic_mou_6"]

    return df


def add_behavioral_features(df):
    
    df = create_recharge_decline(df)
    df = create_outgoing_call_decline(df)
    df = create_incoming_call_decline(df)

    return df

def create_avg_recharge(df):
    """
    Create average recharge for month 6 and 7.
    """

    df["avg_rech_6_7"] = (
        df["total_rech_amt_6"] + df["total_rech_amt_7"]
    ) / 2

    return df