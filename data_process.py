import pandas as pd

def process_mome(roll_seq_data):
    df = roll_seq_data.copy()
    df = df[df.type == 'active']
    df = df[['trading_date', 'adj_close']].sort_values('trading_date')
    df.rename(columns={'trading_date': 'date'}, inplace=True)
    df.set_index('date', inplace=True)
    return df


def process_open_interest(roll_seq_data):
    df = roll_seq_data.copy()
    df = df[df.type == 'active']
    df = df[['trading_date', 'open_interest']].sort_values('trading_date')
    df.rename(columns={'trading_date': 'date'}, inplace=True)
    df.set_index('date', inplace=True)
    return df

def process_term_structure(roll_seq_data, future_description):
    roll_seq_data = roll_seq_data.copy()
    future_description = future_description.copy()
    future_description = future_description[['symbol', 'delistdate']]
    roll_seq_data_active = roll_seq_data[roll_seq_data.type == 'active'][['trading_date', 'adj_close', 'symbol_real']].sort_values('trading_date')
    roll_seq_data_sec_active = roll_seq_data[roll_seq_data.type == 'sec_active'][['trading_date', 'adj_close', 'symbol_real']].sort_values('trading_date')
    df_active = pd.merge(roll_seq_data_active, future_description, left_on='symbol_real', right_on='symbol', how='left').rename(
        columns={'trading_date':'date', 'delistdate':'act_last_trading_day','adj_close':'act_adj_close'})
    df_sec_active = pd.merge(roll_seq_data_sec_active, future_description, left_on='symbol_real', right_on='symbol', how='left').rename(
        columns={'trading_date':'date', 'delistdate':'sec_last_trading_day','adj_close':'sec_adj_close'})
    df = pd.merge(df_active, df_sec_active, on='date', how='inner')
    df = df[['date', 'act_adj_close', 'sec_adj_close', 'act_last_trading_day', 'sec_last_trading_day']].set_index('date')
    df['act_last_trading_day'] = pd.to_datetime(df['act_last_trading_day'])
    df['sec_last_trading_day'] = pd.to_datetime(df['sec_last_trading_day'])
    return df
