# %%
import pandas as pd
import numpy as np


def ts_mome(data: pd.DataFrame, period: int) -> pd.DataFrame:
    """ 周期时序动量因子

    Args:
        data (pd.DataFrame): _description_
        period (int): 周期

    Returns:
        pd.DataFrame: _description_
    """

    def get_volatility_df(data: pd.DataFrame, volatility_period: 60) -> pd.DataFrame:
        """ 年化波动率加权

        Args:
            data (pd.DataFrame): 数据
            volatility_period (60): 波动率周期

        Returns:
            pd.DataFrame: n days * m futures, 每天每个品种的波动率
        """

        # 波动率序列
        volatility_df = data.pct_change().rolling(volatility_period).std() * np.sqrt(volatility_period)
        volatility_df.fillna(method='bfill', inplace=True)

        return volatility_df
    
    ret_df = data.pct_change()
    volatility_df = get_volatility_df(data, period)
    volatility_df = volatility_df.apply(lambda x: np.square(x), axis=0)
    factor_value = ret_df.rolling(period).sum().div(volatility_df).dropna()
    factor_value.rename(columns={'adj_close': 'ts_mome'}, inplace=True)
    factor_value['ts_mome'] = factor_value['ts_mome'].apply(lambda x: 1 if x>0 else -1)
    return factor_value


def ts_rule_mome(data: pd.DataFrame, short_period: int, long_period: int, volatility_period=60) -> pd.DataFrame:
    """ 周期时序长短周期动量因子

    Args:
        short_period (int): 短周期
        long_period (int): 长周期
        volatility_period (int): 波动率周期

    Returns:
        pd.DataFrame: 因子值
    """

    short_ma = data.apply(lambda x: x.rolling(short_period).mean(), axis=0)
    long_ma = data.apply(lambda x: x.rolling(long_period).mean())
    factor_value = (short_ma >= long_ma).astype(int)
    factor_value.rename(columns = {'adj_close': 'ts_rule_mome'}, inplace=True)
    factor_value['ts_rule_mome'] = factor_value['ts_rule_mome'].replace(0, -1)
    return factor_value


def open_interest(data: pd.DataFrame, period: int) -> pd.DataFrame:
    """ 仓单因子 """
    factor_value = data.pct_change(period).dropna()
    factor_value['open_interest'] = factor_value['open_interest'].apply(lambda x: 1 if x>0 else -1)
    return factor_value


def term_structure(data: pd.DataFrame) -> pd.DataFrame:
    """ 期限结构因子
    Args:
        data (pd.DataFrame): 包含主力次主力adj_close, last_trading_day
    Returns:
        pd.DataFrame: 因子值
    """
    
    data['time_delta'] = (data['sec_last_trading_day'] - data['act_last_trading_day']).apply(lambda x: x.days)
    factor_value = -(np.log(data['sec_adj_close']) - np.log(data['act_adj_close'])).div(data['time_delta']) * 365
    factor_value = factor_value.to_frame().rename(columns={0: 'term_structure'})
    factor_value['term_structure'] = factor_value['term_structure'].apply(lambda x: 1 if x>0 else -1)
    return factor_value