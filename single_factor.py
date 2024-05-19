# 样本内回测
from data_process import process_mome, process_open_interest, process_term_structure
from factors import ts_mome, ts_rule_mome, open_interest, term_structure
from utils import Sharpe
import numpy as np
import os
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import datetime


if os.path.exists(os.path.join(os.getcwd(), 'output', 'SingleFactor', 'record.log')):
    os.remove(os.path.join(os.getcwd(), 'output', 'SingleFactor', 'record.log'))
logger.add(os.path.join(os.getcwd(), 'output', 'SingleFactor', 'record.log'))


def get_sharpe(roll_seq_data,future_description, factor_name, param_dict:dict):
    code = param_dict['code']
    roll_seq_data = roll_seq_data[roll_seq_data.code == code].sort_values('trading_date')
    df_active = roll_seq_data[roll_seq_data.type == 'active'].rename(columns={'trading_date': 'date'}).set_index('date')
    df_return = df_active['adj_close'] - df_active['adj_close'].shift(1)
    df_return = df_return.to_frame().rename(columns={'adj_close': 'return'})
    if factor_name == 'ts_mome':
        period = param_dict[factor_name]
        df_factor = process_mome(roll_seq_data)
        factor_value = ts_mome(df_factor, period)
    elif factor_name == 'ts_rule_mome':
        short_period, long_period = param_dict[factor_name]
        df_factor = process_mome(roll_seq_data)
        factor_value = ts_rule_mome(df_factor, short_period, long_period)
    elif factor_name == 'open_interest':
        period = param_dict[factor_name]
        df_factor = process_open_interest(roll_seq_data)
        factor_value = open_interest(df_factor, period)
    elif factor_name == 'term_structure':
        df_factor = process_term_structure(roll_seq_data, future_description)
        factor_value = term_structure(df_factor)

    factor_value = factor_value.shift(1).dropna()
    df = pd.merge(factor_value.reset_index(drop=False), df_return.reset_index(drop=False), on='date', how='inner')
    df['pnl'] = df[factor_name] * df['return']
    df['cum_pnl'] = df['pnl'].cumsum()
    sharpe = Sharpe(df, 'pnl', 'daily')
    return sharpe, df

def find_best_params_single_factor(roll_seq_data, future_description, code, factor_name):
    periods = [1, 20, 60, 180, 240]
    short_periods = [5, 20, 60]
    long_periods = [20, 60, 240]
    param_dict_list = []
    if (factor_name == 'ts_mome') | (factor_name == 'open_interest'):
        for period in periods:
            param_dict = {'code':code, factor_name:period}
            param_dict_list.append(param_dict)
    elif factor_name == 'ts_rule_mome':
        for i in range(len(short_periods)):
            short_period = short_periods[i]
            long_period = long_periods[i]
            param_dict = {'code': code, factor_name:(short_period, long_period)}
            param_dict_list.append(param_dict)

    result = {'code': code, factor_name: None}
    sharpe = -np.inf
    for param_dict in param_dict_list:
        sharpe_temp, df =  get_sharpe(roll_seq_data, future_description, factor_name, param_dict)
        if sharpe_temp > sharpe:
            sharpe = sharpe_temp
            result[factor_name] = param_dict[factor_name]
            result['sharpe'] = sharpe
    logger.info(fr"Single_Factor_InSample ————— factor_name:{factor_name}  code:{code}  params:{result[factor_name]}  sharpe:{result['sharpe']}")
    return result

def fun_single_factor_IS():
    roll_seq_data = pd.read_parquet(os.path.join(os.getcwd(), 'data', 'future_roll_seq_data.parquet'))
    roll_seq_data = roll_seq_data[roll_seq_data.trading_date <= roll_seq_data.trading_date.max() - datetime.timedelta(days=365)]
    future_description = pd.read_parquet(os.path.join(os.getcwd(), 'data', 'future_description.parquet'))
    codes = list(sorted(roll_seq_data.code.unique()))
    codes.remove('TL')
    results = []
    for code in codes:
        for factor_name in ['ts_mome', 'ts_rule_mome', 'open_interest']:
            result = find_best_params_single_factor(roll_seq_data, future_description, code, factor_name)
            results.append(result)
            sharpe, df = get_sharpe(roll_seq_data, future_description, factor_name, result)
            plt.plot(df['date'], df['cum_pnl'])
            title = fr'{code}-{factor_name}-{sharpe}_pnl'
            plt.title(title)
            plt.savefig(os.path.join(os.getcwd(), 'output', 'SingleFactor','IS', fr'{code}-{factor_name}'+'.png'))
            plt.close('all')
    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(os.getcwd(), 'output', 'SingleFactor','IS', 'IS_best_params.csv'))
    return

def fun_single_factors_OS():
    roll_seq_data = pd.read_parquet(os.path.join(os.getcwd(), 'data', 'future_roll_seq_data.parquet'))
    future_description = pd.read_parquet(os.path.join(os.getcwd(), 'data', 'future_description.parquet'))
    best_params = pd.read_csv(os.path.join(os.getcwd(), 'output','SingleFactor','IS','IS_best_params.csv'), index_col=0)
    for code in sorted(best_params.code.unique()):
        for factor_name in ['ts_mome', 'ts_rule_mome', 'open_interest']:
            param = best_params.loc[best_params.code==code, factor_name].dropna().values[0]
            if type(param) == str:
                param = eval(param)
            if type(param) == np.float64:
                param = int(param)
            result = {'code':code, factor_name:param}
            sharpe, df = get_sharpe(roll_seq_data, future_description, factor_name, result)
            df = df[df.date>roll_seq_data.trading_date.max() - datetime.timedelta(days=365)]
            sharpe = Sharpe(df, 'pnl', 'daily')
            logger.info(fr'Single_Factor_OutOfSample ———— code:{code}  factor_name:{factor_name} params:{param} sharpe: {sharpe}')
            plt.plot(df['date'], df['cum_pnl'])
            title = fr'{code}-{factor_name}-{sharpe}_pnl'
            plt.title(title)
            plt.savefig(os.path.join(os.getcwd(), 'output', 'SingleFactor','OS', fr'{code}-{factor_name}'+'.png'))
            plt.close('all')

        factor_name = 'term_structure'
        sharpe, df = get_sharpe(roll_seq_data, future_description, factor_name, {'code':code, factor_name:None})
        logger.info(fr'All ———— code:{code}  TermStructure  Sharpe:{sharpe}')
        plt.plot(df['date'], df['cum_pnl'])
        title = fr'{code}-{factor_name}-{sharpe}_pnl'
        plt.title(title)
        plt.savefig(os.path.join(os.getcwd(), 'output', 'SingleFactor','OS', fr'{code}-{factor_name}'+'.png'))
        plt.close('all')





if __name__ == '__main__':
    fun_single_factor_IS()
    fun_single_factors_OS()

    



