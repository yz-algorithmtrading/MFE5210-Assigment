import os
import pandas as pd
import numpy as np
from loguru import logger
from utils import LassoLogistic, Sharpe, log_output
from factors import ts_mome, ts_rule_mome, open_interest, term_structure 
from data_process import process_mome, process_open_interest, process_term_structure
from pathos.multiprocessing import Pool
from functools import partial
import warnings
import datetime
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '1'

if os.path.exists(os.path.join(os.getcwd(), 'output', 'Logistic', 'record.log')):
    os.remove(os.path.join(os.getcwd(), 'output', 'Logistic', 'record.log'))
logger.add(os.path.join(os.getcwd(), 'output', 'Logistic', 'record.log'))


def fun_lassologistic_IS(params, roll_seq_data, future_description, best_params):
    # best_params = pd.read_csv(os.path.join(os.getcwd(), 'output','SingleFactor','IS','IS_best_params.csv'), index_col=0)
    code, rolling_period, C, penalty = params
    roll_seq_data = roll_seq_data[roll_seq_data.code==code]
    df_active = roll_seq_data[roll_seq_data.type == 'active'].rename(columns={'trading_date': 'date'}).set_index('date')
    df_return = df_active['adj_close'] - df_active['adj_close'].shift(1)
    df_return = df_return.to_frame().rename(columns={'adj_close': 'return'})
    ts_mome_param = int(best_params.loc[best_params.code==code, 'ts_mome'].dropna().values[0])
    ts_rule_mome_param = eval(best_params.loc[best_params.code==code, 'ts_rule_mome'].dropna().values[0])
    open_interest_param = int(best_params.loc[best_params.code==code, 'open_interest'].dropna().values[0])
    factor_ts_mome = ts_mome(process_mome(roll_seq_data), ts_mome_param).shift(1)
    factor_ts_rule_mome = ts_rule_mome(process_mome(roll_seq_data), ts_rule_mome_param[0], ts_rule_mome_param[1]).shift(1)
    factor_open_interest = open_interest(process_open_interest(roll_seq_data), open_interest_param).shift(1)
    factor_term_structure = term_structure(process_term_structure(roll_seq_data, future_description)).shift(1)
    df = pd.concat([factor_ts_mome, factor_ts_rule_mome, factor_open_interest, factor_term_structure],axis=1).dropna()
    df = pd.merge(df.reset_index(drop=False), df_return, on='date', how='inner')
    df_result = LassoLogistic(n=10, df= df, rolling_period=rolling_period, C=C, penalty=penalty, freq='daily')
    sharpe = Sharpe(df_result, 'pnl', 'daily')
    log_output(f"LassoLogistic————IS:\
               Params: code——{code}; rolling_period——{rolling_period}; C——{C}; penalty——{penalty}\
               Sharpe: {sharpe}", if_multiprocessing=True)
    LassoLogistic_params_dict = {'code': code, 'rolling_period':rolling_period, 'C':C, 'penalty':penalty, 'sharpe':sharpe} 
    return LassoLogistic_params_dict

def fun_lassologistic_OS(params, roll_seq_data, future_description, best_params):
    # best_params = pd.read_csv(os.path.join(os.getcwd(), 'output','SingleFactor','IS','IS_best_params.csv'), index_col=0)
    code, rolling_period, C, penalty = params
    roll_seq_data = roll_seq_data[roll_seq_data.code==code]
    df_active = roll_seq_data[roll_seq_data.type == 'active'].rename(columns={'trading_date': 'date'}).set_index('date')
    df_return = df_active['adj_close'] - df_active['adj_close'].shift(1)
    df_return = df_return.to_frame().rename(columns={'adj_close': 'return'})
    ts_mome_param = int(best_params.loc[best_params.code==code, 'ts_mome'].dropna().values[0])
    ts_rule_mome_param = eval(best_params.loc[best_params.code==code, 'ts_rule_mome'].dropna().values[0])
    open_interest_param = int(best_params.loc[best_params.code==code, 'open_interest'].dropna().values[0])
    factor_ts_mome = ts_mome(process_mome(roll_seq_data), ts_mome_param).shift(1)
    factor_ts_rule_mome = ts_rule_mome(process_mome(roll_seq_data), ts_rule_mome_param[0], ts_rule_mome_param[1]).shift(1)
    factor_open_interest = open_interest(process_open_interest(roll_seq_data), open_interest_param).shift(1)
    factor_term_structure = term_structure(process_term_structure(roll_seq_data, future_description)).shift(1)
    df = pd.concat([factor_ts_mome, factor_ts_rule_mome, factor_open_interest, factor_term_structure],axis=1).dropna()
    df = pd.merge(df.reset_index(drop=False), df_return, on='date', how='inner')
    df_result = LassoLogistic(n=10, df= df, rolling_period=rolling_period, C=C, penalty=penalty, freq='daily')
    
    df_result = df_result[df_result.date > roll_seq_data.trading_date.max() - datetime.timedelta(days=365)]
    sharpe = Sharpe(df_result, 'pnl', 'daily')
    log_output(f"LassoLogistic————OS:\
               Params: code——{code}; rolling_period——{rolling_period}; C——{C}; penalty——{penalty}\
               Sharpe: {sharpe}", if_multiprocessing=False)
    plt.plot(df_result['date'], df_result['cum_pnl'])
    title = fr'{code}-lassologistic-{sharpe}_pnl'
    plt.title(title)
    plt.savefig(os.path.join(os.getcwd(), 'output', 'Logistic','OS', fr'{code}'+'.png'))
    plt.close('all')
    return df_result


def fun_IS():
    roll_seq_data = pd.read_parquet(os.path.join(os.getcwd(), 'data', 'future_roll_seq_data.parquet'))
    roll_seq_data = roll_seq_data[roll_seq_data.trading_date <= roll_seq_data.trading_date.max() - datetime.timedelta(days=365)]
    future_description = pd.read_parquet(os.path.join(os.getcwd(), 'data', 'future_description.parquet'))
    best_params = pd.read_csv(os.path.join(os.getcwd(), 'output','SingleFactor','IS','IS_best_params.csv'), index_col=0)
    codes = ['LH', 'A']
    rolling_periods = [52,104,208,312,416,624]
    Cs = [0.1,0.5,1,5,10,50]
    penalties = ['l1', 'l2']
    ps= []
    for code in codes:
        for rolling_period in rolling_periods:
            for C in Cs:
                for penalty in penalties:
                    ps.append((code, rolling_period, C, penalty))
    partial_fun_lassologistic_IS = partial(fun_lassologistic_IS, roll_seq_data=roll_seq_data, future_description=future_description, best_params=best_params)

    with Pool() as pool:
        result = [pool.apply_async(partial_fun_lassologistic_IS, (p,)) for p in ps]
        result = [i.get() for i in result]
    result_list=[]
    for i in range(len(result)):
        temp = pd.DataFrame(result[i], index=[i])
        result_list.append(temp)
    result_df = pd.concat(result_list, axis=0).sort_values(['code', 'sharpe']).reset_index(drop=True)
    result_df.to_csv(os.path.join(os.getcwd(), 'output','Logistic','IS','logistic_best_params.csv'))
    return

def fun_OS():
    roll_seq_data = pd.read_parquet(os.path.join(os.getcwd(), 'data', 'future_roll_seq_data.parquet'))
    future_description = pd.read_parquet(os.path.join(os.getcwd(), 'data', 'future_description.parquet'))
    best_params = pd.read_csv(os.path.join(os.getcwd(), 'output','SingleFactor','IS','IS_best_params.csv'), index_col=0)
    result_df = pd.read_csv(os.path.join(os.getcwd(), 'output','Logistic','IS','logistic_best_params.csv'), index_col=0)
    best_params = pd.read_csv(os.path.join(os.getcwd(), 'output','SingleFactor','IS','IS_best_params.csv'), index_col=0)
    codes = sorted(result_df.code.unique())
    params = []
    for code in codes:
        params = result_df[result_df.code==code]
        params = params[params.sharpe == params.sharpe.max()]
        rolling_period = params['rolling_period'].values[0]
        C = params['C'].values[0]
        penalty = params['penalty'].values[0]
        params = (code, rolling_period, C, penalty)
        fun_lassologistic_OS(params, roll_seq_data, future_description, best_params)
    return
       



if __name__ == '__main__':
    fun_IS()
    fun_OS()





    
