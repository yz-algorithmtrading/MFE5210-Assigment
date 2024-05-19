import os
import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def Sharpe(df,column_name,freq,yearly_risk_free_rate = 0.02):
    if freq == 'monthly':
        IR = (df[column_name].mean() - yearly_risk_free_rate / 12)/ df[column_name].std()
        return 3.464 * IR
    elif freq == 'weekly':
        IR = (df[column_name].mean() - yearly_risk_free_rate / 52)/ df[column_name].std()
        return 7.211 * IR
    elif freq == 'daily':
        IR = (df[column_name].mean() - yearly_risk_free_rate / 252)/ df[column_name].std()
        return 15.8745 * IR   
    
        

def LassoLogistic(n,df,rolling_period,C,penalty = 'l1',freq = 'weekly'):
    # 逻辑回归组合多因子
    x_col = list(df.columns)
    x_col.remove('date')
    x_col.remove('return')
    if 'adj_close' in df.columns:
        x_col.remove('adj_close')
    X = df[x_col]
    y = (df['return'] > 0 ).astype(int)
    y_pred = [None] * n
    for i in range(n,len(X)):
        X_train,y_train = X[max(0,i-rolling_period):i],y[max(0,i-rolling_period):i]
        scalar = StandardScaler()
        X_train_scaled = scalar.fit_transform(X_train)
        model = LogisticRegression(penalty=penalty,solver='liblinear',C = C)
        model.fit(X_train_scaled,y_train)
        X_next = scalar.transform(X.iloc[i,:].values.reshape(1,-1))
        y_next_pred = model.predict(X_next)
        y_pred.append(y_next_pred[0])

    df['predicted'] = y_pred
    df = df.dropna()
  

    df['signal'] = (df['predicted'] - 0.5) * 2
    df['delta_position'] = np.abs(df['signal'] - df['signal'].shift(1))
    df['pnl'] = df['return'] * df['signal'] 
      
    df['cum_pnl'] = df['pnl'].cumsum()
 
    return df


def log_output(content, log_type='info', if_multiprocessing=False):
    if log_type == 'info':
        if if_multiprocessing:
            process_id = os.getpid()
            logger.info(fr'进程{process_id}: ' + content)
        else:
            logger.info(content)
    else:
        print('Log Type not supported')
    return
