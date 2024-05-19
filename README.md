# 5210_Assignment
## data_process.py 
处理本地数据。
## factors.py
定义因子，包括时序动量因子，持仓因子，期限结构因子。
## utils.py
包含计算Sharpe和逻辑回归的功能函数。
## single_factor.py
取最近一年为样本外的时间范围，其它为样本内时间范围。
在样本内对各个单因子进行回测，找到使sharpe最大的参数，用该参数在样本外测试。
## Loggistic.py
用于组合因子，在样本内使用lassologistc的方法训练得到最优参数，在样本外测试。

单因子测试了所有的期货品种
用logistic组合因子，只测试了LH 和 A两个品种
结果都保存在out'put中
