import datetime
import backtrader as bt
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
# 引入修改后的数据读取包
import quantstats as qs
# 引入策略
from backtest.MyStrategy import TestStrategy
from backtest.visualized_res import  FinancialDataVisualizer

from tqdm import tqdm
import matplotlib
import numpy as np

# 切换为图形界面显示的终端TkAgg
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


bin_series = {bin: [] for bin in range(1, 6)}
track_count = 5

# 遍历每个bin和每个track
for bin in range(1, 6):
    for track in range(track_count):
        # 构建文件路径
        sub_track_path = os.path.join('D:/zorro/week7/多颗粒度/config/backtest_res', f'track_{track}')
        file_path = os.path.join(sub_track_path, f'bin{bin}.csv')

        # 读取CSV文件
        cur_df = pd.read_csv(file_path)
        cur_return = cur_df['return'].values
        bin_series[bin].append(cur_return)
        if track == 0 and bin == 1:
            date_series = cur_df['index'].values
for bin, series in bin_series.items():
    # 确定最长序列的长度
    max_length = max(len(s) for s in series)

    # 补0使所有序列长度相同
    adjusted_series = []
    for s in series:
        length_difference = max_length - len(s)
        # 如果序列短于最长序列，则在前面补0
        if length_difference > 0:
            adjusted_series.append(np.concatenate([np.zeros(length_difference), s]))
        else:
            adjusted_series.append(s)

    # 计算平均值
    average_series = np.mean(adjusted_series, axis=0)

    # 保存到CSV文件
    df = pd.DataFrame({
        'index': date_series,
        'return': average_series
    })

    file_path = os.path.join('D:/zorro/week7/多颗粒度/config/backtest_res', f'bin{bin}.csv')
    df.to_csv(file_path, index=False)

start = datetime.datetime(2023, 7, 14)
end = datetime.datetime(2024, 3, 8)
bmk_path = 'D:/zorro/回测数据/bmk.csv'
save_path = 'D:/zorro/week7/多颗粒度/config/backtest_res'
visualizer = FinancialDataVisualizer(start, end, save_path, bmk_path, save_path=save_path,
                                     file_name_prefix='')
visualizer.run()