import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from .my_functions import read_large_csv_parallel
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
import os
from dateutil.relativedelta import relativedelta
import re
import bisect
from multiprocessing import Pool





def get_his_data(his_path):
    df = read_large_csv_parallel(his_path)
    df.rename(columns={
        'trade_date':'date',
        'ts_code':'token',
    },inplace=True)
    df.dropna(inplace=True)
    df['date'] = pd.to_datetime(df['date'].astype(str))
    df.sort_values(['token','date'],inplace=True)
    df.reset_index(drop=True,inplace=True)
    for i in [1,7,20]: #计算未来1,7,20天收益率
        df[f'future_{i}d_price'] = df.groupby('token')['close'].shift(-i)
        df[f'future_{i}d_ret'] = df[f'future_{i}d_price'] / df['close'] - 1
        df.drop(f'future_{i}d_price',axis=1,inplace=True)
    # 删除那些没有足够未来数据的行
    df.dropna(subset=[f'future_{i}d_ret' for i in [1,7,20]], inplace=True)

    def assign_quantile_labels(group):
        # qcut尝试将所有值分成具有相同数量的数据点的箱子，这里是三个箱子
        # labels参数指定了每个箱子的标签
        return pd.qcut(group, q=3, labels=[0, 1, 2])
    for i in [1,7,20]:
        df[f'future_{i}d_ret_rank'] = df.groupby('date')[f'future_{i}d_ret'].rank(method='dense',pct=True)
        df[f'future_{i}d_ret_cat'] = df.groupby('date')[f'future_{i}d_ret'].transform(assign_quantile_labels)

    df.sort_values(['date','token'],inplace=True)
    df.reset_index(drop=True,inplace=True)
    df['date'] = df['date'].dt.strftime('%Y-%m-%d')
    for date,group in df.groupby('date'):
        group.to_csv(f'config/his_data/{date}.csv',index=False,encoding='utf-8-sig')
    return df

def is_visible(y1, y2, y3, i, j, k):
    """判断点3是否在点1和点2之间“不可见”。"""
    # 使用直线方程的判断方法
    x1, x2, x3 = i, j, k
    return y3 > y1 + (y2 - y1) * (x3 - x1) / (x2 - x1)


def build_visibility_matrix(column_data):
    """根据Visibility Graph逻辑构建矩阵。"""
    n = len(column_data)
    matrix = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(i + 1, n):
            visible = True
            for k in range(i + 1, j):
                if is_visible(column_data[i], column_data[j], column_data[k], i, j, k):
                    visible = False
                    break
            if visible:
                matrix[i][j] = 1
                matrix[j][i] = 1

    return matrix

def process_token(df, date, token, time_step, group):
    sub_df = df[(df['token'] == token) & (df['date'] <= date)].tail(time_step)
    if len(sub_df) != time_step:
        return None

    matrices = []
    for feature in ['open', 'high', 'low', 'close', 'vol', 'amount']:
        matrices.append(build_visibility_matrix(sub_df[feature].values))

    y_data = {
        'future_cat_1d': group[group['token'] == token]['future_1d_ret_cat'].values[0],
        'future_return_1d': group[group['token'] == token]['future_1d_ret'].values[0],
        'future_cat_7d': group[group['token'] == token]['future_7d_ret_cat'].values[0],
        'future_return_7d': group[group['token'] == token]['future_7d_ret'].values[0],
        'future_cat_20d': group[group['token'] == token]['future_20d_ret_cat'].values[0],
        'future_return_20d': group[group['token'] == token]['future_20d_ret'].values[0]
    }

    X_tensor = torch.stack([torch.tensor(matrix, dtype=torch.float32) for matrix in matrices])
    return (X_tensor, y_data, token)


def generate_each_day(date,group,pbar,df,time_step,save_path):
    pbar.set_description(f"Processing {date}")

    daily_data = {
        'X': [],
        'Y': {
            'future_cat_1d': [],
            'future_return_1d': [],
            'future_cat_7d': [],
            'future_return_7d': [],
            'future_cat_20d': [],
            'future_return_20d': []
        },
        'tokens': []
    }
    flag = False
    cur_tokens = group['token'].unique()

    # 加一个判断
    # temp_df = df[(df['token'] == '000006.SZ') & (df['date'] <= date)].tail(time_step)
    # if len(temp_df) < time_step:
    #     pbar.update(1)
    #     return

    # with ThreadPoolExecutor() as executor:
    #     futures = {executor.submit(process_token, df, date, token, time_step, group): token for token in cur_tokens}
    #     for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing tokens for {date}"):
    #         result = future.result()
    #         if result:
    #             X_tensor, y_data, token = result
    #             daily_data['X'].append(X_tensor)
    #             daily_data['tokens'].append(token)
    #             for key, value in y_data.items():
    #                 daily_data['Y'][key].append(value)

    for token in tqdm(cur_tokens, desc='processing tokens',position=0,leave=False):
        sub_df = df[(df['token'] == token) & (df['date'] <= date)].tail(time_step)
        if len(sub_df) == time_step:
            flag = True
            matrices = []
            for feature in ['open', 'high', 'low', 'close', 'vol', 'amount']:
                matrices.append(build_visibility_matrix(sub_df[feature].values))
            y_future_cat_20d = group[group['token'] == token]['future_20d_ret_cat'].values[0]
            y_future_return_20d = group[group['token'] == token]['future_20d_ret'].values[0]
            y_future_cat_7d = group[group['token'] == token]['future_7d_ret_cat'].values[0]
            y_future_return_7d = group[group['token'] == token]['future_7d_ret'].values[0]
            y_future_cat_1d = group[group['token'] == token]['future_1d_ret_cat'].values[0]
            y_future_return_1d = group[group['token'] == token]['future_1d_ret'].values[0]

            X_tensor = torch.stack(
                [torch.tensor(matrix, dtype=torch.float32) for matrix in matrices])
            daily_data['X'].append(X_tensor)
            daily_data['Y']['future_cat_1d'].append(y_future_cat_1d)
            daily_data['Y']['future_return_1d'].append(y_future_return_1d)
            daily_data['Y']['future_cat_7d'].append(y_future_cat_7d)
            daily_data['Y']['future_return_7d'].append(y_future_return_7d)
            daily_data['Y']['future_cat_20d'].append(y_future_cat_20d)
            daily_data['Y']['future_return_20d'].append(y_future_return_20d)
            daily_data['tokens'].append(token)

    if flag:
        daily_data['X'] = torch.stack(daily_data['X']) if daily_data['X'] else torch.tensor([])
        for i in [1, 7, 20]:
            daily_data['Y'][f'future_cat_{i}d'] = torch.tensor(daily_data['Y'][f'future_cat_{i}d'], dtype=torch.long) if \
                daily_data['Y'][f'future_cat_{i}d'] else torch.tensor([], dtype=torch.long)

            daily_data['Y'][f'future_return_{i}d'] = torch.tensor(daily_data['Y'][f'future_return_{i}d'],
                                                                  dtype=torch.float) if \
                daily_data['Y'][f'future_return_{i}d'] else torch.tensor([], dtype=torch.float)

        daily_pickle_path = os.path.join(save_path, f"{date}.pickle")
        with open(daily_pickle_path, 'wb') as handle:
            pickle.dump(daily_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    pbar.update(1)

#单进程的版本
def load_data_single(df, save_path, time_step, start_date=None, end_date=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    trade_dates = sorted(df['date'].unique())

    if start_date is not None and end_date is not None:
        # 使用 bisect 找到大于等于 start_date 的第一个交易日的索引位置
        start_idx = bisect.bisect_left(trade_dates, start_date)
        # 确保找到的日期在列表中，且不超过列表边界
        if start_idx == len(trade_dates):  # start_date 大于所有已知交易日
            raise ValueError("start_date is beyond the range of available dates")

        # 计算实际的开始索引，考虑边界问题
        actual_start_idx = max(0, start_idx - time_step)
        actual_start_date = trade_dates[actual_start_idx]

        # 筛选数据
        df = df[(df['date'] >= actual_start_date) & (df['date'] <= end_date)]

    pbar = tqdm(total=df['date'].nunique(), desc='Starting...',position=1,leave=False)

    for date, group in df.groupby('date'):
        generate_each_day(date,group,pbar,df,time_step,save_path)

    pbar.close()




def get_rank(data_df, n):
    '''
    data_df 至少包含date token future_return  factor 列
    '''
    data_df.sort_values(['date','token'], inplace=True)
    data_df.reset_index(inplace=True, drop=True)

    data_df['预测收益率排名'] = data_df.groupby('date')['factor'].rank(pct=True)

    num_groups = n
    bins = [i / num_groups for i in range(num_groups + 1)]
    labels = list(range(1, num_groups + 1))

    data_df['group_factor'] = pd.cut(data_df['预测收益率排名'], bins=bins, labels=labels).astype(int)

    data_df.drop(['预测收益率排名'], axis=1, inplace=True)

    data_df['真实收益率排名'] = data_df.groupby('date')['Y'].rank(pct=True)

    num_groups = n
    bins = [i / num_groups for i in range(num_groups + 1)]
    labels = list(range(1, num_groups + 1))

    data_df['group_future_return'] = pd.cut(data_df['真实收益率排名'], bins=bins, labels=labels).astype(int)

    data_df.drop(['真实收益率排名'], axis=1, inplace=True)

    return data_df


def parse_time_circle(circle_str):
    """
    解析时间周期字符串，如 '12m' 或 '1d'，并返回相应的 relativedelta 对象。
    """
    match = re.match(r"(\d+)([md])", circle_str)
    if not match:
        raise ValueError("Invalid format for time circle. Use format like '12m' for months or '1d' for days.")

    quantity, unit = int(match.group(1)), match.group(2)
    if unit == 'm':
        return relativedelta(months=quantity)
    elif unit == 'd':
        return relativedelta(days=quantity)