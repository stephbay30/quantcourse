import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor

import time
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

    data_df['真实收益率排名'] = data_df.groupby('date')['future_return'].rank(pct=True)

    bins = [i / num_groups for i in range(num_groups + 1)]
    labels = list(range(1, num_groups + 1))

    data_df['group_return'] = pd.cut(data_df['真实收益率排名'], bins=bins, labels=labels).astype(int)

    data_df.drop(['真实收益率排名'], axis=1, inplace=True)

    return data_df


def get_rank_new(data_df, n, group_by_date=True):
    '''
    data_df 至少包含date token future_return  factor 列
    group_by_date: 布尔值，指示是否需要在不同日期上分组排名
    '''
    data_df.sort_values(['date','token'], inplace=True)
    data_df.reset_index(inplace=True, drop=True)

    if group_by_date:
        data_df['预测收益率排名'] = data_df.groupby('date')['factor'].rank(pct=True)
    else:
        data_df['预测收益率排名'] = data_df['factor'].rank(pct=True)

    num_groups = n
    bins = [i / num_groups for i in range(num_groups + 1)]
    labels = list(range(1, num_groups + 1))

    data_df['group_factor'] = pd.cut(data_df['预测收益率排名'], bins=bins, labels=labels).astype(int)

    data_df.drop(['预测收益率排名'], axis=1, inplace=True)

    if group_by_date:
        data_df['真实收益率排名'] = data_df.groupby('date')['future_return'].rank(pct=True)
    else:
        data_df['真实收益率排名'] = data_df['future_return'].rank(pct=True)

    bins = [i / num_groups for i in range(num_groups + 1)]
    labels = list(range(1, num_groups + 1))

    data_df['group_return'] = pd.cut(data_df['真实收益率排名'], bins=bins, labels=labels).astype(int)

    data_df.drop(['真实收益率排名'], axis=1, inplace=True)

    return data_df


def get_bmk(df):
    '''
    df行情数据
    有 date token close 列

    '''

    _df = df.copy()
    _df.rename(columns={
        '开盘时间' :'date',
        '货币对' : 'token',
        '收盘价' :'close'
    },inplace=True)
    _df.sort_values(by=['token','date'],inplace=True)
    _df.reset_index(inplace=True,drop=True)
    _df['return'] = _df.groupby('token')['close'].pct_change()
    _df.dropna(inplace=True) 
    df1 = _df[_df['token'] == 'BTCUSDT'][['date','return','token']]
    df2 = _df[_df['token'] == 'ETHUSDT'][['date','return','token']]
    merged_df = pd.merge(df1, df2, on='date', suffixes=('_BTCUSDT', '_ETHUSDT'))

    merged_df['bmk'] = (merged_df['return_BTCUSDT'] + merged_df['return_ETHUSDT']) / 2 
    merged_df['date'] = merged_df['date']
    merged_df = merged_df[['date','bmk']]
    merged_df['date'] = merged_df['date'].str[:10]
    return merged_df


def read_csv_parallel(file_name, path):
    df = pd.read_csv(os.path.join(path, file_name))
    # df = df[df['货币对'] == 'BTCUSDT'].copy()  # 如果需要筛选
    return df

def read_large_num_parallel(file_list, path):
    # 根据您的机器配置调整max_workers，或者可以省略使用默认值自动决定
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(read_csv_parallel, file, path) for file in file_list]
        results = [future.result() for future in futures]
    return pd.concat(results, ignore_index=True)

def read_large_csv_parallel(data_file_path):
    start_time = time.time()

    file_list = os.listdir(data_file_path)
    df_ml = read_large_num_parallel(file_list, data_file_path)

    end_time = time.time()
    running_time = end_time - start_time
    print("程序运行时间：", running_time, "秒")

    return df_ml

