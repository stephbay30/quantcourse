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
from backtest.MyLongShortStra import LSStrategy
from backtest.visualized_res import  FinancialDataVisualizer
from backtest.GenAverageReport import gen_average_report
from tqdm import tqdm
import matplotlib
import numpy as np

# 切换为图形界面显示的终端TkAgg
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# 回测策略
class BackTest:
    """选股策略"""

    def __init__(self, start, end, cerebro,symbol_df, bin,his_df,bmk_df,save_html_name = None,save_file_name = None):
        self.data_df=None
        self.start = start
        self.end = end
        self.cerebro = cerebro
        self.bin = bin
        self.symbol_df = symbol_df
        self.save_file_name = save_file_name
        self.save_html_name = save_html_name
        self.bmk_df = bmk_df
        self.his_df = his_df


    def feed_data(self):

        trade_data = self.his_df.copy()
        trade_data.rename(
            columns={
                "开盘时间": "date",
                "货币对": "token",
                "开盘价": "open",
                "最高价": "high",
                "最低价": "low",
                "收盘价": "close",
                '成交量': 'volume',
            },
            inplace=True,
        )
        trade_data['date'] = trade_data['date'].str[:10]
        trade_data.index = pd.to_datetime(trade_data.date)
        trade_data = trade_data[trade_data.index >= '2021-01-01']
        trade_data['openinterest'] = 0
        token_lst = trade_data['token'].unique()
        dic = {}
        for token in token_lst:
            cur_df = trade_data[trade_data['token'] == token].copy()
            cur_df = cur_df[['open', 'high', 'low', 'close', 'volume', 'openinterest']]
            dic[token] = cur_df

        max_days_token = max(dic, key=lambda x: len(dic[x]))
        min_days_token = min(dic, key=lambda x: len(dic[x]))
        # 获取最长天数的数据df
        max_days_df = dic[max_days_token]

        # 遍历所有token
        for token, df in dic.items():
            # 合并数据df，使用日期作为键，缺失的数据用0填充
            merged_df = max_days_df.merge(df, how='left', left_index=True, right_index=True, suffixes=('_max_days', ''))
            merged_df.fillna(0, inplace=True)

            # 删除后缀为'_max_days'的列
            merged_df.drop(columns=[col for col in merged_df.columns if col.endswith('_max_days')], inplace=True)

            # 更新原字典中的数据df
            dic[token] = merged_df

        for token, df in dic.items():
            datafeed = bt.feeds.PandasData(dataname=df, fromdate=self.start, todate=self.end)
            self.cerebro.adddata(datafeed, name=token)

    def load_strategy(self):
        self.cerebro.addstrategy(TestStrategy, bin = self.bin, symbol_df = self.symbol_df)
        return

    def cerebro_config(self):
        """
        设置交易回测假设
        :param cerebro:
        :return:
        """

        # 初始资金 100,000,000
        self.cerebro.broker.setcash(1_000_000.0)
        # self.cerebro.broker.set_coc(True)#设置为当天收盘价交易
        self.cerebro.broker.setcommission(commission=0.0004, )

        # 滑点：双边各 0.0001
        # 滑点后超出最高价True仍然会成交
        # 滑点后超出最高价True仍然会以最高价成交
        self.cerebro.broker.set_slippage_perc(
            perc=0.001,
            slip_open=True,
            slip_match=True,
            slip_out=False,
            )

        # 添加分析性工具
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name="_SharpeRatio_A")
        # 返回收益率时序
        self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="_TimeReturn")
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")  # 收益率
        self.cerebro.addanalyzer(bt.analyzers.Transactions, _name="transactions")  # 交易信息
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trade_analyzer")  # 交易分析
        self.cerebro.addanalyzer(bt.analyzers.PyFolio, _name="pyfolio")
        return

    def analyse(self, result):
        portfolio_stats = result[0].analyzers.getbyname("pyfolio")
        returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
        returns.index = returns.index.tz_localize(None)
        # returns.index = pd.to_datetime(returns.index)

        if self.save_file_name is not None:

            returns.to_csv(f'{self.save_file_name}.csv', index=True,
                           encoding='utf-8-sig')

            qs.reports.html(
                returns, benchmark=self.bmk_df, output=f"{self.save_file_name}.html",
                title=self.save_html_name
            )
        else:
            return
        return

    def run(self):
        # 设置交易参数
        self.cerebro_config()

        # 装载数据
        self.feed_data()
        # 装载策略
        self.load_strategy()


        # 启动回测
        #result = self.cerebro.run(maxcpus=20)
        result = self.cerebro.run()
        #self.cerebro.plot()
        self.analyse(result)
        return result


if __name__ == "__main__":
    # 设置起止日期与引擎
    bmk_path = 'D:/zorro/回测数据/bmk.csv'
    his_path = 'D:/zorro/回测数据/all_data.csv'
    his_df = pd.read_csv(his_path)
    bmk_df = pd.read_csv(bmk_path, parse_dates=['date'])
    bmk_df.set_index("date", inplace=True)

    symbol_path = fr'D:/zorro/week11/graphic_CNN/config/train_circle_12m_7d'
    symbol_files = sorted(os.listdir(symbol_path))
    symbol_df = pd.concat([pd.read_csv(f'{symbol_path}/{f}') for f in symbol_files],ignore_index=True)


    save_path_fold = fr'D:/zorro/week11/graphic_CNN/test_res/train_circle_12m_7d'
    if not os.path.exists(save_path_fold):
        os.makedirs(save_path_fold)
    for days in range(7):
        start = datetime.datetime(2024, 1, 1) + datetime.timedelta(days=days)
        end = datetime.datetime(2024,4, 12)
        save_path = fr'{save_path_fold}/track_{days}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for bin in range(1,6):
            cerebro = bt.Cerebro()
            save_html_name = save_path + '/' + f'bin{bin}'
            save_file_name = save_path + '/' + f'bin{bin}'
            print(f'now test bin {bin}')
            backtestor = BackTest(start=start, end=end, cerebro=cerebro, bin =bin, symbol_df= symbol_df,
                                  save_html_name=save_html_name,save_file_name=save_file_name,
                                  his_df = his_df,bmk_df=bmk_df)
            backtestor.run()


        visualizer = FinancialDataVisualizer(start, end, save_path , bmk_df, save_path=save_path,file_name_prefix='test')
        visualizer.run()

    gen_average_report(save_path_fold)
    start = datetime.datetime(2024, 1, 1)
    end = datetime.datetime(2024, 4, 12)

    visualizer_avg = FinancialDataVisualizer(start, end, save_path_fold, bmk_df, save_path=save_path_fold, file_name_prefix='avg')
    visualizer_avg.run()












