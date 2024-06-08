import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import datetime
import quantstats as qs
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class FinancialDataVisualizer:
    def __init__(self, start_date, end_date, csv_folder_path, bmk_df,save_path,file_name_prefix):
        self.start_date = start_date
        self.end_date = end_date
        self.csv_folder_path = csv_folder_path
        self.bmk_df = bmk_df
        self.all_layers_df = None
        self.cumulative_returns = None
        self.save_path = save_path
        self.file_name_prefix = file_name_prefix

    def load_and_process_data(self):
        # 生成日期列表
        date_range = pd.date_range(self.start_date, self.end_date)
        self.all_layers_df = pd.DataFrame(index=date_range)

        # 循环读取每个CSV文件
        for i in range(1, 6):
            file_path = f'{self.csv_folder_path}/bin{i}.csv'
            layer_df = pd.read_csv(file_path, header=0)# 使用第一行作为列标题
            layer_df['index'] = pd.to_datetime(layer_df['index'])
            layer_df.set_index('index', inplace=True)
            self.all_layers_df = pd.concat([self.all_layers_df, layer_df['return']], axis=1)

        # 重新命名列为累计收益率
        self.all_layers_df.columns = [f'layer_{i}_return' for i in range(1, 6)]
        self.all_layers_df.index = date_range

        self.all_layers_df['bmk_return'] = self.bmk_df['bmk']
        self.all_layers_df['bmk_return'][0] = 0
        # 计算每一层的累计收益率并添加到原始DataFrame中
        self.cumulative_returns = (1 + self.all_layers_df).cumprod()

    def plot_bar_chart(self, save_path):
        last_day_return = self.cumulative_returns.iloc[-1] - 1
        fig, ax = plt.subplots()
        width = 0.4

        for i, (layer, ret) in enumerate(last_day_return.items()):
            # 对于 bmk_return 列，设置特殊标签
            if layer == 'bmk_return':
                layer_label = 'BMK'
            else:
                # 对于其他列，假设它们遵循 'layer_x' 的命名方式
                layer_label = 'L' + layer.split('_')[1]

            color = 'green' if ret < 0 else 'red'
            ax.bar(i, ret, width=width, color=color, align='center')
            ax.text(i, ret, f'{ret:.2%}', ha='center', va='bottom', fontsize=10)

        ax.set_title('Layer Returns')
        ax.set_ylabel('Return')
        ax.set_xlabel('Layer')
        ax.set_xticks(range(len(last_day_return)))
        # 为 bmk_return 列设置特殊标签，并为其他列生成标签
        labels = [f'L{i + 1}' if name != 'bmk_return' else 'BMK' for i, name in enumerate(last_day_return.index)]
        ax.set_xticklabels(labels)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.axhline(y=0, color='black', linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{save_path}/{self.file_name_prefix}_bar_value.svg')

    def plot_bar_chart_excess_value(self, save_path):
        last_day_return = self.cumulative_returns.iloc[-1] - 1
        last_day_return = last_day_return - last_day_return[-1]
        fig, ax = plt.subplots()
        width = 0.4

        for i, (layer, ret) in enumerate(last_day_return[:-1].items()):
            layer_label = 'L' + layer.split('_')[1]
            color = 'green' if ret < 0 else 'red'
            ax.bar(i, ret, width=width, color=color, align='center')
            ax.text(i, ret, f'{ret:.2%}', ha='center', va='bottom', fontsize=10)

        ax.set_title('Layer Returns')
        ax.set_ylabel('Return')
        ax.set_xlabel('Layer')
        ax.set_xticks(range(len(last_day_return[:-1])))
        ax.set_xticklabels([f'L{i+1}' for i in range(len(last_day_return[:-1]))])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.axhline(y=0, color='black', linestyle='--')
        plt.tight_layout()
        plt.savefig(f'{save_path}/{self.file_name_prefix}_bar_excess_value.svg')

    def plot_cumulative_returns(self, save_path):
        plt.figure(figsize=(12, 6))
        for column in self.cumulative_returns.columns:
            plt.plot(self.cumulative_returns.index, self.cumulative_returns[column], label=column)

        plt.title('Cumulative Returns Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.savefig(f'{save_path}/{self.file_name_prefix}_layer_return.svg', format='svg')

    def get_qsquant(self):
        ret_df = pd.concat([pd.read_csv(f'{self.save_path}/bin{i}.csv').set_index('index') for i in range(1, 6)],
                           axis=1)
        ret_df.index = pd.to_datetime(ret_df.index)
        ret_df.columns = [f'{i}th group' for i in range(1, 6)]

        self.bmk_df = self.bmk_df.loc[ret_df.index]
        qs.reports.html(
            ret_df, benchmark=self.bmk_df, output=f"{self.save_path}/analysis.html",
            title=f'analysis', match_dates=False
        )

    def run(self):
        self.load_and_process_data()
        self.plot_bar_chart(self.save_path)
        self.plot_bar_chart_excess_value(self.save_path)
        self.plot_cumulative_returns(self.save_path)
        self.get_qsquant()


if __name__ == '__main__':
    start = datetime.datetime(2023, 5, 26)
    end = datetime.datetime(2024, 1, 11)
    csv_folder_path = 'D:/zorro/week6/gpfactor/config/backtest_res'
    benchmark_csv_path = 'D:/zorro/week3/研报复现/config/bmk.csv'
    save_path = 'D:/zorro/week6/gpfactor/backtest_res'
    visualizer = FinancialDataVisualizer(start, end, csv_folder_path, benchmark_csv_path,save_path=save_path)
    visualizer.load_and_process_data()
    visualizer.plot_bar_chart(csv_folder_path)  # 保存条形图
    visualizer.plot_cumulative_returns(csv_folder_path)  # 保存累积收益率图
