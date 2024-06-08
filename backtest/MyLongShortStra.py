import backtrader as bt
import pandas as pd

class LSStrategy(bt.Strategy):
    params = (
        ('maperiod', 20),
        ('nk', 13),
        ('printlog', True),

    )



    def __init__(self , bin, symbol_path):
        self.order = None
        self.buy_price = None
        self.buy_comm = None
        df = pd.read_csv(symbol_path)
        self.symbol_df = df
        self.bin = bin
        self.name_to_data = {d._name: d for d in self.datas}


    def next(self):
        cur_date = self.datas[0].datetime.date(0) # 获取当前日期
        print(f'-------------{cur_date}--------------------')
        symbol_df = self.symbol_df[self.symbol_df['date'] == str(cur_date)] #获取当前日期的symbol
        cur_val = self.broker.get_value() + self.broker.get_cash() #获取当前日期的总仓位价值
        long_df = symbol_df[symbol_df['group_factor'] == 5] #获取当前日期满足分组的可交易的df
        short_df = symbol_df[symbol_df['group_factor'] == 1]

        hold_list = [
            _p._name
            for _p in self.broker.positions
            if self.broker.getposition(_p).size != 0
        ] #获取当前持仓
        long_list = long_df['token'].to_list() #获取今天应该买的持仓
        short_list = short_df['token'].to_list()
        len_long = len(long_list) #今天该买多少只股票
        len_short = len(short_list)

        for token in hold_list: # 对于当前持仓，如果今天不应该买它，那么就清仓
            if token not in long_list and token not in short_list:
                #卖出token
                d = self.name_to_data[token]
                self.close(d)



        for token in long_list: #对于今天要买的票，均仓买入

            d = self.name_to_data[token]
            cur_target = cur_val * ( 0.95/ len_long)
            self.order = self.order_target_value(d, target=cur_target)

        for token in short_list: #对于今天要买的票，均仓买入

            d = self.name_to_data[token]
            cur_target = cur_val * ( -0.95/  len_short)
            self.order = self.order_target_value(d, target=cur_target)


        #print(f'today is {str(cur_date)}, symbol is {cur_symbol},price={cur_price},cur_target={cur_target}')


    def notify_order(self, order):
        # 未被处理的订单
        if order.status in [order.Submitted, order.Accepted]:
            return
        # 已经处理的订单
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order.isbuy():
                self.log(
                    "订单编号ref:%.0f，成交价: %.6f, 成交额: %.2f, 手续费 %.2f, 成交量: %.2f, 证券名称: %s"
                    % (
                        order.ref,  # 订单编号
                        order.executed.price,  # 成交价
                        order.executed.value,  # 成交额
                        order.executed.comm,  # 佣金
                        order.executed.size,  # 成交量
                        order.data._name,  # 股票名称
                    )
                )
            else:  # sell
                self.log(
                    "订单编号ref:%.0f，成交价: %.6f, 成交额: %.2f, 手续费 %.2f, 成交量: %.2f, 证券名称: %s"
                    % (
                        order.ref,
                        order.executed.price,
                        order.executed.value,
                        order.executed.comm,
                        order.executed.size,
                        order.data._name,
                    )
                )

    def log(self, txt, dt=None):
        """可选，构建策略打印日志的函数：可用于打印订单记录或交易记录等"""
        dt = dt or self.datas[0].datetime.date(0)
        print("%s, %s" % (dt.isoformat(), txt))

    def notify_trade(self, trade):  # 记录交易收益情况
        if not trade.isclosed:
            return
        self.log(f"策略收益：\n毛收益 {trade.pnl:.2f}, 净收益 {trade.pnlcomm:.2f}")

    def stop(self):  # 回测结束后输出结果
        self.log("期末总资金 %.2f" % (self.broker.getvalue()))

