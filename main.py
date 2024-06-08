import os.path
import numpy as np
from model.model_builder import SimpleCNN4Cat,SimpleCNN4Ret,GradualWarmupScheduler
from dateutil.relativedelta import relativedelta
import datetime
from model.Trainer import ModelTrainer
from model.FetchingOriData import FetchingOriData
import torch.nn as nn
import torch
from utils.utils import get_rank,parse_time_circle
import gc  # 导入垃圾收集模块
from itertools import product
from model.CustomLoss import RankCorrelationLoss,RMSELoss,SmoothRankingLoss,DistanceCorrelationLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
import copy

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
class ModelRunner:
    def __init__(self, file_path, save_path, train_circle,test_circle,time_step,dropout_rate, start_date,end_date,
                batch_size,num_epoch,input_size,hidden_sizes=None, learning_rates=None,label = None,):
        """
        data_file_path : 即训练的数据
        save_path : signal保存的地址
        train_circle: 训练集的周期，例如12个月为12m. 90天为90d
        test_circle: 同理。测试集的周期。 若训练集为12个月，测试集为1个月。即每一个月滚动一次，使用前12个月的数据训练。
        """
        self.file_path = file_path
        self.save_path = save_path
        self.hidden_sizes = hidden_sizes
        self.learning_rates = learning_rates
        self.predictions = None
        self.pred_loss = None
        self.test_df = None
        self.label = label
        self.time_step = time_step
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.start_date = start_date
        self.end_date = end_date
        self.train_circle = train_circle
        self.test_circle = test_circle
        self.input_size = input_size
        self.device = None

    def run_training_cycles(self):
        current_date = datetime.datetime.strptime(self.start_date, '%Y-%m-%d')
        while True:
            train_delta = parse_time_circle(self.train_circle)
            test_delta = parse_time_circle(self.test_circle)

            train_data_start_date = (current_date - train_delta).strftime('%Y-%m-%d')
            train_data_end_date = (current_date - relativedelta(days=1)).strftime('%Y-%m-%d')
            test_data_start_date = current_date.strftime('%Y-%m-%d')
            test_data_end_date = (current_date + test_delta - relativedelta(days=1)).strftime('%Y-%m-%d')

            # 进行单次训练和测试周期
            self.train_and_evaluate(train_data_start_date, train_data_end_date, test_data_start_date,
                                    test_data_end_date)

            current_date += test_delta  # 移动到下一个测试周期
            if datetime.datetime.strptime(test_data_end_date, '%Y-%m-%d') >= datetime.datetime.strptime(self.end_date,
                                                                                                        '%Y-%m-%d'):
                break

    def create_model_optimizer_scheduler(self, num_channels, activation_function, linear_units):
        model = SimpleCNN4Cat(input_size=self.input_size, dropout_rate=self.dropout_rate, num_channels=num_channels,
                              activation_function=activation_function, linear_units=linear_units).to(self.device) \
            if self.label.startswith('future_cat') else \
            SimpleCNN4Ret(input_size=self.input_size, dropout_rate=self.dropout_rate, num_channels=num_channels,
                          activation_function=activation_function, linear_units=linear_units).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=20, after_scheduler=None)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epoch - 20, eta_min=1e-4)
        warmup_scheduler.after_scheduler = scheduler
        return model, optimizer, warmup_scheduler

    def train_and_evaluate(self, train_data_start_date, train_data_end_date, test_data_start_date, test_data_end_date):
        print(
            f'----------Training from {train_data_start_date} to {train_data_end_date}; Testing from {test_data_start_date} to {test_data_end_date}----------')

        # Setup data loaders
        train_datafetcher = FetchingOriData(self.file_path, batch_size=self.batch_size, mode='train',
                                            start_date=train_data_start_date, end_date=train_data_end_date,
                                            label=self.label)
        train_loader_dic = train_datafetcher.get_loaders(random_state=42)
        train_loader = train_loader_dic['train_loader']
        val_loader = train_loader_dic['val_loader']
        self.device = train_datafetcher.get_device()
        train_criterion = nn.NLLLoss() if self.label.startswith('future_cat') else RMSELoss()
        val_criterion = train_criterion
        # Setup test data loader
        test_datafetcher = FetchingOriData(self.file_path, batch_size=self.batch_size, mode='predict',
                                           start_date=test_data_start_date, end_date=test_data_end_date,
                                           label=self.label)
        test_loader_dic = test_datafetcher.get_loaders()
        test_loader = test_loader_dic['test_loader']
        test_df = test_loader_dic['df']
        best_val_loss = float('inf')
        best_model_params = None
        all_train_losses = []
        all_val_losses = []

        for num_channels, activation_function, linear_units in product(param_grid['num_channels'],
                                                                       param_grid['activation_function'],
                                                                       param_grid['linear_units']):
            print(f'Using Channels: {num_channels}, Activation: {activation_function}, Linear Units: {linear_units}')

            # Setup model with current parameters
            model, optimizer, warmup_scheduler = self.create_model_optimizer_scheduler(num_channels,
                                                                                       activation_function,
                                                                                       linear_units)

            trainer = ModelTrainer(model, train_loader, val_loader, optimizer, train_criterion, val_criterion,
                                   patience=30, scheduler=warmup_scheduler, label=self.label)
            trainer.train(self.num_epoch)  # Execute training and return validation loss
            all_train_losses.append(((num_channels, activation_function, linear_units), trainer.train_losses))
            all_val_losses.append(((num_channels, activation_function, linear_units), trainer.val_losses))

            # Check if this is the best model so far
            if trainer.best_val_loss < best_val_loss:
                best_val_loss = trainer.best_val_loss
                best_model_params = (num_channels, activation_function, linear_units)
                best_model_state = copy.deepcopy(model.state_dict())  # Save best model state

        # Re-setup and evaluate the best model
        print(f"Best Model Params: {best_model_params} with Val Loss: {best_val_loss}")

        model, optimizer, warmup_scheduler = self.create_model_optimizer_scheduler(best_model_params[0],
                                                                                   best_model_params[1],
                                                                                   best_model_params[2])

        trainer = ModelTrainer(model, train_loader, val_loader, optimizer, train_criterion, val_criterion,
                               patience=30, scheduler=warmup_scheduler, label=self.label)
        # Perform final predictions using the best model
        trainer.best_model_state = best_model_state
        predictions, pred_loss = trainer.predict(test_loader)
        best_model = trainer.best_model_state
        self.save_model(best_model, train_data_start_date, train_data_end_date, test_data_start_date,
                        test_data_end_date)  # Save the best model
        self.predictions, self.pred_loss = predictions, pred_loss
        self.test_df = test_df
        plot_path = f'{self.save_path}/plots'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        save_fig_path = f'{plot_path}/best_plot_train_from_{train_data_start_date}_to_{train_data_end_date}_test_from_{test_data_start_date}_to_{test_data_end_date}.svg'
        # trainer.plot_losses(save_fig_path=save_fig_path)  # Save the best model's loss plot
        self.save_results(train_data_start_date, train_data_end_date, test_data_start_date, test_data_end_date)
        para_plot_path =  f'{plot_path}/grid_plot_train_from_{train_data_start_date}_to_{train_data_end_date}_test_from_{test_data_start_date}_to_{test_data_end_date}.svg'
        self.visualize_losses(all_train_losses, all_val_losses, para_plot_path,best_model_params)

        # Cleanup
        del trainer, train_loader, val_loader, train_datafetcher, test_datafetcher
        torch.cuda.empty_cache()
        gc.collect()

    def visualize_losses(self,all_train_losses, all_val_losses, para_plot_path, best_model_params):
        fig, axs = plt.subplots(3, 1, figsize=(12, 15))  # 增加一个额外的子图用于显示最佳模型的损失

        # 绘制训练损失
        for params, losses in all_train_losses:
            num_channels, activation_function, linear_units = params
            label = f'Channels: {num_channels}, Activation: {activation_function}, Units: {linear_units}'
            lw = 1  # 线宽默认为1
            if params == best_model_params:
                label += " (Best)"
                lw = 3  # 最佳模型线宽加粗
                axs[2].plot(losses, label='Best Training Loss', linewidth=lw)  # 绘制最佳模型的训练损失
            axs[0].plot(losses, label=label, linewidth=lw)

        axs[0].set_title('Training Losses')
        axs[0].set_xlabel('Epochs')
        axs[0].set_ylabel('Loss')
        axs[0].legend()
        axs[0].grid(True)

        # 绘制验证损失
        for params, losses in all_val_losses:
            num_channels, activation_function, linear_units = params
            label = f'Channels: {num_channels}, Activation: {activation_function}, Units: {linear_units}'
            lw = 1  # 线宽默认为1
            if params == best_model_params:
                label += " (Best)"
                lw = 3  # 最佳模型线宽加粗
                axs[2].plot(losses, label='Best Validation Loss', linewidth=lw, linestyle='--')  # 绘制最佳模型的验证损失
            axs[1].plot(losses, label=label, linewidth=lw)

        axs[1].set_title('Validation Losses')
        axs[1].set_xlabel('Epochs')
        axs[1].set_ylabel('Loss')
        axs[1].legend()
        axs[1].grid(True)

        # 配置最佳模型损失图的样式
        axs[2].set_title('Best Model Losses')
        axs[2].set_xlabel('Epochs')
        axs[2].set_ylabel('Loss')
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        plt.savefig(para_plot_path)
        plt.close()

    # def train_and_evaluate(self, train_data_start_date, train_data_end_date, test_data_start_date, test_data_end_date):
    #     print(
    #         f'----------Training from {train_data_start_date} to {train_data_end_date}; Testing from {test_data_start_date} to {test_data_end_date}----------')
    #
    #     # Setup data loaders
    #     train_datafetcher = FetchingOriData(self.file_path, batch_size=self.batch_size, mode='train',
    #                                         start_date=train_data_start_date, end_date=train_data_end_date,
    #                                         label=self.label)
    #     train_loader_dic = train_datafetcher.get_loaders(random_state=42)
    #     train_loader = train_loader_dic['train_loader']
    #     val_loader = train_loader_dic['val_loader']
    #     device = train_datafetcher.get_device()
    #
    #     # Setup test data loader
    #     test_datafetcher = FetchingOriData(self.file_path, batch_size=self.batch_size, mode='predict',
    #                                        start_date=test_data_start_date, end_date=test_data_end_date,
    #                                        label=self.label)
    #     test_loader_dic = test_datafetcher.get_loaders()
    #     test_loader = test_loader_dic['test_loader']
    #     test_df = test_loader_dic['df']
    #
    #     # Setup model and training utilities
    #     model = SimpleCNN4Cat(input_size=self.input_size, dropout_rate=self.dropout_rate).to(
    #         device) if self.label.startswith('future_cat') \
    #         else SimpleCNN4Ret(input_size=self.input_size, dropout_rate=self.dropout_rate).to(device)
    #     optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    #     train_criterion = nn.NLLLoss() if self.label.startswith('future_cat') else RMSELoss()
    #     val_criterion = train_criterion  # In case of different validation criterion, adjust here
    #
    #     warmup_scheduler = GradualWarmupScheduler(optimizer, multiplier=10, total_epoch=20, after_scheduler=None)
    #     scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epoch - 20, eta_min=1e-4)
    #     warmup_scheduler.after_scheduler = scheduler
    #
    #     trainer = ModelTrainer(model, train_loader, val_loader, optimizer, train_criterion, val_criterion,
    #                            patience=30, scheduler=warmup_scheduler, label=self.label)
    #     trainer.train(self.num_epoch)  # Execute training
    #     best_model = trainer.best_model_state
    #
    #     # Save model, results, and plots
    #     self.save_model(best_model, train_data_start_date, train_data_end_date, test_data_start_date,
    #                     test_data_end_date)
    #     self.predictions, self.pred_loss = trainer.predict(test_loader)
    #     self.test_df = test_df
    #     plot_path = f'{self.save_path}/plots'
    #     if not os.path.exists(plot_path):
    #         os.makedirs(plot_path)
    #     save_fig_path = (f'{plot_path}/train_from_{train_data_start_date}_to_{train_data_end_date}_test_'
    #                      f'from_{test_data_start_date}_to_{test_data_end_date}.svg')
    #     trainer.plot_losses(save_fig_path=save_fig_path)
    #     self.save_results(train_data_start_date, train_data_end_date, test_data_start_date, test_data_end_date)
    #
    #     # Cleanup
    #     del trainer, train_loader, val_loader, train_datafetcher, test_datafetcher
    #     torch.cuda.empty_cache()
    #     gc.collect()
    def save_model(self, model_state, train_data_start_date, train_data_end_date,
                   test_data_start_date, test_data_end_date):
        model_directory = f'{self.save_path}/checkpoints'
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        model_filename = (f"model_train_from_{train_data_start_date}_to_{train_data_end_date}"
                          f"_test_from_{test_data_start_date}_to_{test_data_end_date}.pth")
        model_path = os.path.join(model_directory, model_filename)
        torch.save(model_state, model_path)
        print(f"Model saved at {model_path}")

    def save_results(self,train_data_start_date, train_data_end_date,test_start_date, test_end_date):
        cur_symbol = self.test_df.copy()
        if self.label.startswith('future_cat'):
            probabilities = np.exp(self.predictions)  # 转换 LogSoftmax 输出为概率
            class_2_probabilities = probabilities[:, 2]  # 选择第三列，即类别2的概率
            cur_symbol['factor'] = class_2_probabilities  # 更新self.predictions以存储特定类别的概率
        else:
            cur_symbol['factor'] = self.predictions

        symbol_df = get_rank(cur_symbol, 5)
        csv_path = f'{self.save_path}/predictions'
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        symbol_df.to_csv(f'{csv_path}/train_from_{train_data_start_date}_to_{train_data_end_date}_'
                         f'test_from_{test_start_date}_to_{test_end_date}.csv',
                         encoding='utf-8-sig', index=False)




if __name__ == '__main__':
    param_grid = {
        'num_channels': [
            [16, 32, 64],
            [32, 64, 128],
            [16, 32, 64,128],
        ],
        'activation_function': ['elu','relu'],
        'linear_units': [64,128]
    }
    time_step = 20
    dropout_rate = 0.25
    num_epochs = 50
    batch_size = 512
    index = 'zz500'
    train_circle = '48m'
    test_circle = '6m'
    input_size = (6,time_step,time_step)

    '''
    label 取值为如下
    ['future_cat_1d', 'future_return_1d', 'future_cat_7d', 'future_return_7d', 'future_cat_20d', 'future_return_20d']
    '''

    start_date = '2022-01-01'
    end_date = '2024-04-30'

    label = 'future_cat_1d'
    file_path = f'config/{index}_data_{time_step}'
    save_path = f'model_res/time_step_{time_step}_train_circle_{train_circle}_{label}_testing_from_{start_date}_to_{end_date}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    runner = ModelRunner(file_path, save_path, dropout_rate=dropout_rate,time_step=time_step,
                         train_circle=train_circle, test_circle=test_circle,label=label,
                         num_epoch=num_epochs, batch_size=batch_size,
                         start_date=start_date, end_date=end_date,input_size=input_size)
    runner.run_training_cycles()

