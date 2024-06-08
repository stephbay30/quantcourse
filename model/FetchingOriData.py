import os
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd

class FetchingOriData:
    def __init__(self, file_path, batch_size, start_date, end_date, label,mode='train'):
        self.file_path = file_path
        self.batch_size = batch_size
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.label = label

    def get_device(self):
        return self.device

    def get_loaders(self, random_state=None):
        # 加载数据
        X, Y, tokens, df_Y_tokens = self._load_data()
        if self.mode == 'train':
            # 分割数据为训练集和验证集
            indices = torch.randperm(X.size(0)).tolist()
            train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=random_state)
            train_dataset = TensorDataset(X[train_indices], Y[train_indices])
            val_dataset = TensorDataset(X[val_indices], Y[val_indices])

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,pin_memory=True)

            return {
                'train_loader': train_loader,
                'val_loader': val_loader
            }
        else:
            test_dataset = TensorDataset(X, Y)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,pin_memory=True)
            return {
                'test_loader': test_loader,
                'df': df_Y_tokens
            }

    def _load_data(self):
        X_list, Y_list, tokens_list, dates_list = [], [], [], []
        for date in pd.date_range(self.start_date, self.end_date):
            pickle_file = f"{self.file_path}/{date.strftime('%Y-%m-%d')}.pickle"
            if os.path.exists(pickle_file):
                with open(pickle_file, 'rb') as handle:
                    daily_data = pickle.load(handle)
                    # 假设Y已经是一个Tensor
                    X_list.append(daily_data['X'])
                    Y_list.extend(daily_data['Y'][self.label].numpy())  # 如果Y是Tensor，转换为numpy数组
                    tokens_list.extend(daily_data['tokens'])
                    # 为这一天的每个样本添加相同的日期
                    dates_list.extend([date.strftime('%Y-%m-%d')] * len(daily_data['Y'][self.label]))

        # 将列表中的张量沿着第一个维度（样本数）拼接
        X = torch.cat(X_list, dim=0)

        # 创建包含Y值、tokens和日期的DataFrame
        df_Y = pd.DataFrame({
            'date': dates_list,
            'token': tokens_list,
            'Y': Y_list
        })

        if self.label.startswith('future_cat'):
            Y = torch.tensor(Y_list, dtype=torch.long)
        else:
            Y = torch.tensor(Y_list, dtype=torch.float32)


        return X, Y, tokens_list, df_Y
