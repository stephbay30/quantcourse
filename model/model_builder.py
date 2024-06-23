import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
class SimpleCNN4Cat(nn.Module):
    def __init__(self, input_size, dropout_rate, num_channels=[32, 64, 128],activation_function='elu',linear_units=128):
        super(SimpleCNN4Cat, self).__init__()
        self.input_size = input_size
        self.input_channels, self.input_height, self.input_width = input_size

        layers = []
        in_channels = self.input_channels
        kernel_sizes = [3 for _ in range(len(num_channels))]
        activations = [ activation_function for _ in range(len(num_channels))]
        # 动态创建卷积层
        for out_channels, kernel_size, activation in zip(num_channels, kernel_sizes, activations):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Calculate the size of the flattened features after all convolutional layers
        self.flattened_size = self.calculate_flattened_size()

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.flattened_size, out_features=linear_units),
            nn.ELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(in_features=linear_units, out_features=3)  # Assuming 3 output classes
        )

        # Softmax layer for the output
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = self.log_softmax(x)
        return x

    def calculate_flattened_size(self):
        temp_tensor = torch.zeros((1, *self.input_size))
        temp_tensor = self.conv_layers(temp_tensor)
        return int(np.prod(temp_tensor.size()))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
class SimpleCNN4Ret(nn.Module):
    def __init__(self, input_size, dropout_rate, num_channels=[32, 64, 128],activation_function='elu',linear_units=128):
        super(SimpleCNN4Ret, self).__init__()
        self.input_size = input_size
        self.input_channels, self.input_height, self.input_width = input_size

        layers = []
        in_channels = self.input_channels
        kernel_sizes = [3 for _ in range(len(num_channels))]
        activations = [ activation_function for _ in range(len(num_channels))]

        # Create convolutional layers based on the num_channels list
        for out_channels, kernel_size, activation in zip(num_channels, kernel_sizes, activations):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Calculate the size of the flattened features after all convolutional layers
        self.flattened_size = self.calculate_flattened_size()

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.flattened_size, out_features=linear_units),
            nn.ELU(),
            nn.Dropout(dropout_rate),

            nn.Linear(in_features=linear_units, out_features=1)
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def calculate_flattened_size(self):
        # Temporarily create a tensor of zeros with the input size
        temp_tensor = torch.zeros((1, *self.input_size))
        # Apply convolutional layers (without the fully connected layers)
        temp_tensor = self.conv_layers(temp_tensor)
        # Return the resulting flattened size
        return int(np.prod(temp_tensor.size()))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)



class SimpleCNN4Ret2d(nn.Module):
    def __init__(self, input_size, dropout_rate, num_channels=[32, 64, 128], activation_function='elu', linear_units=128):
        super(SimpleCNN4Ret2d, self).__init__()
        self.input_size = input_size
        self.input_length, self.input_features = input_size

        layers = []
        in_channels = 1  # For 2D data, the input channel is 1
        kernel_sizes = [3 for _ in range(len(num_channels))]
        activations = [activation_function for _ in range(len(num_channels))]

        # Create convolutional layers based on the num_channels list
        for out_channels, kernel_size, activation in zip(num_channels, kernel_sizes, activations):
            layers.append(nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            # layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(dropout_rate))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Calculate the size of the flattened features after all convolutional layers
        self.flattened_size = self.calculate_flattened_size()

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.flattened_size, out_features=linear_units),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_features=linear_units, out_features=1)
        )
        self._initialize_weights()

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for 2D convolution
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

    def calculate_flattened_size(self):
        # Temporarily create a tensor of zeros with the input size
        temp_tensor = torch.zeros((1, 1, self.input_length, self.input_features))
        # Apply convolutional layers (without the fully connected layers)
        temp_tensor = self.conv_layers(temp_tensor)
        # Return the resulting flattened size
        return int(np.prod(temp_tensor.size()))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

class GradualWarmupScheduler(LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        super().step(epoch)