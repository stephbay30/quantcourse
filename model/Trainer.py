from tqdm import tqdm
import torch
import numpy as np
from model.CustomLoss import RankCorrelationLoss
import copy
import matplotlib.pyplot as plt
import gc

class ModelTrainer:
    def __init__(self, model, train_loader, validation_loader, optimizer, train_criterion,
                 val_criterion, label,scheduler=None, patience=10,):
        self.label = label
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.optimizer = optimizer
        self.train_criterion = train_criterion
        self.val_criterion = val_criterion
        self.scheduler = scheduler
        self.patience = patience
        self.train_losses = []
        self.val_losses = []
        # self.IC_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.ic_cri = RankCorrelationLoss()
        self.best_epoch = None  # 添加一个属性来保存最佳epoch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self, num_epochs):
        no_improve_count = 0

        for epoch in range(num_epochs):
            self.model.train()
            total_train_loss = 0.0

            for batch_X, batch_Y in tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False):
                batch_X = batch_X.to(self.device)
                batch_Y = batch_Y.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                output = torch.squeeze(output)
                loss = self.train_criterion(output, batch_Y)
                loss.backward()

                # 打印梯度范数
                # for name, param in self.model.named_parameters():
                #     if param.grad is not None:
                #         grad_norm = torch.norm(param.grad).item()
                #         print(f"Gradient norm of {name}: {grad_norm}")

                self.optimizer.step()
                total_train_loss += loss.item()
                # del batch_X, batch_Y
                # gc.collect()
                # torch.cuda.empty_cache()


            average_train_loss = total_train_loss / len(self.train_loader)
            self.train_losses.append(average_train_loss)

            val_loss = self.evaluate()
            self.val_losses.append(val_loss)
            # self.IC_losses.append(val_IC_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.best_epoch = epoch  # 更新最佳epoch
                no_improve_count = 0
            else:
                no_improve_count += 1

            if self.scheduler:
                self.scheduler.step()

            if no_improve_count >= self.patience:
                # print(f"Early stopping triggered after {epoch + 1} epochs.")
                break



    def evaluate(self):
        self.model.eval()
        total_val_loss = 0.0
        # total_IC_loss = 0.0

        with torch.no_grad():
            for val_batch_X, val_batch_Y in self.validation_loader:
                val_batch_X = val_batch_X.to(self.device)
                val_batch_Y = val_batch_Y.to(self.device)
                val_output = self.model(val_batch_X)
                val_output = torch.squeeze(val_output)



                val_loss = self.val_criterion(val_output, val_batch_Y)
                # IC_loss = self.ic_cri(val_output, val_batch_Y)
                total_val_loss += val_loss.item()
                # total_IC_loss += IC_loss.item()
                del val_batch_X, val_batch_Y
                gc.collect()

        average_val_loss = total_val_loss / len(self.validation_loader)
        return average_val_loss

    def predict(self, test_loader=None):
        # epoch = self.best_epoch  # 如果没有指定epoch，使用最佳epoch
        model_state = self.best_model_state
        val_loss = self.best_val_loss


        # print(f"Predicting using model  best model with validation loss {val_loss:.4f}")
        print(f"Predicting using model  best model ")

        self.model.load_state_dict(model_state)
        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_Y = batch_Y.to(self.device)
                output = self.model(batch_X)
                output = torch.squeeze(output)
                predictions.append(output.cpu().numpy())
                targets.append(batch_Y.cpu().numpy())


        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)

        # 计算预测和真实值之间的损失
        predictions_tensor = torch.tensor(predictions, dtype=torch.float32)
        if self.label.startswith('future_cat'):
            targets_tensor = torch.tensor(targets, dtype=torch.long)  # 确保是长整型
        else:
            targets_tensor = torch.tensor(targets, dtype=torch.float32)  # 确保是长整型
        prediction_loss = self.val_criterion(predictions_tensor, targets_tensor).item()

        return predictions, prediction_loss


    def plot_losses(self, show_train_loss=True, show_val_loss=True,save_fig=True, save_fig_path=None):
        if not show_train_loss and not show_val_loss:
            print("No data to plot.")
            return

        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(10, 6))

        if show_train_loss:
            plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')

        if show_val_loss:
            plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')

        plt.title('Training and Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        if save_fig:
            plt.savefig(save_fig_path)

