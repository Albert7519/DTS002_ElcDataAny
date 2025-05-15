import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


# 增强版MLP模型定义，支持可变层数和神经元数量
class FlexibleMLPModel(nn.Module):
    def __init__(
        self,
        input_dim=1,
        hidden_layers=None,
        output_dim=1,
        dropout_rate=0.2,
        activation_func=nn.ReLU,
    ):
        super(FlexibleMLPModel, self).__init__()

        if hidden_layers is None:
            hidden_layers = [64, 32]

        # 第一层 - 输入层到第一个隐藏层
        layers = [nn.Linear(input_dim, hidden_layers[0])]

        # 中间层
        for i in range(len(hidden_layers) - 1):
            layers.append(activation_func())
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))

        # 输出层
        layers.append(activation_func())
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_layers[-1], output_dim))

        self.model = nn.Sequential(*layers)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if len(x.shape) == 1:
            x = x.unsqueeze(1)  # 添加特征维度

        return self.model(x)


# 训练模型的函数（带有早停机制）
def train_model_with_early_stopping(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    optimizer,
    criterion,
    patience=30,
    max_epochs=500,
    verbose=False,
):
    best_val_loss = float("inf")
    no_improve_epochs = 0
    best_model_state = None

    for epoch in range(max_epochs):
        # 训练模式
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

        # 评估模式
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            best_model_state = model.state_dict().copy()
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            model.load_state_dict(best_model_state)
            break

        if verbose and epoch % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{max_epochs}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}"
            )

    if best_model_state:
        model.load_state_dict(best_model_state)

    return best_val_loss


# Optuna目标函数，用于超参数优化
def objective(trial, X_train, y_train, X_val, y_val, device, input_dim=1, output_dim=1):
    # 超参数搜索空间
    n_layers = trial.suggest_int("n_layers", 1, 5)
    hidden_layers = []

    for i in range(n_layers):
        # 每一层的神经元数量呈对数分布，从8到256
        n_units = trial.suggest_int(f"n_units_l{i}", 8, 256, log=True)
        hidden_layers.append(n_units)

    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)

    # 选择激活函数
    activation_name = trial.suggest_categorical(
        "activation", ["ReLU", "LeakyReLU", "ELU", "GELU"]
    )
    activation_functions = {
        "ReLU": nn.ReLU,
        "LeakyReLU": lambda: nn.LeakyReLU(0.1),
        "ELU": nn.ELU,
        "GELU": nn.GELU,
    }

    # 选择优化器
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])

    # 定义模型
    model = FlexibleMLPModel(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
        activation_func=activation_functions[activation_name],
    ).to(device)

    # 定义损失函数
    criterion = nn.MSELoss()

    # 定义优化器
    if optimizer_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )

    # 设置早停参数
    patience = trial.suggest_int("patience", 10, 100)
    max_epochs = 2000  # 最大迭代次数

    # 训练模型
    val_loss = train_model_with_early_stopping(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        optimizer,
        criterion,
        patience=patience,
        max_epochs=max_epochs,
    )

    return val_loss


# 评估函数：计算MSE和R²
def evaluate_model(model, X_test, y_test, scaler_y=None):
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        mse = nn.MSELoss()(y_pred, y_test).item()

        # 如果提供了scaler，转换回原始比例计算R²
        if scaler_y is not None:
            y_test_np = y_test.cpu().numpy().reshape(-1, 1)
            y_pred_np = y_pred.cpu().numpy().reshape(-1, 1)
            y_test_original = scaler_y.inverse_transform(y_test_np)
            y_pred_original = scaler_y.inverse_transform(y_pred_np)
            r2 = r2_score(y_test_original, y_pred_original)
        else:
            r2 = r2_score(y_test.cpu().numpy(), y_pred.cpu().numpy())

    return mse, r2


# 运行Optuna优化并返回最佳模型
def run_hyperparameter_optimization(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    scaler_y=None,
    device=None,
    n_trials=50,
    study_name="mlp_optimization",
):
    """
    运行超参数优化并返回最佳模型

    参数:
        X_train, y_train: 训练数据
        X_val, y_val: 验证数据
        X_test, y_test: 测试数据
        scaler_y: 用于反归一化输出的scaler对象
        device: PyTorch设备（cuda或cpu）
        n_trials: Optuna试验次数
        study_name: 研究名称

    返回:
        best_model: 最佳模型
        best_params: 最佳参数
        best_scores: 最佳得分 (MSE, R²)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建Optuna研究
    study = optuna.create_study(direction="minimize", study_name=study_name)

    # 定义目标函数的包装器
    objective_wrapper = lambda trial: objective(
        trial,
        X_train,
        y_train,
        X_val,
        y_val,
        device,
        X_train.shape[1],
        y_train.shape[1] if len(y_train.shape) > 1 else 1,
    )

    # 运行优化
    print(f"\n开始超参数优化，共{n_trials}次试验...")
    study.optimize(objective_wrapper, n_trials=n_trials, show_progress_bar=True)

    # 获取最佳参数
    best_params = study.best_params
    print("\n最佳超参数:")
    for param, value in best_params.items():
        print(f"- {param}: {value}")

    # 创建具有最佳参数的模型
    input_dim = X_train.shape[1] if len(X_train.shape) > 1 else 1
    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1

    # 构建最佳隐藏层架构
    n_layers = best_params["n_layers"]
    hidden_layers = [best_params[f"n_units_l{i}"] for i in range(n_layers)]

    # 选择激活函数
    activation_name = best_params["activation"]
    activation_functions = {
        "ReLU": nn.ReLU,
        "LeakyReLU": lambda: nn.LeakyReLU(0.1),
        "ELU": nn.ELU,
        "GELU": nn.GELU,
    }

    # 创建最佳模型
    best_model = FlexibleMLPModel(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        output_dim=output_dim,
        dropout_rate=best_params["dropout_rate"],
        activation_func=activation_functions[activation_name],
    ).to(device)

    # 定义最佳优化器
    if best_params["optimizer"] == "Adam":
        optimizer = optim.Adam(
            best_model.parameters(),
            lr=best_params["learning_rate"],
            weight_decay=best_params["weight_decay"],
        )
    elif best_params["optimizer"] == "RMSprop":
        optimizer = optim.RMSprop(
            best_model.parameters(),
            lr=best_params["learning_rate"],
            weight_decay=best_params["weight_decay"],
        )
    else:
        optimizer = optim.SGD(
            best_model.parameters(),
            lr=best_params["learning_rate"],
            weight_decay=best_params["weight_decay"],
            momentum=0.9,
        )

    # 使用全部训练集（训练+验证）重新训练最佳模型
    print("\n使用最佳参数在完整训练集上训练最终模型...")
    criterion = nn.MSELoss()

    # 合并训练集和验证集
    X_full_train = (
        torch.cat([X_train, X_val], dim=0)
        if isinstance(X_train, torch.Tensor)
        else torch.cat([torch.tensor(X_train), torch.tensor(X_val)], dim=0)
    )
    y_full_train = (
        torch.cat([y_train, y_val], dim=0)
        if isinstance(y_train, torch.Tensor)
        else torch.cat([torch.tensor(y_train), torch.tensor(y_val)], dim=0)
    )

    # 最终训练（简单训练，无需早停）
    best_model.train()
    max_epochs = 2000
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        y_pred = best_model(X_full_train)
        loss = criterion(y_pred, y_full_train)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch + 1}/{max_epochs}, Loss: {loss.item():.6f}")

    # 在测试集上评估最佳模型
    mse, r2 = evaluate_model(best_model, X_test, y_test, scaler_y)
    print(f"\n最佳模型评估结果 - MSE: {mse:.6f}, R²: {r2:.6f}")

    # 可视化优化历史
    plot_optimization_history(study)
    plot_param_importances(study)
    plot_intermediate_values(study)

    return best_model, best_params, (mse, r2), study


# 可视化函数
def plot_optimization_history(study):
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Optimization History", fontsize=14)
    plt.xlabel("Trial", fontsize=12)
    plt.ylabel("Objective Value (MSE)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_param_importances(study):
    plt.figure(figsize=(12, 8))
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title("Parameter Importances", fontsize=14)
    plt.xlabel("Importance", fontsize=12)
    plt.ylabel("Parameter", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_intermediate_values(study):
    plt.figure(figsize=(10, 6))
    optuna.visualization.matplotlib.plot_intermediate_values(study)
    plt.title("Intermediate Values", fontsize=14)
    plt.xlabel("Trial", fontsize=12)
    plt.ylabel("Objective Value (MSE)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


# 最佳模型预测函数
def predict_with_best_model(best_model, X_future, scaler_y):
    best_model.eval()
    with torch.no_grad():
        future_predictions_scaled = best_model(X_future)
        future_predictions = scaler_y.inverse_transform(
            future_predictions_scaled.cpu().numpy()
        )
    return future_predictions
