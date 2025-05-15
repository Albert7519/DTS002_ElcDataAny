import optuna
from mlp_hyperopt import (
    FlexibleMLPModel,
    train_model_with_early_stopping,
    objective,
    evaluate_model,
    run_hyperparameter_optimization,
    plot_optimization_history,
    plot_param_importances,
    plot_intermediate_values,
    predict_with_best_model,
)

# 准备数据，先进行数据划分为训练、验证和测试集
print("2.4 超参数优化 - MLP模型性能提升")

# 确保数据正确排序
ordered_data = time_series_data.sort_values(by="Year")
X_ordered = ordered_data[["Year"]].values
y_ordered = ordered_data["Net Consumption"].values

# 规范化数据
X_scaled = scaler_X.fit_transform(X_ordered.reshape(-1, 1))
y_scaled = scaler_y.fit_transform(y_ordered.reshape(-1, 1))

# 将数据划分为训练集(70%)、验证集(15%)和测试集(15%)
train_size = int(len(X_scaled) * 0.7)
val_size = int(len(X_scaled) * 0.15)
test_size = len(X_scaled) - train_size - val_size

X_train_hpo = X_scaled[:train_size]
X_val_hpo = X_scaled[train_size : train_size + val_size]
X_test_hpo = X_scaled[train_size + val_size :]

y_train_hpo = y_scaled[:train_size]
y_val_hpo = y_scaled[train_size : train_size + val_size]
y_test_hpo = y_scaled[train_size + val_size :]

# 转换为PyTorch张量
X_train_tensor_hpo = torch.FloatTensor(X_train_hpo).to(device)
y_train_tensor_hpo = torch.FloatTensor(y_train_hpo).to(device)
X_val_tensor_hpo = torch.FloatTensor(X_val_hpo).to(device)
y_val_tensor_hpo = torch.FloatTensor(y_val_hpo).to(device)
X_test_tensor_hpo = torch.FloatTensor(X_test_hpo).to(device)
y_test_tensor_hpo = torch.FloatTensor(y_test_hpo).to(device)

print(f"训练集大小: {X_train_tensor_hpo.shape}, {y_train_tensor_hpo.shape}")
print(f"验证集大小: {X_val_tensor_hpo.shape}, {y_val_tensor_hpo.shape}")
print(f"测试集大小: {X_test_tensor_hpo.shape}, {y_test_tensor_hpo.shape}")

# 运行超参数优化
n_trials = 50  # 试验次数，可以根据你的时间和计算资源调整
best_model, best_params, best_scores, study = run_hyperparameter_optimization(
    X_train_tensor_hpo,
    y_train_tensor_hpo,
    X_val_tensor_hpo,
    y_val_tensor_hpo,
    X_test_tensor_hpo,
    y_test_tensor_hpo,
    scaler_y=scaler_y,
    device=device,
    n_trials=n_trials,
    study_name="mlp_electricity_forecasting",
)

# 获取最佳MSE和R²
best_mse, best_r2 = best_scores
print(f"\n超参数优化后的模型评估结果:")
print(f"- MSE: {best_mse:.6f}")
print(f"- R²: {best_r2:.6f}")

# 原始模型评估结果
print(f"\n原始增强版MLP模型评估结果:")
original_mse, original_r2 = evaluate_model(
    mlp_model, X_test_tensor_hpo, y_test_tensor_hpo, scaler_y
)
print(f"- MSE: {original_mse:.6f}")
print(f"- R²: {original_r2:.6f}")

# 性能提升百分比
mse_improvement = (original_mse - best_mse) / original_mse * 100
r2_improvement = (
    (best_r2 - original_r2) * 100
    if original_r2 < 0
    else (best_r2 - original_r2) / original_r2 * 100
)
print(f"\n性能提升:")
print(f"- MSE减少了 {mse_improvement:.2f}%")
print(f"- R² {'提高了' if r2_improvement > 0 else '降低了'} {abs(r2_improvement):.2f}%")

# 使用最佳模型进行预测（2022-2024年）
future_years = np.array([[2022], [2023], [2024]])
future_years_scaled = scaler_X.transform(future_years)
future_years_tensor = torch.FloatTensor(future_years_scaled).to(device)

# 使用最佳模型进行预测
print("\n使用优化后的模型预测2022-2024年的电力消费:")
optimized_predictions = predict_with_best_model(
    best_model, future_years_tensor, scaler_y
)

# 创建包含原始模型和优化模型预测结果的DataFrame
comparison_df = pd.DataFrame(
    {
        "Year": future_years.flatten(),
        "Optimized Model Prediction (GWh)": optimized_predictions.flatten(),
        "Original Model Prediction (GWh)": future_predictions.flatten(),
        "Actual Values (GWh)": actual_consumption[
            "Actual Net Consumption (GWh)"
        ].values,
    }
)

print("\n预测结果对比:")
print(comparison_df)

# 计算预测误差
comparison_df["Optimized Model Error (%)"] = (
    abs(
        comparison_df["Optimized Model Prediction (GWh)"]
        - comparison_df["Actual Values (GWh)"]
    )
    / comparison_df["Actual Values (GWh)"]
    * 100
)
comparison_df["Original Model Error (%)"] = (
    abs(
        comparison_df["Original Model Prediction (GWh)"]
        - comparison_df["Actual Values (GWh)"]
    )
    / comparison_df["Actual Values (GWh)"]
    * 100
)

# 平均误差
optimized_mape = comparison_df["Optimized Model Error (%)"].mean()
original_mape = comparison_df["Original Model Error (%)"].mean()

print(f"\n平均绝对百分比误差 (MAPE):")
print(f"- 优化后的模型: {optimized_mape:.2f}%")
print(f"- 原始增强版MLP模型: {original_mape:.2f}%")
print(f"- 误差减少: {(original_mape - optimized_mape):.2f}%")

# 可视化对比
plt.figure(figsize=(15, 8))

# 历史数据
plt.plot(
    years,
    values,
    marker="o",
    linestyle="-",
    color="blue",
    label="历史数据 (1980-2021)",
    alpha=0.6,
)

# 实际值
plt.plot(
    comparison_df["Year"],
    comparison_df["Actual Values (GWh)"],
    marker="*",
    markersize=12,
    linestyle="-",
    color="black",
    linewidth=2,
    label="实际值 (2022-2024)",
)

# 原始模型预测
plt.plot(
    comparison_df["Year"],
    comparison_df["Original Model Prediction (GWh)"],
    marker="^",
    linestyle="--",
    color="green",
    label="原始增强版MLP预测",
)

# 优化模型预测
plt.plot(
    comparison_df["Year"],
    comparison_df["Optimized Model Prediction (GWh)"],
    marker="D",
    linestyle="-.",
    color="red",
    linewidth=2,
    label="优化后的MLP预测",
)

plt.title(f"电力消费预测对比 - 超参数优化前后", fontsize=16)
plt.xlabel("年份", fontsize=14)
plt.ylabel("电力消费 (GWh)", fontsize=14)
plt.grid(True, linestyle="--", alpha=0.7)
plt.legend(loc="best")
plt.xticks(np.append(years[::5], comparison_df["Year"]), rotation=45)
plt.tight_layout()
plt.show()

# 可视化预测误差对比
plt.figure(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(comparison_df))

plt.bar(
    index,
    comparison_df["Original Model Error (%)"],
    bar_width,
    color="green",
    alpha=0.7,
    label="原始增强版MLP模型",
)
plt.bar(
    index + bar_width,
    comparison_df["Optimized Model Error (%)"],
    bar_width,
    color="red",
    alpha=0.7,
    label="优化后的MLP模型",
)

plt.title("预测误差对比 (原始 vs 优化后的模型)", fontsize=16)
plt.xlabel("预测年份", fontsize=14)
plt.ylabel("百分比误差 (%)", fontsize=14)
plt.xticks(index + bar_width / 2, comparison_df["Year"])
plt.legend(loc="best")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()

# 总结优化结果和最佳参数
print("\n====== 超参数优化总结 ======")
print("\n最佳模型超参数:")
for param, value in best_params.items():
    print(f"- {param}: {value}")

if best_params.get("n_layers"):
    hidden_layers = [
        best_params.get(f"n_units_l{i}") for i in range(best_params.get("n_layers"))
    ]
    print(f"\n最佳隐藏层架构: {hidden_layers}")

print(f"\n最佳优化器: {best_params.get('optimizer', 'N/A')}")
print(f"最佳学习率: {best_params.get('learning_rate', 'N/A')}")
print(f"最佳Dropout率: {best_params.get('dropout_rate', 'N/A')}")
print(f"最佳权重衰减: {best_params.get('weight_decay', 'N/A')}")
print(f"最佳激活函数: {best_params.get('activation', 'N/A')}")

print("\n-------- 超参数优化结果 --------")
print(f"原始模型 MSE: {original_mse:.6f}, R²: {original_r2:.6f}")
print(f"优化后模型 MSE: {best_mse:.6f}, R²: {best_r2:.6f}")
print(f"实际预测 MAPE - 原始: {original_mape:.2f}%, 优化后: {optimized_mape:.2f}%")
print("-----------------------------")
