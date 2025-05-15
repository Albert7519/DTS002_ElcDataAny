import optuna
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# 导入超参数优化相关模块
#!pip install optuna -q  # 安装Optuna库

# 注意：单元格运行后，你可能需要重启内核才能使用新安装的库
try:
    from mlp_hyperopt import (FlexibleMLPModel, train_model_with_early_stopping, 
                            evaluate_model, run_hyperparameter_optimization,
                            predict_with_best_model)
    from hpo_visualization import plot_hyperparameter_visualization, summarize_findings
    
    print("成功导入超参数优化相关模块！")
except Exception as e:
    print(f"导入模块时出错: {e}")
    print("如果你刚刚安装了Optuna，可能需要重启Jupyter内核。")

# 说明如何使用
print("\n===== 超参数优化使用指南 =====")
print("请从hpo_implementation.py文件中复制相关代码到新单元格中运行超参数优化过程。")
print("\n主要步骤:")
print("1. 数据准备与划分")
print("2. 运行超参数优化")
print("3. 评估优化后的模型性能")
print("4. 使用优化后的模型进行预测")
print("5. 可视化和总结结果")

print("\n要使用可视化和总结功能，请在运行完优化后执行:")
print("plot_hyperparameter_visualization(best_params, (best_mse, best_r2), (original_mse, original_r2))")
print("summarize_findings(best_params, (best_mse, best_r2), (original_mse, original_r2), original_mape, optimized_mape)")
