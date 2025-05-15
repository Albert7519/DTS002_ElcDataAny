import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 总结结果和超参数优化的好处
def plot_hyperparameter_visualization(best_params, best_scores, original_scores):
    """
    可视化超参数优化结果

    参数:
        best_params: 最佳超参数字典
        best_scores: 优化后的性能指标元组 (mse, r2)
        original_scores: 原始模型的性能指标元组 (mse, r2)
    """
    # 提取分数
    best_mse, best_r2 = best_scores
    original_mse, original_r2 = original_scores

    # 创建性能对比条形图
    plt.figure(figsize=(15, 6))

    # MSE对比
    plt.subplot(1, 2, 1)
    performance_data = pd.DataFrame(
        {"Model": ["原始增强版MLP", "超参数优化MLP"], "MSE": [original_mse, best_mse]}
    )
    sns.barplot(x="Model", y="MSE", data=performance_data, palette=["green", "red"])
    plt.title("均方误差 (MSE) 对比", fontsize=14)
    plt.ylabel("MSE (越低越好)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # R²对比
    plt.subplot(1, 2, 2)
    performance_data = pd.DataFrame(
        {"Model": ["原始增强版MLP", "超参数优化MLP"], "R²": [original_r2, best_r2]}
    )
    sns.barplot(x="Model", y="R²", data=performance_data, palette=["green", "red"])
    plt.title("决定系数 (R²) 对比", fontsize=14)
    plt.ylabel("R² (越高越好)", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()

    # 展示网络结构对比
    if "n_layers" in best_params:
        best_hidden_layers = [
            best_params.get(f"n_units_l{i}") for i in range(best_params.get("n_layers"))
        ]

        plt.figure(figsize=(12, 6))

        # 原始网络结构
        plt.subplot(1, 2, 1)
        original_layers = [1, 64, 128, 256, 128, 64, 1]  # 输入层、隐藏层、输出层
        plt.plot(
            range(len(original_layers)),
            original_layers,
            "go-",
            linewidth=2,
            markersize=10,
        )
        plt.title("原始MLP网络结构", fontsize=14)
        plt.xlabel("层索引", fontsize=12)
        plt.ylabel("神经元数量", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(
            range(len(original_layers)),
            ["输入"]
            + [f"隐藏{i + 1}" for i in range(len(original_layers) - 2)]
            + ["输出"],
        )
        plt.xticks(rotation=45)

        # 优化后的网络结构
        plt.subplot(1, 2, 2)
        optimized_layers = [1] + best_hidden_layers + [1]  # 输入层、隐藏层、输出层
        plt.plot(
            range(len(optimized_layers)),
            optimized_layers,
            "ro-",
            linewidth=2,
            markersize=10,
        )
        plt.title("优化后MLP网络结构", fontsize=14)
        plt.xlabel("层索引", fontsize=12)
        plt.ylabel("神经元数量", fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xticks(
            range(len(optimized_layers)),
            ["输入"]
            + [f"隐藏{i + 1}" for i in range(len(optimized_layers) - 2)]
            + ["输出"],
        )
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    # 超参数重要性分析表格
    print("\n超参数重要性分析:")
    params_importance = {
        "网络深度(层数)": "影响模型复杂度和表达能力，适当的深度能捕捉数据中的复杂模式。",
        "隐藏层宽度": "影响模型容量，过大容易过拟合，过小可能欠拟合。",
        "学习率": f"控制训练速度和稳定性，最佳值为{best_params.get('learning_rate', 'N/A')}。",
        "Dropout比例": f"控制正则化强度，防止过拟合，最佳值为{best_params.get('dropout_rate', 'N/A')}。",
        "优化器选择": f"不同优化器适合不同问题，此问题最佳选择是{best_params.get('optimizer', 'N/A')}。",
        "激活函数": f"影响非线性表达能力，最佳选择是{best_params.get('activation', 'N/A')}。",
        "权重衰减": f"控制L2正则化强度，最佳值为{best_params.get('weight_decay', 'N/A')}。",
        "早停耐心值": f"防止过拟合的重要策略，最佳值为{best_params.get('patience', 'N/A')}。",
    }

    for param, importance in params_importance.items():
        print(f"- {param}: {importance}")


# 总结超参数优化的收益和实际应用建议
def summarize_findings(
    best_params, best_scores, original_scores, original_mape, optimized_mape
):
    """
    总结超参数优化的结果和收益
    """
    print("\n====== 模型优化总结 ======")
    print("1. 性能提升:")
    best_mse, best_r2 = best_scores
    original_mse, original_r2 = original_scores

    mse_improvement = (original_mse - best_mse) / original_mse * 100
    r2_improvement = (
        best_r2 - original_r2
        if original_r2 < 0
        else (best_r2 - original_r2) / original_r2 * 100
    )
    mape_improvement = (
        (original_mape - optimized_mape) / original_mape * 100
        if original_mape != 0
        else 0
    )

    print(f"   - MSE减少: {mse_improvement:.2f}%")
    print(f"   - R²: 从 {original_r2:.4f} 提高到 {best_r2:.4f}")
    print(f"   - 实际预测误差(MAPE)减少: {mape_improvement:.2f}%")

    if "n_layers" in best_params:
        best_hidden_layers = [
            best_params.get(f"n_units_l{i}") for i in range(best_params.get("n_layers"))
        ]
        print("\n2. 最优网络结构:")
        print(
            f"   - 隐藏层: {best_hidden_layers} (对比原始模型: [64, 128, 256, 128, 64])"
        )
        print(f"   - 总层数: {len(best_hidden_layers) + 2} (包括输入和输出层)")

    print("\n3. 最佳训练策略:")
    print(f"   - 优化器: {best_params.get('optimizer', 'N/A')}")
    print(f"   - 学习率: {best_params.get('learning_rate', 'N/A')}")
    print(f"   - Dropout率: {best_params.get('dropout_rate', 'N/A')}")
    print(f"   - 激活函数: {best_params.get('activation', 'N/A')}")

    print("\n4. 关键发现:")

    # 根据优化结果生成关键发现
    if len(best_hidden_layers) < 5:
        print("   - 较浅的网络结构比原始5层网络表现更好，说明原模型可能过于复杂")
    else:
        print("   - 深层网络结构对此问题至关重要")

    if best_params.get("dropout_rate", 0) < 0.2:
        print("   - 较低的Dropout率表明模型不需要强正则化")
    else:
        print("   - 较高的Dropout率对防止过拟合很重要")

    if best_params.get("optimizer", "") == "Adam":
        print("   - Adam优化器适合此类时间序列预测问题")

    print("\n5. 应用建议:")
    print("   - 建议使用优化后的模型参数进行未来电力消费预测")
    print("   - 定期重新训练模型以适应新数据模式")
    print("   - 考虑增加更多相关特征（如GDP、人口数据等）进一步提升预测准确性")
    print("   - 建议每6-12个月进行一次超参数重新优化，以适应潜在的数据分布变化")


# 使用上述函数可视化并总结超参数优化结果
# plot_hyperparameter_visualization(best_params, (best_mse, best_r2), (original_mse, original_r2))
# summarize_findings(best_params, (best_mse, best_r2), (original_mse, original_r2), original_mape, optimized_mape)
