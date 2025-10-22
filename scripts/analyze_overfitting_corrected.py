#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析训练和测试结果的差异，检查过拟合问题（修正版）
使用正确的step值
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置全局字体和兼容性（可选，但推荐）
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14,
    'figure.titlesize': 22
})

def load_metrics_data(base_path):
    """加载所有子文件夹的完整metrics数据"""
    data = {}
    
    for subfolder in os.listdir(base_path):
        # if 'vatp' not in subfolder:
        #     continue
        subfolder_path = os.path.join(base_path, subfolder)
        if os.path.isdir(subfolder_path):
            csv_path = os.path.join(subfolder_path, 'csv', 'version_0', 'metrics.csv')
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    data[subfolder] = df
                    print(f"Successfully loaded {subfolder}: {len(df)} rows")
                except Exception as e:
                    print(f"Error loading {subfolder}: {e}")
    
    return data

def analyze_overfitting(data):
    """分析过拟合情况"""
    print("\n=== 过拟合分析 ===")
    
    for folder_name, df in data.items():
        print(f"\n--- {folder_name} ---")
        
        # 获取最后一行（测试结果）
        test_row = df[df['test/recall@5'].notna()].iloc[-1] if len(df[df['test/recall@5'].notna()]) > 0 else None
        if test_row is not None:
            test_recall5 = test_row['test/recall@5']
            test_ndcg5 = test_row['test/ndcg@5']
            test_recall10 = test_row['test/recall@10']
            test_ndcg10 = test_row['test/ndcg@10']
            test_step = test_row['step']
        else:
            test_recall5 = test_ndcg5 = test_recall10 = test_ndcg10 = test_step = None
        
        # 获取验证集最后几个值
        val_data = df[df['val/recall@5'].notna()]
        if len(val_data) > 0:
            last_val_recall5 = val_data['val/recall@5'].iloc[-1]
            last_val_ndcg5 = val_data['val/ndcg@5'].iloc[-1]
            last_val_recall10 = val_data['val/recall@10'].iloc[-1]
            last_val_ndcg10 = val_data['val/ndcg@10'].iloc[-1]
            last_val_step = val_data['step'].iloc[-1]
            
            # 计算验证集最后5个值的平均值
            last_5_val_recall5 = val_data['val/recall@5'].tail(5).mean()
            last_5_val_ndcg5 = val_data['val/ndcg@5'].tail(5).mean()
        else:
            last_val_recall5 = last_val_ndcg5 = last_val_recall10 = last_val_ndcg10 = last_val_step = None
            last_5_val_recall5 = last_5_val_ndcg5 = None
        
        print(f"验证集最后值 (步数: {last_val_step}) - Recall@5: {last_val_recall5:.4f}, NDCG@5: {last_val_ndcg5:.4f}")
        print(f"验证集最后5个平均值 - Recall@5: {last_5_val_recall5:.4f}, NDCG@5: {last_5_val_ndcg5:.4f}")
        
        if test_recall5 is not None:
            print(f"测试集结果 (步数: {test_step}) - Recall@5: {test_recall5:.4f}, NDCG@5: {test_ndcg5:.4f}")
            print(f"测试集结果 - Recall@10: {test_recall10:.4f}, NDCG@10: {test_ndcg10:.4f}")
            
            # 计算过拟合程度
            if last_val_recall5 is not None:
                overfitting_recall5 = last_val_recall5 - test_recall5
                overfitting_ndcg5 = last_val_ndcg5 - test_ndcg5
                print(f"过拟合程度 - Recall@5: {overfitting_recall5:.4f}, NDCG@5: {overfitting_ndcg5:.4f}")
        else:
            print("无测试集结果")

def plot_training_vs_test(data):
    """绘制训练和测试结果对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))  # 增大图形尺寸以适应大字体
    
    # 定义颜色
    colors = [
        '#1f77b4',  # 蓝色
        '#ff7f0e',  # 橙色
        '#2ca02c',  # 绿色
        '#d62728',  # 红色
        '#9467bd',  # 紫色
        '#8c564b',  # 棕色
        '#e377c2',  # 粉紫色
        '#7f7f7f',  # 灰色
        '#bcbd22',  # 黄绿色
        '#17becf',  # 青蓝色
        '#aec7e8',  # 浅蓝
        '#ffbb78'   # 浅橙
    ]
    
    for i, (folder_name, df) in enumerate(data.items()):
        color = colors[i % len(colors)]
        
        # 获取验证集数据
        val_data = df[df['val/recall@5'].notna()]
        if len(val_data) == 0:
            continue
            
        x_val = val_data['step'].values
        
        # 获取测试集数据
        test_data = df[df['test/recall@5'].notna()]
        if len(test_data) == 0:
            continue
            
        x_test = test_data['step'].values
        
        # Plot 1: Recall@5
        axes[0, 0].plot(x_val, val_data['val/recall@5'], 
                       color=color, linestyle='-', linewidth=2, 
                       label=f'{folder_name} (val)', alpha=0.8)
        axes[0, 0].scatter(x_test, test_data['test/recall@5'], 
                          color=color, s=100, marker='s', 
                          label=f'{folder_name} (test)', alpha=0.8)
        
        # Plot 2: NDCG@5
        axes[0, 1].plot(x_val, val_data['val/ndcg@5'], 
                       color=color, linestyle='-', linewidth=2, 
                       label=f'{folder_name} (val)', alpha=0.8)
        axes[0, 1].scatter(x_test, test_data['test/ndcg@5'], 
                          color=color, s=100, marker='s', 
                          label=f'{folder_name} (test)', alpha=0.8)
        
        # Plot 3: Recall@10
        axes[1, 0].plot(x_val, val_data['val/recall@10'], 
                       color=color, linestyle='-', linewidth=2, 
                       label=f'{folder_name} (val)', alpha=0.8)
        axes[1, 0].scatter(x_test, test_data['test/recall@10'], 
                          color=color, s=100, marker='s', 
                          label=f'{folder_name} (test)', alpha=0.8)
        
        # Plot 4: NDCG@10
        axes[1, 1].plot(x_val, val_data['val/ndcg@10'], 
                       color=color, linestyle='-', linewidth=2, 
                       label=f'{folder_name} (val)', alpha=0.8)
        axes[1, 1].scatter(x_test, test_data['test/ndcg@10'], 
                          color=color, s=100, marker='s', 
                          label=f'{folder_name} (test)', alpha=0.8)
    
    # 设置子图属性 —— 显式增大字体（覆盖全局设置，确保清晰）
    titles = ['Recall@5', 'NDCG@5', 'Recall@10', 'NDCG@10']
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontsize=25, fontweight='bold')
        ax.set_xlabel('Training Steps', fontsize=25, fontweight='bold')
        ax.set_ylabel('')  # 删除y轴标签
        ax.tick_params(axis='both', which='major', labelsize=28)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0.02)
        ax.set_xlim(left=1000)  # 设置x轴从1000步开始
        # 设置y轴刻度更稀疏
        ax.locator_params(axis='y', nbins=5)
        # 调整x轴标签位置，向下移动
        ax.xaxis.label.set_y(-0.15)
    
    # 仅在 Recall@10 子图上显示图例（axes[1, 0] 对应 Recall@10）
    axes[1, 0].legend(fontsize=18, loc='lower left')

    # 调整整体布局间距
    plt.tight_layout(pad=1.2, w_pad=2.0, h_pad=2.0)
    plt.savefig('/data/zhantianyu/Project/GRID/logs/image/strategy_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    base_path = "logs/train/runs/beauty/gr/exp3"
    
    print("Loading metrics data...")
    data = load_metrics_data(base_path)
    
    if not data:
        print("Error: No valid data files found")
        return
    
    # 分析过拟合情况
    analyze_overfitting(data)
    
    # 绘制训练vs测试对比图
    print("\nPlotting training vs test comparison...")
    plot_training_vs_test(data)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()