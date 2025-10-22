#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析训练和测试结果的差异，检查过拟合问题
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Set font for better compatibility
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def load_metrics_data(base_path):
    """加载所有子文件夹的完整metrics数据"""
    data = {}
    
    for subfolder in os.listdir(base_path):
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
        else:
            test_recall5 = test_ndcg5 = test_recall10 = test_ndcg10 = None
        
        # 获取验证集最后几个值
        val_data = df[df['val/recall@5'].notna()]
        if len(val_data) > 0:
            last_val_recall5 = val_data['val/recall@5'].iloc[-1]
            last_val_ndcg5 = val_data['val/ndcg@5'].iloc[-1]
            last_val_recall10 = val_data['val/recall@10'].iloc[-1]
            last_val_ndcg10 = val_data['val/ndcg@10'].iloc[-1]
            
            # 计算验证集最后5个值的平均值
            last_5_val_recall5 = val_data['val/recall@5'].tail(5).mean()
            last_5_val_ndcg5 = val_data['val/ndcg@5'].tail(5).mean()
        else:
            last_val_recall5 = last_val_ndcg5 = last_val_recall10 = last_val_ndcg10 = None
            last_5_val_recall5 = last_5_val_ndcg5 = None
        
        print(f"验证集最后值 - Recall@5: {last_val_recall5:.4f}, NDCG@5: {last_val_ndcg5:.4f}")
        print(f"验证集最后5个平均值 - Recall@5: {last_5_val_recall5:.4f}, NDCG@5: {last_5_val_ndcg5:.4f}")
        
        if test_recall5 is not None:
            print(f"测试集结果 - Recall@5: {test_recall5:.4f}, NDCG@5: {test_ndcg5:.4f}")
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
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 定义颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (folder_name, df) in enumerate(data.items()):
        color = colors[i % len(colors)]
        
        # 获取验证集数据
        val_data = df[df['val/recall@5'].notna()]
        if len(val_data) == 0:
            continue
            
        x_val = np.arange(len(val_data))
        
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
    
    # 设置子图属性
    titles = ['Recall@5', 'NDCG@5', 'Recall@10', 'NDCG@10']
    for i, (ax, title) in enumerate(zip(axes.flat, titles)):
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig('/data/zhantianyu/Project/GRID/training_vs_test_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """主函数"""
    base_path = "/data/zhantianyu/Project/GRID/logs/train/runs/2025-09-11"
    
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
