#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建详细的对比表格
"""

import pandas as pd
import numpy as np

def create_comparison_table():
    """创建对比表格"""
    
    # 数据
    data = {
        'Configuration': ['max_lbl_3', 'max_lbl_2', 'original_lbl', 'max_lbl_1'],
        'Val_Recall@5_Final': [0.0535, 0.0530, 0.0575, 0.0487],
        'Val_Recall@5_Avg5': [0.0529, 0.0526, 0.0567, 0.0493],
        'Test_Recall@5': [0.0385, 0.0400, 0.0427, 0.0374],
        'Val_NDCG@5_Final': [0.0358, 0.0351, 0.0386, 0.0327],
        'Val_NDCG@5_Avg5': [0.0355, 0.0348, 0.0379, 0.0324],
        'Test_NDCG@5': [0.0256, 0.0257, 0.0285, 0.0240],
        'Overfitting_Recall@5': [0.0150, 0.0130, 0.0148, 0.0112],
        'Overfitting_NDCG@5': [0.0102, 0.0094, 0.0101, 0.0087],
        'Training_Steps': [60, 43, 52, 54]
    }
    
    df = pd.DataFrame(data)
    
    # 计算排名
    df['Val_Rank'] = df['Val_Recall@5_Final'].rank(ascending=False)
    df['Test_Rank'] = df['Test_Recall@5'].rank(ascending=False)
    df['Overfitting_Rank'] = df['Overfitting_Recall@5'].rank(ascending=True)  # 越小越好
    
    print("=== 详细对比分析 ===")
    print(df.to_string(index=False, float_format='%.4f'))
    
    print("\n=== 关键发现 ===")
    print("1. 验证集性能排名 (Recall@5):")
    val_ranking = df.sort_values('Val_Recall@5_Final', ascending=False)
    for i, (_, row) in enumerate(val_ranking.iterrows(), 1):
        print(f"   {i}. {row['Configuration']}: {row['Val_Recall@5_Final']:.4f}")
    
    print("\n2. 测试集性能排名 (Recall@5):")
    test_ranking = df.sort_values('Test_Recall@5', ascending=False)
    for i, (_, row) in enumerate(test_ranking.iterrows(), 1):
        print(f"   {i}. {row['Configuration']}: {row['Test_Recall@5']:.4f}")
    
    print("\n3. 过拟合程度排名 (Recall@5, 越小越好):")
    overfitting_ranking = df.sort_values('Overfitting_Recall@5', ascending=True)
    for i, (_, row) in enumerate(overfitting_ranking.iterrows(), 1):
        print(f"   {i}. {row['Configuration']}: {row['Overfitting_Recall@5']:.4f}")
    
    print("\n4. 为什么max_lbl_3测试结果比max_lbl_2差？")
    lbl3_row = df[df['Configuration'] == 'max_lbl_3'].iloc[0]
    lbl2_row = df[df['Configuration'] == 'max_lbl_2'].iloc[0]
    
    print(f"   - max_lbl_3: 验证集 {lbl3_row['Val_Recall@5_Final']:.4f} → 测试集 {lbl3_row['Test_Recall@5']:.4f} (下降 {lbl3_row['Overfitting_Recall@5']:.4f})")
    print(f"   - max_lbl_2: 验证集 {lbl2_row['Val_Recall@5_Final']:.4f} → 测试集 {lbl2_row['Test_Recall@5']:.4f} (下降 {lbl2_row['Overfitting_Recall@5']:.4f})")
    print(f"   - max_lbl_3的过拟合程度更高: {lbl3_row['Overfitting_Recall@5']:.4f} vs {lbl2_row['Overfitting_Recall@5']:.4f}")
    print(f"   - 训练步数更多: {lbl3_row['Training_Steps']} vs {lbl2_row['Training_Steps']} 步")
    
    return df

if __name__ == "__main__":
    create_comparison_table()
