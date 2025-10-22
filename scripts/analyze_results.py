#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

def analyze_rkmeans_results():
    """分析rkmeans original和rastp模式的test性能结果"""
    
    # 定义实验路径
    base_path = "/data/zhantianyu/Project/GRID/logs/train/runs/toys/gr"
    
    # 实验配置
    experiments = {
        'original': ['seed_42', 'seed_1', 'seed_999', 'seed_1024', 'seed_2025'],
        'rastp': ['seed_42', 'seed_1', 'seed_999', 'seed_1024', 'seed_2025']
    }
    
    # 存储结果
    results = {'original': [], 'rastp': []}
    
    # 读取每个实验的test结果
    for mode in ['original', 'rastp']:
        for seed in experiments[mode]:
            exp_name = f"rkmeans_{mode}_{seed}"
            csv_path = os.path.join(base_path, exp_name, "csv/version_0/metrics.csv")
            
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    # 获取最后一行test结果
                    test_row = df[df['test/loss'].notna()].iloc[-1]
                    
                    test_metrics = {
                        'seed': seed,
                        'loss': test_row['test/loss'],
                        'ndcg@10': test_row['test/ndcg@10'],
                        'ndcg@5': test_row['test/ndcg@5'],
                        'recall@10': test_row['test/recall@10'],
                        'recall@5': test_row['test/recall@5']
                    }
                    results[mode].append(test_metrics)
                    print(f"✓ 读取 {exp_name}: loss={test_metrics['loss']:.4f}, ndcg@10={test_metrics['ndcg@10']:.4f}")
                except Exception as e:
                    print(f"✗ 读取 {exp_name} 失败: {e}")
            else:
                print(f"✗ 文件不存在: {csv_path}")
    
    # 统计分析
    print("\n" + "="*80)
    print("RKMEANS 实验结果统计分析")
    print("="*80)
    
    for mode in ['original', 'rastp']:
        if not results[mode]:
            print(f"\n{mode.upper()} 模式: 无有效数据")
            continue
            
        print(f"\n{mode.upper()} 模式结果:")
        print("-" * 50)
        
        # 转换为DataFrame便于计算
        df_mode = pd.DataFrame(results[mode])
        
        # 计算均值和标准差
        metrics = ['loss', 'ndcg@10', 'ndcg@5', 'recall@10', 'recall@5']
        
        for metric in metrics:
            values = df_mode[metric].values
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)  # 样本标准差
            
            print(f"{metric:>10}: {mean_val:.4f} ± {std_val:.4f}")
            
            # 显示各seed的详细数值
            print(f"            详细: ", end="")
            for i, (seed, val) in enumerate(zip(df_mode['seed'], values)):
                print(f"{seed}={val:.4f}", end="  ")
            print()
    
    # 对比分析
    print("\n" + "="*80)
    print("对比分析 (rastp vs ORIGINAL)")
    print("="*80)
    
    if results['original'] and results['rastp']:
        df_original = pd.DataFrame(results['original'])
        df_rastp = pd.DataFrame(results['rastp'])
        
        print(f"{'指标':<12} {'ORIGINAL':<20} {'rastp':<20} {'提升':<15}")
        print("-" * 70)
        
        for metric in ['loss', 'ndcg@10', 'ndcg@5', 'recall@10', 'recall@5']:
            orig_mean = np.mean(df_original[metric].values)
            rastp_mean = np.mean(df_rastp[metric].values)
            
            if metric == 'loss':
                # loss越小越好
                improvement = ((orig_mean - rastp_mean) / orig_mean) * 100
                direction = "↓" if improvement > 0 else "↑"
            else:
                # 其他指标越大越好
                improvement = ((rastp_mean - orig_mean) / orig_mean) * 100
                direction = "↑" if improvement > 0 else "↓"
            
            print(f"{metric:<12} {orig_mean:.4f} ± {np.std(df_original[metric].values, ddof=1):.4f}  {rastp_mean:.4f} ± {np.std(df_rastp[metric].values, ddof=1):.4f}  {direction}{abs(improvement):.2f}%")

if __name__ == "__main__":
    analyze_rkmeans_results()
