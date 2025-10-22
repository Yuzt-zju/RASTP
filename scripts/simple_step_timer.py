#!/usr/bin/env python3
"""
简单的Step时间计算器

通过监控验证时间来估算每个step的平均时间
每1600个step进行一次验证，统计10次验证的时间，然后除以16000得到平均step时间
"""

import time
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict


class SimpleStepTimer:
    """简单的step时间计算器"""
    
    def __init__(self, save_dir: str = "logs/step_timing"):
        """
        初始化计时器
        
        Args:
            save_dir: 保存时间数据的目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 验证时间记录
        self.validation_times: List[float] = []
        self.validation_steps: List[int] = []
        
        # 当前验证开始时间
        self.val_start_time = None
        
        # 统计信息
        self.total_steps = 16000
        self.steps_per_validation = 1600
        self.expected_validations = 10
        
        print(f"简单Step计时器初始化完成")
        print(f"目标: 记录{self.expected_validations}次验证，计算{self.total_steps}个step的平均时间")
        print(f"每次验证间隔: {self.steps_per_validation} steps")
        print(f"数据保存目录: {self.save_dir}")
    
    def start_validation(self, step: int):
        """
        开始验证计时
        
        Args:
            step: 当前step数
        """
        self.val_start_time = time.time()
        print(f"开始验证 (Step {step})")
    
    def end_validation(self, step: int):
        """
        结束验证计时
        
        Args:
            step: 当前step数
        """
        if self.val_start_time is None:
            print("警告: 没有开始的验证时间")
            return
        
        validation_time = time.time() - self.val_start_time
        
        # 记录验证时间
        self.validation_times.append(validation_time)
        self.validation_steps.append(step)
        
        print(f"验证完成 (Step {step}): {validation_time:.2f}秒")
        
        # 计算当前统计信息
        self._print_current_stats()
        
        # 重置
        self.val_start_time = None
        
        # 如果完成了10次验证，计算最终结果
        if len(self.validation_times) >= self.expected_validations:
            self._calculate_final_results()
    
    def _print_current_stats(self):
        """打印当前统计信息"""
        num_validations = len(self.validation_times)
        
        if num_validations == 0:
            return
        
        # 计算当前的平均验证时间
        avg_val_time = sum(self.validation_times) / num_validations
        
        # 估算每个step的时间
        estimated_step_time = avg_val_time / self.steps_per_validation
        
        # 估算剩余时间
        remaining_validations = self.expected_validations - num_validations
        estimated_remaining_time = remaining_validations * avg_val_time
        
        print(f"当前进度: {num_validations}/{self.expected_validations} 次验证")
        print(f"平均验证时间: {avg_val_time:.2f}秒")
        print(f"估算每step时间: {estimated_step_time:.4f}秒")
        
        if remaining_validations > 0:
            print(f"预计剩余时间: {estimated_remaining_time:.2f}秒")
    
    def _calculate_final_results(self):
        """计算最终结果"""
        print("\n" + "="*50)
        print("最终计算结果")
        print("="*50)
        
        # 计算统计信息
        total_val_time = sum(self.validation_times)
        avg_val_time = total_val_time / len(self.validation_times)
        min_val_time = min(self.validation_times)
        max_val_time = max(self.validation_times)
        
        # 计算每个step的平均时间
        avg_step_time = total_val_time / self.total_steps
        
        # 计算总训练时间估算
        total_training_time = avg_step_time * self.total_steps
        
        print(f"验证次数: {len(self.validation_times)}")
        print(f"总验证时间: {total_val_time:.2f}秒")
        print(f"平均验证时间: {avg_val_time:.2f}秒")
        print(f"最短验证时间: {min_val_time:.2f}秒")
        print(f"最长验证时间: {max_val_time:.2f}秒")
        print(f"")
        print(f"🎯 每个step平均时间: {avg_step_time:.4f}秒")
        print(f"🎯 总训练时间估算: {total_training_time:.2f}秒 ({total_training_time/60:.1f}分钟)")
        
        # 保存结果
        self._save_results({
            "validation_times": self.validation_times,
            "validation_steps": self.validation_steps,
            "total_validation_time": total_val_time,
            "average_validation_time": avg_val_time,
            "min_validation_time": min_val_time,
            "max_validation_time": max_val_time,
            "average_step_time": avg_step_time,
            "estimated_total_training_time": total_training_time,
            "total_steps": self.total_steps,
            "steps_per_validation": self.steps_per_validation,
            "calculation_time": datetime.now().isoformat()
        })
    
    def _save_results(self, results: Dict):
        """保存结果到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存JSON格式
        json_file = self.save_dir / f"step_timing_results_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存CSV格式（详细数据）
        csv_file = self.save_dir / f"step_timing_details_{timestamp}.csv"
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['validation_number', 'step', 'validation_time', 'avg_step_time'])
            
            for i, (step, val_time) in enumerate(zip(self.validation_steps, self.validation_times)):
                step_time = val_time / self.steps_per_validation
                writer.writerow([i+1, step, val_time, step_time])
        
        # 保存总结文件
        summary_file = self.save_dir / f"step_timing_summary_{timestamp}.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("Step时间统计结果\n")
            f.write("="*30 + "\n\n")
            f.write(f"验证次数: {len(self.validation_times)}\n")
            f.write(f"总验证时间: {results['total_validation_time']:.2f}秒\n")
            f.write(f"平均验证时间: {results['average_validation_time']:.2f}秒\n")
            f.write(f"最短验证时间: {results['min_validation_time']:.2f}秒\n")
            f.write(f"最长验证时间: {results['max_validation_time']:.2f}秒\n")
            f.write(f"\n🎯 每个step平均时间: {results['average_step_time']:.4f}秒\n")
            f.write(f"🎯 总训练时间估算: {results['estimated_total_training_time']:.2f}秒\n")
            f.write(f"🎯 总训练时间估算: {results['estimated_total_training_time']/60:.1f}分钟\n")
            f.write(f"\n计算时间: {results['calculation_time']}\n")
        
        print(f"\n结果已保存到:")
        print(f"  JSON: {json_file}")
        print(f"  CSV:  {csv_file}")
        print(f"  总结: {summary_file}")
    
    def get_current_progress(self) -> Dict:
        """获取当前进度"""
        if not self.validation_times:
            return {"progress": 0, "message": "尚未开始验证"}
        
        num_validations = len(self.validation_times)
        progress = num_validations / self.expected_validations * 100
        
        avg_val_time = sum(self.validation_times) / num_validations
        estimated_step_time = avg_val_time / self.steps_per_validation
        
        return {
            "progress_percent": progress,
            "completed_validations": num_validations,
            "expected_validations": self.expected_validations,
            "average_validation_time": avg_val_time,
            "estimated_step_time": estimated_step_time
        }


# 全局计时器实例
_timer = None


def get_timer() -> SimpleStepTimer:
    """获取全局计时器实例"""
    global _timer
    if _timer is None:
        _timer = SimpleStepTimer()
    return _timer


def start_validation_timing(step: int):
    """开始验证计时"""
    timer = get_timer()
    timer.start_validation(step)


def end_validation_timing(step: int):
    """结束验证计时"""
    timer = get_timer()
    timer.end_validation(step)


def get_progress() -> Dict:
    """获取当前进度"""
    timer = get_timer()
    return timer.get_current_progress()


if __name__ == "__main__":
    # 测试脚本
    print("简单Step计时器测试")
    print("="*30)
    
    timer = SimpleStepTimer()
    
    # 模拟10次验证
    for i in range(10):
        step = (i + 1) * 1600
        timer.start_validation(step)
        time.sleep(0.5 + (i % 3) * 0.2)  # 模拟不同的验证时间
        timer.end_validation(step)
    
    print("\n测试完成！")












