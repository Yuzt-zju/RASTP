#!/usr/bin/env python3
"""
Step时间统计集成脚本

这个脚本展示了如何将简单的时间统计功能集成到现有的训练代码中。
只需要在验证开始和结束时调用相应的函数即可。
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.simple_step_timer import start_validation_timing, end_validation_timing, get_progress


def example_usage():
    """示例用法"""
    print("Step时间统计集成示例")
    print("="*40)
    
    # 模拟训练过程中的验证调用
    print("模拟训练过程中的验证调用...")
    
    # 在验证开始时调用
    for step in [1600, 3200, 4800, 6400, 8000, 9600, 11200, 12800, 14400, 16000]:
        print(f"\nStep {step} - 开始验证")
        start_validation_timing(step)
        
        # 这里是你实际的验证代码
        # ... 验证逻辑 ...
        
        print(f"Step {step} - 验证完成")
        end_validation_timing(step)
        
        # 显示进度
        progress = get_progress()
        print(f"进度: {progress['progress_percent']:.1f}% ({progress['completed_validations']}/{progress['expected_validations']})")
        if progress['estimated_step_time'] > 0:
            print(f"当前估算每step时间: {progress['estimated_step_time']:.4f}秒")


def integration_guide():
    """集成指南"""
    print("\n" + "="*50)
    print("集成指南")
    print("="*50)
    print("""
要在你的训练代码中集成时间统计功能，只需要：

1. 导入计时函数：
   from scripts.simple_step_timer import start_validation_timing, end_validation_timing

2. 在验证开始时调用：
   start_validation_timing(current_step)

3. 在验证结束时调用：
   end_validation_timing(current_step)

4. 可选：获取进度信息：
   from scripts.simple_step_timer import get_progress
   progress = get_progress()

示例代码：
```python
# 在你的验证循环中
for step in range(0, 16001, 1600):
    if step > 0:  # 跳过step 0
        # 开始验证计时
        start_validation_timing(step)
        
        # 你的验证代码
        # ... validation logic ...
        
        # 结束验证计时
        end_validation_timing(step)
```

完成后，结果会自动保存到 logs/step_timing/ 目录中，包括：
- JSON格式的详细结果
- CSV格式的验证时间数据
- 文本格式的总结报告
""")


if __name__ == "__main__":
    import time
    
    # 运行示例
    example_usage()
    
    # 显示集成指南
    integration_guide()
    
    print("\n✅ 集成示例完成！")












