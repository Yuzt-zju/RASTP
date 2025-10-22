#!/bin/bash

# SCP传输脚本 - 将GRID项目传输到远程主机lyg0204
# 远程主机配置：
# Host: lyg0204
# HostName: js1.blockelite.cn
# User: zhantianyu
# Port: 31400

# 源目录
SOURCE_DIR="/data/zhantianyu/Project/GRID"

# 目标目录（远程主机）
TARGET_DIR="/home/zhantianyu/GRID"

# 创建临时目录用于存放要传输的文件
TEMP_DIR="/tmp/grid_transfer_$(date +%s)"
mkdir -p "$TEMP_DIR"

# 复制要传输的文件到临时目录（排除不需要的文件）
echo "📦 准备传输文件..."
cp -r "$SOURCE_DIR"/* "$TEMP_DIR/"

# 删除不需要的文件和目录
rm -f "$TEMP_DIR/notices.txt"
rm -rf "$TEMP_DIR/.git"
rm -rf "$TEMP_DIR/__MACOSX"
rm -rf "$TEMP_DIR/logs"
rm -rf "$TEMP_DIR/outputs"
rm -rf "$TEMP_DIR/data"

# SCP命令 - 传输临时目录的内容
scp -r -P 31400 \
    "$TEMP_DIR/" \
    zhantianyu@js1.blockelite.cn:"$TARGET_DIR"

# 检查传输是否成功
if [ $? -eq 0 ]; then
    echo "✅ SCP传输成功完成！"
    echo "项目已传输到：$TARGET_DIR"
    
    # 清理临时目录
    rm -rf "$TEMP_DIR"
else
    echo "❌ SCP传输失败，请检查网络连接和权限"
    # 保留临时目录用于调试
    echo "临时文件保存在：$TEMP_DIR"
    exit 1
fi

echo ""
echo "📋 传输内容概览："
echo "- 项目源代码 (src/)"
echo "- 配置文件 (configs/)"
echo "- 脚本文件 (scripts/)"
echo "- 模型文件 (qwen/, google/)"
echo "- 文档文件 (README.md, LICENSE)"
echo "- 训练脚本 (step*.sh, train_gr*.sh)"
echo ""
echo "🚫 已排除的文件："
echo "- notices.txt (26MB大文件)"
echo "- .git/ (版本控制)"
echo "- logs/ (日志文件)"
echo "- outputs/ (输出文件)"
echo "- data/ (数据文件)"
echo ""
echo "💡 提示：如果需要传输notices.txt文件，请单独执行："
echo "scp -P 31400 $SOURCE_DIR/notices.txt zhantianyu@js1.blockelite.cn:$TARGET_DIR/"
