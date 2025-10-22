#!/bin/bash

# GRID项目打包脚本
# 将当前目录打包成压缩文件，便于传输

# 当前目录
CURRENT_DIR="/data/zhantianyu/Project/GRID"

# 打包文件名（包含时间戳）
PACKAGE_NAME="grid_project_$(date +%Y%m%d_%H%M%S).tar.gz"

# 排除的文件和目录模式
EXCLUDE_PATTERNS=(
    "--exclude=notices.txt"
    "--exclude=.git"
    "--exclude=__MACOSX"
    "--exclude=logs"
    "--exclude=outputs"
    "--exclude=*.tar.gz"
    "--exclude=google"
    "--exclude=*.zip"
)

echo "📦 开始打包GRID项目..."
echo "打包目录: $CURRENT_DIR"
echo "输出文件: $PACKAGE_NAME"
echo ""

# 执行打包命令
cd "$CURRENT_DIR" && tar -czvf "$PACKAGE_NAME" ${EXCLUDE_PATTERNS[@]} .

# 检查打包是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 打包成功完成！"
    echo "压缩文件: $PACKAGE_NAME"
    
    # 显示文件大小
    FILE_SIZE=$(du -h "$PACKAGE_NAME" | cut -f1)
    echo "文件大小: $FILE_SIZE"
    
    # 显示包含的文件列表
    echo ""
    echo "📋 包含的文件："
    tar -tzf "$PACKAGE_NAME" | head -10
    echo "... (更多文件)"
    
    # 显示排除的文件
    echo ""
    echo "🚫 已排除的文件："
    echo "- notices.txt (26MB大文件)"
    echo "- .git/ (版本控制)"
    echo "- logs/ (日志文件)"
    echo "- outputs/ (输出文件)"
    echo "- data/ (数据文件)"
    echo "- __MACOSX/ (macOS系统文件)"
    
else
    echo "❌ 打包失败"
    exit 1
fi

echo ""
echo "💡 使用说明："
echo "1. 传输压缩文件到远程主机："
echo "   scp -P 31400 $PACKAGE_NAME zhantianyu@js1.blockelite.cn:/home/zhantianyu/"
echo "2. 在远程主机解压："
echo "   tar -xzvf $PACKAGE_NAME -C /home/zhantianyu/GRID"
echo "3. 安装依赖："
echo "   pip install -r requirements.txt"
