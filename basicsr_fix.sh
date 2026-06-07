#!/bin/bash

# basicsr_fix.sh - 修复basicsr与新版torchvision的兼容性问题
# 将 functional_tensor 替换为 functional

# 设置搜索路径，默认当前目录
SEARCH_PATH="${1:-.}"

echo "=== basicsr_fix.sh ==="
echo "搜索路径: $SEARCH_PATH"
echo ""

# 搜索 degradations.py 文件
echo "正在搜索 degradations.py 文件..."
files=$(find "$SEARCH_PATH" -name "degradations.py" -type f 2>/dev/null)

if [ -z "$files" ]; then
    echo "未找到 degradations.py 文件"
    exit 1
fi

echo "找到以下文件:"
echo "$files"
echo ""

# 对每个找到的文件执行替换
count=0
for file in $files; do
    echo "正在处理: $file"
    
    # 使用 sed 替换，使用 # 作为分隔符避免冲突
    sed -i 's#functional_tensor#functional#g' "$file"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ 替换成功"
        count=$((count + 1))
    else
        echo "  ✗ 替换失败"
    fi
done

echo ""
echo "=== 完成 ==="
echo "成功修复 $count 个文件"
