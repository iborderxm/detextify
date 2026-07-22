"""PIL/Pillow 文字渲染示例 - 使用微软雅黑字体在图像上渲染商品信息"""
import os
import sys
from PIL import Image, ImageDraw, ImageFont


def find_microsoft_yahei_font():
    """查找微软雅黑字体文件路径"""
    font_paths = [
        r"C:\Windows\Fonts\msyh.ttc",      # 常规
        r"C:\Windows\Fonts\msyhbd.ttc",    # 粗体
        r"C:\Windows\Fonts\msyhl.ttc",     # 细体
        "/Library/Fonts/Microsoft YaHei.ttc",  # macOS
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux fallback
    ]
    for path in font_paths:
        if os.path.exists(path):
            return path
    return None


def draw_text_with_stroke(draw, text, position, font, text_color, stroke_color, stroke_width=2):
    """在指定位置绘制带描边的文字（使用Pillow原生描边功能）"""
    x, y = position
    # Pillow 8.0+ 原生支持 stroke_width 和 stroke_fill 参数
    draw.text(
        (x, y), text, font=font,
        fill=text_color,
        stroke_width=stroke_width,
        stroke_fill=stroke_color
    )
    # 获取文字尺寸
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def measure_multiline_text_height(draw, text, font, max_width):
    """测量多行文字的总高度（不实际绘制）"""
    total_height = 0
    
    # 按行分割
    lines = text.split('\n')
    for line in lines:
        if not line:
            total_height += font.size + 4
            continue
        
        # 如果单行过长，按空格分割
        words = line.split(' ')
        current_line = ''
        for word in words:
            test_line = current_line + (word if not current_line else ' ' + word)
            bbox = draw.textbbox((0, 0), test_line, font=font)
            test_width = bbox[2] - bbox[0]
            
            if test_width <= max_width or not current_line:
                current_line = test_line
            else:
                # 计算当前行高度
                bbox = draw.textbbox((0, 0), current_line, font=font)
                total_height += (bbox[3] - bbox[1]) + 8
                current_line = word
        
        # 计算最后一行高度
        if current_line:
            bbox = draw.textbbox((0, 0), current_line, font=font)
            total_height += (bbox[3] - bbox[1]) + 8
    
    return total_height


def draw_multiline_text(draw, text, position, font, text_color, stroke_color, stroke_width, max_width):
    """绘制多行文字，自动换行"""
    x, y = position
    current_x, current_y = x, y
    
    # 按行分割
    lines = text.split('\n')
    for line in lines:
        if not line:
            current_y += font.size + 4
            continue
        
        # 如果单行过长，按空格分割
        words = line.split(' ')
        current_line = ''
        for word in words:
            test_line = current_line + (word if not current_line else ' ' + word)
            bbox = draw.textbbox((0, 0), test_line, font=font)
            test_width = bbox[2] - bbox[0]
            
            if test_width <= max_width or not current_line:
                current_line = test_line
            else:
                # 绘制当前行
                tw, th = draw_text_with_stroke(draw, current_line, (current_x, current_y), font, text_color, stroke_color, stroke_width)
                current_y += th + 8
                current_line = word
        
        # 绘制最后一行
        if current_line:
            tw, th = draw_text_with_stroke(draw, current_line, (current_x, current_y), font, text_color, stroke_color, stroke_width)
            current_y += th + 8
    
    return current_y - y


def render_product_info(image_path, product_info, output_path, position=(0.05, 0.95), text_scale=1.0, use_bold=True, position_anchor='bottom-left'):
    """在图像上渲染商品信息文字
    
    Args:
        image_path: 输入图像路径
        product_info: 商品信息文本
        output_path: 输出图像路径
        position: 文字锚点位置比例 (x_ratio, y_ratio)，范围 0.0-1.0
        text_scale: 文字缩放因子，默认为 1.0
        use_bold: 是否使用粗体
        position_anchor: 锚点类型，支持 'top-left' 和 'bottom-left'
    """
    image = Image.open(image_path).convert('RGBA')
    draw = ImageDraw.Draw(image)
    
    base_font_size = image.height // 40
    font_size = max(12, int(base_font_size * text_scale))
    
    base_stroke_width = max(1, font_size // 10)
    stroke_width = int(base_stroke_width * text_scale)
    
    font_path = find_microsoft_yahei_font()
    if font_path is None:
        print("警告：未找到微软雅黑字体，使用默认字体")
        font = ImageFont.truetype('arial.ttf', font_size) if os.path.exists('arial.ttf') else ImageFont.load_default()
    else:
        font_index = 1 if use_bold and 'msyh.ttc' in font_path.lower() else 0
        font = ImageFont.truetype(font_path, font_size, index=font_index)
    
    x = int(image.width * position[0])
    max_width = image.width - x - int(image.width * 0.05)
    
    # 预计算文字总高度
    text_height = measure_multiline_text_height(draw, product_info, font, max_width)
    
    # 根据锚点类型计算起始 y 坐标
    if position_anchor == 'bottom-left':
        # 底部锚点：从底部向上偏移文字高度和边距
        y = int(image.height * position[1]) - text_height
    else:
        # 顶部锚点：直接使用 position[1] 作为起始位置
        y = int(image.height * position[1])
    
    rendered_position = (x, y)
    
    text_color = (255, 255, 255, 77)
    stroke_color = (0, 0, 0, 200)
    
    draw_multiline_text(draw, product_info, rendered_position, font, text_color, stroke_color, stroke_width, max_width)
    
    image.save(output_path, quality=95)
    print(f"图像已保存到: {output_path}")
    return image


if __name__ == "__main__":
    test_image_path = "./data/1.jpg"
    test_output_path = sys.argv[2] if len(sys.argv) >= 3 else "test_output.png"
    print(f"使用自定义图像: {test_image_path}")
    
    # 商品信息示例
    product_info = """商品名称：无线蓝牙耳机 Pro Max
品牌：TechSound
型号：TS-BT2024
颜色：深空灰
价格：¥299.00
特点：主动降噪 | 40小时续航 | Hi-Fi音质"""
    
    # 渲染商品信息到图像（位置：左下角）
    render_product_info(
        image_path=test_image_path,
        product_info=product_info,
        output_path=test_output_path,
        position=(0.02, 0.98),
        text_scale=1.2,
        use_bold=True,
        position_anchor='bottom-left'
    )
    
    print("\n渲染完成！")