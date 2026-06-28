from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor

model_name = "/tmp/Qwen3-VL-8B-Instruct"

model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    local_files_only=True
)
processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)

ocr_text = """Detected 7 text boxes.
   Text Box 1:
 	 Text: 'CZPINCHEN'
 	 Position: x=50, y=63, w=187, h=35
   Text Box 2:
 	 Text: '江苏常州市品晨配件厂'
 	 Position: x=87, y=53, w=200, h=22
   Text Box 3:
 	 Text: '车型：大众高尔夫6'
 	 Position: x=577, y=317, w=290, h=30
   Text Box 4:
 	 Text: '源头工厂'
 	 Position: x=633, y=26, w=227, h=59
   Text Box 5:
 	 Text: '年份：'
 	 Position: x=639, y=313, w=96, h=34
   Text Box 6:
 	 Text: '2009-2013年'
 	 Position: x=639, y=423, w=205, h=28
   Text Box 7:
 	 Text: '支持来样定制，跨境专供，量大价优，专业生产汽车配件'
 	 Position: x=686, y=265, w=479, h=23"""

prompt = f"""请从以下OCR识别结果中提取商品信息，移除营销词（如工厂名、联系方式、广告语等），只保留商品相关描述。所有输出内容必须翻译为英文。

OCR识别结果：
{ocr_text}

请根据商品类型自动识别并提取相关属性，输出格式如下：
属性名1: 属性值1
属性名2: 属性值2
...

注意：
1. 只输出包含商品信息的字段，没有的字段不要输出
2. 属性名和属性值都必须使用英文
3. 根据商品类型选择合适的属性名（如 Brand、Model、Year、Size、Color、Material 等）"""

messages = [
    {"role": "system", "content": "你是一个专业的商品信息提取助手，擅长从OCR识别结果中提取关键商品信息并过滤营销内容。"},
    {"role": "user", "content": prompt}
]
model_inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
model_inputs = model_inputs.to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(response)