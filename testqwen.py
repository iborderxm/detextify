from modelscope import AutoModelForCausalLM, AutoTokenizer

model_name = "/tmp/Qwen2.5-7B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

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

prompt = f"""请从以下OCR识别结果中提取商品信息，移除营销词（如工厂名、联系方式、广告语等），只保留商品相关描述。

OCR识别结果：
{ocr_text}

请参考以下格式输出纯文本：
品牌: xxx
车型: xxx
年份: xxx

注意：只输出包含商品信息的字段，没有的字段不要输出。"""

messages = [
    {"role": "system", "content": "你是一个专业的商品信息提取助手，擅长从OCR识别结果中提取关键商品信息并过滤营销内容。"},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]