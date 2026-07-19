from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor

model_name = "/tmp/Qwen3-VL-8B-Instruct"

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_name, dtype="auto", device_map="auto", local_files_only=True
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-8B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)

# 请识别提取图像中的文本,另外需要移除营销广告词(工厂或公司名称、电子邮箱、手机号码、电话号码、qq号、微信、抖音、快手、网址等)，只保留商品相关信息(没有商品信息的输出空字符串).
# 请识别提取图像中的文本,提取后参照原图和语义进一步补全优化,需要移除营销广告词(工厂或公司名称、电子邮箱、手机号码、电话号码、qq号、微信、抖音、快手、网址等),只留商品相关信息
# 请识别提取图像中的全部文本,识别结果需要移除营销广告词(工厂或公司名称、电子邮箱、手机号码、电话号码、qq号、微信、抖音、快手、网址等)，只保留商品相关信息(没有商品信息的输出none)
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "./data/1.jpg",
            },
            {"type": "text", "text": "请识别提取图像中的全部文本,识别结果需要移除营销广告词(工厂或公司名称、电子邮箱、手机号码、电话号码、qq号、微信、抖音、快手、网址等)，只保留商品相关信息(没有商品信息的输出none)"},
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)