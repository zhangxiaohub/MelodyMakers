from transformers import AutoProcessor, AutoModel
import scipy.io.wavfile
import torch
import numpy as np

# 指定本地模型路径
model_path = "/home/zhangxiao/huggingface_models/suno/bark-small"

# 初始化Bark模型
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# 将模型移至GPU
if torch.cuda.is_available():
    model = model.to("cuda:1")

lyrics = "小酒窝长睫毛，你有最美的微笑"
# 生成演唱音频
inputs = processor(
    text=[lyrics],
    return_tensors="pt",
)

# 如果使用GPU，将输入数据也移至GPU
if torch.cuda.is_available():
    inputs = {k: v.to("cuda:1") for k, v in inputs.items()}
    
try:
    # 生成音频
    speech_values = model.generate(
        **inputs, 
        do_sample=True,
        temperature=0.7,
        use_cache=True)

    # Bark 模型的采样率固定为 24kHz
    SAMPLE_RATE = 24000

    # 保存人声
    scipy.io.wavfile.write(
        "human_out.wav", 
        rate=SAMPLE_RATE,
        data=speech_values.cpu().numpy().squeeze()
    )

    # 添加调试信息
    print(f"音频生成完成！")
    print(f"采样率: {SAMPLE_RATE}")
    print(f"音频形状: {speech_values.shape}")

except Exception as e:
    print(f"生成音频时发生错误: {str(e)}")
    raise