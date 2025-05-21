from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import torch
import numpy as np

# 指定本地模型路径
model_path = "/home/zhangxiao/huggingface_models/facebook/musicgen-small"

# 组合风格与歌词信息
#lyrics = "Fly me to the moon, let me sing among the stars"
#lyrics = "In the quiet of the night, stars whisper secrets,Guiding dreams through the shadows, where hope never rests.Every heartbeat, a melody,Echoes of a story yet to be told.With every step, I feel the rhythm,Guiding me through the unknown."
#style_prompt = f"Classical style music with piano accompaniment, suitable for singing: {lyrics}"
style = input("请输入音乐风格（古典、流行、摇滚、电子）: ")
if style == "古典":
    style_prompt = "Create a soft melodic piano accompaniment with gentle strings in the background, " \
                   "suitable for singing. The music should have a peaceful and dreamy atmosphere, " \
                   "in the style of a romantic ballad with a slow tempo around 70-80 BPM."  
elif style == "流行":
    style_prompt = "Create an upbeat pop arrangement with modern electronic beats, catchy synthesizer melodies, "\
                   "and clear rhythmic structure. The music should have a contemporary pop sound with a bright and energetic atmosphere,"\
                   "featuring a strong bass line and dynamic drums. Tempo should be around 120-130 BPM in a radio-friendly style."
elif style == "摇滚":
    style_prompt = "Create a powerful rock instrumental with distorted electric guitars, driving drum beats," \
                   "and intense bass lines. The music should have an energetic and raw atmosphere, featuring "\
                   "classic rock elements with heavy guitar riffs and powerful drum fills. Include some guitar "\
                   "solos in a stadium rock style with tempo around 140-150 BPM."
elif style == "电子":
    style_prompt = "Create a modern electronic dance track with pulsing synthesizers, deep bass drops, "\
                   "and progressive house elements. The music should have an immersive and futuristic "\
                   "atmosphere, featuring layered electronic sounds, rhythmic arpeggios, and modern production "\
                   "effects. Build up tension and release with a steady tempo around 128-130 BPM."

# 初始化模型和处理器
processor = AutoProcessor.from_pretrained(model_path)
model = MusicgenForConditionalGeneration.from_pretrained(model_path)

# 设置设备
device = "cuda:1" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 准备输入
inputs = processor(
    text=[style_prompt],
    padding=True,
    return_tensors="pt",
)
inputs = {k: v.to(device) for k, v in inputs.items()}

# 生成音频
audio_values = model.generate(
    **inputs,
    max_new_tokens=1500,  # 控制生成音频的长度
    do_sample=True,
    guidance_scale=3.0
)

# 获取采样率并保存音频
sampling_rate = model.config.audio_encoder.sampling_rate
# 注意这里使用 audio_values[0, 0] 而不是 audio_values[0]
output_data = audio_values[0, 0].cpu().numpy()

# 保存音频文件
scipy.io.wavfile.write(
    "backMusic_out.wav", 
    rate=sampling_rate, 
    data=output_data
)

# 添加调试信息
print(f"音频形状: {output_data.shape}")
print(f"采样率: {sampling_rate}")
print(f"音频范围: min={output_data.min():.3f}, max={output_data.max():.3f}")