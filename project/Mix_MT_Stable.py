from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from diffusers import StableDiffusionPipeline
import logging
import os
from datetime import datetime
import sys

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
OUTPUT_DIR = "outputs"

'''
def check_dependencies():
    """检查必要的依赖包"""
    try:
        import sentencepiece
        logger.info("所有依赖包已正确安装")
    except ImportError:
        logger.error("缺少必要的依赖包 sentencepiece，正在尝试安装...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece"])
            logger.info("sentencepiece 安装成功")
        except Exception as e:
            logger.error(f"安装失败: {str(e)}")
            raise
'''

def load_translation_model(model_path):
    """加载翻译模型"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        return tokenizer, model
    except Exception as e:
        logger.error(f"翻译模型加载失败: {str(e)}")
        raise


def setup_pipeline():
    """加载本地 Stable Diffusion 模型"""
    # 本地模型路径
    local_sd_dir = "/mnt/sdc/huggingface/model_hub/stable-diffusion-v1-5"
    
    try:
        # 从本地加载模型（启用半精度以节省显存）
        pipe = StableDiffusionPipeline.from_pretrained(
            local_sd_dir,
            torch_dtype=torch.float16  # 半精度加速（需 CUDA 支持）
        )
        
        # 将模型移动到 GPU（如果可用）
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
            # 可选：启用 xformers 显存优化（需安装 xformers）
            # pipe.enable_xformers_memory_efficient_attention()
        
        logger.info("Stable Diffusion 模型加载成功")
        return pipe
    
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise

def generate_image(pipe, prompt):
    """根据提示生成并保存图像"""
    try:
         # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 生成图像（调整参数按需）
        image = pipe(
            prompt,
            num_inference_steps=50,  # 推理步数（默认 50）
            guidance_scale=7.5        # 提示相关性系数（默认 7.5）
        ).images[0]
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #filename = f"output_{timestamp}.png"
        filename = os.path.join(OUTPUT_DIR, f"generated_{timestamp}.png")
        image.save(filename)
        logger.info(f"图像已保存为: {filename}")
    
    except Exception as e:
        logger.error(f"图像生成失败: {str(e)}")
        raise


if __name__ == "__main__":
    try:
        # 检查依赖
        #check_dependencies()
        
        # 加载翻译模型
        tokenizer, model = load_translation_model("/mnt/sdc/huggingface/model_hub/opus-mt-zh-en")
        
        user_input = input("请描述你想生成的歌曲封面~: ")
        
        # 翻译处理
        inputs = tokenizer(user_input, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model.generate(**inputs)
        translated_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"翻译结果: {translated_prompt}")
        
        # 初始化 Stable Diffusion
        pipe = setup_pipeline()
        
        # 生成图像
        generate_image(pipe, translated_prompt)
        
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        sys.exit(1)