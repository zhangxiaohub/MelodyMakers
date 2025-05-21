from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from diffusers import StableDiffusionPipeline
import logging
import os
import sys

# --------------------------
# 初始化Flask应用
# --------------------------
app = Flask(__name__, 
          static_folder='static',        # 静态文件目录
          template_folder='templates')   # 模板目录

# 启用跨域（需在app创建后立即配置）
#CORS(app, supports_credentials=True)
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

# --------------------------
# 全局配置
# --------------------------
# 配置日志
logging.basicConfig(level=logging.INFO)  #配置日志记录级别为 INFO，方便调试
logger = logging.getLogger(__name__)    # 获取当前模块的日志记录器

# 全局模型实例
# 这里的模型实例在初始化时加载，避免每次请求都重新加载
# 这会显著提高性能
tokenizer = None
translation_model = None
sd_pipeline = None

# --------------------------
# 模型初始化函数
# --------------------------
def initialize_models():
    """初始化所有模型（服务启动时调用）"""
    global tokenizer, translation_model, sd_pipeline
    
    try:
        # 1. 加载翻译模型
        translation_model_path = "/mnt/sdc/huggingface/model_hub/opus-mt-zh-en"
        logger.info("正在加载翻译模型...")
        tokenizer = AutoTokenizer.from_pretrained(translation_model_path)
        translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_path)
        translation_model.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. 加载Stable Diffusion
        sd_model_path = "/mnt/sdc/huggingface/model_hub/stable-diffusion-v1-5"
        logger.info("正在加载Stable Diffusion模型...")
        sd_pipeline = StableDiffusionPipeline.from_pretrained(
             sd_model_path,
             torch_dtype=torch.float16
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sd_pipeline = sd_pipeline.to(device)
                
        logger.info("所有模型加载完成！")
        
    except Exception as e:
        logger.error(f"模型初始化失败: {str(e)}")
        sys.exit(1)

# --------------------------
# 核心业务逻辑
# --------------------------
'''
def generate_cover_image(description):
    """生成封面核心逻辑"""
    data = request.get_json()
    if not data or 'description' not in data:
        logger.error("缺少描述参数")
        return jsonify({"error": "缺少描述参数"}), 400

    description = data['description']
    logger.info(f"接收到的描述: {description}")

    try:
        # 1. 翻译输入
        inputs = tokenizer(description, return_tensors="pt")
        inputs = {k: v.to(translation_model.device) for k, v in inputs.items()}
        outputs = translation_model.generate(**inputs)
        translated_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"翻译结果: {translated_prompt}")

        # 2. 生成图像
        image = sd_pipeline(
            translated_prompt,
            num_inference_steps=50,  # 迭代次数,50步推理
            guidance_scale=7.5      # 引导比例
        ).images[0]
        
        if image:
            logger.info("图像生成成功！")
        else:
            logger.error("图像生成失败，返回值为空！")

        # 3. 返回图像
        return image
    
    except Exception as e:
        logger.error(f"生成过程中出现错误: {str(e)}")
        raise
'''


# --------------------------
# 路由定义
# --------------------------

#测试路由
@app.route('/test-write', methods=['GET'])
def test_write():
    try:
        test_path = os.path.join('static', 'generated_covers', 'test.txt')
        with open(test_path, 'w') as f:
            f.write('测试写入权限成功！')
        return jsonify({"message": "写入成功", "path": test_path}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/')
def index():
    """主页路由"""
    return render_template('index.html')   #返回 index.html 模板，作为前端页面


def generate_cover_image(description):
    """生成封面核心逻辑"""
    try:
        # 1. 翻译输入
        inputs = tokenizer(description, return_tensors="pt")
        inputs = {k: v.to(translation_model.device) for k, v in inputs.items()}
        outputs = translation_model.generate(**inputs)
        translated_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"翻译结果: {translated_prompt}")

        # 2. 生成图像
        logger.info("开始生成图片...")
        image = sd_pipeline(
            translated_prompt,
            num_inference_steps=50,
            guidance_scale=7.5
        ).images[0]
        
        if image:
            logger.info("图像生成成功！")
            return image
        else:
            logger.error("图像生成失败，返回值为空！")
            raise Exception("图像生成失败")

    except Exception as e:
        logger.error(f"生成过程中出现错误: {str(e)}")
        raise

@app.route('/generate-cover', methods=['POST'])
def handle_generate_cover():
    """封面生成API"""
    try:
        # 验证输入
        data = request.get_json()
        if not data or 'description' not in data:
            logger.error("缺少描述参数")
            return jsonify({"error": "缺少描述参数"}), 400

        description = data['description']
        logger.info(f"接收到描述: {description}")

        # 生成图像
        pil_image = generate_cover_image(description)
        
        # 保存图片
        image_dir = os.path.join('static', 'generated_covers')
        os.makedirs(image_dir, exist_ok=True)
        
        # 转换为Base64
        buffered = BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        logger.info("图片已转换为Base64格式")
        
        return jsonify({
            "image": f"data:image/png;base64,{img_str}",
            "status": "success"
        })
    
    except Exception as e:
        logger.error(f"API请求失败: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500


'''
@app.route('/generate-cover', methods=['POST'])
def handle_generate_cover():
    """封面生成API"""
    try:
        # 验证输入
        data = request.get_json()
        if not data or 'description' not in data:
            return jsonify({"error": "缺少描述参数"}), 400
        
        description = data['description']
        logger.info(f"接收到的描述: {description}")

        # 生成图像
        pil_image = generate_cover_image(data['description'])
        
        # 确保目录存在
        image_dir = os.path.join('static', 'generated_covers')
        os.makedirs(image_dir, exist_ok=True)
        # 定义图片路径
        image_path = os.path.join(image_dir, 'cover.png')
        # 保存图片
        pil_image.save(image_path, format="PNG")

        # 记录保存路径
        logger.info(f"图片已保存到: {image_path}")

        # 返回图片的 URL
        return jsonify({
            "image_url": f"/{image_path}",
            "status": "success"
        })

    except Exception as e:
        logger.error(f"API请求失败: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500
'''


# --------------------------
# 主程序入口
# --------------------------
if __name__ == '__main__':
    # 初始化模型
    initialize_models()
    # 启动Web服务（重要！）
    app.run(
        host='0.0.0.0', 
        port=5000, 
        threaded=True, 
        use_reloader=False  # 避免重复初始化模型
    )
