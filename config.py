"""
LGF (Local-Global Fusion) — 全局配置文件

本文件是唯一需要根据你的运行环境修改的文件。
所有模型脚本统一从此处读取路径与 LLM 服务配置。
"""
import os

# ============================================================
# 路径配置
# ============================================================
# 开源包根目录（config.py 所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据文件（NLI 验证样本，JSON 列表，每个样本含 premise/hypothesis/is_negative 等字段）
DATA_PATH = os.path.join(BASE_DIR, "data", "NLI_Input_total_1_sample.json")

# RST 解析结果缓存文件（pkl，训练/推理时自动生成）
RST_CACHE_PATH = os.path.join(BASE_DIR, "data", "NLI_Input_total_1_rst_cache.pkl")

# 输出目录与输出文件
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
GLOBAL_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pt")
FUSION_RESULT_PATH = os.path.join(OUTPUT_DIR, "fusion_results.json")
LOCAL_RESULT_PATH = os.path.join(OUTPUT_DIR, "NLI_Output_total_1.json")

# ============================================================
# 本地 LLM 服务配置（局部模块用）
# ============================================================
# 该地址指向一个 OpenAI 兼容的 Chat Completions 服务（vLLM / llama.cpp 等均可）。
# 请在发布前替换为你自己的服务地址。
LLM_API_ENDPOINT = os.environ.get("LLM_API_ENDPOINT", "http://localhost:8001/v1")
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "Llama3.1-8B-Instruct")

# ============================================================
# 训练/推理配置
# ============================================================
# 若未安装 CUDA 设备，请改为 "cpu"。注意：RST 解析（RSTConfig.cuda_device）也需相应调整。
DEVICE = os.environ.get("LGF_DEVICE", "cpu")

TRAIN_BATCH_SIZE = 8
TRAIN_LEARNING_RATE = 2e-4
TRAIN_NUM_EPOCHS = 20
TRAIN_RATIO = 0.7
TRAIN_VAL_RATIO = 0.0
TRAIN_SEED = 42