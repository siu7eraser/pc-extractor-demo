"""
缓存 BERT 模型到本地，避免每次启动都从 HuggingFace 下载
"""

import os
from transformers import BertTokenizer, BertModel

# 设置本地缓存目录
cache_dir = os.path.join(os.path.dirname(__file__), "weights", "bert_cache")
os.makedirs(cache_dir, exist_ok=True)

print(f"正在缓存 BERT 模型到: {cache_dir}")
print("这可能需要几分钟...")

try:
    # 缓存 tokenizer
    print("下载 tokenizer...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
    print("Tokenizer 缓存完成")

    # 缓存模型
    print("下载 model...")
    model = BertModel.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
    print("Model 缓存完成")

    print(f"\nBERT 模型已缓存到: {cache_dir}")
    print("\n请在代码中使用:")
    print(f'  text_encoder_type = "{cache_dir}"')
    print("或者在 grounded_sam.py 中设置 HF_HOME 环境变量")

except Exception as e:
    print(f"缓存失败: {e}")
