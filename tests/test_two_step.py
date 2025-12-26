"""
测试两步式分割流程
"""
import os
import sys
import cv2
import torch
import numpy as np

# 添加项目根目录到 Python 路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# 使用离线模式，避免网络连接
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from groundingdino.util.inference import load_image, predict, annotate
from segment_anything import sam_model_registry, SamPredictor

# 配置路径（相对于项目根目录）
WEIGHTS_FOLDER = os.path.join(ROOT_DIR, 'weights')
UPLOADS_FOLDER = os.path.join(ROOT_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(ROOT_DIR, 'results')
os.makedirs(RESULTS_FOLDER, exist_ok=True)

image_path = os.path.join(UPLOADS_FOLDER, "b5404591-5ade-41f3-8b84-a5eb2f57a18a.jpg")
text_prompt = "crane arm"

print("=" * 50)
print("步骤 1: GroundingDINO 检测（显示边界框）")
print("=" * 50)

# 第一步：加载 GroundingDINO 并检测
from groundingdino.util.inference import load_model
dino_model = load_model(
    os.path.join(WEIGHTS_FOLDER, "GroundingDINO_SwinT_OGC.py"),
    os.path.join(WEIGHTS_FOLDER, "groundingdino_swint_ogc.pth")
)

image_source, image = load_image(image_path)

boxes, logits, phrases = predict(
    model=dino_model,
    image=image,
    caption=text_prompt,
    box_threshold=0.35,
    text_threshold=0.25
)

print(f"\n检测到 {len(phrases)} 个目标:")
for i, (phrase, logit) in enumerate(zip(phrases, logits)):
    print(f"  [{i}] {phrase}: {logit:.3f}")

# 生成预览图（仅边界框）
annotated_frame = annotate(
    image_source=image_source,
    boxes=boxes,
    logits=logits,
    phrases=phrases
)
preview_path = os.path.join(RESULTS_FOLDER, "step1_detection_preview.jpg")
cv2.imwrite(preview_path, annotated_frame)
print(f"\n预览图已保存: {preview_path}")

print("\n" + "=" * 50)
print("用户确认后，执行步骤 2: SAM 精确分割")
print("=" * 50)

# 第二步：用户确认后，只加载 SAM 模型（不重复加载 GroundingDINO）
print("\n加载 SAM 模型...")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=os.path.join(WEIGHTS_FOLDER, "sam_vit_b_01ec64.pth"))
sam.to(device=device)
sam_predictor = SamPredictor(sam)

# 准备图像
image_rgb = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
sam_predictor.set_image(image_rgb)

# 将归一化的 [cx, cy, w, h] 转换为像素坐标 [x1, y1, x2, y2]
h, w = image_rgb.shape[:2]
masks = []
for box in boxes:
    box_np = box.cpu().numpy() if hasattr(box, 'cpu') else np.array(box)
    cx, cy, bw, bh = box_np
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    box_xyxy = np.array([x1, y1, x2, y2])

    # SAM 分割
    mask, _, _ = sam_predictor.predict(
        box=box_xyxy,
        multimask_output=False
    )
    masks.append(mask[0])

# 生成最终结果图
result_path = os.path.join(RESULTS_FOLDER, "step2_sam_segmentation.jpg")
result_image = image_source.copy()

for i, mask in enumerate(masks):
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(
        mask_uint8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    color = [0, 255, 0]  # 绿色

    # 填充掩码
    mask_overlay = ((mask > 0)[:, :, None] * color).astype(np.uint8)
    result_image = cv2.addWeighted(result_image, 0.7, mask_overlay, 0.3, 0)

    # 绘制轮廓
    cv2.drawContours(result_image, contours, -1, color, 2)

# 添加边界框和标签
result_image = annotate(
    image_source=result_image,
    boxes=boxes,
    logits=logits,
    phrases=phrases
)

cv2.imwrite(result_path, result_image)

print(f"\n分割完成! 结果已保存: {result_path}")
print("\n对比:")
print(f"  预览图（仅边界框）: {preview_path}")
print(f"  分割图（精确掩码）: {result_path}")
