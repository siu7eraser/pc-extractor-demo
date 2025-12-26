"""
Grounded-SAM: GroundingDINO + SAM 集成模块

使用 GroundingDINO 进行文本提示的目标检测，
然后使用 SAM 进行精确分割。
"""

import os
import torch
import numpy as np
from typing import List, Tuple, Optional
import cv2


class GroundedSAM:
    """Grounded-SAM 模型封装"""

    def __init__(
        self,
        groundingdino_config: str = "weights/GroundingDINO_SwinT_OGC.py",
        groundingdino_checkpoint: str = "weights/groundingdino_swint_ogc.pth",
        sam_checkpoint: str = "weights/sam_vit_b_01ec64.pth",
        device: Optional[str] = None
    ):
        """
        初始化 Grounded-SAM

        Args:
            groundingdino_config: GroundingDINO 配置文件路径
            groundingdino_checkpoint: GroundingDINO 权重文件路径
            sam_checkpoint: SAM 权重文件路径
            device: 设备 ('cuda', 'cpu' 或 None 自动检测)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 加载 GroundingDINO
        from groundingdino.util.inference import load_model as load_dino_model
        self.groundingdino = load_dino_model(
            groundingdino_config,
            groundingdino_checkpoint
        )

        # 加载 SAM
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        self.sam_predictor = SamPredictor(sam)

    def detect_with_groundingdino(
        self,
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        使用 GroundingDINO 检测目标

        Args:
            image: 输入图像 (RGB numpy array)
            text_prompt: 文本提示
            box_threshold: 边界框置信度阈值
            text_threshold: 文本置信度阈值

        Returns:
            boxes: 边界框 (N, 4) 格式 [x1, y1, x2, y2]
            logits: 置信度分数
            phrases: 检测到的短语
        """
        from groundingdino.util.inference import predict, load_image
        import torchvision.transforms as T

        # 图像已经是 RGB 格式的 numpy array，需要转换回 PIL Image
        from PIL import Image
        image_pil = Image.fromarray(image.astype('uint8'))

        # GroundingDINO 预处理
        transform = T.Compose([
            T.Resize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        image_processed = transform(image_pil).to(self.device)

        boxes, logits, phrases = predict(
            model=self.groundingdino,
            image=image_processed,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        return boxes, logits, phrases

    def segment_with_sam(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        boxes_normalized: bool = False
    ) -> List[np.ndarray]:
        """
        使用 SAM 进行分割

        Args:
            image: 输入图像 (RGB)
            boxes: 边界框，格式取决于 boxes_normalized 参数
            boxes_normalized: 如果为 True，boxes 是归一化的 [cx, cy, w, h] 格式 (0-1)
                              如果为 False，boxes 是像素坐标 [x1, y1, x2, y2] 格式

        Returns:
            masks: 分割掩码列表，每个掩码形状为 (H, W)
        """
        self.sam_predictor.set_image(image)
        h, w = image.shape[:2]

        masks = []
        for box in boxes:
            box_np = box.cpu().numpy() if hasattr(box, 'cpu') else np.array(box)

            if boxes_normalized:
                # 从归一化的 [cx, cy, w, h] 转换为像素坐标 [x1, y1, x2, y2]
                cx, cy, bw, bh = box_np
                x1 = (cx - bw / 2) * w
                y1 = (cy - bh / 2) * h
                x2 = (cx + bw / 2) * w
                y2 = (cy + bh / 2) * h
                box_xyxy = np.array([x1, y1, x2, y2])
            else:
                # 已经是 [x1, y1, x2, y2] 格式
                box_xyxy = box_np

            # SAM 期望 [x1, y1, x2, y2] 格式
            mask, _, _ = self.sam_predictor.predict(
                box=box_xyxy,
                multimask_output=False
            )
            masks.append(mask[0])  # 取第一个掩码

        return masks

    def predict(
        self,
        image_path: str,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ) -> dict:
        """
        完整的 Grounded-SAM 预测流程

        Args:
            image_path: 图像路径
            text_prompt: 文本提示（如 "crane arm"）
            box_threshold: 边界框置信度阈值
            text_threshold: 文本置信度阈值

        Returns:
            result: 包含以下键的字典:
                - boxes: 边界框
                - masks: 分割掩码
                - logits: 置信度分数
                - phrases: 检测到的短语
        """
        # 读取图像
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. GroundingDINO 检测
        boxes, logits, phrases = self.detect_with_groundingdino(
            image_rgb,
            text_prompt,
            box_threshold,
            text_threshold
        )

        if len(boxes) == 0:
            return {
                "boxes": [],
                "masks": [],
                "logits": [],
                "phrases": []
            }

        # 2. SAM 分割
        masks = self.segment_with_sam(image_rgb, boxes)

        return {
            "boxes": boxes,
            "masks": masks,
            "logits": logits,
            "phrases": phrases
        }

    def annotate(
        self,
        image_path: str,
        boxes: np.ndarray,
        masks: List[np.ndarray],
        logits: np.ndarray,
        phrases: List[str],
        output_path: str,
        draw_boxes: bool = True,
        draw_masks: bool = True,
        random_color: bool = False
    ):
        """
        可视化预测结果

        Args:
            image_path: 原始图像路径
            boxes: 边界框
            masks: 分割掩码
            logits: 置信度分数
            phrases: 检测到的短语
            output_path: 输出路径
            draw_boxes: 是否绘制边界框
            draw_masks: 是否绘制掩码
            random_color: 是否使用随机颜色
        """
        image = cv2.imread(image_path)

        if draw_masks:
            for i, mask in enumerate(masks):
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(
                    mask_uint8,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                color = np.random.randint(0, 256, 3).tolist() if random_color else [0, 255, 0]

                # 填充掩码
                mask_overlay = ((mask > 0)[:, :, None] * color).astype(np.uint8)
                image = cv2.addWeighted(image, 0.7, mask_overlay, 0.3, 0)

                # 绘制轮廓
                cv2.drawContours(image, contours, -1, color, 2)

        if draw_boxes:
            from groundingdino.util.inference import annotate as annotate_dino
            image = annotate_dino(
                image_source=image,
                boxes=boxes,
                logits=logits,
                phrases=phrases
            )

        cv2.imwrite(output_path, image)


# 便捷函数
def load_grounded_sam(
    groundingdino_config: str = "weights/GroundingDINO_SwinT_OGC.py",
    groundingdino_checkpoint: str = "weights/groundingdino_swint_ogc.pth",
    sam_checkpoint: str = "weights/sam_vit_b_01ec64.pth",
    device: Optional[str] = None
) -> GroundedSAM:
    """
    加载 Grounded-SAM 模型

    Args:
        groundingdino_config: GroundingDINO 配置文件路径
        groundingdino_checkpoint: GroundingDINO 权重文件路径
        sam_checkpoint: SAM 权重文件路径
        device: 设备 ('cuda', 'cpu' 或 None 自动检测)

    Returns:
        GroundedSAM 实例
    """
    return GroundedSAM(
        groundingdino_config=groundingdino_config,
        groundingdino_checkpoint=groundingdino_checkpoint,
        sam_checkpoint=sam_checkpoint,
        device=device
    )


if __name__ == "__main__":
    # 测试代码
    import sys

    if len(sys.argv) < 3:
        print("Usage: python grounded_sam.py <image_path> <text_prompt>")
        sys.exit(1)

    image_path = sys.argv[1]
    text_prompt = sys.argv[2]

    print("Loading Grounded-SAM model...")
    model = load_grounded_sam()

    print(f"Running prediction on {image_path} with prompt '{text_prompt}'...")
    result = model.predict(image_path, text_prompt)

    print(f"Detected {len(result['phrases'])} objects:")
    for phrase, logit in zip(result['phrases'], result['logits']):
        print(f"  - {phrase}: {logit:.3f}")

    output_path = image_path.rsplit('.', 1)[0] + "_grounded_sam_result.jpg"
    model.annotate(
        image_path=image_path,
        boxes=result['boxes'],
        masks=result['masks'],
        logits=result['logits'],
        phrases=result['phrases'],
        output_path=output_path
    )

    print(f"Result saved to {output_path}")
