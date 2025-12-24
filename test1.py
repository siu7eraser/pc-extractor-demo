from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2

  # 1. 加载模型
model = load_model(
    "weights/GroundingDINO_SwinT_OGC.py",  # 配置文件
    "weights/groundingdino_swint_ogc.pth"# 权重文件
)

  # 2. 加载图片
image_source, image = load_image("1.jpg")

  # 3. 设置检测参数
TEXT_PROMPT = "building"  # 用. 分隔多个类别
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

  # 4. 推理
boxes, logits, phrases = predict(
      model=model,
      image=image,
      caption=TEXT_PROMPT,
      box_threshold=BOX_THRESHOLD,
      text_threshold=TEXT_THRESHOLD
  )

  # 5. 可视化结果
annotated_frame = annotate(
      image_source=image_source,
      boxes=boxes,
      logits=logits,
      phrases=phrases
  )
cv2.imwrite("result.jpg", annotated_frame)