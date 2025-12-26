import os
import sys
import json
import uuid
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

# 添加项目根目录到 Python 路径
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# 使用离线模式，避免网络连接问题
os.environ['TRANSFORMERS_OFFLINE'] = '1'

app = Flask(__name__)
CORS(app)

# 配置目录（相对于项目根目录）
UPLOAD_FOLDER = os.path.join(ROOT_DIR, 'uploads')
RESULT_FOLDER = os.path.join(ROOT_DIR, 'results')
WEIGHTS_FOLDER = os.path.join(ROOT_DIR, 'weights')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# DeepSeek 客户端
client = OpenAI(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 会话存储 {session_id: {"messages": [...], "image_path": "...", "result_count": 0, "detection_cache": {...}}}
sessions = {}

# 检测结果缓存（用于用户确认后的 SAM 分割）
_detection_cache = {}  # {session_id: {"boxes": ..., "logits": ..., "phrases": ..., "image_path": ...}}

# Grounded-SAM 模型实例（延迟加载）
_grounded_sam_model = None


def get_grounded_sam_model():
    """获取或创建 Grounded-SAM 模型单例"""
    global _grounded_sam_model
    if _grounded_sam_model is None:
        from backend.grounded_sam import load_grounded_sam
        print("Loading Grounded-SAM model...")
        _grounded_sam_model = load_grounded_sam(
            groundingdino_config=os.path.join(WEIGHTS_FOLDER, "GroundingDINO_SwinT_OGC.py"),
            groundingdino_checkpoint=os.path.join(WEIGHTS_FOLDER, "groundingdino_swint_ogc.pth"),
            sam_checkpoint=os.path.join(WEIGHTS_FOLDER, "sam_vit_b_01ec64.pth")
        )
        print("Grounded-SAM model loaded successfully!")
    return _grounded_sam_model

# 工具定义
tools = [
    {
        "type": "function",
        "function": {
            "name": "detect_objects",
            "description": "在图像中检测指定物体，显示边界框预览（第一步：检测）",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "图像路径"},
                    "object_prompt": {"type": "string", "description": "物体描述，如 'crane arm'"}
                },
                "required": ["image_path", "object_prompt"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "segment_with_sam",
            "description": "对已检测的物体进行精确分割（第二步：用户确认后执行 SAM 分割）",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "会话ID"},
                    "object_indices": {"type": "array", "items": {"type": "integer"}, "description": "要分割的物体索引列表，如 [0, 2] 表示分割第1和第3个检测到的物体。不提供则分割所有检测到的物体。"}
                },
                "required": ["session_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "segment_object_with_sam",
            "description": "一次性完成检测和分割（不显示中间步骤，直接输出精确掩码）",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "图像路径"},
                    "object_prompt": {"type": "string", "description": "物体描述，如 'crane arm'"}
                },
                "required": ["image_path", "object_prompt"]
            }
        }
    }
]


def handle_tool(name: str, inputs: dict, result_path: str, session_id: str = None) -> dict:
    """工具处理函数"""
    if name == "detect_objects":
        # 第一步：GroundingDINO 检测，缓存结果供后续 SAM 使用
        from groundingdino.util.inference import load_model, load_image, predict, annotate
        import cv2

        model = load_model(
            os.path.join(WEIGHTS_FOLDER, "GroundingDINO_SwinT_OGC.py"),
            os.path.join(WEIGHTS_FOLDER, "groundingdino_swint_ogc.pth")
        )

        image_source, image = load_image(inputs['image_path'])

        TEXT_PROMPT = inputs['object_prompt']
        BOX_THRESHOLD = 0.35
        TEXT_THRESHOLD = 0.25

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        # 生成预览图（仅边界框）
        annotated_frame = annotate(
            image_source=image_source,
            boxes=boxes,
            logits=logits,
            phrases=phrases
        )
        cv2.imwrite(result_path, annotated_frame)

        # 缓存检测结果供后续 SAM 使用
        if session_id:
            _detection_cache[session_id] = {
                "boxes": boxes,
                "logits": logits,
                "phrases": phrases,
                "image_path": inputs['image_path'],
                "image_source": image_source
            }

        return {
            "success": True,
            "result_saved": result_path,
            "detected": phrases,
            "num_objects": len(phrases),
            "method": "detection_only",
            "message": f"检测到 {len(phrases)} 个目标，已显示边界框预览。确认后请使用 '确认分割' 或 'segment_with_sam' 进行精确分割。"
        }

    elif name == "segment_with_sam":
        # 第二步：用户确认后，使用缓存的检测结果进行 SAM 分割
        if session_id not in _detection_cache:
            return {"error": "请先执行检测 (detect_objects)"}

        cached = _detection_cache[session_id]

        # 获取用户指定的物体索引，默认分割所有
        object_indices = inputs.get('object_indices', list(range(len(cached['phrases']))))

        # 过滤出用户选择的物体
        selected_boxes = cached['boxes'][object_indices] if len(object_indices) < len(cached['boxes']) else cached['boxes']
        selected_phrases = [cached['phrases'][i] for i in object_indices] if len(object_indices) < len(cached['phrases']) else cached['phrases']
        selected_logits = cached['logits'][object_indices] if len(object_indices) < len(cached['logits']) else cached['logits']

        if len(selected_boxes) == 0:
            return {
                "success": True,
                "result_saved": None,
                "detected": [],
                "num_objects": 0,
                "method": "sam_segmentation",
                "message": "没有选择任何物体"
            }

        # 使用 Grounded-SAM 的 SAM 部分进行分割
        model = get_grounded_sam_model()

        import cv2
        image_rgb = cv2.cvtColor(cached['image_source'], cv2.COLOR_BGR2RGB)

        # SAM 分割（boxes 是归一化的 [cx, cy, w, h] 格式）
        masks = model.segment_with_sam(image_rgb, selected_boxes, boxes_normalized=True)

        # 生成结果图
        model.annotate(
            image_path=cached['image_path'],
            boxes=selected_boxes,
            masks=masks,
            logits=selected_logits,
            phrases=selected_phrases,
            output_path=result_path
        )

        # 清除缓存（可选）
        # del _detection_cache[session_id]

        return {
            "success": True,
            "result_saved": result_path,
            "detected": selected_phrases,
            "num_objects": len(selected_phrases),
            "method": "sam_segmentation",
            "message": f"已完成 {len(selected_phrases)} 个目标的精确分割"
        }

    elif name == "segment_object_with_sam":
        # 一次性完成检测和分割（原有功能保留）
        model = get_grounded_sam_model()

        result = model.predict(
            image_path=inputs['image_path'],
            text_prompt=inputs['object_prompt'],
            box_threshold=0.35,
            text_threshold=0.25
        )

        if len(result['phrases']) == 0:
            return {
                "success": True,
                "result_saved": None,
                "detected": [],
                "num_objects": 0,
                "method": "grounded_sam",
                "message": "未检测到目标"
            }

        model.annotate(
            image_path=inputs['image_path'],
            boxes=result['boxes'],
            masks=result['masks'],
            logits=result['logits'],
            phrases=result['phrases'],
            output_path=result_path
        )

        return {
            "success": True,
            "result_saved": result_path,
            "detected": result['phrases'],
            "num_objects": len(result['phrases']),
            "method": "grounded_sam"
        }

    return {"error": "未知工具"}


def run_agent_turn(session_id: str, user_message: str) -> dict:
    """执行一轮对话，支持多轮交互"""
    session = sessions.get(session_id)
    if not session:
        return {"error": "会话不存在"}

    image_path = session["image_path"]
    messages = session["messages"]

    # 添加用户消息
    messages.append({"role": "user", "content": user_message})

    result_image = None

    max_iterations = 5
    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )

        choice = response.choices[0]
        message = choice.message

        if message.tool_calls:
            # 添加 assistant 消息（带工具调用）
            messages.append(message)

            for tool_call in message.tool_calls:
                inputs = json.loads(tool_call.function.arguments)
                inputs['image_path'] = image_path

                # 生成结果图片路径
                session["result_count"] += 1
                result_path = os.path.join(
                    RESULT_FOLDER,
                    f"{session_id}_result_{session['result_count']}.jpg"
                )

                tool_result = handle_tool(tool_call.function.name, inputs, result_path, session_id)

                if tool_result.get("success"):
                    result_image = result_path

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False)
                })
        else:
            # 普通回复，添加到历史并返回
            messages.append({"role": "assistant", "content": message.content})
            return {
                "answer": message.content or "",
                "result_image": result_image,
                "session_id": session_id
            }

    return {
        "answer": "处理超时",
        "result_image": result_image,
        "session_id": session_id
    }


@app.route('/api/session/create', methods=['POST'])
def create_session():
    """
    创建新会话，上传图片

    请求格式 (multipart/form-data):
    - image: 图片文件

    返回:
    - session_id: 会话ID
    """
    if 'image' not in request.files:
        return jsonify({"error": "缺少图片"}), 400

    image_file = request.files['image']

    # 生成会话ID
    session_id = str(uuid.uuid4())
    image_ext = os.path.splitext(image_file.filename)[1] or '.jpg'
    image_path = os.path.join(UPLOAD_FOLDER, f"{session_id}{image_ext}")

    # 保存图片
    image_file.save(image_path)

    # 创建会话
    sessions[session_id] = {
        "messages": [
            {"role": "system", "content": f"""你是图像分割助手。用户已上传图片，路径为: {image_path}。

工作流程（两步式）：
1. detect_objects - 检测目标并显示边界框预览（快速预览）
   - 当用户请求分割时，先调用此工具显示检测结果
   - 检测完成后，提示用户确认是否进行精确分割

2. segment_with_sam - 对已检测的目标进行精确分割（用户确认后执行）
   - 使用 SAM 生成精确掩码
   - 可选择性指定要分割的物体索引（如 [0, 2] 表示分割第1和第3个物体）
   - 不指定索引则分割所有检测到的物体

3. segment_object_with_sam - 一次性完成检测和分割（跳过预览）
   - 直接输出精确分割结果

默认使用两步流程：先 detect_objects 预览，用户确认后再 segment_with_sam。
当用户请求分割物体时，直接调用 detect_objects 工具，image_path 使用 '{image_path}'，object_prompt 使用用户描述的物体名称。"""}
        ],
        "image_path": image_path,
        "result_count": 0
    }

    return jsonify({
        "session_id": session_id,
        "message": "会话已创建，请发送分割需求"
    })


@app.route('/api/session/chat', methods=['POST'])
def chat():
    """
    在会话中发送消息

    请求格式 (JSON):
    - session_id: 会话ID
    - message: 用户消息

    返回:
    - answer: 文本回答
    - result_image: base64 编码的结果图片（如果有）
    - session_id: 会话ID
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "请求格式错误"}), 400

    session_id = data.get("session_id")
    message = data.get("message")

    if not session_id or not message:
        return jsonify({"error": "缺少 session_id 或 message"}), 400

    if session_id not in sessions:
        return jsonify({"error": "会话不存在或已过期"}), 404

    try:
        result = run_agent_turn(session_id, message)

        response_data = {
            "answer": result["answer"],
            "result_image": None,
            "session_id": session_id
        }

        # 如果有结果图片，转为 base64
        if result.get("result_image") and os.path.exists(result["result_image"]):
            with open(result["result_image"], "rb") as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
                response_data["result_image"] = f"data:image/jpeg;base64,{image_data}"

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/session/delete', methods=['POST'])
def delete_session():
    """删除会话"""
    data = request.get_json()
    session_id = data.get("session_id") if data else None

    if session_id and session_id in sessions:
        del sessions[session_id]
        return jsonify({"message": "会话已删除"})

    return jsonify({"error": "会话不存在"}), 404


@app.route('/api/health', methods=['GET'])
def health():
    """健康检查"""
    return jsonify({"status": "ok", "active_sessions": len(sessions)})


if __name__ == '__main__':
    print("启动图像分割服务...")
    print("\nAPI 端点:")
    print("  POST /api/session/create  - 创建会话，上传图片")
    print("  POST /api/session/chat    - 发送消息，进行对话")
    print("  POST /api/session/delete  - 删除会话")
    print("  GET  /api/health          - 健康检查")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
