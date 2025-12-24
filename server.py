import os
import json
import uuid
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI

app = Flask(__name__)
CORS(app)

# 配置上传目录
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# DeepSeek 客户端
client = OpenAI(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 会话存储 {session_id: {"messages": [...], "image_path": "...", "result_count": 0}}
sessions = {}

# 工具定义
tools = [
    {
        "type": "function",
        "function": {
            "name": "segment_object",
            "description": "在图像中分割指定物体",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "图像路径"},
                    "object_prompt": {"type": "string", "description": "物体描述"}
                },
                "required": ["image_path", "object_prompt"]
            }
        }
    }
]


def handle_tool(name: str, inputs: dict, result_path: str) -> dict:
    """工具处理函数"""
    if name == "segment_object":
        from groundingdino.util.inference import load_model, load_image, predict, annotate
        import cv2

        model = load_model(
            "weights/GroundingDINO_SwinT_OGC.py",
            "weights/groundingdino_swint_ogc.pth"
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

        annotated_frame = annotate(
            image_source=image_source,
            boxes=boxes,
            logits=logits,
            phrases=phrases
        )
        cv2.imwrite(result_path, annotated_frame)

        return {
            "success": True,
            "result_saved": result_path,
            "detected": phrases,
            "num_objects": len(phrases)
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

                tool_result = handle_tool(tool_call.function.name, inputs, result_path)

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
            {"role": "system", "content": f"你是图像分割助手。用户已上传图片，路径为: {image_path}。当用户请求分割物体时，直接调用 segment_object 工具，image_path 使用 '{image_path}'，object_prompt 使用用户描述的物体名称。不要询问图片路径，直接执行分割。"}
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
    app.run(host='0.0.0.0', port=5000, debug=True)
