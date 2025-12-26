import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),  # 使用环境变量中的 DeepSeek key
    base_url="https://api.deepseek.com"
)

# 定义工具 (OpenAI 格式)
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
                    "object_prompt": {"type": "string", "description": "物体描述，如 'crane arm'"}
                },
                "required": ["image_path", "object_prompt"]
            }
        }
    }
]

# 工具处理函数
def handle_tool(name: str, inputs: dict) -> str:
    if name == "segment_object":
        from groundingdino.util.inference import load_model, load_image, predict, annotate
        import cv2

        # 1. 加载模型
        model = load_model(
            "weights/GroundingDINO_SwinT_OGC.py",  # 配置文件
            "weights/groundingdino_swint_ogc.pth"  # 权重文件
        )

        # 2. 加载图片
        image_source, image = load_image(inputs['image_path'])

        # 3. 设置检测参数
        TEXT_PROMPT = inputs['object_prompt']
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

        return json.dumps({"success": True, "result_saved": "result.jpg", "detected": phrases})
    return json.dumps({"error": "未知工具"})

# Agent 主循环
def run_agent(user_message: str):
    messages = [
        {"role": "system", "content": "你是图像分割助手。当用户请求分割图片中的物体时，直接调用 segment_object 工具，不要询问更多信息。"},
        {"role": "user", "content": user_message}
    ]

    while True:
        response = client.chat.completions.create(
            model="deepseek-chat",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )

        choice = response.choices[0]
        message = choice.message

        # 检查是否需要调用工具
        if message.tool_calls:
            # 添加 assistant 消息
            messages.append(message)

            for tool_call in message.tool_calls:
                print(f"调用工具: {tool_call.function.name}")
                inputs = json.loads(tool_call.function.arguments)
                result = handle_tool(tool_call.function.name, inputs)

                # 添加工具结果
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        else:
            # 输出回复
            if message.content:
                print(f"Agent: {message.content}")

            # 添加 assistant 消息到历史
            messages.append({"role": "assistant", "content": message.content})

            # 等待用户输入继续对话
            user_input = input("You: ").strip()
            if user_input.lower() in ['exit', 'quit', '退出']:
                print("再见！")
                break
            messages.append({"role": "user", "content": user_input})

if __name__ == "__main__":
    print("图像分割助手已启动，输入 'exit' 退出")
    run_agent("请帮我分割图片1.jpg 中的塔吊")
