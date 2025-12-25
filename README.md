# PC-Extractor Demo

基于 GroundingDINO 的图像分割 Agent，使用 DeepSeek 大模型进行自然语言交互。

## 功能

- 自然语言描述要分割的物体
- 自动调用 GroundingDINO 进行目标检测和分割
- 支持多轮对话

## 安装

```bash
```bash
pip install -r requirements.txt
```

### 前端安装

```bash
cd frontend
npm install
```


## 配置

设置 DeepSeek API Key 环境变量：

```powershell
$env:ANTHROPIC_API_KEY="your-deepseek-api-key"
```

## 使用

```bash
### 启动后端 API

```bash
python server.py
```

### 启动前端应用

```bash
cd frontend
npm run dev
```

访问 `http://localhost:5174` (端口可能随 Vite 在终端的输出而变化) 使用 Web 界面。

### 命令行模式

```

示例对话：
```
图像分割助手已启动，输入 'exit' 退出
Agent: 请告诉我要分割的图片和物体
You: 请帮我分割图片1.jpg中的塔吊
调用工具: segment_object
Agent: 已完成分割，结果保存在 result.jpg
```

## 依赖

- groundingdino
- openai
- opencv-python
- torch

## 模型权重

需要下载 GroundingDINO 权重文件到 `weights/` 目录：
- `GroundingDINO_SwinT_OGC.py` - 配置文件
- `groundingdino_swint_ogc.pth` - 模型权重
