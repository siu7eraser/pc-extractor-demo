# PC-Extractor

基于 GroundingDINO + SAM 的图像分割系统，使用 DeepSeek 大模型进行自然语言交互。

## 功能

- 自然语言描述要分割的物体
- 两步式分割流程：先检测预览，确认后精确分割
- GroundingDINO 目标检测 + SAM 精确分割
- 支持多轮对话
- React 前端界面

## 项目结构

```
pc-extractor/
├── backend/                # 后端代码
│   ├── server.py          # Flask API 服务
│   ├── grounded_sam.py    # Grounded-SAM 模型封装
│   └── agent.py           # Agent 逻辑
├── frontend/              # React 前端
│   ├── src/
│   └── package.json
├── tests/                 # 测试代码
│   ├── test_two_step.py   # 两步式分割测试
│   ├── test1.py
│   └── request_test.py
├── scripts/               # 工具脚本
│   ├── download_sam_weights.py
│   ├── download_sam_vitb.py
│   └── cache_bert_model.py
├── weights/               # 模型权重
├── uploads/               # 上传的图片
├── results/               # 分割结果
└── requirements.txt
```

## 安装

### 后端依赖

```bash
pip install -r requirements.txt
```

### 前端依赖

```bash
cd frontend
npm install
```

### 模型权重

下载以下文件到 `weights/` 目录：

1. **GroundingDINO**
   - `GroundingDINO_SwinT_OGC.py` - 配置文件
   - `groundingdino_swint_ogc.pth` - 模型权重

2. **SAM (Segment Anything)**
   - `sam_vit_b_01ec64.pth` - SAM ViT-B 权重

可使用脚本下载 SAM 权重：
```bash
python scripts/download_sam_vitb.py
```

## 配置

设置 DeepSeek API Key 环境变量：

```powershell
# Windows PowerShell
$env:ANTHROPIC_API_KEY="your-deepseek-api-key"
```

```bash
# Linux/Mac
export ANTHROPIC_API_KEY="your-deepseek-api-key"
```

## 使用

### 启动后端 API

```bash
python backend/server.py
```

API 端点：
- `POST /api/session/create` - 创建会话，上传图片
- `POST /api/session/chat` - 发送消息，进行对话
- `POST /api/session/delete` - 删除会话
- `GET /api/health` - 健康检查

### 启动前端

```bash
cd frontend
npm run dev
```

访问 `http://localhost:5174` 使用 Web 界面。

## 工作流程

### 两步式分割（推荐）

1. **检测阶段** - 使用 GroundingDINO 检测目标，显示边界框预览
2. **分割阶段** - 用户确认后，使用 SAM 进行精确分割

### 一次性分割

直接完成检测和分割，跳过预览步骤。

## 测试

```bash
# 运行两步式分割测试
python tests/test_two_step.py
```

## 依赖

- groundingdino
- segment-anything
- flask
- flask-cors
- openai
- opencv-python
- torch
- numpy

## API 文档

详见 [API_DOC.md](API_DOC.md)
