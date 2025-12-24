# 图像分割服务 API 文档

## 基本信息

- **Base URL**: `http://localhost:5000`
- **协议**: HTTP
- **数据格式**: JSON / multipart/form-data

---

## 接口列表

### 1. 创建会话（上传图片）

创建新的对话会话并上传待处理的图片。

**请求**

```
POST /api/session/create
Content-Type: multipart/form-data
```

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| image | File | 是 | 图片文件（支持 jpg、png 等常见格式） |

**响应**

```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message": "会话已创建，请发送分割需求"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| session_id | string | 会话唯一标识，后续请求需要携带 |
| message | string | 提示信息 |

**错误响应**

| 状态码 | 错误信息 | 说明 |
|--------|----------|------|
| 400 | `{"error": "缺少图片"}` | 未上传图片文件 |

---

### 2. 发送消息（对话分割）

在已创建的会话中发送消息，进行图像分割对话。

**请求**

```
POST /api/session/chat
Content-Type: application/json
```

```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message": "帮我分割出图片中的猫"
}
```

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| session_id | string | 是 | 会话ID |
| message | string | 是 | 用户消息，描述要分割的物体 |

**响应**

```json
{
  "answer": "我已经帮你分割出了图片中的猫，共检测到 2 只猫。",
  "result_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| answer | string | AI 的文本回复 |
| result_image | string \| null | Base64 编码的结果图片（带 data URI 前缀），如无分割结果则为 null |
| session_id | string | 会话ID |

**错误响应**

| 状态码 | 错误信息 | 说明 |
|--------|----------|------|
| 400 | `{"error": "请求格式错误"}` | 请求体不是有效 JSON |
| 400 | `{"error": "缺少 session_id 或 message"}` | 缺少必填参数 |
| 404 | `{"error": "会话不存在或已过期"}` | session_id 无效 |
| 500 | `{"error": "错误详情"}` | 服务器内部错误 |

---

### 3. 删除会话

删除指定的会话，释放资源。

**请求**

```
POST /api/session/delete
Content-Type: application/json
```

```json
{
  "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| session_id | string | 是 | 要删除的会话ID |

**响应**

```json
{
  "message": "会话已删除"
}
```

**错误响应**

| 状态码 | 错误信息 | 说明 |
|--------|----------|------|
| 404 | `{"error": "会话不存在"}` | session_id 无效 |

---

### 4. 健康检查

检查服务是否正常运行。

**请求**

```
GET /api/health
```

**响应**

```json
{
  "status": "ok",
  "active_sessions": 3
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| status | string | 服务状态，"ok" 表示正常 |
| active_sessions | number | 当前活跃会话数量 |

---

## 使用流程

```
1. 调用 /api/session/create 上传图片，获取 session_id
           ↓
2. 调用 /api/session/chat 发送分割请求（可多轮对话）
           ↓
3. 前端展示 result_image（Base64 图片）和 answer（文本回复）
           ↓
4. 使用完毕调用 /api/session/delete 清理会话
```

## 前端调用示例

### 创建会话（上传图片）

```javascript
async function createSession(imageFile) {
  const formData = new FormData();
  formData.append('image', imageFile);

  const response = await fetch('http://localhost:5000/api/session/create', {
    method: 'POST',
    body: formData
  });

  return await response.json();
}
```

### 发送消息

```javascript
async function sendMessage(sessionId, message) {
  const response = await fetch('http://localhost:5000/api/session/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      session_id: sessionId,
      message: message
    })
  });

  return await response.json();
}
```

### 完整使用示例

```javascript
// 1. 用户选择图片后创建会话
const fileInput = document.getElementById('imageInput');
const imageFile = fileInput.files[0];

const { session_id } = await createSession(imageFile);

// 2. 发送分割请求
const result = await sendMessage(session_id, '帮我分割出图片中的人');

// 3. 展示结果
if (result.result_image) {
  document.getElementById('resultImg').src = result.result_image;
}
document.getElementById('answerText').textContent = result.answer;

// 4. 继续对话（支持多轮）
const result2 = await sendMessage(session_id, '再帮我分割出车辆');
```

---

## 注意事项

1. **CORS**: 服务端已启用 CORS，前端可直接跨域访问
2. **会话管理**: 会话数据存储在内存中，服务重启后会丢失
3. **图片格式**: 结果图片以 Base64 格式返回，前缀为 `data:image/jpeg;base64,`
4. **多轮对话**: 同一会话支持多次分割请求，上下文会保留
