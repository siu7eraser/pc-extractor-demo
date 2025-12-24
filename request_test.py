
import requests

# 创建会话
resp = requests.post(
    'http://localhost:5000/api/session/create',
    files={'image': open('1.jpg', 'rb')}
)
session_id = resp.json()['session_id']
print(f"会话ID: {session_id}")


# 第一轮对话
resp = requests.post(
    'http://localhost:5000/api/session/chat',
    json={'session_id': session_id, 'message': '分割banner'}
)
print(resp.json()['answer'])

# 第二轮对话（继续交互）
resp = requests.post(
    'http://localhost:5000/api/session/chat',
    json={'session_id': session_id, 'message': '分割buildings'}
)
print(resp.json()['answer'])


requests.post(
    'http://localhost:5000/api/session/delete',
    json={'session_id': session_id}
)