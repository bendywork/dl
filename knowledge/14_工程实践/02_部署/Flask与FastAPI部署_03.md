# Flask 与 FastAPI 模型部署

## 📌 核心问题
> 训练好的模型如何包装成 HTTP 接口供业务调用？

## 🌱 根源与动机

模型训练完 → 保存为 `.pt` / `.onnx` → 用 Web 框架包装成 REST API → 前端/业务服务调用。

**Flask**：轻量同步框架，适合原型和小并发  
**FastAPI**：异步高性能，自带 OpenAPI 文档，生产首选

## 📐 理论推导

### Flask 基本结构

```python
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    result = model_infer(text)   # 调用模型
    return jsonify({'label': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### FastAPI 异步结构

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictRequest(BaseModel):
    text: str

@app.post('/predict')
async def predict(req: PredictRequest):
    result = await model_infer_async(req.text)
    return {'label': result}
```

**FastAPI 优势**：
- 自动 `/docs` Swagger 界面
- Pydantic 参数验证（类型注解即文档）
- `async/await` 支持高并发

### ONNX 导出加速推理

```python
# 导出
torch.onnx.export(model, dummy_input, 'model.onnx',
                  input_names=['input'], output_names=['output'])

# 推理（CPU 比 PyTorch 快 2~4x）
import onnxruntime as ort
sess = ort.InferenceSession('model.onnx')
output = sess.run(['output'], {'input': input_array})
```

## 💡 关键理解

- **Flask 同步**：每个请求独占线程，并发高时用 gunicorn 多 worker
- **FastAPI 异步**：I/O 密集任务（如调用外部 API）效率高，CPU 密集（模型推理）仍需多进程
- **模型只加载一次**：在 `if __name__ == '__main__'` 之前加载，不要在请求函数内加载
- **ONNX 导出**：固化计算图，跨框架推理，CPU 推理比 PyTorch 快得多

## 🔧 代码实现

对应代码：`knowledge/工程实践/Flask部署/`
- `01_flask_demo基础.py` — Flask 最小示例
- `02_flask_app完整示例.py` — 含模型加载的完整 Flask 应用
- `03_http_client调用.py` — Python 调用接口的客户端代码
- `04_fastapi_app示例.py` — FastAPI + Pydantic 示例

## ⚠️ 易错点与常见误解

1. **model.eval() 必须调用**：关闭 Dropout 和 BatchNorm 的训练模式
2. **with torch.no_grad()：推理时必加**，不然会保留梯度计算图浪费显存
3. **Flask 默认开发模式不能用于生产**：用 `gunicorn -w 4 app:app` 启动
4. **ONNX 导出时 dynamic_axes**：如果 batch size 或序列长度可变，必须声明动态轴

## 🔗 知识延伸

- [[LLM基础与训练]] — LLM 的部署需要更复杂的推理引擎（vLLM、TGI）
- [[异步编程]] — FastAPI 的 async/await 基础

## 📚 参考资料
- PDF课件：`knowledge/PDF课件/扩展_01_Web基础Flask初识.pdf`
- 对应代码：`knowledge/工程实践/Flask部署/`
