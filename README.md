
```bash
pip install uv
uv sync
# .venv\Scripts\activate
# deactivate
```

## 启动 APP

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload --log-level info
```

## API 接口调用

详细的 API 调用文档请查看 [API_USAGE.md](API_USAGE.md)

### 快速示例

#### 1. 启动任务
```bash
curl -X POST "http://localhost:8000/start" \
  -H "Content-Type: application/json" \
  -d "{\"task_id\": \"task_001\", \"file_path\": \"data/test_examples.xlsx\"}"
```

#### 2. 查询状态
```bash
curl -X GET "http://localhost:8000/status/task_001"
```

#### 3. 下载文件
```bash
curl -X GET "http://localhost:8000/download/task_001"
```

#### 4. 中断任务
```bash
curl -X POST "http://localhost:8000/interrupt/task_001"
```

#### PowerShell 示例
```powershell
# 启动任务
$body = @{task_id="task_001"; file_path="data/test_examples.xlsx"} | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/start" -Method Post -Body $body -ContentType "application/json"

# 查询状态
Invoke-RestMethod -Uri "http://localhost:8000/status/task_001" -Method Get

# 下载文件
Invoke-RestMethod -Uri "http://localhost:8000/download/task_001" -Method Get

# 中断任务
Invoke-RestMethod -Uri "http://localhost:8000/interrupt/task_001" -Method Post
```