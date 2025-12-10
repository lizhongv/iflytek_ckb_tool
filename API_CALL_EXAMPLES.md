# API 接口调用示例

本文档提供所有 API 接口的调用命令示例，包括 curl 和 Python requests 两种方式。

**基础信息：**
- 服务地址：`http://localhost:8000`
- 默认端口：`8000`

---

## 1. 根接口 - 获取 API 信息

### GET /

获取 API 基本信息和可用端点列表。

**curl 命令：**
```bash
curl -X GET "http://localhost:8000/"
```

**Python requests：**
```python
import requests

response = requests.get("http://localhost:8000/")
print(response.json())
```

**响应示例：**
```json
{
  "message": "CKB QA Tool API",
  "version": "1.0.0",
  "endpoints": {
    "/start": "POST - Start integrated workflow",
    "/status/{task_id}": "GET - Get task status and progress",
    "/download/{task_id}": "GET - Download result files",
    "/interrupt/{task_id}": "POST - Interrupt running task",
    "/health": "GET - Health check"
  }
}
```

---

## 2. 健康检查接口

### GET /health

检查服务健康状态。

**curl 命令：**
```bash
curl -X GET "http://localhost:8000/health"
```

**Python requests：**
```python
import requests

response = requests.get("http://localhost:8000/health")
print(response.json())
```

**响应示例：**
```json
{
  "status": "healthy",
  "service": "CKB QA Tool API"
}
```

---

## 3. 启动任务接口

### POST /start

启动一个集成工作流任务（批量处理 → 数据分析 → 指标分析）。

**curl 命令：**
```bash
curl -X POST "http://localhost:8000/start" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task-12345",
    "file_path": "data/test_examples.xlsx",
    "query_selected": true,
    "chunk_selected": true,
    "answer_selected": true,
    "problem_analysis": true,
    "norm_analysis": true,
    "set_analysis": true,
    "recall_analysis": true,
    "reply_analysis": true,
    "scene_config_file": "data/scene_config.xlsx",
    "parallel_execution": true
  }'
```

**Python requests：**
```python
import requests

url = "http://localhost:8000/start"
payload = {
    "task_id": "task-12345",
    "file_path": "data/input.xlsx",
    "query_selected": True,
    "chunk_selected": True,
    "answer_selected": True,
    "problem_analysis": True,
    "norm_analysis": True,
    "set_analysis": True,
    "recall_analysis": True,
    "reply_analysis": True,
    "scene_config_file": "data/scene_config.xlsx",
    "parallel_execution": True
}

response = requests.post(url, json=payload)
print(response.json())
```

**请求参数说明：**
- `task_id` (必填): 唯一任务标识符
- `file_path` (必填): 输入 Excel 文件路径
- `query_selected` (可选, 默认 true): 是否使用查询字段（必须为 true）
- `chunk_selected` (可选, 默认 true): 是否使用参考溯源字段
- `answer_selected` (可选, 默认 true): 是否使用参考答案字段
- `problem_analysis` (可选, 默认 true): 是否启用问题侧分析
- `norm_analysis` (可选, 默认 true): 是否启用规范性分析
- `set_analysis` (可选, 默认 true): 是否启用集内集外分析
- `recall_analysis` (可选, 默认 true): 是否启用召回侧分析
- `reply_analysis` (可选, 默认 true): 是否启用回复侧分析
- `scene_config_file` (可选, 默认 "data/scene_config.xlsx"): 场景配置文件路径
- `parallel_execution` (可选, 默认 true): 是否使用并行执行

**响应示例（成功）：**
```json
{
  "code": "0000",
  "message": "Operation completed successfully",
  "success": true,
  "task_id": "task-12345"
}
```

**响应示例（失败）：**
```json
{
  "code": "9999",
  "message": "Unknown error occurred: Task task-12345 already exists",
  "success": false,
  "task_id": "task-12345"
}
```

---

## 4. 查询任务状态接口

### GET /status/{task_id}

获取指定任务的状态和进度信息。

**curl 命令：**
```bash
curl -X GET "http://localhost:8000/status/task-12345"
```

**Python requests：**
```python
import requests

task_id = "task-12345"
response = requests.get(f"http://localhost:8000/status/{task_id}")
print(response.json())
```

**响应示例（成功）：**
```json
{
  "code": "0000",
  "message": "Operation completed successfully",
  "success": true,
  "task_id": "task-12345",
  "total_progress": 65.5,
  "status": {
    "batch_status": "完成",
    "norm_status": "完成",
    "set_status": "完成",
    "recall_status": "正在进行",
    "reply_status": "未执行",
    "metrics_status": "未执行"
  },
  "progress": {
    "batch_progress": 100.0,
    "norm_progress": 100.0,
    "set_progress": 100.0,
    "recall_progress": 55.0,
    "reply_progress": 0.0,
    "metrics_progress": 0.0
  },
  "files": {
    "excel_file": "data/input_integrated_result_task-12345.xlsx",
    "json_file": "data/input_metrics_task-12345.json",
    "report_file": "data/input_质量分析报告_task-12345.md",
    "intermediate_file": "data/input_batch_result_task-12345.xlsx"
  },
  "error_message": null,
  "created_at": "2024-01-01T10:00:00",
  "updated_at": "2024-01-01T10:05:00",
  "cancelled": false
}
```

**状态值说明：**
- `未执行`: 任务尚未开始
- `正在进行`: 任务正在执行中
- `完成`: 任务已完成
- `不执行`: 任务被跳过
- `失败`: 任务执行失败
- `已中断`: 任务被中断

---

## 5. 下载文件接口

### GET /download/{task_id}

获取已完成任务的输出文件路径。

**curl 命令：**
```bash
curl -X GET "http://localhost:8000/download/task-12345"
```

**Python requests：**
```python
import requests

task_id = "task-12345"
response = requests.get(f"http://localhost:8000/download/{task_id}")
print(response.json())
```

**响应示例（成功）：**
```json
{
  "code": "0000",
  "message": "Operation completed successfully",
  "success": true,
  "excel_file": "data/input_integrated_result_task-12345.xlsx",
  "json_file": "data/input_metrics_task-12345.json",
  "report_file": "data/input_质量分析报告_task-12345.md"
}
```

**响应示例（失败 - 文件未找到）：**
```json
{
  "code": "2001",
  "message": "Input file not found: Task task-12345 has not generated output files yet",
  "success": false
}
```

**文件说明：**
- `excel_file`: 集成分析结果 Excel 文件（包含所有分析结果）
- `json_file`: 指标分析 JSON 文件（包含所有指标数据）
- `report_file`: 质量分析报告 Markdown 文件（LLM 生成的综合分析报告）

---

## 6. 中断任务接口

### POST /interrupt/{task_id}

中断一个正在运行的任务。

**curl 命令：**
```bash
curl -X POST "http://localhost:8000/interrupt/task-12345"
```

**Python requests：**
```python
import requests

task_id = "task-12345"
response = requests.post(f"http://localhost:8000/interrupt/{task_id}")
print(response.json())
```

**响应示例（成功）：**
```json
{
  "code": "0000",
  "message": "Operation completed successfully",
  "success": true,
  "excel_file": "data/input_integrated_result_task-12345.xlsx",
  "json_file": null,
  "report_file": null,
  "intermediate_file": "data/input_batch_result_task-12345.xlsx"
}
```

**响应示例（任务已取消）：**
```json
{
  "code": "0000",
  "message": "Operation completed successfully",
  "success": true,
  "excel_file": "data/input_integrated_result_task-12345.xlsx",
  "json_file": null,
  "report_file": null,
  "intermediate_file": null
}
```

---

## 完整工作流示例

以下是一个完整的工作流示例，展示如何启动任务、查询状态和下载结果：

### Python 完整示例

```python
import requests
import time
import json

# 配置
BASE_URL = "http://localhost:8000"
TASK_ID = "task-12345"
INPUT_FILE = "data/input.xlsx"

# 1. 启动任务
print("1. 启动任务...")
start_url = f"{BASE_URL}/start"
start_payload = {
    "task_id": TASK_ID,
    "file_path": INPUT_FILE,
    "query_selected": True,
    "chunk_selected": True,
    "answer_selected": True,
    "problem_analysis": True,
    "norm_analysis": True,
    "set_analysis": True,
    "recall_analysis": True,
    "reply_analysis": True,
    "parallel_execution": True
}

response = requests.post(start_url, json=start_payload)
result = response.json()
print(f"启动结果: {json.dumps(result, indent=2, ensure_ascii=False)}")

if not result.get("success"):
    print(f"启动失败: {result.get('message')}")
    exit(1)

# 2. 轮询任务状态
print("\n2. 查询任务状态...")
status_url = f"{BASE_URL}/status/{TASK_ID}"

while True:
    response = requests.get(status_url)
    status = response.json()
    
    if not status.get("success"):
        print(f"查询状态失败: {status.get('message')}")
        break
    
    total_progress = status.get("total_progress", 0)
    batch_status = status.get("status", {}).get("batch_status", "")
    metrics_status = status.get("status", {}).get("metrics_status", "")
    
    print(f"总进度: {total_progress:.1f}% | "
          f"批量处理: {batch_status} | "
          f"指标分析: {metrics_status}")
    
    # 检查是否完成
    if metrics_status == "完成" or metrics_status == "失败":
        break
    
    # 等待 2 秒后再次查询
    time.sleep(2)

# 3. 下载结果文件
print("\n3. 获取文件路径...")
download_url = f"{BASE_URL}/download/{TASK_ID}"
response = requests.get(download_url)
download_result = response.json()

if download_result.get("success"):
    print(f"Excel 文件: {download_result.get('excel_file')}")
    print(f"JSON 文件: {download_result.get('json_file')}")
    print(f"报告文件: {download_result.get('report_file')}")
else:
    print(f"获取文件路径失败: {download_result.get('message')}")
```

### Bash 完整示例

```bash
#!/bin/bash

BASE_URL="http://localhost:8000"
TASK_ID="task-12345"
INPUT_FILE="data/input.xlsx"

# 1. 启动任务
echo "1. 启动任务..."
START_RESPONSE=$(curl -s -X POST "${BASE_URL}/start" \
  -H "Content-Type: application/json" \
  -d "{
    \"task_id\": \"${TASK_ID}\",
    \"file_path\": \"${INPUT_FILE}\",
    \"query_selected\": true,
    \"chunk_selected\": true,
    \"answer_selected\": true,
    \"problem_analysis\": true,
    \"norm_analysis\": true,
    \"set_analysis\": true,
    \"recall_analysis\": true,
    \"reply_analysis\": true,
    \"parallel_execution\": true
  }")

echo "$START_RESPONSE" | jq '.'

SUCCESS=$(echo "$START_RESPONSE" | jq -r '.success')
if [ "$SUCCESS" != "true" ]; then
    echo "启动失败"
    exit 1
fi

# 2. 轮询任务状态
echo -e "\n2. 查询任务状态..."
while true; do
    STATUS_RESPONSE=$(curl -s "${BASE_URL}/status/${TASK_ID}")
    TOTAL_PROGRESS=$(echo "$STATUS_RESPONSE" | jq -r '.total_progress')
    METRICS_STATUS=$(echo "$STATUS_RESPONSE" | jq -r '.status.metrics_status')
    
    echo "总进度: ${TOTAL_PROGRESS}% | 指标分析状态: ${METRICS_STATUS}"
    
    if [ "$METRICS_STATUS" == "完成" ] || [ "$METRICS_STATUS" == "失败" ]; then
        break
    fi
    
    sleep 2
done

# 3. 下载结果文件
echo -e "\n3. 获取文件路径..."
DOWNLOAD_RESPONSE=$(curl -s "${BASE_URL}/download/${TASK_ID}")
echo "$DOWNLOAD_RESPONSE" | jq '.'
```

---

## 错误码说明

所有业务接口（除 `/` 和 `/health`）都返回统一的响应格式，包含以下字段：

- `code`: 状态码（字符串）
- `message`: 消息描述
- `success`: 是否成功（布尔值）

**常见错误码：**
- `0000`: 操作成功
- `1001`: 输入文件未配置
- `2001`: 文件未找到
- `2002`: 文件读取错误
- `2003`: 文件写入错误
- `2004`: 文件被锁定
- `6001`: 未找到对话组
- `7001-7009`: 分析相关错误
- `8001-8006`: 指标分析相关错误
- `9001`: 系统异常
- `9999`: 未知错误

完整的错误码列表请参考 `conf/error_codes.py`。

---

## 注意事项

1. **文件路径**：`file_path` 和 `scene_config_file` 需要使用服务器端的绝对路径或相对于项目根目录的相对路径。

2. **任务 ID**：每个任务必须使用唯一的 `task_id`，建议使用 UUID 或时间戳。

3. **异步执行**：`/start` 接口是异步的，任务会在后台执行。需要轮询 `/status` 接口来获取进度。

4. **文件下载**：`/download` 接口返回的是文件路径，不是文件内容。需要根据返回的路径在服务器上访问文件。

5. **任务中断**：中断任务后，已生成的文件仍然可以通过 `/download` 接口获取。

