#!/usr/bin/env bash
set -e

echo "=================================================="
echo " 🚀 Starting Jupyter Notebook (GitHub Codespaces)"
echo "=================================================="

# ========= 可配置项 =========
PORT=8888
IP=0.0.0.0
LOG_FILE=jupyter.log
# ===========================

# 确认 python & jupyter
echo "[INFO] Python path: $(which python)"
python --version

if ! command -v jupyter >/dev/null 2>&1; then
  echo "[ERROR] Jupyter is not installed in this environment"
  exit 1
fi

# 如果端口被占用，先杀掉
if lsof -i :"$PORT" >/dev/null 2>&1; then
  echo "[WARN] Port $PORT is already in use, killing old process..."
  lsof -ti :"$PORT" | xargs kill -9
fi

# 启动 Jupyter
echo "[INFO] Launching Jupyter Notebook..."
echo "[INFO] Logs will be written to $LOG_FILE"

nohup jupyter notebook \
  --ip="$IP" \
  --port="$PORT" \
  --no-browser \
  --NotebookApp.token='' \
  --NotebookApp.password='' \
  --NotebookApp.allow_origin='*' \
  --NotebookApp.allow_remote_access=True \
  > "$LOG_FILE" 2>&1 &

JUPYTER_PID=$!

echo "[INFO] Jupyter PID: $JUPYTER_PID"
echo "[INFO] Waiting for Jupyter to become available..."

# 等待端口起来
for i in {1..30}; do
  if lsof -i :"$PORT" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

# 输出启动日志（最近 40 行）
echo
echo "================ Jupyter Startup Logs ================"
tail -n 40 "$LOG_FILE"
echo "======================================================"

# 打印关键信息
echo
echo "✅ Jupyter Notebook started successfully!"
echo "--------------------------------------------------"
echo "📌 PID        : $JUPYTER_PID"
echo "📌 Port       : $PORT"
echo "📌 Log file   : $LOG_FILE"
echo "📌 Access URL : https://<your-codespace-name>-$PORT.githubpreview.dev"
echo "--------------------------------------------------"
echo "💡 In GitHub Codespaces:"
echo "   - Go to PORTS tab"
echo "   - Forward port $PORT"
echo "   - Set visibility to Public (if needed)"
echo "=================================================="