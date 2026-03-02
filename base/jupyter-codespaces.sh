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

echo "[INFO] Python path: $(which python)"
python --version

if ! command -v jupyter >/dev/null 2>&1; then
  echo "[ERROR] Jupyter is not installed in this environment"
  exit 1
fi

# 如果端口被占用，先杀掉
if lsof -i :"$PORT" >/dev/null 2>&1; then
  echo "[WARN] Port $PORT is already in use, killing old process..."
  lsof -ti :"$PORT" | xargs kill -9 || true
  sleep 2
fi

echo "[INFO] Launching Jupyter Notebook in background..."
echo "[INFO] Logs will be written to $LOG_FILE"

# 删除旧日志
rm -f "$LOG_FILE"

# 后台启动（适配 Jupyter Server 2.x）
jupyter notebook \
  --ip="$IP" \
  --port="$PORT" \
  --no-browser \
  --ServerApp.token='' \
  --ServerApp.password='' \
  --ServerApp.allow_origin='*' \
  --ServerApp.allow_remote_access=True \
  > "$LOG_FILE" 2>&1 &

JUPYTER_PID=$!
disown

echo "[INFO] Jupyter PID: $JUPYTER_PID"

# 等待端口就绪
echo "[INFO] Waiting for Jupyter to become available..."
for i in {1..30}; do
  if lsof -i :"$PORT" >/dev/null 2>&1; then
    echo "[INFO] Port $PORT is now open!"
    break
  fi
  sleep 1
done

# 再次确认进程是否还在
if ! ps -p "$JUPYTER_PID" >/dev/null 2>&1; then
  echo "[ERROR] Jupyter process exited unexpectedly!"
  echo "-------- Full Log --------"
  cat "$LOG_FILE"
  exit 1
fi

# 输出最近 20 行日志
echo
echo "================ Jupyter Startup Logs (Last 20 lines) ================"
tail -n 20 "$LOG_FILE"
echo "======================================================================="

# 关键信息提示
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

echo "[INFO] Jupyter is running in background."
echo "[INFO] To watch logs: tail -f $LOG_FILE"