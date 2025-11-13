#!/bin/sh
# 1. 在背景啟動 Ollama 伺服器
ollama serve &
# 2. 抓取伺服器的 PID
PID=$!

echo "Ollama server started in background (PID: $PID)..."
echo "Waiting for server to be ready (up to 40s)..."

# 3. 輪詢檢查伺服器是否就緒 (最多 20 次，每次 2 秒)
attempts=0
max_attempts=20
until $(curl --output /dev/null --silent --fail http://localhost:11434/api/tags); do
    if [ $attempts -ge $max_attempts ]; then
        echo "Ollama server failed to start."
        kill $PID
        exit 1
    fi
    printf '.'
    attempts=$((attempts+1))
    sleep 2
done

echo "\nOllama server is ready."
echo "Pulling gemma:2b model (this will only happen if it's missing)..."

# 4. 伺服器已就緒，現在才 pull 模型
#    (這次會把模型 pull 到掛載的 Volume 中)
ollama pull gemma:2b

echo "Model pull complete. Server is running."

# 5. 讓 container 保持存活 (等待背景的伺服器)
wait $PID