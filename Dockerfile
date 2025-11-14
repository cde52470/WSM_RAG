# 使用官方 Python 3.9 映像
FROM python:3.9-slim

# 在容器中設定工作目錄
WORKDIR /app

# 複製 requirements.txt 並安裝相依套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製並安裝 rageval/evaluation 的相依套件
COPY rageval/evaluation/requirements.txt ./evaluation-requirements.txt
RUN pip install --no-cache-dir -r evaluation-requirements.txt

# 複製專案內所有檔案到工作目錄
COPY . .

# --- [終極修正] 移除 Windows 的換行符號 (CRLF) 和 BOM ---
# 1. 移除 Windows (CRLF) 換行符號
RUN sed -i 's/\r$//' run.sh

# 2. 移除 UTF-8 BOM (Byte Order Mark) (如果存在)
RUN sed -i '1s/^\xEF\xBB\xBF//' run.sh
# --------------------------------------------------------

# [修正] 賦予兩個腳本執行權限
# 這必須在 COPY 檔案之後才能執行
RUN chmod +x run.sh
RUN chmod +x wait_and_run.sh

# 設定容器啟動時要執行的預設指令，先執行延遲腳本
CMD ["/bin/sh", "./wait_and_run.sh"]