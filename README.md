# WSM RAG 專案評估指南

本專案使用 `docker-compose` 建立一個可攜式、自包含的 RAG（Retrieval-Augmented Generation）評估環境。它包含兩個服務：
1.  **`app`**：執行主要 RAG 程式 (`My_RAG`) 和評估工具 (`rageval`) 的 Python 服務。
2.  **`ollama`**：一個專門用來跑「裁判模型 (Judge LLM)」的服務，`rageval` 會呼叫它來為 RAG 答案評分。

使用 `docker-compose` 可以確保所有協作者都有一致的執行環境，無需在本機手動安裝 Ollama 或處理複雜的網路設定。

## 執行需求 (Prerequisites)

* [Docker Desktop](https://www.docker.com/products/docker-desktop/)
* 一台有 **NVIDIA GPU** 的主機 (本專案已針對 6GB VRAM 進行優化)
* 確保 Docker Desktop 已設定使用 NVIDIA GPU (通常是預設)

## ⚡ 如何執行 (How to Run)

**只需要這一個指令。**

在專案的根目錄 (包含 `docker-compose.yml` 的地方)，打開你的終端機 (PowerShell / Terminal) 並執行：

```bash
docker-compose up --build
```

### 第一次執行會發生什麼？

1.  **Build Services:** `docker-compose` 會使用 `Dockerfile` 和 `ollama_service/Dockerfile` 分別建立 `app` 和 `ollama_service` 的映像。這個過程現在非常快，因為模型不會在 build 階段下載。
2.  **Start Services:** 啟動 `ollama_service` 和 `app` 兩個 container。
3.  **Run `entrypoint.sh` & Download Model:**
    *   `ollama_service` 容器啟動後，會執行 `entrypoint.sh` 腳本。
    *   此腳本會先啟動 Ollama 伺服器，然後**自動下載 `gemma:2b` 模型** (約 2.5GB)。
    *   模型會被下載到 `ollama_storage` 這個 Volume 中，這意味著**未來再啟動就無需重新下載**。
    *   **(請在此時保持耐心，這只需要一次)**
4.  **Run `run.sh`:** 當 `ollama_service` 準備就緒後，`app` 服務會自動開始執行 `run.sh` 腳本 (包含 RAG 預測和評估)。
5.  **Run Evaluation:** `ollama_service` 會將模型載入 VRAM，`rageval` 的評估進度條就會開始跑了。

### 未來執行

1. **重新跑一次評估 (無修改程式碼):**
   ```bash
   docker-compose up
   ```

2. **重新建置並執行 (有修改程式碼):**
   ```bash
   docker-compose up --build --force-recreate
   ```

3. **跑完自動關閉並清除所有服務:**
   ```bash
   docker-compose up --build --force-recreate --abort-on-container-exit
   ```







## 🧹 如何停止與清理

1.  **停止服務：** 在 `docker-compose up` 正在運行的終端機中，按下 `Ctrl + C`。
2.  **停止並移除 Container：** (如果服務是在背景 `-d` 執行，或你想徹底清理)
    ```bash
    docker-compose down
    ```
3.  **移除 Ollama 模型快取 (非必要)：** 如果你想刪除下載的 `gemma:2b` 模型，執行：
    ```bash
    docker-compose down -v
    ```
    (`-v` 會連同 `ollama_storage` volume 一起刪除)

## 🚀 優化 (Optimization)

### lixiang1201_2323_optimize-rag-performance

**目標：** 優化 Ollama 客戶端 (Client) 的實例化，以提高 RAG 管道的整體執行效能。

**改動內容：**

1.  **`My_RAG/main.py` (修改):**
    *   將 `ollama.Client` 的實例化邏輯和主機備援 (fallback) 邏輯從 `My_RAG/generator.py` 移至 `main.py`。
    *   此邏輯現在位於查詢處理迴圈之前，確保 `ollama.Client` 物件只會被建立一次。
    *   在 `main.py` 中，新增了 `ollama.Client` 和 `os` 的 import。
    *   `generate_answer` 函式的呼叫現在會傳遞這個已經實例化好的 `ollama_client` 物件。
    *   新增了連線失敗時的錯誤處理，如果所有主機都無法連線，會拋出 `ConnectionError`。

2.  **`My_RAG/generator.py` (修改):**
    *   移除了 `from ollama import Client` 的 import (因為 client 會從 `main.py` 傳入)。
    *   修改了 `generate_answer` 函式的簽名 (signature)，使其接受一個 `ollama_client` 物件作為參數。
    *   移除了函式內部重複建立 `ollama.Client` 物件和主機備援的邏輯。
    *   直接使用傳入的 `ollama_client` 物件來進行 `generate` 操作。

**優化效益：**
*   避免了在每個查詢中重複實例化 `ollama.Client` 物件和執行連線檢查，大幅減少了不必要的開銷。
*   提高了 RAG 管道在處理大量查詢時的執行效率。

### esdese-feature-1
**目標：** 引入混合檢索 (Hybrid Retrieval) 與重排序 (Reranking) 機制，提升檢索準確度。

**改動內容：**

1.  **Hybrid Retrieval (混合檢索):**
    *   結合 **BM25** (關鍵字檢索) 與 **Embedding Similarity** (向量語意檢索)。
    *   使用 `ollama` 的 embedding 模型 (或 `SentenceTransformer`) 生成文件與查詢的向量。
    *   透過加權平均 (`alpha` 參數) 合併兩者的分數，兼顧精確匹配與語意理解。

2.  **LLM Reranker (重排序):**
    *   在初步檢索出候選文件後，使用 LLM (如 `granite4:3b`) 對候選文件進行再一次的相關性評分。
    *   根據 LLM 的評分對結果進行重新排序，將最相關的文件排在最前面。

3.  **整合優化:**
    *   將上述功能整合進 `My_RAG/retriever.py`。
    *   配合 `lixiang1201_2323_optimize-rag-performance` 的優化，重用 `main.py` 中建立的單一 `ollama.Client` 實例，避免重複連線開銷。

### uuuu
**目標：** 吸收 `uuuu` 分支的優化項目，包含智能文件切分與精細提示工程。

**改動內容：**

1.  **智能文件切分 (`My_RAG/chunker.py`):**
    *   引入了基於句子邊界的切分方法，取代了原有的固定長度切分。
    *   新方法會根據句號、換行符等標點符號來分割文本，確保每個文件區塊（chunk）的語意完整性，對提升檢索品質有正面幫助。

2.  **精細提示工程 (`My_RAG/generator.py`):**
    *   針對中、英文設計了更詳細、更具指導性的 Prompt。
    *   新的 Prompt 明確指示 LLM 必須嚴格根據上下文回答、仔細比對公司名稱與年份等實體，並完整包含所有關鍵資訊，以提高生成答案的忠實度和準確性。

3.  **流程整合 (`My_RAG/main.py`):**
    *   調整了 `main.py` 中的函式呼叫，以兼容新的 `generator.py`（傳遞 `language` 參數）。

### wang
**目標：** 吸收 `wang` 分支的優化項目，包含新的混合檢索策略、答案引用來源以及更精準的參考資料篩選。

**改動內容：**

1.  **兩階段混合檢索 (`My_RAG/retriever.py`):**
    *   引入了新的 `HybridBM25EmbeddingRetriever`。
    *   **策略：** 首先使用 BM25 從大量文件中快速篩選出 30 個候選，然後僅對這些候選文件計算向量，最後使用向量相似度進行精確的重排序（Re-ranking）。
    *   **效益：** 這是一種高效的兩階段檢索策略，兼顧了速度與準確性。
    *   **向量模型：** 中文使用 `qwen3-embedding:0.6b`，英文使用 `embeddinggemma:300m`。(註：`nomic-embed-text` 是另一個可能表現更好的英文模型，但需另外執行 `ollama pull nomic-embed-text` 下載)。

2.  **答案引用來源 (`My_RAG/generator.py`):**
    *   修改了 Prompt，要求 LLM 在生成答案後，必須明確標示出答案是參考了哪些上下文（例如 `Sources: [1], [3]`）。
    *   這項改動大幅提升了答案的可追溯性和可信度。

3.  **句子級參考資料 (`My_RAG/main.py`):**
    *   新增了 `_select_reference_sentences` 函式。
    *   它不再將整個文件區塊（chunk）作為參考資料，而是進一步從區塊中，挑選出與問題最相關的幾個**句子**作為最終的 `references`。
    *   這使得參考資料更為精準、簡潔。

## 🚀 未來工作 (Future Work)

梳理流程
---

### 🏆 RAG 建議工作流程框架

以下是一個標準的 RAG 工作流程框架，旨在系統化地提升 RAG 系統的效能和準確性。

#### 階段 I：數據處理與索引 (Ingestion & Indexing)
這是流程的離線階段，主要目標是建立一個高品質、高效可檢索的知識庫。

| 步驟  | 工作內容              | 具體操作 (Preprocessing)                                       |
| ----- | --------------------- | ------------------------------------------------------------ |
| I-1   | 數據清洗              | **移除噪音：** 處理原始數據中的不必要格式、標籤和干擾資訊。              |
|       |                       | **正規化：** 統一文字大小寫，處理特殊符號，確保數據一致性。          |
| I-2   | 文本優化              | **語言學簡化：** 進行詞幹提取 (Stemming) 或詞形還原 (Lemmatization)；中文則進行精確分詞。 |
| I-3   | 切割 (Chunking)       | **語義分割：** 將長篇文件分割成語義完整且連貫的獨立區塊。          |
|       |                       | **策略決定：** 確定最佳 Chunk 大小（例如 512 tokens）和重疊區間（例如 50 tokens），避免語義斷裂。 |
| I-4   | 建立向量              | **語義表示：** 將每個文本 Chunk 轉換為高維向量。                |
|       |                       | **模型選擇：** 選擇適合目標語言和領域的 Embedding Model (例如 BGE-Large-zh-v1.5)。 |
| I-5   | 索引儲存              | **高效檢索：** 將 Chunk 文本及其對應向量存入向量資料庫。            |
|       |                       | **建立索引：** 在向量資料庫中建立高效的索引結構，以加速相似度搜索。       |

*匯出到試算表*

#### 階段 II：用戶查詢優化 (Query Processing)
這是流程的線上階段，核心是將用戶的原始查詢轉化為檢索系統最能理解、最有效率的形式。

| 步驟  | 工作內容              | 具體操作 (Query Rewriting)                                       |
| ----- | --------------------- | ------------------------------------------------------------ |
| II-1  | 查詢重寫              | **LLM 輔助優化：** 根據查詢的複雜度和目標，利用 LLM 進行重寫。      |
|       |                       | **Multi-query：** 生成多個不同角度的查詢，以提高檢索的覆蓋率和召回率。 |
| II-2  | 查詢轉化              | **語義向量化：** 將最終優化後的查詢轉換為向量。                    |
|       |                       | **HyDE：** 利用 LLM 生成一個假設性的答案，再將該假設答案轉化為向量進行檢索，以捕捉深層語義。 |

*匯出到試算表*

#### 階段 III：檢索與精煉 (Retrieval & Filtering)
本階段目標是從龐大的知識庫中，精準地找出與用戶查詢最相關、最準確的參考資料。

| 步驟  | 工作內容              | 具體操作                                                     |
| ----- | --------------------- | ------------------------------------------------------------ |
| III-1 | 初步檢索              | **混合搜索：** 同時運用 Sparse Search (關鍵詞匹配，例如 BM25) 和 Dense Search (向量相似度) 兩種方法。 |
|       |                       | **Top-K 篩選：** 從兩種方法中各抓取一定數量的 Top-K 候選結果。      |
| III-2 | 精煉排序 (Re-ranking) | **二次評估：** 使用獨立的、通常較小的 LLM 或專門的排序模型（Ranker Model），對初步檢索到的 Top-K 結果進行二次相關性評估。 |
|       |                       | **最佳 M 個 Chunk：** 根據語義和上下文連貫性，重新排序並選出最相關的 M 個 Chunk（M < K）。 |
| III-3 | 內容提取              | **核心資訊聚焦：** 從 M 個最佳 Chunk 中，提取出最核心、不冗餘且適合生成模型使用的資訊。 |
|       |                       | **數據清理：** 清理 Chunk 中的元數據（metadata）和任何重複內容。  |

*匯出到試算表*

#### 階段 IV：生成與輸出 (Generation & Output)
本階段的目標是確保 LLM 能夠根據用戶查詢和精煉後的上下文，生成出高品質、符合需求的答案。

| 步驟  | 工作內容              | 具體操作                                                     |
| ----- | --------------------- | ------------------------------------------------------------ |
| IV-1  | 上下文組裝            | **Prompt 構建：** 將用戶查詢、清晰的系統指令和檢索到的內容組合成最終的 Prompt。 |
|       |                       | **模板應用：** 設置標準的 Prompt 模板，將處理後的檢索內容插入指定位置。 |
| IV-2  | 窗口優化              | **高效利用：** 調整 LLM 的上下文窗口，確保在模型有限的記憶體中傳遞最關鍵資訊。 |
|       |                       | **技術應用：** 考慮實施動態截斷或上下文壓縮技術 (例如 LongLoRA 或 RAGAS 的 Context Compression)。 |
| IV-3  | 最終生成              | **高質量回答：** LLM 根據組裝好的 Prompt 生成最終答案。          |
|       |                       | **格式強制：** 確保答案嚴格遵守預設的格式限制（例如 JSON 格式）。   |
| IV-4  | 輸出                  | **呈現給用戶：** 將最終答案清晰、有效地呈現給用戶。                |


現在我們已經成功整合了混合檢索與 LLM 重排序機制，大幅提升了檢索的精準度。接下來，我們將專注於以下更進階的優化和探索：

#### 1. RAG 流程強化 (RAG Workflow Enhancements)
- **進階重排序模型 (Advanced Reranking Models):** 探索並整合更專業的重排序模型 (例如 BGE Reranker)，以進一步提升排序精準度並可能降低延遲。
- **查詢改寫與擴展 (Query Rewriting & Expansion):**
  - **Multi-Query Generation:** 利用 LLM 從單一查詢生成多個視角的問題，擴展檢索範圍。
  - **HyDE (Hypothetical Document Embeddings):** 生成假設性文件並用於檢索，捕捉查詢的語意意圖。
  - **Decomposition:** 將複雜查詢分解成可獨立回答的子問題，提升RAG處理複雜問題的能力。
  - **Step-Back Prompting:** 讓模型從具體問題回溯到更高層次的抽象概念，幫助檢索更相關的背景資訊。

#### 2. 文件處理與管理 (Document Processing & Management)
- **智能切分策略 (Intelligent Chunking Strategies):** 深入研究並實作如 `RecursiveCharacterTextSplitter` 等基於內容結構的智能切分方法，以確保資訊單元完整性，並系統性地測試 `chunk_size` 和 `chunk_overlap` 對 RAG 表現的影響。
- **知識圖譜整合 (Knowledge Graph Integration):** 探索從文件內容構建知識圖譜的可能性，並利用圖譜檢索來處理複雜、多跳的查詢。

#### 3. 生成模型與互動 (Generation & Interaction)
- **上下文視窗優化 (Context Window Optimization):** 精煉傳遞給生成模型的上下文，在有限的 Token 視窗內提供最關鍵且無冗餘的資訊，可能涉及摘要、資訊壓縮等技術。
- **動態 Prompt Engineering (Dynamic Prompt Engineering):** 開發能夠根據查詢類型、檢索結果等動態調整提示詞的機制，以引導生成模型產生更精確、符合要求的答案。
- **答案格式規範與驗證 (Answer Formatting & Validation):** 強化生成答案的格式控制與內容驗證，確保輸出的一致性與品質。

#### 4. 系統級優化與評估 (System-Level Optimization & Evaluation)
- **實時效能監控 (Real-time Performance Monitoring):** 建立 RAG 系統各環節的實時監控，識別瓶頸並進行優化。
- **持續評估框架 (Continuous Evaluation Framework):** 建立自動化的評估流程，能夠快速反饋不同優化策略的效果。
- **模型微調探索 (Model Fine-tuning Exploration):** 針對特定任務和資料集，微調嵌入模型、重排序模型或生成模型，以獲得更高的專案效能。
