import sys
import os
import json
import logging
import re
import time
from tqdm import tqdm
from collections import defaultdict
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from My_RAG.chunker import chunk_documents
from ollama import Client

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DOCS_PATH = os.path.join("dragonball_dataset", "dragonball_docs.jsonl")
OUTPUT_KG_PATH = "kg_index.json"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = "granite4:3b"

# --- [Optimization] Stopwords & Filters ---
STOPWORDS = {
    "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "but", 
    "company", "limited", "ltd", "inc", "corp", "corporation", "group", "holdings",
    "report", "annual", "financial", "statement", "result", "results", "fiscal", "year",
    "table", "figure", "note", "page", "total", "amount", "value", "rate", "number",
    "公司", "有限公司", "股份", "集團", "報告", "年度", "財務", "報表", "金額", "合計"
}

def clean_entity(ent: str) -> str:
    """Cleans and validates an entity string. Returns None if invalid."""
    ent = ent.strip().lower()
    # 1. Length check
    if len(ent) < 2: return None
    # 2. Stopwords check
    if ent in STOPWORDS: return None
    # 3. Numeric check (pure numbers or currency-like)
    if re.match(r'^[\d.,%]+$', ent): return None # "2023", "1,000", "50%"
    # 4. Remove generic prefixes if present (LLM sometimes adds "the ...")
    if ent.startswith("the "): ent = ent[4:]
    
    return ent

# --- LLM Extraction Logic ---
def llm_extract_entities(text_chunk: str, client: Client, model: str) -> List[str]:
    # Optimized Prompt: Ask for specific types to reduce noise
    prompt = f"""Identify the key entities in the text: Company Names (e.g., TSMC), Locations, and Product Names.
Ignore generic terms (e.g., report, year, profit).
Output comma-separated values only.
Text: {text_chunk}
Entities:"""
    try:
        response = client.generate(model=model, prompt=prompt, stream=False)
        content = response.get("response", "").strip()
        raw_entities = [e.strip() for e in content.split(',')]
        
        valid_entities = []
        for e in raw_entities:
            cleaned = clean_entity(e)
            if cleaned:
                valid_entities.append(cleaned)
        return valid_entities
    except Exception as e:
        return []

# --- Regex Extraction Logic ---
def extract_generic_properties(content: str, domain: str = None) -> List[str]:
    entities = []
    if domain:
        entities.append(f"Domain:{domain}")
    
    # Locations
    location_pattern = re.compile(r"注册地为(位于)?(.*?)[，,。]")
    location_match = location_pattern.search(content)
    if location_match and location_match.group(2):
        location = location_match.group(2).strip()
        if len(location) < 15 and ' ' not in location:
            entities.append(f"Loc:{location}")
        
    # Years (Strict 4-digit)
    years = re.findall(r"\b(?:19|20)\d{2}\b", content)
    for y in years:
        entities.append(f"Year:{y}")
            
    return entities

def main():
    logging.info(f"--- Starting Optimized KG Build (No Pandas, LLM: {LLM_MODEL}) ---")
    
    # 1. Connect to Ollama
    use_llm = False
    try:
        client = Client(host=OLLAMA_HOST)
        client.show(LLM_MODEL) 
        logging.info(f"Connected to Ollama. Using model: {LLM_MODEL}")
        use_llm = True
    except Exception as e:
        logging.warning(f"Ollama connection issue: {e}. Building with Regex ONLY.")

    # 2. Load Docs (Standard JSONL - Matching Runtime Order)
    docs_list = []
    try:
        if not os.path.exists(DOCS_PATH):
            raise FileNotFoundError(f"Documents file not found: {DOCS_PATH}")
        
        logging.info("Loading documents (standard line-by-line)...")
        with open(DOCS_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    docs_list.append(json.loads(line))
        logging.info(f"Loaded {len(docs_list)} documents.")
        
    except Exception as e:
        logging.error(f"Data loading failed: {e}")
        sys.exit(1)

    # 3. Chunking
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 300
    logging.info(f"Chunking documents (Size={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP})...")
    chunks = chunk_documents(docs_list, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    logging.info(f"Generated {len(chunks)} chunks.")

    # 4. Build Index
    entity_map = defaultdict(set)
    logging.info("Extracting entities...")
    start_time = time.time()
    
    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Processing Chunks"):
        content = chunk.get("page_content", "")
        meta = chunk.get("metadata", {})
        domain = meta.get("domain") or chunk.get("domain")
        
        # A. Regex (High Precision)
        regex_ents = extract_generic_properties(content, domain)
        for ent in regex_ents:
            entity_map[ent].add(i)
            
        # B. LLM (Semantic Recall)
        if use_llm and len(content) > 50:
             try:
                llm_ents = llm_extract_entities(content, client, LLM_MODEL)
                for ent in llm_ents:
                     # Check if it looks like a year (avoid duplicate/wrong weight)
                     if re.match(r"^(19|20)\d{2}$", ent):
                         entity_map[f"Year:{ent}"].add(i)
                     else:
                         entity_map[f"Term:{ent}"].add(i)
             except Exception:
                 pass
    
    # 5. Initialize Baseline Generic KG (Jieba)
    from My_RAG.knowledge_graph import SimpleKnowledgeGraph
    logging.info("Merging with baseline Jieba extraction...")
    kg = SimpleKnowledgeGraph(chunks) 
    
    # Merge our findings
    for ent, refs in entity_map.items():
        for ref in refs:
            kg.entity_map[ent].add(ref)
            
    # 6. Save
    logging.info(f"Final Entity Count: {len(kg.entity_map)}")
    logging.info(f"Saving index to {OUTPUT_KG_PATH}...")
    kg.save(OUTPUT_KG_PATH)
    
    elapsed = time.time() - start_time
    logging.info(f"Build complete in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
