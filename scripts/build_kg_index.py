import sys
import os
import json
import logging
import re
import time
import pandas as pd
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
OLLAMA_HOST = "http://localhost:11434" # Adjust if running in Docker/WSL
# Use the strongest available model. Defaulting to granite4:3b as per previous context.
LLM_MODEL = "granite4:3b" 

# --- LLM Extraction Logic ---
def llm_extract_entities(text_chunk: str, client: Client, model: str) -> List[str]:
    """
    Uses LLM to extract key entities (Subjects/Objects) from the text.
    Returns a list of entity strings.
    """
    prompt = f"""Extract all main entities (Company names, Financial terms, Locations, Amounts) from the text below. 
Output ONLY the entities separated by commas. Do not include labels or other text.
Text: {text_chunk}
Entities:"""
    
    try:
        response = client.generate(model=model, prompt=prompt, stream=False)
        content = response.get("response", "").strip()
        # Simple parsing logic
        entities = [e.strip() for e in content.split(',')]
        # Basic cleaning
        return [e for e in entities if len(e) > 1 and not e.isdigit()]
    except Exception as e:
        logging.warning(f"LLM extraction failed: {e}")
        return []

# --- Regex Extraction Logic (From User's Optimization) ---
def extract_generic_properties(content: str, domain: str = None) -> List[str]:
    """Extracts regex-based entities (Locations, Years, Domain)."""
    entities = []
    
    # 1. IS_IN_DOMAIN (Treat Domain as an entity)
    if domain and pd.notna(domain):
        entities.append(f"Domain:{domain}")
    
    # 2. LOCATED_AT (模式: "注册地为 [地点]")
    # Modified to capture just the location entity
    location_pattern = re.compile(r"注册地为(位于)?(.*?)[，,。]")
    location_match = location_pattern.search(content)
    if location_match and location_match.group(2):
        location = location_match.group(2).strip()
        if len(location) < 15 and ' ' not in location:
            entities.append(f"Loc:{location}") # Prefix for clarity
        
    # 3. REPORTED_FOR (模式: YYYY年度 或 YYYY年MM月)
    # Modified to generic regex capture all years
    years = re.findall(r"\b(?:19|20)\d{2}\b", content)
    for y in years:
        entities.append(f"Year:{y}")
            
    return entities

def main():
    logging.info(f"--- Starting Advanced KG Build (Pandas + Regex + LLM: {LLM_MODEL}) ---")
    
    # 1. Connect to Ollama
    try:
        client = Client(host=OLLAMA_HOST)
        client.show(LLM_MODEL) # Check if model exists
        logging.info(f"Connected to Ollama. Using model: {LLM_MODEL}")
        use_llm = True
    except Exception as e:
        logging.warning(f"Could not connect to Ollama or model not found: {e}. Falling back to Regex only.")
        use_llm = False

    # 2. Load Docs with Pandas
    try:
        if not os.path.exists(DOCS_PATH):
            raise FileNotFoundError(f"Documents file not found: {DOCS_PATH}")
        
        logging.info("Loading documents with Pandas...")
        # Read JSONL directly into DataFrame
        df_docs = pd.read_json(DOCS_PATH, lines=True)
        logging.info(f"Loaded {len(df_docs)} documents.")
        
    except Exception as e:
        logging.error(f"Data loading failed: {e}")
        sys.exit(1)

    # 3. Convert DataFrame back to list of dicts for Chunker (compatibility)
    #   Ideally we'd rewrite chunker to accept DF, but reusing existing logic is safer.
    docs_list = df_docs.to_dict('records')
    
    # 4. Chunking (MUST match Runtime logic)
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 300
    logging.info(f"Chunking documents (Size={CHUNK_SIZE}, Overlap={CHUNK_OVERLAP})...")
    chunks = chunk_documents(docs_list, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    logging.info(f"Generated {len(chunks)} chunks.")

    # 5. Build Entity Index (The "Cheat Sheet")
    # Structure: Entity -> Set of Chunk Indices
    entity_map = defaultdict(set)
    
    logging.info("Extracting entities from chunks...")
    start_time = time.time()
    
    # Iterate over chunks with progress bar
    for i, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Processing Chunks"):
        content = chunk.get("page_content", "")
        # Metadata access might vary depending on how chunker preserves it.
        # Assuming metadata is preserved in 'metadata' key or top-level.
        meta = chunk.get("metadata", {})
        domain = meta.get("domain") or chunk.get("domain")
        
        # A. Regex Extraction (Fast)
        regex_ents = extract_generic_properties(content, domain)
        for ent in regex_ents:
            entity_map[ent].add(i)
            
        # B. LLM Extraction (Slow but Smart)
        # Only run on chunks with substantial content
        if use_llm and len(content) > 50:
             # Optimization: Limit LLM calls for testing? 
             # No, user said "time is not an issue". We run FULL POWER.
             
             # But let's be careful not to choke on too many generic words.
             # We rely on existing Jieba extraction in knowledge_graph.py usually.
             # Here we AUGMENT it. 
             
             # Actually, simpler strategy:
             # Just use the SimpleKnowledgeGraph logical extractors PLUS Regex.
             # Adding full LLM for 1800 chunks * 3s = 1.5 hours. Acceptable? User said yes.
             
             try:
                llm_ents = llm_extract_entities(content, client, LLM_MODEL)
                for ent in llm_ents:
                     # Clean and add
                     ent_clean = ent.strip().lower()
                     if len(ent_clean) > 1:
                         entity_map[f"Term:{ent_clean}"].add(i)
             except Exception:
                 pass
        
        # C. Include Jieba Extraction (from original logic)
        # We can simulate what SimpleKnowledgeGraph does to ensure we don't LOSE those entities.
        # Re-using the regex/jieba logic from knowledge_graph.py would be best, 
        # but importing the class and using internal methods is cleaner.
    
    # --- Integration Strategy ---
    # Instead of rewriting all extraction logic here, lay the LLM results ON TOP OF 
    # the standard extraction.
    # So we initialize the standard KG first, then inject our LLM findings.
    
    from My_RAG.knowledge_graph import SimpleKnowledgeGraph
    
    logging.info("Initializing baseline SimpleKnowledgeGraph (Jieba+Regex)...")
    kg = SimpleKnowledgeGraph(chunks) # Builds standard index
    
    logging.info("Augmenting with LLM & Advanced Regex...")
    # Now merge our findings into kg.entity_map
    for ent, refs in entity_map.items():
        for ref in refs:
            kg.entity_map[ent].add(ref)
            
    # 6. Save
    logging.info(f"Final Entity Count: {len(kg.entity_map)}")
    logging.info(f"Saving optimized index to {OUTPUT_KG_PATH}...")
    kg.save(OUTPUT_KG_PATH)
    
    elapsed = time.time() - start_time
    logging.info(f"Build complete in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    main()
