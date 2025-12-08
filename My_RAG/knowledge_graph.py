import re
import jieba.posseg as pseg
from collections import defaultdict
from typing import List, Dict, Set, Any

class SimpleKnowledgeGraph:
    """
    A lightweight Knowledge Graph implementation using an inverted index structure.
    Nodes are Entities (Terms, Years) and Documents (Chunks).
    Edges represent the occurrence of an Entity in a Chunk.
    """
    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.entity_map = defaultdict(set) # Entity -> Set[ChunkIndex]
        self._build_graph()

    def _is_contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        for ch in text:
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False

    def _extract_entities(self, text: str, is_query: bool = False) -> Set[str]:
        """
        Extracts entities from text.
        
        Args:
            text: The text to extract from.
            is_query: If True, uses looser extraction rules.
        """
        entities = set()
        
        # 1. Extract Years (4-digit numbers) - Works for both languages
        years = re.findall(r"\b(19|20)\d{2}\b", text)
        for y in years:
            entities.add(f"Year:{y}")

        # 2. Chinese Entity Extraction
        if self._is_contains_chinese(text):
            # Use jieba POS tagging to extract specific entity types
            # nt: Organization, nr: Person, ns: Location, eng: English
            words = pseg.cut(text)
            valid_pos = {'nt', 'nr', 'ns', 'eng', 'nz'} # nz: other proper nouns
            
            # Simple Chinese Stopwords (mostly to filter query noise)
            cn_stopwords = {"公司", "營收", "年報", "報告", "什麼", "多少", "為何", "如何"}
            
            for word, flag in words:
                if (flag in valid_pos or (is_query and flag in ['n', 'x'])) and len(word) > 1:
                     if word not in cn_stopwords:
                        entities.add(f"Term:{word.lower()}")
            
            # Also try to catch English terms in Chinese text using regex (often cleaner than jieba's 'eng')
            if is_query:
                 tokens = re.findall(r"\b[A-Za-z][a-zA-Z0-9&'\-\.]*\b", text)
            else:
                 tokens = re.findall(r"\b[A-Z][a-zA-Z0-9&'\-\.]*\b", text)
            
            for t in tokens:
                if len(t) > 2:
                    entities.add(f"Term:{t.lower()}")

            return entities

        # 3. English Entity Extraction (Original Logic)
        if is_query:
            # Looser regex for Query: Allow lowercase letters
            tokens = re.findall(r"\b[A-Za-z][a-zA-Z0-9&'\-\.]*\b", text)
        else:
            # Strict regex for Document Indexing: Capitalized words only
            tokens = re.findall(r"\b[A-Z][a-zA-Z0-9&'\-\.]*\b", text)
        
        # Expanded Stopwords list (Case-INsensitive checked below)
        stopwords = {
            "the", "a", "an", "in", "on", "at", "to", "for", "of", "and", "or", "but", "with", "by", 
            "from", "as", "if", "while", "where", "when", "then", "it", "this", "that", "these", "those",
            "he", "she", "they", "we", "you", "i", "is", "are", "was", "were", "be", "have", "has", "had",
            "do", "does", "did", "can", "could", "will", "would", "should", "may", "might", "must",
            "question", "answer", "context", "note", "table", "figure", "page",
            "how", "what", "which", "who", "whom", "whose", "why", "limit", "show", "tell", "me"
        }

        for t in tokens:
            t_lower = t.lower()
            # Filter distinct terms (length > 2) and skip stopwords
            if len(t) > 2 and t_lower not in stopwords and not t.isdigit():
                 # Always store as lowercase key for matching
                entities.add(f"Term:{t_lower}")

        return entities

    def _build_graph(self):
        """Constructs the Entity-Document graph."""
        for i, chunk in enumerate(self.chunks):
            text = chunk.get("page_content", "")
            # Indexing time: Strict Mode (is_query=False)
            entities = self._extract_entities(text, is_query=False)
            for ent in entities:
                self.entity_map[ent].add(i)

    def search(self, query: str) -> Dict[int, float]:
        """
        Traverses the graph to find chunks related to entities in the query.
        Returns a dictionary of {chunk_index: score}.
        Score is currently based on the number of matched entities (Intersection).
        """
        # Query time: Loose Mode (is_query=True)
        query_entities = self._extract_entities(query, is_query=True)
        scores = defaultdict(float)
        
        for ent in query_entities:
            if ent in self.entity_map:
                # 1-Hop: Entity -> Chunks
                # If we had a deeper graph, we could do Entity -> Related Entity -> Chunks
                related_chunk_indices = self.entity_map[ent]
                
                # Weighting Refinement:
                # Terms (Entities like 'TSMC') are much more discriminatory than Years (like '2023').
                # If we weight Year > Term, we get lots of noise (documents matching only the year).
                # Fix: Term = 5.0, Year = 1.0
                weight = 5.0 
                if ent.startswith("Year:"):
                    weight = 1.0 
                
                for idx in related_chunk_indices:
                    scores[idx] += weight
                    
        return scores
