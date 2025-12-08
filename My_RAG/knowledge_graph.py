import re
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

    def _extract_entities(self, text: str) -> Set[str]:
        """
        Extracts simple entities (Years, Capitalized Terms/Key Terms) from text.
        This serves as a heuristic Entity Extraction for financial domains.
        """
        entities = set()
        
        # Normalize text
        # text_lower = text.lower() # We keep casing for extraction logic but normalize for storage if needed
        
        # 1. Extract Years (4-digit numbers)
        # This is crucial for distinguishing financial reports from different periods.
        years = re.findall(r"\b(19|20)\d{2}\b", text)
        for y in years:
            entities.add(f"Year:{y}")

        # 2. Extract Potential Entities (English)
        # Matches words starting with letters, allowing for some special chars like & (e.g., AT&T)
        # We focus on terms that likely represent proper nouns or significant financial terms.
        # This regex mimics the one used in the 'wang' branch optimization but structured as graph nodes.
        tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9&'\-\.]*\b", text)
        
        for t in tokens:
            # Filter distinct terms (length > 2)
            if len(t) > 2 and not t.isdigit():
                 # Use lowercase for case-insensitive matching in the graph
                entities.add(f"Term:{t.lower()}")

        return entities

    def _build_graph(self):
        """Constructs the Entity-Document graph."""
        for i, chunk in enumerate(self.chunks):
            text = chunk.get("page_content", "")
            entities = self._extract_entities(text)
            for ent in entities:
                self.entity_map[ent].add(i)

    def search(self, query: str) -> Dict[int, float]:
        """
        Traverses the graph to find chunks related to entities in the query.
        Returns a dictionary of {chunk_index: score}.
        Score is currently based on the number of matched entities (Intersection).
        """
        query_entities = self._extract_entities(query)
        scores = defaultdict(float)
        
        for ent in query_entities:
            if ent in self.entity_map:
                # 1-Hop: Entity -> Chunks
                # If we had a deeper graph, we could do Entity -> Related Entity -> Chunks
                related_chunk_indices = self.entity_map[ent]
                
                # Weighting: Years are often critical constraints, give them a slight boost?
                # For now, uniform weighting is fine, as it aggregates well.
                weight = 1.0
                if ent.startswith("Year:"):
                    weight = 1.5 
                
                for idx in related_chunk_indices:
                    scores[idx] += weight
                    
        return scores
