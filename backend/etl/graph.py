"""
backend/etl/graph.py
Knowledge Graph Engine for Multi-Layer GraphRAG.
Handles extraction of Entity-Relationship triplets and NetworkX graph storage.
"""
import networkx as nx
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
from backend.core.config import get_settings
from backend.core.logging_config import get_logger

log = get_logger(__name__)
settings = get_settings()

GRAPH_PATH = settings.data_dir / "knowledge_graph.gpickle"

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.load()

    def load(self):
        if GRAPH_PATH.exists():
            try:
                with open(GRAPH_PATH, "rb") as f:
                    self.graph = pickle.load(f)
                log.info("Knowledge Graph loaded: %d nodes, %d edges", 
                         self.graph.number_of_nodes(), self.graph.number_of_edges())
            except Exception as e:
                log.error("Failed to load graph: %s", e)
                self.graph = nx.MultiDiGraph()
        else:
            log.info("No existing Knowledge Graph found. Starting fresh.")
            self.graph = nx.MultiDiGraph()

    def save(self):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        with open(GRAPH_PATH, "wb") as f:
            pickle.dump(self.graph, f)
        log.info("Knowledge Graph saved to %s", GRAPH_PATH)

    def add_triplet(self, entity1: str, relation: str, entity2: str, metadata: Dict[str, Any]):
        e1 = entity1.strip().upper()
        e2 = entity2.strip().upper()
        rel = relation.strip().upper()
        
        self.graph.add_node(e1, type="ENTITY")
        self.graph.add_node(e2, type="ENTITY")
        self.graph.add_edge(e1, e2, relation=rel, **metadata)

    def extract_triplets_from_text(self, text: str, chunk_metadata: Dict[str, Any]) -> int:
        """
        Uses LLM to extract Entity-Relation-Entity triplets.
        Returns number of triplets extracted.
        """
        if not text or len(text.strip()) < 20:
            return 0
            
        prompt = (
            "You are a Knowledge Graph specialized in Indian Government Regulations.\n"
            "Extract all unique entities and their relationships from the text below.\n"
            "Output ONLY a JSON list of objects: [{\"e1\": \"Entity1\", \"rel\": \"Relationship\", \"e2\": \"Entity2\"}].\n"
            "Keep entity names short and concise.\n\n"
            f"Text: {text[:2000]}\n\n"
            "JSON Output:"
        )

        try:
            full_response = ""
            if settings.use_local_llm:
                from backend.rag.llm import _ollama_generate_stream
                for token in _ollama_generate_stream(prompt, model=settings.local_llm_model):
                    full_response += token
            else:
                from backend.rag.llm import _gemini_generate
                full_response = _gemini_generate(prompt)
            
            # Clean response to find JSON
            start = full_response.find("[")
            end = full_response.rfind("]") + 1
            if start == -1 or end == 0:
                return 0
            
            triplets = json.loads(full_response[start:end])
            count = 0
            for t in triplets:
                if all(k in t for k in ("e1", "rel", "e2")):
                    self.add_triplet(t["e1"], t["rel"], t["e2"], chunk_metadata)
                    count += 1
            return count
        except Exception as e:
            log.error("Extraction error: %s", e)
            return 0

    def query_graph(self, entities: List[str], max_hops: int = 2) -> List[str]:
        """
        Finds context by traversing the graph from seed entities.
        """
        context_snippets = []
        seen_edges = set()
        
        for entity in entities:
            entity_node = entity.strip().upper()
            if not self.graph.has_node(entity_node):
                continue
                
            # Get subgraph within max_hops
            subgraph = nx.ego_graph(self.graph, entity_node, radius=max_hops, center=True, undirected=True)
            
            for u, v, k, data in subgraph.edges(data=True, keys=True):
                edge_id = tuple(sorted((u, v))) + (data.get("relation"),)
                if edge_id in seen_edges:
                    continue
                seen_edges.add(edge_id)
                
                snippet = f"Relationship: {u} -> {data.get('relation')} -> {v} (Source: {data.get('file_name', 'Unknown')})"
                context_snippets.append(snippet)
                
        return context_snippets

# Singleton instance
_kg_instance = None

def get_kg() -> KnowledgeGraph:
    global _kg_instance
    if _kg_instance is None:
        _kg_instance = KnowledgeGraph()
    return _kg_instance
