"""
Identity Resolution Engine
Performs:
1. Deterministic graph resolution across Exact Identifiers (Email, Phone, CRM ID).
2. Probabilistic linkage with confidence scoring across Digital Identifiers (Cookie, IDFA, Device).
3. Constructs unified canonical profile mappings and resolves fragmented touches into persistent Golden Records (Customer 360).
"""
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple

class IdentityResolutionEngine:
    def __init__(self, confidence_threshold: float = 0.70):
        self.confidence_threshold = confidence_threshold
        self.graph = nx.Graph()
        
    def build_identity_graph(self, df_identities: pd.DataFrame) -> nx.Graph:
        """
        Builds a connected bipartite/identity graph where nodes are either 
        raw identifiers (e.g. email:foo@bar.com, cookie:ck_123) or source records,
        and edges represent observed co-occurrences with confidence weights.
        """
        self.graph.clear()
        
        for _, row in df_identities.iterrows():
            id_type = row["identifier_type"]
            id_val = str(row["identifier_value"]).strip().lower()
            confidence = float(row.get("confidence_score", 1.0))
            source_sys = row.get("source_system", "unknown")
            entity_ref = row["canonical_entity_id"]
            
            # Form unique node identifier: type:val
            node_id = f"{id_type}:{id_val}"
            
            # Only connect if confidence satisfies threshold
            if confidence >= self.confidence_threshold:
                # Add node with metadata
                self.graph.add_node(node_id, id_type=id_type, id_value=id_val, source_system=source_sys)
                self.graph.add_node(entity_ref, node_type="entity")
                
                # Add edge connecting the identifier with the entity session
                self.graph.add_edge(node_id, entity_ref, weight=confidence)
                
        return self.graph

    def resolve_identities(self) -> pd.DataFrame:
        """
        Extracts connected components to resolve fragmented identities into 
        Canonical Golden IDs (UCID: Unified Customer ID).
        """
        components = list(nx.connected_components(self.graph))
        mappings = []
        
        for cluster_idx, comp in enumerate(components):
            canonical_ucid = f"UCID_{cluster_idx+1:06d}"
            
            # Find constituent entities and identifiers
            for node in comp:
                if self.graph.nodes[node].get("node_type") == "entity":
                    mappings.append({
                        "unified_customer_id": canonical_ucid,
                        "raw_entity_id": node,
                        "node_type": "entity",
                        "identifier_type": "entity_id",
                        "identifier_value": node
                    })
                else:
                    id_type = self.graph.nodes[node].get("id_type", "unknown")
                    id_val = self.graph.nodes[node].get("id_value", node)
                    mappings.append({
                        "unified_customer_id": canonical_ucid,
                        "raw_entity_id": None,
                        "node_type": "identifier",
                        "identifier_type": id_type,
                        "identifier_value": id_val
                    })
                    
        return pd.DataFrame(mappings)

if __name__ == "__main__":
    from src.ingestion.synthetic_generator import generate_cdp_dataset
    data = generate_cdp_dataset(num_customers=50)
    engine = IdentityResolutionEngine()
    engine.build_identity_graph(data["identities"])
    resolved = engine.resolve_identities()
    print("Resolved identity mappings sample:")
    print(resolved.head(10))
    print(f"Total resolved mapping rows: {len(resolved)}")
    print(f"Unique Unified Customer IDs: {resolved['unified_customer_id'].nunique()}")
