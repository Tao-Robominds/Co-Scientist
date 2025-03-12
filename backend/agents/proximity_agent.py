from typing import Dict, List, Any, Optional, Tuple
import uuid
from langchain_openai import ChatOpenAI

class ProximityAgent:
    """
    The Proximity agent computes similarity between hypotheses for clustering
    and efficient exploration of the hypothesis space.
    """
    
    SIMILARITY_PROMPT = """You are the Proximity agent in the AI Co-Scientist system. Your role is to analyze the similarity between research hypotheses addressing this goal:

"{research_goal}"

Compare the following pair of hypotheses:

Hypothesis A:
Title: {hypothesis_a_title}
Description:
{hypothesis_a_description}

Hypothesis B:
Title: {hypothesis_b_title}
Description:
{hypothesis_b_description}

Analyze their similarity across these dimensions:
1. Core Concepts (How similar are the fundamental ideas?)
2. Approach/Methodology (Do they use similar methods?)
3. Target Outcomes (Are they trying to achieve similar results?)
4. Implementation Strategy (How similar are their practical approaches?)
5. Resource Requirements (Do they need similar resources?)

For each dimension:
1. Identify specific points of overlap
2. Note key differences
3. Rate similarity on a scale of 0-1 (0 = completely different, 1 = identical)

Format your response as:
SIMILARITY_SCORES:
Core Concepts: [0-1]
Approach: [0-1]
Outcomes: [0-1]
Implementation: [0-1]
Resources: [0-1]

OVERALL_SIMILARITY: [0-1]

ANALYSIS:
[Your detailed comparison]
"""

    def __init__(self, research_goal: str, hypotheses: List[Dict[str, Any]], 
                 proximity_graph: Dict[str, Any], context_memory: Dict[str, Any]):
        """
        Initialize the Proximity agent.
        
        Args:
            research_goal: The scientist's research goal
            hypotheses: List of hypotheses to analyze
            proximity_graph: Current proximity graph state
            context_memory: The context memory containing system state
        """
        self.research_goal = research_goal
        self.hypotheses = hypotheses
        self.proximity_graph = proximity_graph
        self.context_memory = context_memory
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    def perceiver(self, hypothesis_a: Dict[str, Any], hypothesis_b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the context for similarity analysis between two hypotheses.
        """
        return {
            "research_goal": self.research_goal,
            "hypothesis_a_title": hypothesis_a.get("title", "Untitled"),
            "hypothesis_a_description": hypothesis_a.get("description", ""),
            "hypothesis_b_title": hypothesis_b.get("title", "Untitled"),
            "hypothesis_b_description": hypothesis_b.get("description", "")
        }
    
    def actor(self) -> Dict[str, Any]:
        """
        Compute similarity between hypotheses and update the proximity graph.
        """
        # Initialize or get existing edges
        edges = self.proximity_graph.get("edges", [])
        
        # Track new comparisons
        new_edges = []
        comparisons_made = 0
        max_comparisons = 20  # Limit number of comparisons per round
        
        # Get pairs of hypotheses to compare
        pairs = self._select_comparison_pairs()
        
        for hypothesis_a, hypothesis_b in pairs[:max_comparisons]:
            # Skip if we already have this edge
            edge_exists = any(
                (e["source"] == hypothesis_a["id"] and e["target"] == hypothesis_b["id"]) or
                (e["source"] == hypothesis_b["id"] and e["target"] == hypothesis_a["id"])
                for e in edges
            )
            
            if edge_exists:
                continue
            
            # Prepare context for similarity analysis
            context = self.perceiver(hypothesis_a, hypothesis_b)
            
            # Run similarity analysis
            messages = [
                {
                    "role": "system",
                    "content": self.SIMILARITY_PROMPT.format(**context)
                }
            ]
            
            response = self.model.invoke(messages)
            
            # Parse similarity scores
            similarity = self._parse_similarity_result(response.content)
            
            if similarity is not None:
                # Create edge with similarity score
                edge = {
                    "id": str(uuid.uuid4()),
                    "source": hypothesis_a["id"],
                    "target": hypothesis_b["id"],
                    "similarity": similarity,
                    "created_at": "now"  # This would be replaced with actual timestamp
                }
                
                new_edges.append(edge)
                comparisons_made += 1
        
        # Update proximity graph
        updated_edges = edges + new_edges
        updated_proximity_graph = {
            "edges": updated_edges,
            "last_update": "now"  # This would be replaced with actual timestamp
        }
        
        # Update context memory
        updated_context_memory = self.context_memory.copy()
        if "agent_states" not in updated_context_memory:
            updated_context_memory["agent_states"] = {}
        
        updated_context_memory["agent_states"]["proximity"] = {
            "last_analysis": {
                "comparisons_made": comparisons_made,
                "new_edges": len(new_edges)
            }
        }
        
        return {
            "updated_proximity_graph": updated_proximity_graph,
            "updated_context_memory": updated_context_memory
        }
    
    def _select_comparison_pairs(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Select pairs of hypotheses for similarity comparison.
        Prioritizes comparing new hypotheses and those without many existing edges.
        """
        pairs = []
        existing_edges = self.proximity_graph.get("edges", [])
        
        # Count existing edges for each hypothesis
        edge_counts = {}
        for edge in existing_edges:
            edge_counts[edge["source"]] = edge_counts.get(edge["source"], 0) + 1
            edge_counts[edge["target"]] = edge_counts.get(edge["target"], 0) + 1
        
        # Sort hypotheses by number of existing edges (ascending)
        sorted_hypotheses = sorted(
            self.hypotheses,
            key=lambda h: edge_counts.get(h.get("id"), 0)
        )
        
        # Create pairs prioritizing hypotheses with fewer edges
        for i, h1 in enumerate(sorted_hypotheses):
            for h2 in sorted_hypotheses[i+1:]:
                # Check if edge already exists
                edge_exists = any(
                    (e["source"] == h1["id"] and e["target"] == h2["id"]) or
                    (e["source"] == h2["id"] and e["target"] == h1["id"])
                    for e in existing_edges
                )
                
                if not edge_exists:
                    pairs.append((h1, h2))
        
        return pairs
    
    def _parse_similarity_result(self, content: str) -> Optional[float]:
        """
        Parse the LLM response to extract similarity scores.
        """
        try:
            # Extract overall similarity score
            for line in content.split('\n'):
                if line.startswith('OVERALL_SIMILARITY:'):
                    score = float(line.split(':')[1].strip())
                    return min(1.0, max(0.0, score))  # Ensure score is between 0 and 1
        except (ValueError, IndexError):
            pass
        
        return None
    
    @classmethod
    def create(cls, research_goal: str, hypotheses: List[Dict[str, Any]], 
               proximity_graph: Dict[str, Any], context_memory: Dict[str, Any]) -> 'ProximityAgent':
        """Factory method to create a ProximityAgent instance"""
        return cls(research_goal, hypotheses, proximity_graph, context_memory) 