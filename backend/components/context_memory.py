from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime

class ContextMemory:
    """
    A persistent context memory system for storing and retrieving agent states.
    Enables iterative scientific reasoning over long time horizons.
    """
    
    def __init__(self, storage_path: str = "context_memory", research_id: Optional[str] = None):
        """
        Initialize the context memory.
        
        Args:
            storage_path: Directory path for storing memory
            research_id: Unique identifier for this research project
        """
        self.storage_path = storage_path
        self.research_id = research_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.memory_file = os.path.join(storage_path, f"{self.research_id}_memory.json")
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize memory if it doesn't exist
        if not os.path.exists(self.memory_file):
            self._init_memory()
    
    def _init_memory(self):
        """Initialize the memory file with default structure."""
        default_memory = {
            "research_id": self.research_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "research_goal": "",
            "hypotheses": [],
            "reviews": [],
            "tournament_state": {"rankings": {}},
            "proximity_graph": {"edges": []},
            "iterations": 0,
            "statistics": {
                "hypotheses_generated": 0,
                "hypotheses_reviewed": 0,
                "tournament_matches": 0,
                "hypotheses_evolved": 0
            },
            "agent_states": {
                "supervisor": {},
                "generation": {},
                "reflection": {},
                "ranking": {},
                "evolution": {},
                "proximity": {},
                "meta_review": {}
            }
        }
        
        self._save_memory(default_memory)
    
    def _load_memory(self) -> Dict[str, Any]:
        """Load the memory from file."""
        try:
            with open(self.memory_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is corrupted, initialize it
            self._init_memory()
            with open(self.memory_file, 'r') as f:
                return json.load(f)
    
    def _save_memory(self, memory: Dict[str, Any]):
        """Save the memory to file."""
        # Update the timestamp
        memory["updated_at"] = datetime.now().isoformat()
        
        with open(self.memory_file, 'w') as f:
            json.dump(memory, f, indent=2)
    
    def get_full_memory(self) -> Dict[str, Any]:
        """Get the entire memory state."""
        return self._load_memory()
    
    def update_full_memory(self, memory: Dict[str, Any]):
        """Update the entire memory state."""
        self._save_memory(memory)
    
    def get_research_goal(self) -> str:
        """Get the research goal."""
        memory = self._load_memory()
        return memory.get("research_goal", "")
    
    def set_research_goal(self, goal: str):
        """Set the research goal."""
        memory = self._load_memory()
        memory["research_goal"] = goal
        self._save_memory(memory)
    
    def get_hypotheses(self) -> List[Dict[str, Any]]:
        """Get all hypotheses."""
        memory = self._load_memory()
        return memory.get("hypotheses", [])
    
    def add_hypotheses(self, new_hypotheses: List[Dict[str, Any]]):
        """Add new hypotheses to memory."""
        memory = self._load_memory()
        
        # Update statistics
        memory["statistics"]["hypotheses_generated"] += len(new_hypotheses)
        
        # Add hypotheses
        memory["hypotheses"].extend(new_hypotheses)
        self._save_memory(memory)
    
    def get_reviews(self) -> List[Dict[str, Any]]:
        """Get all hypothesis reviews."""
        memory = self._load_memory()
        return memory.get("reviews", [])
    
    def add_reviews(self, new_reviews: List[Dict[str, Any]]):
        """Add new reviews to memory."""
        memory = self._load_memory()
        
        # Update statistics
        memory["statistics"]["hypotheses_reviewed"] += len(new_reviews)
        
        # Add reviews
        memory["reviews"].extend(new_reviews)
        self._save_memory(memory)
    
    def get_tournament_state(self) -> Dict[str, Any]:
        """Get the current tournament state."""
        memory = self._load_memory()
        return memory.get("tournament_state", {"rankings": {}})
    
    def update_tournament_state(self, tournament_state: Dict[str, Any], matches_count: int = 0):
        """Update the tournament state."""
        memory = self._load_memory()
        
        # Update statistics
        memory["statistics"]["tournament_matches"] += matches_count
        
        # Update tournament state
        memory["tournament_state"] = tournament_state
        self._save_memory(memory)
    
    def get_proximity_graph(self) -> Dict[str, Any]:
        """Get the proximity graph for hypotheses."""
        memory = self._load_memory()
        return memory.get("proximity_graph", {"edges": []})
    
    def update_proximity_graph(self, proximity_graph: Dict[str, Any]):
        """Update the proximity graph."""
        memory = self._load_memory()
        memory["proximity_graph"] = proximity_graph
        self._save_memory(memory)
    
    def get_agent_state(self, agent_type: str) -> Dict[str, Any]:
        """Get the state for a specific agent."""
        memory = self._load_memory()
        return memory.get("agent_states", {}).get(agent_type, {})
    
    def update_agent_state(self, agent_type: str, state: Dict[str, Any]):
        """Update the state for a specific agent."""
        memory = self._load_memory()
        
        if "agent_states" not in memory:
            memory["agent_states"] = {}
        
        memory["agent_states"][agent_type] = state
        self._save_memory(memory)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get the current statistics."""
        memory = self._load_memory()
        return memory.get("statistics", {})
    
    def increment_iteration(self):
        """Increment the iteration counter."""
        memory = self._load_memory()
        memory["iterations"] += 1
        self._save_memory(memory)
    
    def get_iteration(self) -> int:
        """Get the current iteration."""
        memory = self._load_memory()
        return memory.get("iterations", 0) 