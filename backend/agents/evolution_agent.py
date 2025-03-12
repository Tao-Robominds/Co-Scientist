from typing import Dict, List, Any, Optional
import uuid
from langchain_openai import ChatOpenAI

class EvolutionAgent:
    """
    The Evolution agent refines and improves top-ranked hypotheses through
    iterative improvement, synthesis, and creative exploration.
    """
    
    EVOLUTION_PROMPT = """You are the Evolution agent in the AI Co-Scientist system. Your role is to refine and improve promising research hypotheses addressing this goal:

"{research_goal}"

You will evolve the following hypotheses through these strategies:
1. Synthesis: Combine elements from multiple hypotheses
2. Specialization: Focus and elaborate on specific aspects
3. Generalization: Broaden applicability or scope
4. Cross-pollination: Apply ideas from other fields
5. Constraint relaxation: Challenge assumptions
6. Mechanism elaboration: Detail specific processes

For each hypothesis, consider:
1. Core strengths to preserve and enhance
2. Weaknesses to address
3. Opportunities for novel combinations
4. Potential for practical implementation

Input Hypotheses:
{hypotheses_text}

For each hypothesis, generate 1-2 evolved variants that:
1. Maintain the valuable core ideas
2. Address identified limitations
3. Incorporate novel improvements
4. Enhance practical feasibility

Format each evolved hypothesis with:
1. Title: Clear and descriptive
2. Parent: ID of the original hypothesis
3. Evolution Strategy: Which approach(es) were used
4. Description: Detailed explanation
5. Improvements: List specific enhancements
6. Validation: How to test the improvements

Be creative while maintaining scientific rigor.
"""

    def __init__(self, research_goal: str, hypotheses: List[Dict[str, Any]], context_memory: Dict[str, Any]):
        """
        Initialize the Evolution agent.
        
        Args:
            research_goal: The scientist's research goal
            hypotheses: List of hypotheses to evolve (typically top-ranked ones)
            context_memory: The context memory containing system state
        """
        self.research_goal = research_goal
        self.hypotheses = hypotheses
        self.context_memory = context_memory
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.7)  # Higher temperature for creativity
    
    def perceiver(self) -> Dict[str, Any]:
        """
        Prepare the context for hypothesis evolution.
        """
        # Format hypotheses for evolution
        hypotheses_text = ""
        for i, hypothesis in enumerate(self.hypotheses, 1):
            hypotheses_text += f"\nHypothesis {i} (ID: {hypothesis.get('id', 'unknown')}):\n"
            hypotheses_text += f"Title: {hypothesis.get('title', 'Untitled')}\n"
            hypotheses_text += f"Description:\n{hypothesis.get('description', '')}\n"
            hypotheses_text += "-" * 80 + "\n"
        
        return {
            "research_goal": self.research_goal,
            "hypotheses_text": hypotheses_text
        }
    
    def actor(self) -> Dict[str, Any]:
        """
        Generate evolved variants of the input hypotheses.
        """
        context = self.perceiver()
        
        messages = [
            {
                "role": "system",
                "content": self.EVOLUTION_PROMPT.format(**context)
            }
        ]
        
        # Call LLM for hypothesis evolution
        response = self.model.invoke(messages)
        
        # Parse the response to extract evolved hypotheses
        content = response.content
        evolved_hypotheses = self._parse_evolved_hypotheses(content)
        
        # Update context memory with evolution agent state
        evolution_state = {
            "last_evolution": content,
            "evolved_from": [h["id"] for h in self.hypotheses],
            "evolved_hypotheses": [h["id"] for h in evolved_hypotheses]
        }
        
        # Update the full context memory
        updated_context_memory = self.context_memory.copy()
        if "agent_states" not in updated_context_memory:
            updated_context_memory["agent_states"] = {}
        updated_context_memory["agent_states"]["evolution"] = evolution_state
        
        return {
            "evolved_hypotheses": evolved_hypotheses,
            "updated_context_memory": updated_context_memory
        }
    
    def _parse_evolved_hypotheses(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response to extract evolved hypotheses.
        """
        evolved = []
        
        # Split content by evolved hypothesis sections
        sections = content.split("Hypothesis")
        sections = [s.strip() for s in sections if s.strip()]
        
        for section in sections:
            # Initialize hypothesis fields
            hypothesis = {
                "id": str(uuid.uuid4()),
                "title": "",
                "description": "",
                "parent_id": None,
                "evolution_strategy": [],
                "improvements": [],
                "validation_approach": "",
                "research_goal": self.research_goal,
                "created_at": "now",  # This would be replaced with actual timestamp
                "source": "evolution_agent"
            }
            
            # Parse section line by line
            current_field = None
            lines = section.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for field markers
                if line.lower().startswith('title:'):
                    current_field = "title"
                    hypothesis["title"] = line.split(':', 1)[1].strip()
                elif line.lower().startswith('parent:'):
                    current_field = None
                    parent_info = line.split(':', 1)[1].strip()
                    # Try to extract parent ID
                    if '(ID:' in parent_info:
                        parent_id = parent_info.split('(ID:', 1)[1].strip(')')
                        hypothesis["parent_id"] = parent_id
                elif line.lower().startswith('evolution strategy:'):
                    current_field = "evolution_strategy"
                    strategies = line.split(':', 1)[1].strip()
                    hypothesis["evolution_strategy"] = [s.strip() for s in strategies.split(',')]
                elif line.lower().startswith('description:'):
                    current_field = "description"
                elif line.lower().startswith('improvements:'):
                    current_field = "improvements"
                elif line.lower().startswith('validation:'):
                    current_field = "validation"
                else:
                    # Append content to current field
                    if current_field == "description":
                        if hypothesis["description"]:
                            hypothesis["description"] += "\n"
                        hypothesis["description"] += line
                    elif current_field == "improvements":
                        if line.startswith('-'):
                            hypothesis["improvements"].append(line[1:].strip())
                    elif current_field == "validation":
                        if hypothesis["validation_approach"]:
                            hypothesis["validation_approach"] += "\n"
                        hypothesis["validation_approach"] += line
            
            # Only add hypotheses that have at least a title and description
            if hypothesis["title"] and hypothesis["description"]:
                evolved.append(hypothesis)
        
        return evolved
    
    @classmethod
    def create(cls, research_goal: str, hypotheses: List[Dict[str, Any]], context_memory: Dict[str, Any]) -> 'EvolutionAgent':
        """Factory method to create an EvolutionAgent instance"""
        return cls(research_goal, hypotheses, context_memory) 