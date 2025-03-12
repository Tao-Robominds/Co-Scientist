from typing import Dict, List, Any, Optional
import uuid
from langchain_openai import ChatOpenAI

class GenerationAgent:
    """
    The Generation agent initiates the research process by generating initial hypotheses
    that address the research goal.
    """
    
    GENERATION_PROMPT = """You are the Generation agent in the AI Co-Scientist system. Your role is to generate novel, scientifically plausible hypotheses addressing this research goal:

"{research_goal}"

A good scientific hypothesis should:
1. Be specific and testable
2. Address the research goal directly
3. Be grounded in scientific principles
4. Offer a novel approach or perspective
5. Suggest potential experimental validation methods

Generate {num_hypotheses} distinct hypotheses that approach the research goal from different angles. For each hypothesis:
1. Provide a clear title
2. Explain the core idea in 2-3 paragraphs
3. Describe how it addresses the research goal
4. Outline potential implementation or validation approaches
5. Identify potential challenges or limitations

Be creative but scientifically rigorous. Consider interdisciplinary approaches and emerging technologies where appropriate.
"""

    def __init__(self, research_goal: str, context_memory: Dict[str, Any]):
        """
        Initialize the Generation agent.
        
        Args:
            research_goal: The scientist's research goal
            context_memory: The context memory containing system state
        """
        self.research_goal = research_goal
        self.context_memory = context_memory
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.7)  # Higher temperature for creativity
    
    def perceiver(self) -> Dict[str, Any]:
        """
        Prepare the context for hypothesis generation.
        """
        # Extract existing hypotheses to avoid duplication
        existing_hypotheses = self.context_memory.get("hypotheses", [])
        existing_titles = [h.get("title", "") for h in existing_hypotheses]
        
        # Determine how many hypotheses to generate
        # Generate more in early iterations, fewer in later ones
        iteration = self.context_memory.get("iteration", 0)
        if iteration == 0:
            num_hypotheses = 5  # Generate more initially
        else:
            num_hypotheses = 3  # Generate fewer in subsequent iterations
        
        return {
            "research_goal": self.research_goal,
            "num_hypotheses": num_hypotheses,
            "existing_titles": existing_titles
        }
    
    def actor(self) -> Dict[str, Any]:
        """
        Generate new hypotheses based on the research goal.
        """
        context = self.perceiver()
        
        messages = [
            {
                "role": "system",
                "content": self.GENERATION_PROMPT.format(**context)
            }
        ]
        
        # Call LLM for hypothesis generation
        response = self.model.invoke(messages)
        
        # Parse the response to extract hypotheses
        content = response.content
        new_hypotheses = self._parse_hypotheses(content)
        
        # Update context memory with generation agent state
        generation_state = {
            "last_generation": content,
            "generated_hypotheses": [h["title"] for h in new_hypotheses]
        }
        
        # Update the full context memory
        updated_context_memory = self.context_memory.copy()
        if "agent_states" not in updated_context_memory:
            updated_context_memory["agent_states"] = {}
        updated_context_memory["agent_states"]["generation"] = generation_state
        
        return {
            "new_hypotheses": new_hypotheses,
            "updated_context_memory": updated_context_memory
        }
    
    def _parse_hypotheses(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response to extract structured hypotheses.
        """
        hypotheses = []
        
        # Split by hypothesis markers (assuming numbered format like "1.", "2.", etc.)
        sections = []
        current_section = ""
        
        for line in content.split('\n'):
            # Check if line starts a new hypothesis
            if line.strip() and line[0].isdigit() and line[1:].startswith('. '):
                if current_section:
                    sections.append(current_section)
                current_section = line
            else:
                current_section += '\n' + line
        
        # Add the last section
        if current_section:
            sections.append(current_section)
        
        # Process each section into a structured hypothesis
        for section in sections:
            lines = section.split('\n')
            
            # Extract title from the first line
            title = lines[0].strip()
            if ':' in title:
                title = title.split(':', 1)[1].strip()
            elif '.' in title:
                title = title.split('.', 1)[1].strip()
            
            # Extract description from remaining lines
            description = '\n'.join(lines[1:]).strip()
            
            # Create a unique ID for the hypothesis
            hypothesis_id = str(uuid.uuid4())
            
            hypotheses.append({
                "id": hypothesis_id,
                "title": title,
                "description": description,
                "research_goal": self.research_goal,
                "created_at": "now",  # This would be replaced with actual timestamp
                "source": "generation_agent"
            })
        
        return hypotheses
    
    @classmethod
    def create(cls, research_goal: str, context_memory: Dict[str, Any]) -> 'GenerationAgent':
        """Factory method to create a GenerationAgent instance"""
        return cls(research_goal, context_memory) 