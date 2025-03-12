from typing import Dict, List, Any, Optional
import uuid
from langchain_openai import ChatOpenAI

class ReflectionAgent:
    """
    The Reflection agent reviews hypotheses for correctness, quality, and novelty.
    It acts as a scientific peer reviewer.
    """
    
    REFLECTION_PROMPT = """You are the Reflection agent in the AI Co-Scientist system. Your role is to critically review research hypotheses addressing this goal:

"{research_goal}"

For each hypothesis, evaluate:

1. Scientific Merit:
   - Is it grounded in established scientific principles?
   - Does it make logical sense?
   - Are the proposed mechanisms plausible?

2. Novelty and Innovation:
   - Does it offer a unique perspective?
   - How does it advance current understanding?
   - Is it sufficiently different from existing approaches?

3. Testability and Feasibility:
   - Can it be experimentally validated?
   - Are the proposed methods realistic?
   - What resources would be required?

4. Potential Impact:
   - If proven true, how significant would the impact be?
   - What applications could benefit?
   - Are there broader implications?

5. Limitations and Risks:
   - What are the key assumptions?
   - What could go wrong?
   - Are there ethical considerations?

For each hypothesis, provide:
1. A numerical score (1-10) for each criterion
2. Detailed justification for each score
3. Specific suggestions for improvement
4. Overall recommendation (Accept/Revise/Reject)

Hypotheses to review:
{hypotheses_text}

Be thorough and constructive in your criticism while maintaining scientific rigor.
"""

    def __init__(self, research_goal: str, hypotheses: List[Dict[str, Any]], context_memory: Dict[str, Any]):
        """
        Initialize the Reflection agent.
        
        Args:
            research_goal: The scientist's research goal
            hypotheses: List of hypotheses to review
            context_memory: The context memory containing system state
        """
        self.research_goal = research_goal
        self.hypotheses = hypotheses
        self.context_memory = context_memory
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.3)  # Lower temperature for consistent evaluation
    
    def perceiver(self) -> Dict[str, Any]:
        """
        Prepare the context for hypothesis review.
        """
        # Format hypotheses for review
        hypotheses_text = ""
        for i, hypothesis in enumerate(self.hypotheses, 1):
            hypotheses_text += f"\nHypothesis {i}:\n"
            hypotheses_text += f"Title: {hypothesis.get('title', 'Untitled')}\n"
            hypotheses_text += f"Description:\n{hypothesis.get('description', '')}\n"
            hypotheses_text += "-" * 80 + "\n"
        
        return {
            "research_goal": self.research_goal,
            "hypotheses_text": hypotheses_text
        }
    
    def actor(self) -> Dict[str, Any]:
        """
        Review hypotheses and generate structured feedback.
        """
        context = self.perceiver()
        
        messages = [
            {
                "role": "system",
                "content": self.REFLECTION_PROMPT.format(**context)
            }
        ]
        
        # Call LLM for hypothesis review
        response = self.model.invoke(messages)
        
        # Parse the response to extract reviews
        content = response.content
        new_reviews = self._parse_reviews(content)
        
        # Update context memory with reflection agent state
        reflection_state = {
            "last_review": content,
            "reviewed_hypotheses": [h["id"] for h in self.hypotheses]
        }
        
        # Update the full context memory
        updated_context_memory = self.context_memory.copy()
        if "agent_states" not in updated_context_memory:
            updated_context_memory["agent_states"] = {}
        updated_context_memory["agent_states"]["reflection"] = reflection_state
        
        return {
            "new_reviews": new_reviews,
            "updated_context_memory": updated_context_memory
        }
    
    def _parse_reviews(self, content: str) -> List[Dict[str, Any]]:
        """
        Parse the LLM response to extract structured reviews.
        """
        reviews = []
        
        # Split content by hypothesis sections
        sections = content.split("Hypothesis")
        sections = [s.strip() for s in sections if s.strip()]
        
        # Process each section
        for i, section in enumerate(sections):
            if i >= len(self.hypotheses):
                break
                
            hypothesis = self.hypotheses[i]
            
            # Extract scores
            scores = {
                "scientific_merit": 0,
                "novelty": 0,
                "testability": 0,
                "impact": 0,
                "limitations": 0
            }
            
            # Simple score extraction (could be made more robust)
            for line in section.split('\n'):
                line = line.lower()
                for criterion in scores.keys():
                    if criterion.replace('_', ' ') in line and ':' in line:
                        try:
                            score = int(line.split(':')[1].strip().split('/')[0])
                            scores[criterion] = min(10, max(1, score))  # Ensure 1-10 range
                        except (ValueError, IndexError):
                            continue
            
            # Extract overall recommendation
            recommendation = "Revise"  # Default
            if "accept" in section.lower():
                recommendation = "Accept"
            elif "reject" in section.lower():
                recommendation = "Reject"
            
            # Create review object
            review_id = str(uuid.uuid4())
            reviews.append({
                "id": review_id,
                "hypothesis_id": hypothesis["id"],
                "scores": scores,
                "overall_score": sum(scores.values()) / len(scores),
                "recommendation": recommendation,
                "full_review": section.strip(),
                "created_at": "now",  # This would be replaced with actual timestamp
                "source": "reflection_agent"
            })
        
        return reviews
    
    @classmethod
    def create(cls, research_goal: str, hypotheses: List[Dict[str, Any]], context_memory: Dict[str, Any]) -> 'ReflectionAgent':
        """Factory method to create a ReflectionAgent instance"""
        return cls(research_goal, hypotheses, context_memory) 