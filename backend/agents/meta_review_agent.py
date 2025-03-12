from typing import Dict, List, Any, Optional
import uuid
from langchain_openai import ChatOpenAI

class MetaReviewAgent:
    """
    The Meta-review agent synthesizes insights from reviews and tournament debates
    to provide comprehensive research overviews and recommendations.
    """
    
    META_REVIEW_PROMPT = """You are the Meta-review agent in the AI Co-Scientist system. Your role is to synthesize insights and provide a comprehensive overview of the research addressing this goal:

"{research_goal}"

Top Hypotheses to Review:
{hypotheses_text}

Reviews and Tournament Performance:
{reviews_text}

Analyze and synthesize:

1. Research Progress:
   - Key themes and patterns across hypotheses
   - Evolution of ideas through iterations
   - Emerging consensus or divergent approaches

2. Scientific Merit:
   - Strength of theoretical foundations
   - Quality of proposed methodologies
   - Potential for experimental validation

3. Innovation Assessment:
   - Novel concepts and approaches
   - Creative combinations of ideas
   - Unique perspectives or insights

4. Implementation Pathways:
   - Most promising directions
   - Resource requirements
   - Potential challenges and solutions

5. Impact Analysis:
   - Potential scientific contributions
   - Practical applications
   - Broader implications

6. Recommendations:
   - Priority research directions
   - Suggested improvements
   - Areas needing more exploration

Format your response as a structured research overview with:
1. Executive Summary
2. Analysis of Key Findings
3. Detailed Assessment of Top Hypotheses
4. Strategic Recommendations
5. Future Directions

Be thorough, analytical, and forward-looking in your synthesis.
"""

    def __init__(self, research_goal: str, hypotheses: List[Dict[str, Any]], 
                 reviews: List[Dict[str, Any]], tournament_state: Dict[str, Any], 
                 context_memory: Dict[str, Any]):
        """
        Initialize the Meta-review agent.
        
        Args:
            research_goal: The scientist's research goal
            hypotheses: List of top hypotheses to review
            reviews: List of all reviews
            tournament_state: Current tournament rankings
            context_memory: The context memory containing system state
        """
        self.research_goal = research_goal
        self.hypotheses = hypotheses
        self.reviews = reviews
        self.tournament_state = tournament_state
        self.context_memory = context_memory
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    def perceiver(self) -> Dict[str, Any]:
        """
        Prepare the context for meta-review synthesis.
        """
        # Format hypotheses with their rankings
        hypotheses_text = ""
        for i, hypothesis in enumerate(self.hypotheses, 1):
            hypothesis_id = hypothesis.get("id", "")
            ranking = self.tournament_state.get("rankings", {}).get(hypothesis_id, 0)
            
            hypotheses_text += f"\nHypothesis {i} (Ranking Score: {ranking:.1f}):\n"
            hypotheses_text += f"Title: {hypothesis.get('title', 'Untitled')}\n"
            hypotheses_text += f"Description:\n{hypothesis.get('description', '')}\n"
            hypotheses_text += "-" * 80 + "\n"
        
        # Format reviews and tournament insights
        reviews_text = ""
        for hypothesis in self.hypotheses:
            hypothesis_id = hypothesis.get("id", "")
            relevant_reviews = [r for r in self.reviews if r.get("hypothesis_id") == hypothesis_id]
            
            reviews_text += f"\nReviews for {hypothesis.get('title', 'Untitled')}:\n"
            
            if not relevant_reviews:
                reviews_text += "No reviews available.\n"
            else:
                for review in relevant_reviews:
                    reviews_text += f"Overall Score: {review.get('overall_score', 0):.1f}/10\n"
                    reviews_text += f"Recommendation: {review.get('recommendation', 'Unknown')}\n"
                    
                    scores = review.get("scores", {})
                    reviews_text += "Detailed Scores:\n"
                    for criterion, score in scores.items():
                        reviews_text += f"- {criterion.replace('_', ' ').title()}: {score}/10\n"
                    
                    reviews_text += "\nKey Points:\n"
                    reviews_text += review.get("full_review", "No detailed review available.")
                    reviews_text += "\n" + "-" * 80 + "\n"
        
        return {
            "research_goal": self.research_goal,
            "hypotheses_text": hypotheses_text,
            "reviews_text": reviews_text
        }
    
    def actor(self) -> Dict[str, Any]:
        """
        Generate a comprehensive research overview and recommendations.
        """
        context = self.perceiver()
        
        messages = [
            {
                "role": "system",
                "content": self.META_REVIEW_PROMPT.format(**context)
            }
        ]
        
        # Call LLM for meta-review synthesis
        response = self.model.invoke(messages)
        
        # Extract the research overview
        research_overview = response.content
        
        # Update context memory with meta-review state
        meta_review_state = {
            "last_review": research_overview,
            "reviewed_hypotheses": [h["id"] for h in self.hypotheses],
            "timestamp": "now"  # This would be replaced with actual timestamp
        }
        
        # Update the full context memory
        updated_context_memory = self.context_memory.copy()
        if "agent_states" not in updated_context_memory:
            updated_context_memory["agent_states"] = {}
        updated_context_memory["agent_states"]["meta_review"] = meta_review_state
        
        return {
            "research_overview": research_overview,
            "updated_context_memory": updated_context_memory
        }
    
    @classmethod
    def create(cls, research_goal: str, hypotheses: List[Dict[str, Any]], 
               reviews: List[Dict[str, Any]], tournament_state: Dict[str, Any], 
               context_memory: Dict[str, Any]) -> 'MetaReviewAgent':
        """Factory method to create a MetaReviewAgent instance"""
        return cls(research_goal, hypotheses, reviews, tournament_state, context_memory) 