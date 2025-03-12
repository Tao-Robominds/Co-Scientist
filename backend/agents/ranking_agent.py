from typing import Dict, List, Any, Optional, Tuple
import random
from langchain_openai import ChatOpenAI

class RankingAgent:
    """
    The Ranking agent conducts tournament-based evaluation to prioritize hypotheses.
    Uses an Elo-like rating system and simulated scientific debates.
    """
    
    DEBATE_PROMPT = """You are the Ranking agent in the AI Co-Scientist system. Your role is to evaluate two competing hypotheses addressing this research goal:

"{research_goal}"

Hypothesis A:
Title: {hypothesis_a_title}
Description:
{hypothesis_a_description}

Reviews:
{hypothesis_a_reviews}

Hypothesis B:
Title: {hypothesis_b_title}
Description:
{hypothesis_b_description}

Reviews:
{hypothesis_b_reviews}

Compare these hypotheses based on:
1. Scientific Merit and Rigor
2. Innovation and Novelty
3. Feasibility and Practicality
4. Potential Impact
5. Clarity and Completeness

For each criterion:
1. Analyze how each hypothesis performs
2. Compare their relative strengths and weaknesses
3. Determine which hypothesis is stronger

Then:
1. Provide an overall winner (A or B)
2. Justify your decision with specific points
3. Estimate the confidence in your decision (0-100%)

Format your response as:
WINNER: [A/B]
CONFIDENCE: [0-100]
JUSTIFICATION:
[Your detailed analysis]
"""

    def __init__(self, research_goal: str, hypotheses: List[Dict[str, Any]], 
                 reviews: List[Dict[str, Any]], tournament_state: Dict[str, Any], 
                 context_memory: Dict[str, Any]):
        """
        Initialize the Ranking agent.
        
        Args:
            research_goal: The scientist's research goal
            hypotheses: List of all hypotheses
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
    
    def perceiver(self, hypothesis_a: Dict[str, Any], hypothesis_b: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare the context for a tournament match between two hypotheses.
        """
        # Get reviews for each hypothesis
        reviews_a = [r for r in self.reviews if r.get("hypothesis_id") == hypothesis_a.get("id")]
        reviews_b = [r for r in self.reviews if r.get("hypothesis_id") == hypothesis_b.get("id")]
        
        # Format reviews
        def format_reviews(reviews: List[Dict[str, Any]]) -> str:
            if not reviews:
                return "No reviews available."
            
            formatted = ""
            for review in reviews:
                scores = review.get("scores", {})
                formatted += f"Overall Score: {review.get('overall_score', 0):.1f}/10\n"
                formatted += f"Recommendation: {review.get('recommendation', 'Unknown')}\n"
                formatted += "Detailed Scores:\n"
                for criterion, score in scores.items():
                    formatted += f"- {criterion.replace('_', ' ').title()}: {score}/10\n"
                formatted += "\n"
            return formatted
        
        return {
            "research_goal": self.research_goal,
            "hypothesis_a_title": hypothesis_a.get("title", "Untitled"),
            "hypothesis_a_description": hypothesis_a.get("description", ""),
            "hypothesis_a_reviews": format_reviews(reviews_a),
            "hypothesis_b_title": hypothesis_b.get("title", "Untitled"),
            "hypothesis_b_description": hypothesis_b.get("description", ""),
            "hypothesis_b_reviews": format_reviews(reviews_b)
        }
    
    def actor(self) -> Dict[str, Any]:
        """
        Run tournament matches and update rankings.
        """
        # Initialize or get existing rankings
        rankings = self.tournament_state.get("rankings", {})
        
        # Initialize rankings for new hypotheses
        for hypothesis in self.hypotheses:
            hypothesis_id = hypothesis.get("id")
            if hypothesis_id not in rankings:
                rankings[hypothesis_id] = 1500  # Initial Elo rating
        
        # Run tournament matches
        matches_played = 0
        max_matches = 10  # Limit number of matches per round
        
        # Select pairs of hypotheses to compare
        pairs = self._select_tournament_pairs(rankings)
        
        for hypothesis_a, hypothesis_b in pairs[:max_matches]:
            # Prepare context for the debate
            context = self.perceiver(hypothesis_a, hypothesis_b)
            
            # Run the debate
            messages = [
                {
                    "role": "system",
                    "content": self.DEBATE_PROMPT.format(**context)
                }
            ]
            
            response = self.model.invoke(messages)
            
            # Parse results
            winner, confidence = self._parse_debate_result(response.content)
            
            # Update Elo ratings
            if winner is not None:
                self._update_elo_ratings(
                    rankings,
                    hypothesis_a.get("id"),
                    hypothesis_b.get("id"),
                    winner == "A",
                    confidence
                )
                matches_played += 1
        
        # Update tournament state
        updated_tournament_state = {
            "rankings": rankings,
            "total_matches": self.tournament_state.get("total_matches", 0) + matches_played
        }
        
        # Update context memory
        updated_context_memory = self.context_memory.copy()
        if "agent_states" not in updated_context_memory:
            updated_context_memory["agent_states"] = {}
        
        updated_context_memory["agent_states"]["ranking"] = {
            "last_tournament": {
                "matches_played": matches_played,
                "rankings": rankings
            }
        }
        
        return {
            "updated_tournament_state": updated_tournament_state,
            "updated_context_memory": updated_context_memory
        }
    
    def _select_tournament_pairs(self, rankings: Dict[str, float]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Select pairs of hypotheses for tournament matches.
        Uses a mix of random and rating-based pairing.
        """
        pairs = []
        hypotheses = self.hypotheses.copy()
        
        # Sort hypotheses by rating
        hypotheses.sort(key=lambda h: rankings.get(h.get("id"), 1500))
        
        # Mix of strategies:
        # 1. Adjacent pairs (similar ratings)
        # 2. Random pairs
        # 3. Top vs Bottom pairs
        
        # Strategy 1: Adjacent pairs (40% of matches)
        for i in range(0, len(hypotheses)-1, 2):
            pairs.append((hypotheses[i], hypotheses[i+1]))
        
        # Strategy 2: Random pairs (30% of matches)
        random_pairs = []
        remaining = hypotheses.copy()
        while len(remaining) >= 2:
            a = remaining.pop(random.randrange(len(remaining)))
            b = remaining.pop(random.randrange(len(remaining)))
            random_pairs.append((a, b))
        pairs.extend(random_pairs)
        
        # Strategy 3: Top vs Bottom (30% of matches)
        n = len(hypotheses) // 2
        top_vs_bottom = list(zip(hypotheses[:n], reversed(hypotheses[n:])))
        pairs.extend(top_vs_bottom)
        
        # Shuffle all pairs
        random.shuffle(pairs)
        
        return pairs
    
    def _parse_debate_result(self, content: str) -> Tuple[Optional[str], float]:
        """
        Parse the debate response to extract winner and confidence.
        """
        winner = None
        confidence = 0.5  # Default confidence
        
        # Extract winner
        for line in content.split('\n'):
            if line.startswith('WINNER:'):
                winner_str = line.split(':')[1].strip().upper()
                if winner_str in ['A', 'B']:
                    winner = winner_str
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split(':')[1].strip()) / 100.0
                except (ValueError, IndexError):
                    pass
        
        return winner, confidence
    
    def _update_elo_ratings(self, rankings: Dict[str, float], id_a: str, id_b: str, 
                           a_won: bool, confidence: float):
        """
        Update Elo ratings based on match result.
        
        Args:
            rankings: Current Elo rankings
            id_a: ID of first hypothesis
            id_b: ID of second hypothesis
            a_won: Whether hypothesis A won
            confidence: Confidence in the result (0-1)
        """
        K = 32  # Base K-factor
        
        # Get current ratings
        rating_a = rankings.get(id_a, 1500)
        rating_b = rankings.get(id_b, 1500)
        
        # Calculate expected scores
        exp_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
        exp_b = 1 - exp_a
        
        # Adjust K-factor based on confidence
        adjusted_k = K * confidence
        
        # Update ratings
        if a_won:
            rankings[id_a] = rating_a + adjusted_k * (1 - exp_a)
            rankings[id_b] = rating_b + adjusted_k * (0 - exp_b)
        else:
            rankings[id_a] = rating_a + adjusted_k * (0 - exp_a)
            rankings[id_b] = rating_b + adjusted_k * (1 - exp_b)
    
    @classmethod
    def create(cls, research_goal: str, hypotheses: List[Dict[str, Any]], 
               reviews: List[Dict[str, Any]], tournament_state: Dict[str, Any], 
               context_memory: Dict[str, Any]) -> 'RankingAgent':
        """Factory method to create a RankingAgent instance"""
        return cls(research_goal, hypotheses, reviews, tournament_state, context_memory) 