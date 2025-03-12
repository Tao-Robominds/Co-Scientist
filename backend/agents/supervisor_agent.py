from typing import Dict, List, Any, Optional
import random
import uuid
from langchain_openai import ChatOpenAI

class SupervisorAgent:
    """
    The Supervisor agent manages the Co-Scientist workflow. 
    It assigns specialized agents to tasks and allocates resources.
    """
    
    SUPERVISOR_PROMPT = """You are the Supervisor agent in the AI Co-Scientist system. Your role is to manage and coordinate the research process based on the goal:

"{research_goal}"

Current system state:
- Iteration: {iteration}
- Total hypotheses generated: {hypotheses_count}
- Hypotheses awaiting review: {unreviewed_count}
- Tournament state: {tournament_summary}

Your task is to determine which specialized agents to run next. The available agents are:

1. Generation agent: Generates initial hypotheses addressing the research goal
2. Reflection agent: Reviews hypotheses for correctness, quality, and novelty
3. Ranking agent: Conducts tournament-based evaluation to prioritize hypotheses
4. Evolution agent: Refines top-ranked hypotheses through iterative improvement
5. Proximity agent: Computes similarity between hypotheses for clustering
6. Meta-review agent: Synthesizes research overview from top hypotheses and reviews

Based on the current state, create a prioritized task queue. Focus on:
1. Maintaining a balance between generating new ideas and improving existing ones
2. Ensuring hypotheses are properly reviewed and ranked
3. Periodically synthesizing findings for the scientist to review

Provide a brief explanation of your reasoning, followed by a prioritized list of tasks.
"""

    def __init__(self, research_goal: str, context_memory: Dict[str, Any], iteration: int = 0):
        """
        Initialize the Supervisor agent.
        
        Args:
            research_goal: The scientist's research goal
            context_memory: The context memory containing system state
            iteration: Current iteration number
        """
        self.research_goal = research_goal
        self.context_memory = context_memory
        self.iteration = iteration
        self.model = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    def perceiver(self) -> Dict[str, Any]:
        """
        Prepare the context for the Supervisor's decision making.
        """
        # Extract statistics from context memory
        hypotheses = self.context_memory.get("hypotheses", [])
        reviews = self.context_memory.get("reviews", [])
        tournament_state = self.context_memory.get("tournament_state", {})
        
        # Calculate statistics
        hypotheses_count = len(hypotheses)
        
        # Count hypotheses awaiting review
        reviewed_ids = [review.get("hypothesis_id") for review in reviews]
        unreviewed_count = sum(1 for h in hypotheses if h.get("id") not in reviewed_ids)
        
        # Summarize tournament state
        top_hypotheses = []
        if tournament_state and "rankings" in tournament_state:
            rankings = tournament_state["rankings"]
            # Sort hypothesis IDs by ranking score
            sorted_ids = sorted(rankings.keys(), key=lambda k: rankings[k], reverse=True)
            # Get top 3 IDs if available
            top_ids = sorted_ids[:3] if len(sorted_ids) >= 3 else sorted_ids
            # Find hypotheses with these IDs
            top_hypotheses = [h for h in hypotheses if h.get("id") in top_ids]
        
        tournament_summary = f"{len(tournament_state.get('rankings', {}))} hypotheses ranked"
        if top_hypotheses:
            tournament_summary += f", top hypothesis: {top_hypotheses[0].get('title', 'Untitled')}"
        
        return {
            "research_goal": self.research_goal,
            "iteration": self.iteration,
            "hypotheses_count": hypotheses_count,
            "unreviewed_count": unreviewed_count,
            "tournament_summary": tournament_summary
        }
    
    def actor(self) -> Dict[str, Any]:
        """
        Generate supervision decisions based on the current state.
        """
        context = self.perceiver()
        
        messages = [
            {
                "role": "system",
                "content": self.SUPERVISOR_PROMPT.format(**context)
            }
        ]
        
        # Call LLM for supervision decisions
        response = self.model.invoke(messages)
        
        # Parse the response to extract task queue
        content = response.content
        task_queue = self._parse_task_queue(content)
        
        # Update context memory with supervisor state
        supervisor_state = {
            "last_decision": content,
            "task_queue": task_queue
        }
        
        # Update the full context memory
        updated_context_memory = self.context_memory.copy()
        if "agent_states" not in updated_context_memory:
            updated_context_memory["agent_states"] = {}
        updated_context_memory["agent_states"]["supervisor"] = supervisor_state
        
        return {
            "summary": content,
            "task_queue": task_queue,
            "next_task": task_queue[0] if task_queue else "end",
            "updated_context_memory": updated_context_memory
        }
    
    def _parse_task_queue(self, content: str) -> List[str]:
        """
        Parse the LLM response to extract the task queue.
        """
        # Default task queue if parsing fails
        default_queue = ["generation", "reflection", "ranking"]
        
        # Simple heuristic parsing based on list numbers and agent keywords
        agent_keywords = {
            "generation": ["generation", "generate", "create", "new ideas", "new hypotheses"],
            "reflection": ["reflection", "review", "evaluate"],
            "ranking": ["ranking", "tournament", "prioritize", "rank"],
            "evolution": ["evolution", "refine", "improve", "iterate"],
            "proximity": ["proximity", "cluster", "similarity", "graph"],
            "meta_review": ["meta", "meta-review", "synthesize", "overview", "summarize"]
        }
        
        # Try to extract tasks from the content
        tasks = []
        
        # Look for numbers followed by agent keywords
        for line in content.lower().split('\n'):
            for agent_type, keywords in agent_keywords.items():
                if any(keyword in line for keyword in keywords):
                    tasks.append(agent_type)
                    break
        
        # If we couldn't extract tasks, just run a standard sequence based on state
        if not tasks:
            # Fallback logic - create a sensible sequence based on iteration
            if self.iteration == 0:
                tasks = ["generation", "reflection", "ranking"]
            elif self.iteration % 5 == 0:  # Every 5 iterations, do a full meta-review
                tasks = ["meta_review"]
            else:
                # Mix of tasks based on iteration number
                if self.iteration % 3 == 0:
                    tasks = ["evolution", "reflection", "ranking", "proximity"]
                elif self.iteration % 3 == 1:
                    tasks = ["generation", "reflection", "ranking"]
                else:
                    tasks = ["proximity", "evolution", "reflection", "ranking"]
        
        # Ensure the task queue isn't empty
        if not tasks:
            tasks = default_queue
        
        return tasks
    
    @classmethod
    def create(cls, research_goal: str, context_memory: Dict[str, Any], iteration: int = 0) -> 'SupervisorAgent':
        """Factory method to create a SupervisorAgent instance"""
        return cls(research_goal, context_memory, iteration) 