from typing import Literal, Dict, List, TypedDict, Optional, cast, Any, Annotated
import json

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_core.runnables.config import get_executor_for_config
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END

# Import agents
from backend.agents.supervisor_agent import SupervisorAgent
from backend.agents.generation_agent import GenerationAgent
from backend.agents.reflection_agent import ReflectionAgent
from backend.agents.ranking_agent import RankingAgent
from backend.agents.evolution_agent import EvolutionAgent
from backend.agents.proximity_agent import ProximityAgent
from backend.agents.meta_review_agent import MetaReviewAgent

# Import shared components if needed
from backend.components.context_memory import ContextMemory

# Initialize the LLM
model = ChatOpenAI(model="gpt-4o", temperature=0.3)

# Define the state schema for our workflow
class CoScientistState(TypedDict):
    """State for the Co-Scientist workflow"""
    messages: List[SystemMessage | HumanMessage | AIMessage]  # Chat history
    research_goal: str  # The scientist's research goal
    hypotheses: List[Dict[str, Any]]  # Generated research hypotheses
    reviews: List[Dict[str, Any]]  # Reviews of hypotheses
    tournament_state: Dict[str, Any]  # State of the tournament for ranking hypotheses
    proximity_graph: Dict[str, Any]  # Proximity graph for similar hypotheses
    context_memory: Dict[str, Any]  # Persistent context memory for agent states
    current_task: str  # Current task being executed
    task_queue: List[str]  # Queue of pending tasks
    iteration: int  # Current iteration number

# Define the functions for each agent
def run_supervisor(state: Dict[str, Any], config: RunnableConfig):
    """
    Supervisor agent that manages the workflow, prioritizes tasks, and allocates resources.
    """
    # Initialize any missing fields with default values
    state.setdefault("hypotheses", [])
    state.setdefault("reviews", [])
    state.setdefault("tournament_state", {"rankings": {}})
    state.setdefault("proximity_graph", {"edges": []})
    state.setdefault("context_memory", {})
    state.setdefault("task_queue", [])
    state.setdefault("iteration", 0)
    
    messages = state["messages"]
    research_goal = state.get("research_goal", "")
    context_memory = state.get("context_memory", {})
    iteration = state.get("iteration", 0)
    
    # If research goal is not set, extract it from the first human message
    if not research_goal and messages:
        for msg in messages:
            if isinstance(msg, HumanMessage):
                research_goal = msg.content
                break
        
        # Update research goal in state
        state["research_goal"] = research_goal
    
    # Create the supervisor agent
    agent = SupervisorAgent(
        research_goal=research_goal,
        context_memory=context_memory,
        iteration=iteration
    )
    
    # Get supervisor decisions
    result = agent.actor()
    
    # Update the state with supervisor decisions
    return {
        "messages": state["messages"] + [AIMessage(content=f"Supervisor update: {result['summary']}")],
        "research_goal": research_goal,
        "task_queue": result["task_queue"],
        "current_task": result["next_task"] if result["task_queue"] else "",
        "context_memory": result["updated_context_memory"],
        "iteration": iteration
    }

def run_generation_agent(state: Dict[str, Any], config: RunnableConfig):
    """
    Generation agent that produces initial hypotheses for the research goal.
    """
    research_goal = state.get("research_goal", "")
    context_memory = state.get("context_memory", {})
    existing_hypotheses = state.get("hypotheses", [])
    
    # Create the generation agent
    agent = GenerationAgent(
        research_goal=research_goal,
        context_memory=context_memory
    )
    
    # Generate new hypotheses
    result = agent.actor()
    
    # Update the state with new hypotheses
    combined_hypotheses = existing_hypotheses + result["new_hypotheses"]
    
    return {
        "messages": state["messages"] + [AIMessage(content=f"Generated {len(result['new_hypotheses'])} new hypotheses.")],
        "hypotheses": combined_hypotheses,
        "context_memory": result["updated_context_memory"],
        "current_task": "",  # Clear current task as it's completed
    }

def run_reflection_agent(state: Dict[str, Any], config: RunnableConfig):
    """
    Reflection agent that reviews the quality of generated hypotheses.
    """
    research_goal = state.get("research_goal", "")
    context_memory = state.get("context_memory", {})
    hypotheses = state.get("hypotheses", [])
    existing_reviews = state.get("reviews", [])
    
    # Select hypotheses that need review
    unreviewed_hypotheses = [h for h in hypotheses if h.get("id") not in [r.get("hypothesis_id") for r in existing_reviews]]
    
    if not unreviewed_hypotheses:
        return {
            "messages": state["messages"] + [AIMessage(content="No new hypotheses to review.")],
            "current_task": "",  # Clear current task as it's completed
        }
    
    # Create the reflection agent
    agent = ReflectionAgent(
        research_goal=research_goal,
        hypotheses=unreviewed_hypotheses,
        context_memory=context_memory
    )
    
    # Review hypotheses
    result = agent.actor()
    
    # Update the state with new reviews
    combined_reviews = existing_reviews + result["new_reviews"]
    
    return {
        "messages": state["messages"] + [AIMessage(content=f"Reviewed {len(result['new_reviews'])} hypotheses.")],
        "reviews": combined_reviews,
        "context_memory": result["updated_context_memory"],
        "current_task": "",  # Clear current task as it's completed
    }

def run_ranking_agent(state: Dict[str, Any], config: RunnableConfig):
    """
    Ranking agent that evaluates and prioritizes hypotheses in a tournament.
    """
    research_goal = state.get("research_goal", "")
    context_memory = state.get("context_memory", {})
    hypotheses = state.get("hypotheses", [])
    reviews = state.get("reviews", [])
    tournament_state = state.get("tournament_state", {})
    
    # Create the ranking agent
    agent = RankingAgent(
        research_goal=research_goal,
        hypotheses=hypotheses,
        reviews=reviews,
        tournament_state=tournament_state,
        context_memory=context_memory
    )
    
    # Run tournament ranking
    result = agent.actor()
    
    return {
        "messages": state["messages"] + [AIMessage(content=f"Updated hypothesis rankings through tournament evaluation.")],
        "tournament_state": result["updated_tournament_state"],
        "context_memory": result["updated_context_memory"],
        "current_task": "",  # Clear current task as it's completed
    }

def run_evolution_agent(state: Dict[str, Any], config: RunnableConfig):
    """
    Evolution agent that refines and improves top-ranked hypotheses.
    """
    research_goal = state.get("research_goal", "")
    context_memory = state.get("context_memory", {})
    hypotheses = state.get("hypotheses", [])
    tournament_state = state.get("tournament_state", {})
    
    # Get top-ranked hypotheses to evolve
    ranked_hypotheses = sorted(
        hypotheses, 
        key=lambda h: tournament_state.get("rankings", {}).get(h.get("id", ""), 0),
        reverse=True
    )
    top_hypotheses = ranked_hypotheses[:3]  # Take top 3 for evolution
    
    # Create the evolution agent
    agent = EvolutionAgent(
        research_goal=research_goal,
        hypotheses=top_hypotheses,
        context_memory=context_memory
    )
    
    # Evolve hypotheses
    result = agent.actor()
    
    # Update the state with evolved hypotheses
    combined_hypotheses = [h for h in hypotheses if h.get("id") not in [evolved.get("parent_id") for evolved in result["evolved_hypotheses"]]]
    combined_hypotheses.extend(result["evolved_hypotheses"])
    
    return {
        "messages": state["messages"] + [AIMessage(content=f"Evolved {len(result['evolved_hypotheses'])} hypotheses.")],
        "hypotheses": combined_hypotheses,
        "context_memory": result["updated_context_memory"],
        "current_task": "",  # Clear current task as it's completed
    }

def run_proximity_agent(state: Dict[str, Any], config: RunnableConfig):
    """
    Proximity agent that computes similarity between hypotheses for clustering.
    """
    research_goal = state.get("research_goal", "")
    context_memory = state.get("context_memory", {})
    hypotheses = state.get("hypotheses", [])
    proximity_graph = state.get("proximity_graph", {})
    
    # Create the proximity agent
    agent = ProximityAgent(
        research_goal=research_goal,
        hypotheses=hypotheses,
        proximity_graph=proximity_graph,
        context_memory=context_memory
    )
    
    # Compute proximity
    result = agent.actor()
    
    return {
        "messages": state["messages"] + [AIMessage(content=f"Updated proximity graph for hypothesis clustering.")],
        "proximity_graph": result["updated_proximity_graph"],
        "context_memory": result["updated_context_memory"],
        "current_task": "",  # Clear current task as it's completed
    }

def run_meta_review_agent(state: Dict[str, Any], config: RunnableConfig):
    """
    Meta-review agent that synthesizes insights from reviews and tournament debates.
    """
    research_goal = state.get("research_goal", "")
    context_memory = state.get("context_memory", {})
    hypotheses = state.get("hypotheses", [])
    reviews = state.get("reviews", [])
    tournament_state = state.get("tournament_state", {})
    
    # Get top hypotheses
    ranked_hypotheses = sorted(
        hypotheses, 
        key=lambda h: tournament_state.get("rankings", {}).get(h.get("id", ""), 0),
        reverse=True
    )
    top_hypotheses = ranked_hypotheses[:5]  # Take top 5 for meta-review
    
    # Create the meta-review agent
    agent = MetaReviewAgent(
        research_goal=research_goal,
        hypotheses=top_hypotheses,
        reviews=reviews,
        tournament_state=tournament_state,
        context_memory=context_memory
    )
    
    # Generate meta-review
    result = agent.actor()
    
    # Prepare the research overview
    research_overview = result["research_overview"]
    
    return {
        "messages": state["messages"] + [AIMessage(content=f"Generated comprehensive research overview:\n\n{research_overview}")],
        "context_memory": result["updated_context_memory"],
        "current_task": "",  # Clear current task as it's completed
        "iteration": state.get("iteration", 0) + 1  # Increment iteration counter
    }

def route_next_task(state: Dict[str, Any], config: RunnableConfig) -> str:
    """
    Route to the next task based on the supervisor's task queue.
    """
    current_task = state.get("current_task", "")
    task_queue = state.get("task_queue", [])
    
    # If no current task, we need the supervisor to assign one
    if not current_task:
        if not task_queue:
            return "run_supervisor"
        else:
            # Update current task from the queue
            current_task = task_queue[0]
            state["current_task"] = current_task
            state["task_queue"] = task_queue[1:]  # Remove the first task
    
    # Map task names to node names
    task_to_node = {
        "generation": "run_generation_agent",
        "reflection": "run_reflection_agent",
        "ranking": "run_ranking_agent",
        "evolution": "run_evolution_agent",
        "proximity": "run_proximity_agent",
        "meta_review": "run_meta_review_agent",
        "supervisor": "run_supervisor",
        "end": END,
        # Also include direct node names for backward compatibility
        "run_generation_agent": "run_generation_agent",
        "run_reflection_agent": "run_reflection_agent",
        "run_ranking_agent": "run_ranking_agent",
        "run_evolution_agent": "run_evolution_agent",
        "run_proximity_agent": "run_proximity_agent",
        "run_meta_review_agent": "run_meta_review_agent",
        "run_supervisor": "run_supervisor"
    }
    
    # Return the appropriate node name or default to supervisor
    return task_to_node.get(current_task, "run_supervisor")

def get_initial_state(query: str = "Research goal: Develop novel methods for carbon capture and sequestration."):
    """
    Initialize the state with the research goal only.
    Other fields will be initialized as needed by the agents.
    """
    return {
        "messages": [
            SystemMessage(content="I am a Co-Scientist AI assistant designed to help with scientific research."),
            HumanMessage(content=query)
        ],
        "research_goal": query,
        "current_task": "",
        "task_queue": ["supervisor"],  # Start with supervisor
    }

# Build the graph
workflow = StateGraph(CoScientistState)

# Add nodes for each agent
workflow.add_node("run_supervisor", run_supervisor)
workflow.add_node("run_generation_agent", run_generation_agent)
workflow.add_node("run_reflection_agent", run_reflection_agent)
workflow.add_node("run_ranking_agent", run_ranking_agent)
workflow.add_node("run_evolution_agent", run_evolution_agent)
workflow.add_node("run_proximity_agent", run_proximity_agent)
workflow.add_node("run_meta_review_agent", run_meta_review_agent)

# Add edges based on the routing function
workflow.add_conditional_edges(
    START,
    route_next_task,
    {
        "run_supervisor": "run_supervisor",
        "run_generation_agent": "run_generation_agent",
        "run_reflection_agent": "run_reflection_agent",
        "run_ranking_agent": "run_ranking_agent",
        "run_evolution_agent": "run_evolution_agent",
        "run_proximity_agent": "run_proximity_agent",
        "run_meta_review_agent": "run_meta_review_agent",
        "end": END
    }
)

workflow.add_conditional_edges(
    "run_supervisor",
    route_next_task,
    {
        "run_generation_agent": "run_generation_agent",
        "run_reflection_agent": "run_reflection_agent",
        "run_ranking_agent": "run_ranking_agent",
        "run_evolution_agent": "run_evolution_agent",
        "run_proximity_agent": "run_proximity_agent",
        "run_meta_review_agent": "run_meta_review_agent",
        "end": END
    }
)

workflow.add_conditional_edges(
    "run_generation_agent",
    route_next_task,
    {
        "run_supervisor": "run_supervisor",
        "run_reflection_agent": "run_reflection_agent",
        "run_ranking_agent": "run_ranking_agent",
        "run_evolution_agent": "run_evolution_agent",
        "run_proximity_agent": "run_proximity_agent",
        "run_meta_review_agent": "run_meta_review_agent",
        "end": END
    }
)

workflow.add_conditional_edges(
    "run_reflection_agent",
    route_next_task,
    {
        "run_supervisor": "run_supervisor",
        "run_generation_agent": "run_generation_agent",
        "run_ranking_agent": "run_ranking_agent",
        "run_evolution_agent": "run_evolution_agent",
        "run_proximity_agent": "run_proximity_agent",
        "run_meta_review_agent": "run_meta_review_agent",
        "end": END
    }
)

workflow.add_conditional_edges(
    "run_ranking_agent",
    route_next_task,
    {
        "run_supervisor": "run_supervisor",
        "run_generation_agent": "run_generation_agent",
        "run_reflection_agent": "run_reflection_agent",
        "run_evolution_agent": "run_evolution_agent",
        "run_proximity_agent": "run_proximity_agent",
        "run_meta_review_agent": "run_meta_review_agent",
        "end": END
    }
)

workflow.add_conditional_edges(
    "run_evolution_agent",
    route_next_task,
    {
        "run_supervisor": "run_supervisor",
        "run_generation_agent": "run_generation_agent",
        "run_reflection_agent": "run_reflection_agent",
        "run_ranking_agent": "run_ranking_agent",
        "run_proximity_agent": "run_proximity_agent",
        "run_meta_review_agent": "run_meta_review_agent",
        "end": END
    }
)

workflow.add_conditional_edges(
    "run_proximity_agent",
    route_next_task,
    {
        "run_supervisor": "run_supervisor",
        "run_generation_agent": "run_generation_agent",
        "run_reflection_agent": "run_reflection_agent",
        "run_ranking_agent": "run_ranking_agent",
        "run_evolution_agent": "run_evolution_agent",
        "run_meta_review_agent": "run_meta_review_agent",
        "end": END
    }
)

workflow.add_conditional_edges(
    "run_meta_review_agent",
    route_next_task,
    {
        "run_supervisor": "run_supervisor",
        "run_generation_agent": "run_generation_agent",
        "run_reflection_agent": "run_reflection_agent",
        "run_ranking_agent": "run_ranking_agent",
        "run_evolution_agent": "run_evolution_agent",
        "run_proximity_agent": "run_proximity_agent",
        "end": END
    }
)

# Compile the graph
graph = workflow.compile()

# Define entry point for our workflow
def run(query: str = "Research goal: Develop novel methods for carbon capture and sequestration."):
    """Run the Co-Scientist workflow with the given research goal."""
    config = {"recursion_limit": 50}
    initial_state = get_initial_state(query)
    return graph.invoke(initial_state, config) 