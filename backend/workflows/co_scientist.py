from dotenv import load_dotenv
import asyncio
from dataclasses import dataclass
from typing import Literal

from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace, TResponseInputItem, WebSearchTool
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import RawResponsesStreamEvent

load_dotenv()

"""
Co-Scientist System Implementation
"""

# Define specialized agents
generation_agent = Agent(
    name="generation_agent",
    instructions=(
        "You are a Generation agent that initiates the research process by generating initial focus areas, "
        "extending them, and generating hypotheses that address the research goal. Explore relevant literature "
        "using web search, synthesize existing findings into novel directions, and engage in simulated scientific "
        "debates for iterative improvement. Return a list of innovative hypotheses related to the research goal."
        "\n\nWhen generating hypotheses, first search the web for the latest research and findings related to the "
        "research goal. Use these search results to inform your hypothesis generation, ensuring they reflect "
        "current scientific understanding and identify potential gaps in existing research."
    ),
    handoff_description="Generates initial hypotheses for research goals",
    tools=[WebSearchTool()],
)

# Create a specialized agent for hypothesis selection
hypothesis_selector_agent = Agent(
    name="hypothesis_selector_agent",
    instructions=(
        "You are a Hypothesis Selector agent that evaluates multiple sets of hypotheses and selects the most "
        "promising ones. Review all provided hypotheses and select the most promising, diverse, and innovative "
        "candidates. Provide a consolidated list of the strongest and most diverse hypotheses, and explain why "
        "you selected each one. Aim to select 3-5 hypotheses that together represent the strongest research directions."
    ),
    handoff_description="Selects the most promising hypotheses from multiple generated sets",
)

reflection_agent = Agent(
    name="reflection_agent",
    instructions=(
        "You are a Reflection agent that acts as a scientific peer reviewer. Critically examine the correctness, "
        "quality, and novelty of hypotheses and research proposals. Evaluate each hypothesis's potential to provide "
        "improved explanations for existing research observations. Provide detailed and constructive feedback."
        "\n\nUse web search to find recent publications, preprints, and scientific discussions related to the "
        "hypotheses. Use this information to assess the currency, novelty, and scientific merit of each hypothesis."
        "\n\nIf provided with feedback about active/current research areas, focus on improving hypotheses to "
        "align with cutting-edge research and eliminate outdated lines of inquiry."
        "\n\nWhen suggesting improvements, cite specific papers or research findings from your web searches."
    ),
    handoff_description="Reviews and provides feedback on hypotheses",
    tools=[WebSearchTool()],
)

ranking_agent = Agent(
    name="ranking_agent",
    instructions=(
        "You are a Ranking agent that employs an Elo-based tournament to assess and prioritize hypotheses. "
        "Conduct pairwise comparisons of hypotheses through simulated scientific debates, evaluating their "
        "relative merits. Rank hypotheses based on their scientific validity, novelty, and potential impact."
    ),
    handoff_description="Ranks hypotheses through tournament-style evaluation",
)

proximity_agent = Agent(
    name="proximity_agent",
    instructions=(
        "You are a Proximity agent that computes a proximity graph for generated hypotheses. Identify clusters "
        "of similar ideas, perform de-duplication, and enable efficient exploration of the hypothesis landscape. "
        "Provide a map of how hypotheses relate to each other in the research space."
    ),
    handoff_description="Maps relationships between hypotheses",
)

evolution_agent = Agent(
    name="evolution_agent",
    instructions=(
        "You are an Evolution agent that continuously refines top-ranked hypotheses. Use strategies like "
        "synthesizing existing ideas, leveraging analogies, incorporating literature for support, exploring "
        "unconventional reasoning, and simplifying concepts for clarity. Return improved versions of the hypotheses."
    ),
    handoff_description="Refines and improves promising hypotheses",
)

meta_review_agent = Agent(
    name="meta_review_agent",
    instructions=(
        "You are a Meta-review agent that synthesizes insights from all reviews and tournament debates. "
        "Identify recurring patterns, use findings to optimize performance, and enhance the quality of generated "
        "hypotheses. Synthesize top-ranked hypotheses and reviews into a comprehensive research overview."
    ),
    handoff_description="Synthesizes insights and creates research overviews",
)

# Define a dataclass for hypothesis evaluation feedback
@dataclass
class HypothesisEvaluation:
    status: Literal["all_active", "needs_refinement"]
    feedback: str
    active_areas: list[str]
    outdated_areas: list[str]

# Create a hypothesis evaluator agent
hypothesis_evaluator = Agent[None](
    name="hypothesis_evaluator",
    instructions=(
        "You are a Hypothesis Evaluator agent that assesses whether proposed hypotheses are in active, "
        "current research areas. For each hypothesis, determine if it represents a field with ongoing active "
        "research or if it's an outdated or settled area of inquiry."
        "\n\nUse web search to find the latest research papers and scientific discussions related to each hypothesis. "
        "This will help you accurately determine if a research area is currently active."
        "\n\nProvide specific feedback on which hypotheses are in active research areas and which are not. "
        "Be precise about why a hypothesis might be outdated or why it represents cutting-edge research. "
        "\n\nClassify your evaluation as 'all_active' only if ALL hypotheses represent current active research. "
        "Otherwise, classify as 'needs_refinement' and provide detailed feedback on what needs to change."
        "\n\nEarly iterations should typically be classified as 'needs_refinement' to encourage exploration of "
        "more recent research directions."
        "\n\nInclude citations to recent papers or research when providing feedback on hypothesis currency."
    ),
    output_type=HypothesisEvaluation,
    tools=[WebSearchTool()],
)

# Supervisor agent with parallel hypothesis generation capability
supervisor_agent = Agent(
    name="supervisor_agent",
    instructions=(
        "You are the Supervisor agent for the Co-Scientist system. Your role is to orchestrate specialized agents "
        "to generate, evaluate, and refine scientific hypotheses. Based on the research goal, you should:"
        "\n1. Parse the research goal and design a research plan"
        "\n2. Strategically call specialized agents as tools when needed"
        "\n3. Generate multiple sets of hypotheses in parallel and select the most promising ones"
        "\n4. Track progress and maintain statistics on hypothesis generation and evaluation"
        "\n5. Guide the iterative improvement of hypotheses"
        "\n6. Synthesize final results into a comprehensive research overview"
        "\nYou should always use your tools rather than attempting to perform their functions yourself."
    ),
    tools=[
        generation_agent.as_tool(
            tool_name="generate_hypotheses",
            tool_description="Generate initial hypotheses for the research goal",
        ),
        hypothesis_selector_agent.as_tool(
            tool_name="select_hypotheses",
            tool_description="Select the most promising hypotheses from multiple sets",
        ),
        reflection_agent.as_tool(
            tool_name="review_hypotheses",
            tool_description="Review and provide feedback on the proposed hypotheses",
        ),
        ranking_agent.as_tool(
            tool_name="rank_hypotheses",
            tool_description="Rank hypotheses through tournament-style evaluation",
        ),
        proximity_agent.as_tool(
            tool_name="map_hypothesis_relationships",
            tool_description="Analyze relationships and similarities between hypotheses",
        ),
        evolution_agent.as_tool(
            tool_name="refine_hypotheses",
            tool_description="Refine and improve promising hypotheses",
        ),
        meta_review_agent.as_tool(
            tool_name="synthesize_research",
            tool_description="Create a comprehensive research overview from top hypotheses",
        ),
    ],
)

async def evaluate_and_refine_hypotheses(research_goal, initial_hypotheses):
    """Iteratively evaluate and refine hypotheses until all are in active research areas"""
    print("\n--- Evaluating if hypotheses are in active research areas ---\n")
    
    current_hypotheses = initial_hypotheses
    iteration = 1
    input_items: list[TResponseInputItem] = [
        {"content": f"Research Goal: {research_goal}\n\nHypotheses:\n{current_hypotheses}", "role": "user"}
    ]
    
    while True:
        print(f"\nEvaluation iteration #{iteration}")
        
        # Evaluate if hypotheses are in active research areas
        evaluation_result = await Runner.run(hypothesis_evaluator, input_items)
        evaluation: HypothesisEvaluation = evaluation_result.final_output
        
        print(f"\nEvaluation status: {evaluation.status}")
        print(f"Active research areas: {', '.join(evaluation.active_areas)}")
        print(f"Outdated/inactive areas: {', '.join(evaluation.outdated_areas)}")
        
        if evaluation.status == "all_active":
            print("\nAll hypotheses are in active research areas. Moving forward.")
            break
        
        print("\nRefining hypotheses based on evaluation feedback...")
        
        # Add feedback to the input for reflection agent
        input_items.append({"content": f"Feedback on research currency: {evaluation.feedback}", "role": "user"})
        
        # Run reflection agent to refine hypotheses
        reflection_result = await Runner.run(reflection_agent, input_items)
        
        # Update current hypotheses
        input_items = reflection_result.to_input_list()
        current_hypotheses = ItemHelpers.text_message_outputs(reflection_result.new_items)
        
        print(f"\nRefined hypotheses (iteration #{iteration}):\n{current_hypotheses}")
        
        iteration += 1
        if iteration > 3:  # Limit to 3 iterations to prevent infinite loops
            print("\nReached maximum iterations. Proceeding with current hypotheses.")
            break
    
    return current_hypotheses

async def generate_hypotheses_in_parallel(research_goal):
    """Generate three sets of hypotheses in parallel"""
    print("\n--- Generating hypotheses in parallel ---\n")
    
    hypothesis_results = await asyncio.gather(
        Runner.run(generation_agent, f"Research Goal: {research_goal}\nGeneration Set #1: Generate innovative and diverse hypotheses. First, search the web for recent research related to this goal to inform your hypotheses."),
        Runner.run(generation_agent, f"Research Goal: {research_goal}\nGeneration Set #2: Focus on potentially transformative hypotheses. Use web search to find cutting-edge research in this area before generating your hypotheses."),
        Runner.run(generation_agent, f"Research Goal: {research_goal}\nGeneration Set #3: Consider unconventional approaches to the research goal. Search the web for emerging or cross-disciplinary approaches to this research area.")
    )
    
    # Extract outputs from each generation
    hypothesis_sets = []
    for i, result in enumerate(hypothesis_results, 1):
        hypotheses = ItemHelpers.text_message_outputs(result.new_items)
        hypothesis_sets.append(f"Hypothesis Set #{i}:\n{hypotheses}")
    
    # Join the sets with separators
    all_hypotheses = "\n\n".join(hypothesis_sets)
    
    # Select the best hypotheses using the selector agent
    selection_result = await Runner.run(
        hypothesis_selector_agent,
        f"Research Goal: {research_goal}\n\nGenerated Hypotheses:\n{all_hypotheses}\n\nPlease select and consolidate the most promising hypotheses."
    )
    
    return selection_result.final_output

async def main():
    # Get research goal from user
    research_goal = input("Enter your scientific research goal: ")
    
    # Run the entire orchestration in a single trace
    with trace("Co-Scientist workflow"):
        # Generate hypotheses in parallel
        print("\nGenerating hypotheses in parallel...\n")
        selected_hypotheses = await generate_hypotheses_in_parallel(research_goal)
        print(f"\n--- Selected Hypotheses ---\n{selected_hypotheses}\n")
        
        # Evaluate and refine hypotheses to ensure they're in active research areas
        active_hypotheses = await evaluate_and_refine_hypotheses(research_goal, selected_hypotheses)
        print(f"\n--- Active Research Hypotheses ---\n{active_hypotheses}\n")
        
        # Pass the active hypotheses to the supervisor for further processing
        research_input = f"Research Goal: {research_goal}\n\nActive Research Hypotheses:\n{active_hypotheses}\n\nPlease continue the research process with these hypotheses."
        
        print("\nSupervisor agent starting orchestration process...\n")
        
        # Use streaming instead of waiting for complete results
        result = Runner.run_streamed(supervisor_agent, research_input)
        
        # Process and display results as they stream in
        current_agent = "supervisor_agent"
        print(f"\n--- Now working: {current_agent} ---")
        
        final_output = ""
        async for event in result.stream_events():
            # Display raw text as it's generated
            if isinstance(event, RawResponsesStreamEvent):
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)
                    # Capture the final output as it's being streamed
                    if current_agent == "supervisor_agent":
                        final_output += data.delta
                elif isinstance(data, ResponseContentPartDoneEvent):
                    print("\n")
            
            # When an agent changes, announce it
            if hasattr(event, 'agent_name') and event.agent_name != current_agent:
                current_agent = event.agent_name
                print(f"\n\n--- Now working: {current_agent} ---\n")
        
        # Display final research overview
        print(f"\n\n=== Final Research Overview ===\n{final_output}")

if __name__ == "__main__":
    asyncio.run(main())



