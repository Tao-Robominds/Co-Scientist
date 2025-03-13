from dotenv import load_dotenv
import asyncio

from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import RawResponsesStreamEvent

load_dotenv()


# Define specialized agents
generation_agent = Agent(
    name="generation_agent",
    instructions=(
        "You are a Generation agent that initiates the research process by generating initial focus areas, "
        "extending them, and generating hypotheses that address the research goal. Explore relevant literature "
        "using web search, synthesize existing findings into novel directions, and engage in simulated scientific "
        "debates for iterative improvement. Return a list of innovative hypotheses related to the research goal."
    ),
    handoff_description="Generates initial hypotheses for research goals",
)

reflection_agent = Agent(
    name="reflection_agent",
    instructions=(
        "You are a Reflection agent that acts as a scientific peer reviewer. Critically examine the correctness, "
        "quality, and novelty of hypotheses and research proposals. Evaluate each hypothesis's potential to provide "
        "improved explanations for existing research observations. Provide detailed and constructive feedback."
    ),
    handoff_description="Reviews and provides feedback on hypotheses",
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

# Supervisor agent that orchestrates other agents as tools
supervisor_agent = Agent(
    name="supervisor_agent",
    instructions=(
        "You are the Supervisor agent for the Co-Scientist system. Your role is to orchestrate specialized agents "
        "to generate, evaluate, and refine scientific hypotheses. Based on the research goal, you should:"
        "\n1. Parse the research goal and design a research plan"
        "\n2. Strategically call specialized agents as tools when needed"
        "\n3. Track progress and maintain statistics on hypothesis generation and evaluation"
        "\n4. Guide the iterative improvement of hypotheses"
        "\n5. Synthesize final results into a comprehensive research overview"
        "\nYou should always use your tools rather than attempting to perform their functions yourself."
    ),
    tools=[
        generation_agent.as_tool(
            tool_name="generate_hypotheses",
            tool_description="Generate initial hypotheses for the research goal",
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

async def main():
    # Get research goal from user
    research_goal = input("Enter your scientific research goal: ")
    
    # Run the entire orchestration in a single trace
    with trace("Co-Scientist workflow"):
        print("\nSupervisor agent starting orchestration process...\n")
        
        # Use streaming instead of waiting for complete results
        result = Runner.run_streamed(supervisor_agent, research_goal)
        
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



