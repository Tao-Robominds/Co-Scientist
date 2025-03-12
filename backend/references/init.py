from dotenv import load_dotenv
load_dotenv()

from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner, RunContextWrapper
from agents.exceptions import InputGuardrailTripwireTriggered
from pydantic import BaseModel
import asyncio

# Define UserContext
class UserContext(BaseModel):
    name: str = "Student"

class HomeworkOutput(BaseModel):
    is_homework: bool
    reasoning: str

guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking about homework.",
    output_type=HomeworkOutput,
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)


async def homework_guardrail(ctx, agent, input_data):
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    final_output = result.final_output_as(HomeworkOutput)
    return GuardrailFunctionOutput(
        output_info=final_output,
        tripwire_triggered=not final_output.is_homework,
    )

def dynamic_instructions(
    context: RunContextWrapper[UserContext], agent: Agent[UserContext]
) -> str:
    return f"You determine which agent to use based on user's homework question. Always say hi to {context.context.name} in your response."


triage_agent = Agent(
    name="Triage Agent",
    instructions=dynamic_instructions,
    handoffs=[history_tutor_agent, math_tutor_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=homework_guardrail),
    ],
)

async def main():
    user_context = UserContext(name="Boringtao")
    
    try:
        print("Processing first query...")
        result = await Runner.run(triage_agent, "who was the first president of the united states?", context=user_context)
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print(f"First query was rejected by guardrail: Not a homework question")
    except Exception as e:
        print(f"Error processing first query: {e}")

    try:
        print("\nProcessing second query...")
        result = await Runner.run(triage_agent, "what is life", context=user_context)
        print(result.final_output)
    except InputGuardrailTripwireTriggered as e:
        print(f"Second query was rejected by guardrail: Not a homework question")
    except Exception as e:
        print(f"Error processing second query: {e}")

if __name__ == "__main__":
    asyncio.run(main())
