# Google Co-Scientist
[Google Co-scientist](https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/)

Given a scientist’s research goal that has been specified in natural language, the AI co-scientist is designed to generate novel research hypotheses, a detailed research overview, and experimental protocols. To do so, it uses a coalition of specialized agents — Generation, Reflection, Ranking, Evolution, Proximity and Meta-review — that are inspired by the scientific method itself. These agents use automated feedback to iteratively generate, evaluate, and refine hypotheses, resulting in a self-improving cycle of increasingly high-quality and novel outputs.

![image](https://github.com/user-attachments/assets/3b6708eb-8c43-4721-aad1-cc424724f8c8)

The AI co-scientist parses the assigned goal into a research plan configuration, managed by a Supervisor agent. The Supervisor agent assigns the specialized agents to the worker queue and allocates resources. This design enables the system to flexibly scale compute and to iteratively improve its scientific reasoning towards the specified research goal.

![image](https://github.com/user-attachments/assets/beaf7569-faf9-47d7-b088-ac9ed79eab87)

At a high level, the co-scientist system comprises four key components:

## Natural language interface
Scientists interact with and supervise the system primarily through
natural language. This allows them to not only define the initial research goal but also refine it at any
time, provide feedback on generated hypotheses (including their own solutions), and generally guide the
system’s progress.

## Asynchronous task framework 
The co-scientist employs a multi-agent system where specialized
agents operate as worker processes within an asynchronous, continuous, and configurable task execution
framework. A dedicated Supervisor agent manages the worker task queue, assigns specialized agents to
these processes, and allocates resources. This design enables the system to flexibly and effectively utilize
computational resources and iteratively improve its scientific reasoning capabilities.

## Specialized agents 
Following inductive biases and scientific priors derived from the scientific method,
the process of scientific reasoning and hypothesis generation is broken down into sub-tasks. Individual,
specialized agents, each equipped with customized instruction prompts, are designed to execute these
sub-tasks. These agents operate as workers coordinated by the Supervisor agent.
In summary, the Generation agent curates an initial list of research hypotheses satisfying a research goal.
These are then reviewed by the Reflection agent and evaluated in a tournament by the Ranking agent. The
Evolution, Proximity, and Meta-review agents operate on the tournament state to help improve the quality of
the system outputs.

### Supervisor agent
The Supervisor agent’s seamless orchestration of these specialized agents enables the development of valid,
novel, and testable hypotheses and research plans tailored to the input research goal.
The Supervisor agent periodically computes and writes to the context memory, a comprehensive suite of
statistics, including the number of hypotheses generated and requiring review, and the progress of the
tournament. These statistics also include analyses of the effectiveness of different hypothesis generation
methodologies (e.g., generating new ideas via the Generation agent vs. improving existing ideas via the
Evolution agent). Based on these statistics, the Supervisor agent then orchestrates subsequent system
operations, i.e., generating new hypotheses, reviews, tournaments, and improvements to existing hypotheses,
by strategically weighting and sampling the specialized agents for execution via the worker process

### Generation agent
The agent initiates the research process by generating the initial focus areas,
iteratively extending them and generating a set of initial hypotheses and proposals that address the
research goal. This involves exploring relevant literature using web search, synthesizing existing findings
into novel directions, and engaging in simulated scientific debates for iterative improvement.

### Reflection agent
This agent simulates the role of a scientific peer reviewer, critically examining the
correctness, quality, and novelty of the generated hypotheses and research proposals. Furthermore, it
evaluates the potential of each hypothesis to provide an improved explanation for existing research
observations (identified via literature search and review), particularly those that may be under explained.

### Ranking agent
An important abstraction in the co-scientist system is the notion of a tournament
where different research proposals are evaluated and ranked enabling iterative improvements. The
Ranking agent employs and orchestrates an Elo-based tournament [61] to assess and prioritize the
generated hypotheses at any given time. This involves pairwise comparisons, facilitated by simulated
scientific debates, which allow for a nuanced evaluation of the relative merits of each proposal.

### Proximity agent
This agent asynchronously computes a proximity graph for generated hypotheses,
enabling clustering of similar ideas, de-duplication, and efficient exploration of the hypothesis landscape.

### Evolution agent
The co-scientist’s iterative improvement capability relies heavily on this agent, which
continuously refines the top-ranked hypotheses emerging from the tournament. Its refinement strategies
include synthesizing existing ideas, using analogies, leveraging literature for supporting details, exploring
unconventional reasoning, and simplifying concepts for clarity.

### Meta-review agent
This agent also enables the co-scientist’s continuous improvement by synthesizing
insights from all reviews, identifying recurring patterns in tournament debates, and using these findings
to optimize other agents’ performance in subsequent iterations. This also enhances the quality and
relevance of generated hypotheses and reviews in subsequent iterations. The agent also synthesizes
top-ranked hypotheses and reviews into a comprehensive research overview for review by the scientist

## Context memory
In order to enable iterative computation and scientific reasoning over long time
horizons, the co-scientist uses a persistent context memory to store and retrieve states of the agents and
the system during the course of the computation.
The Gemini 2.0 model is the foundational LLM underpinning all agents in the co-scientist system. The specific
co-scientist design was arrived at with iterative developments and is reflective of the current capabilities of
the underlying LLMs.

