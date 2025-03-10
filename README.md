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

## Context memory
In order to enable iterative computation and scientific reasoning over long time
horizons, the co-scientist uses a persistent context memory to store and retrieve states of the agents and
the system during the course of the computation.
The Gemini 2.0 model is the foundational LLM underpinning all agents in the co-scientist system. The specific
co-scientist design was arrived at with iterative developments and is reflective of the current capabilities of
the underlying LLMs.

