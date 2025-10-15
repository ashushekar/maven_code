# Perplexia AI Curriculum

## Overview
Perplexia AI is a multi-week LangChain and LangGraph learning project that walks through progressively more capable chat agents. Each week adds orchestration patterns, retrieval workflows, and tool integrations that students can explore through a Gradio front end or by importing the Python modules directly.

## Key Capabilities by Week
- **Week 1 – Prompting Foundations**: Guided response formatting, calculator tool planning, and stateful memory branching.
- **Week 2 – Retrieval Systems**: Live web search via Tavily, PDF-based RAG with structured outputs, and corrective retrieval that merges document and web evidence.
- **Week 3 – Agentic Workflows**: Tool-using agents, iterative research graphs, and multi-agent deep research with bookmarking MCP integrations.

## Repository Layout
```
code/
├── run.py                 # CLI entry point that launches the Gradio demo
├── perplexia_ai/
│   ├── app.py             # Gradio wiring and factory dispatch
│   ├── core/              # Shared chat abstractions
│   ├── tools/             # Reusable LangChain-compatible utilities
│   ├── week1/             # Prompting, tools, and memory agents
│   ├── week2/             # Retrieval augmented generation agents
│   ├── week3/             # Agentic and multi-agent workflows
│   └── docs/              # Architecture diagram and reference PDFs
└── requirements.txt       # Python dependencies for all lessons
```

## Getting Started
1. **Install dependencies**
   ```bash
   pip install -r code/requirements.txt
   ```
2. **Set API credentials**
   - Copy `.env.example` (if available) or create a `.env` file alongside `code/perplexia_ai/app.py`.
   - Provide credentials for OpenAI (or compatible) chat models, Tavily search, and any optional MCP servers you plan to use.

## Running the Demo
Launch a given week and part from the project root:
```bash
python code/run.py --week 2 --mode part3
```
The Gradio interface exposes preset examples and streams responses from the selected agent implementation.

## Architecture & Documentation
- Editable system diagram: [`code/perplexia_ai/docs/system_architecture.drawio`](code/perplexia_ai/docs/system_architecture.drawio)
- Narrative walkthrough: [`code/perplexia_ai/docs/system_architecture.md`](code/perplexia_ai/docs/system_architecture.md)

These documents describe how the UI layer, orchestration factories, and week-specific modules collaborate, including shared tooling and external dependencies.

## Development Tips
- Every chat mode implements the `ChatInterface` base class located in `code/perplexia_ai/core/chat_interface.py`.
- Week factories expose `create_chat_implementation` helpers so new experiments can swap in alternative agents without changing the UI wiring.
- Use tracing (e.g., Opik or LangSmith) when debugging LangGraph flows; hooks can be configured where the graphs are instantiated inside the week modules.
