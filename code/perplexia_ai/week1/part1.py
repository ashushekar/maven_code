# part1.py
"""
Part 1 - Query Understanding (LangGraph version)

Graph:
    START -> classify -> (conditional route) -> respond_* -> END

State carries:
- question: str (input)
- category: Literal[...] (set by classifier)
- answer: str (final output)
"""

from typing import Dict, List, Optional, Literal, TypedDict


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, END
from perplexia_ai.core.chat_interface import ChatInterface


# ---------- Graph State ----------

Category = Literal["factual", "analytical", "comparison", "definition", "default"]

class QAState(TypedDict, total=False):
    question: str
    category: Category
    answer: str


# ---------- Prompts ----------

CLASSIFIER_PROMPT = ChatPromptTemplate.from_template(
    """
    Given the user question, classify into one of the following 4 categories with expected definitions:

    - Factual Questions ("What is...?", "Who invented...?") : Return 'factual'
    - Analytical Questions ("How does...?", "Why do...?") : Return 'analytical'
    - Comparison Questions ("What's the difference between...?") : Return 'comparison'
    - Definition Requests ("Define...", "Explain...") : Return 'definition'

    ONLY return the category word as the answer and nothing else. Do not add quotes.
    Return 'default' if the question does not fit into any of the categories.

    Prompting by examples:
    * Question: What is the highest mountain in the world?
    * Response: factual

    * Question: What's the difference between OpenAI and Anthropic?
    * Response: comparison

    User question: {question}
    """
)

RESPONSE_PROMPTS: Dict[Category, ChatPromptTemplate] = {
    "factual": ChatPromptTemplate.from_template(
        """
        Answer the following question concisely with a direct fact. Avoid unnecessary details.

        User question: "{question}"
        Answer:
        """
    ),
    "analytical": ChatPromptTemplate.from_template(
        """
        Provide a detailed explanation with reasoning for the following question. Break down the response into logical steps.

        User question: "{question}"
        Explanation:
        """
    ),
    "comparison": ChatPromptTemplate.from_template(
        """
        Compare the following concepts. Present the answer in a structured format using bullet points or a table for clarity.

        User question: "{question}"
        Comparison:
        """
    ),
    "definition": ChatPromptTemplate.from_template(
        """
        Define the following term and provide relevant examples and use cases for better understanding.

        User question: "{question}"
        Definition:
        Examples:
        Use Cases:
        """
    ),
    "default": ChatPromptTemplate.from_template(
        """
        Respond your best to answer the following question but keep it very brief.

        User question: "{question}"
        Answer:
        """
    ),
}


# ---------- Implementation ----------

class QueryUnderstandingChat(ChatInterface):
    """
    Week 1 Part 1 rewritten using LangGraph.

    - Node `classify` sets category in state
    - Conditional edges route to the correct responder node
    - Responder node writes the final `answer` to state
    """

    def __init__(self, model: str = "gpt-5", temperature: float = 0.0):
        self.model_name = model
        self.temperature = temperature
        self.llm = None
        self.app = None  # LangGraph compiled app

    def initialize(self) -> None:
        # LLM (OpenAI via langchain-openai)
        self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)

        # Chains
        self.classifier_chain = CLASSIFIER_PROMPT | self.llm | StrOutputParser()
        self.response_chains = {
            key: (tmpl | self.llm | StrOutputParser())
            for key, tmpl in RESPONSE_PROMPTS.items()
        }

        # Graph definition
        graph = StateGraph(QAState)

        # --- Nodes ---
        def classify(state: QAState) -> QAState:
            category = self.classifier_chain.invoke({"question": state["question"]}).strip().lower()
            # Safety clamp
            if category not in RESPONSE_PROMPTS:
                category = "default"
            return {"category": category}

        def respond_factual(state: QAState) -> QAState:
            ans = self.response_chains["factual"].invoke({"question": state["question"]})
            return {"answer": ans}

        def respond_analytical(state: QAState) -> QAState:
            ans = self.response_chains["analytical"].invoke({"question": state["question"]})
            return {"answer": ans}

        def respond_comparison(state: QAState) -> QAState:
            ans = self.response_chains["comparison"].invoke({"question": state["question"]})
            return {"answer": ans}

        def respond_definition(state: QAState) -> QAState:
            ans = self.response_chains["definition"].invoke({"question": state["question"]})
            return {"answer": ans}

        def respond_default(state: QAState) -> QAState:
            ans = self.response_chains["default"].invoke({"question": state["question"]})
            return {"answer": ans}

        graph.add_node("classify", classify)
        graph.add_node("respond_factual", respond_factual)
        graph.add_node("respond_analytical", respond_analytical)
        graph.add_node("respond_comparison", respond_comparison)
        graph.add_node("respond_definition", respond_definition)
        graph.add_node("respond_default", respond_default)

        # --- Edges ---
        graph.set_entry_point("classify")

        def route_by_category(state: QAState) -> str:
            cat = state.get("category", "default")
            return {
                "factual": "respond_factual",
                "analytical": "respond_analytical",
                "comparison": "respond_comparison",
                "definition": "respond_definition",
                "default": "respond_default",
            }.get(cat, "respond_default")

        graph.add_conditional_edges(
            "classify",
            route_by_category,
            {
                "respond_factual": "respond_factual",
                "respond_analytical": "respond_analytical",
                "respond_comparison": "respond_comparison",
                "respond_definition": "respond_definition",
                "respond_default": "respond_default",
            },
        )

        # All responder nodes go to END
        graph.add_edge("respond_factual", END)
        graph.add_edge("respond_analytical", END)
        graph.add_edge("respond_comparison", END)
        graph.add_edge("respond_definition", END)
        graph.add_edge("respond_default", END)

        # Compile
        self.app = graph.compile()

    def process_message(
        self, message: str, chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Runs the LangGraph app end-to-end:
        - Sets question in state
        - Classifies and routes
        - Returns generated answer
        """
        if self.app is None:
            # Be defensive if initialize() wasn't called
            self.initialize()

        initial_state: QAState = {"question": message}
        final_state: QAState = self.app.invoke(initial_state)

        # Optional: Debug print for observability
        # print(f"message: {message}, category: {final_state.get('category')}")

        return final_state.get("answer", "")
