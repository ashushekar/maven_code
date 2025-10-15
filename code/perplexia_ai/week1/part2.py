# part2.py
"""
Part 2 - Basic Tools (LangGraph version)

Graph:
    START -> classify -> (conditional route)
           -> respond_factual | respond_analytical | respond_comparison | respond_definition
           -> respond_calculation (LLM->expr -> Calculator.evaluate_expression)
           -> respond_datetime   (LLM->code -> exec sandbox)
           -> respond_default -> END
"""

import contextlib
import io
from typing import Dict, List, Optional, Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator


# ---------- Graph State ----------

Category = Literal[
    "factual",
    "analytical",
    "comparison",
    "definition",
    "calculation",
    "datetime",
    "default",
]

class ToolState(TypedDict, total=False):
    question: str
    category: Category
    answer: str
    expr: str     # intermediate for calculation
    code: str     # intermediate for datetime code


# ---------- Prompts (unchanged from your template) ----------

QUERY_CLASSIFIER_PROMPT = ChatPromptTemplate.from_template(
    """
    Given the user question, classify into one of the following 4 categories with expected definitions:

    - Factual Questions ("What is...?", "Who invented...?") : Return 'factual'
    - Analytical Questions ("How does...?", "Why do...?") : Return 'analytical'
    - Comparison Questions ("What's the difference between...?") : Return 'comparison'
    - Definition Requests ("Define...", "Explain...") : Return 'definition'
    - Any question related to date or time computation: Return "datetime"
    - Any question that requires mathematical calculation to be done: Return "calculation". Only return calculation if the calculation is not associated with doing computations on dates or time etc.

    ONLY return the category word as the answer and nothing else. Do not add quotes in the final output.
    Return 'default' (without quotes) if the question does not fit into any of the categories.

    Prompting by examples:
    * Question: What is the highest mountain in the world?
    * Response: factual

    * Question: What's the difference between OpenAI and Anthropic?
    * Response: comparison

    * Question: What's a 18% tip of $105 bill?
    * Response: calculation

    * Question: What day is it today? or What is the date of 30 days from now?
    * Response: datetime

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
    "calculation": ChatPromptTemplate.from_template(
        """
        You are a smart AI model but cannot do any complex calculations. You are very good at
        translating a math question to a simple equation which can be solved by a calculator.

        Convert the user question below to a math calculation.
        Remember that the calculator can only use +, -, *, /, //, % operators,
        so only use those operators and output the final math equation.

        User Query: "{question}"

        The final output should ONLY contain the valid math equation, no words or any other text.
        Otherwise the calculator tool will error out.

        Examples:
        Question: What is 5 times 20?
        Answer: 5 * 20

        Question: What is the split of each person for a 4 person dinner of $100 with 20*% tip?
        Answer: (100 + 0.2*100) / 4

        Question: Round 100.5 to the nearest integer.
        Answer: 100.5 // 1
        """
    ),
    "datetime": ChatPromptTemplate.from_template(
        """You are a smart AI which is very good at translating a question in english
        to a simple python code to output the result. You'll only be given queries related
        to date and time, for which generate the python code required to get the answer.
        Your code will be sent to a Python interpreter and the expectation is to print the output on the final line.

        These are the ONLY python libraries you have access to - math, datetime, time.

        User Query: "{question}"

        The final output should ONLY contain valid Python code, no words or any other text.
        Otherwise the Python interpreter tool will error out. Avoid returning ``` or python
        in the output, just return the code directly.

        Examples:
        Question: What day is it today?
        Answer: print(datetime.now().strftime("%A"))

        Question: What is the date of 30 days from now?
        Answer: print(datetime.now() + timedelta(days=30))
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


# ---------- Helpers (same logic as your class methods) ----------

def _calculate_answer(expression: str) -> str:
    """Evaluates the math expression and returns the result as a string."""
    # print(f"Evaluating expression: {expression}")
    return str(Calculator.evaluate_expression(expression))

def _datetime_answer(code: str) -> str:
    """
    Executes the given python code and returns stdout as a string.
    SECURITY NOTE: This mirrors your assignment behavior and assumes trusted code.
    """
    # print(f"Executing code: {code}")
    output_buffer = io.StringIO()
    code = f"from datetime import *\nimport datetime\nimport time\n{code}"
    with contextlib.redirect_stdout(output_buffer):
        exec(code, {})
    return output_buffer.getvalue().strip()


# ---------- Implementation ----------

class BasicToolsChat(ChatInterface):
    """Week 1 Part 2 implemented via LangGraph with calculator & datetime branches."""

    def __init__(self, model: str = "gpt-5", temperature: float = 0.0):
        self.model_name = model
        self.temperature = temperature
        self.llm = None
        self.app = None  # compiled LangGraph app

    def initialize(self) -> None:
        self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)

        # Chains
        self.classifier_chain = QUERY_CLASSIFIER_PROMPT | self.llm | StrOutputParser()

        # Normal response chains (non-tool)
        self.response_chains = {
            key: (tmpl | self.llm | StrOutputParser())
            for key, tmpl in RESPONSE_PROMPTS.items()
            if key in {"factual", "analytical", "comparison", "definition", "default"}
        }

        # Tool-generating chains (LLM → text → tool)
        self.calc_expression_chain = RESPONSE_PROMPTS["calculation"] | self.llm | StrOutputParser()
        self.datetime_code_chain   = RESPONSE_PROMPTS["datetime"]   | self.llm | StrOutputParser()

        # Build graph
        graph = StateGraph(ToolState)

        # --- Nodes ---
        def classify(state: ToolState) -> ToolState:
            cat = self.classifier_chain.invoke({"question": state["question"]}).strip().lower()
            if cat not in RESPONSE_PROMPTS:
                cat = "default"
            return {"category": cat}

        def respond_generic(category: Category):
            def _node(state: ToolState) -> ToolState:
                ans = self.response_chains[category].invoke({"question": state["question"]})
                return {"answer": ans}
            return _node

        def respond_calculation(state: ToolState) -> ToolState:
            expr = self.calc_expression_chain.invoke({"question": state["question"]}).strip()
            # Safety: keep only allowed characters (digits, ops, spaces, dots, parentheses)
            # (keeps behavior close to your assignment while reducing accidental junk)
            # You can relax/tighten this if needed.
            allowed = set("0123456789+-*/()% .")
            if not expr or any(ch not in allowed for ch in expr):
                return {"answer": "Sorry, I couldn't parse a valid calculation."}
            result = _calculate_answer(expr)
            return {"expr": expr, "answer": result}

        def respond_datetime(state: ToolState) -> ToolState:
            code = self.datetime_code_chain.invoke({"question": state["question"]}).strip()
            # Basic guardrails: only allow a small subset of characters to mimic assignment constraints
            # (letters, digits, underscores, parentheses, quotes, spaces, colon, comma, dot, plus, minus, star)
            # This is still not a sandbox—just a nudge toward safety for the exercise.
            allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_()[]{}\"' :,.+-*/%\n=\r\t")
            if not code or any(ch not in allowed for ch in code):
                return {"answer": "Sorry, I couldn't produce safe datetime code."}
            try:
                result = _datetime_answer(code)
            except Exception as e:
                result = f"Execution error: {e}"
            return {"code": code, "answer": result}

        # Register nodes
        graph.add_node("classify", classify)
        graph.add_node("respond_factual",     respond_generic("factual"))
        graph.add_node("respond_analytical",  respond_generic("analytical"))
        graph.add_node("respond_comparison",  respond_generic("comparison"))
        graph.add_node("respond_definition",  respond_generic("definition"))
        graph.add_node("respond_default",     respond_generic("default"))
        graph.add_node("respond_calculation", respond_calculation)
        graph.add_node("respond_datetime",    respond_datetime)

        # Entry & routing
        graph.set_entry_point("classify")

        def route(state: ToolState) -> str:
            return {
                "factual": "respond_factual",
                "analytical": "respond_analytical",
                "comparison": "respond_comparison",
                "definition": "respond_definition",
                "calculation": "respond_calculation",
                "datetime": "respond_datetime",
                "default": "respond_default",
            }.get(state.get("category", "default"), "respond_default")

        graph.add_conditional_edges(
            "classify",
            route,
            {
                "respond_factual": "respond_factual",
                "respond_analytical": "respond_analytical",
                "respond_comparison": "respond_comparison",
                "respond_definition": "respond_definition",
                "respond_calculation": "respond_calculation",
                "respond_datetime": "respond_datetime",
                "respond_default": "respond_default",
            },
        )

        # All responders -> END
        for node in [
            "respond_factual",
            "respond_analytical",
            "respond_comparison",
            "respond_definition",
            "respond_calculation",
            "respond_datetime",
            "respond_default",
        ]:
            graph.add_edge(node, END)

        self.app = graph.compile()

    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        if self.app is None:
            self.initialize()
        final: ToolState = self.app.invoke({"question": message})
        # Uncomment for debugging:
        # print(f"message: {message}, category: {final.get('category')}, expr: {final.get('expr')}, code: {final.get('code')}")
        return final.get("answer", "")
