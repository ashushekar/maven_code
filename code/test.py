# Install required dependencies
# uv pip install langgraph langchain-openai python-dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from typing import TypedDict

# -------------------------
# 1. Define the state schema
# -------------------------
class SentimentState(TypedDict):
    sentence: str
    sentiment: str
    message: str

# -------------------------
# 2. Define the LLM
# -------------------------
llm = ChatOpenAI(model="gpt-5-mini", temperature=0)

# -------------------------
# 3. Define nodes
# -------------------------

def analyze_sentiment(state: SentimentState):
    """Node to analyze sentiment from sentence."""
    prompt = f"Classify the sentiment of the following sentence as Positive, Negative, or Neutral:\n\n{state['sentence']}"
    response = llm.invoke(prompt)
    sentiment = response.content.strip()
    return {"sentiment": sentiment}

def generate_message(state: SentimentState):
    """Node to generate a friendly message based on sentiment."""
    sentiment = state["sentiment"]
    prompt = f"Generate a short, empathetic message for someone expressing a {sentiment.lower()} sentiment."
    response = llm.invoke(prompt)
    return {"message": response.content.strip()}

# -------------------------
# 4. Build the LangGraph
# -------------------------
graph_builder = StateGraph(SentimentState)

# Add nodes
graph_builder.add_node("analyze", analyze_sentiment)
graph_builder.add_node("message", generate_message)

# Set entry and transitions
graph_builder.set_entry_point("analyze")
graph_builder.add_edge("analyze", "message")
graph_builder.add_edge("message", END)

# Compile
sentiment_graph = graph_builder.compile()

# -------------------------
# 5. Example test
# -------------------------
result = sentiment_graph.invoke({"sentence": "I got the job! I'm so excited!"})
print(f"Sentiment: {result['sentiment']}")
print(f"Message: {result['message']}")
