"""Part 1 - Tool-Using Agent implementation.

This implementation focuses on:
- Converting tools from Assignment 1 to use with LangGraph
- Using the ReAct pattern for autonomous tool selection
- Comparing manual workflow vs agent approaches
"""

from typing import Dict, List, Optional, Any, Annotated
import io
import contextlib
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults

from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator

class ToolUsingAgentChat(ChatInterface):
    """Week 3 Part 1 implementation focusing on tool-using agents."""
    
    def __init__(self):
        self.llm = None
        self.tools = []
        self.graph = None
    
    def initialize(self) -> None:
        """Initialize components for the tool-using agent.
        
        Students should:
        - Initialize the chat model
        - Define tools for calculator, DateTime, and weather
        - Create the ReAct agent using LangGraph
        """
        # Initialize chat model
        self.llm = init_chat_model("gpt-5-nano", model_provider="openai")
        
        # Create tools
        self.tools = self._create_tools()
        
        # NOTE: create_react_agent as described in the tutorial is a pre-built function
        # which uses a very simple AgentState. If you need to define a complex agent state,
        # you can pass in an argument called 'state_schema' to create_react_agent. See
        # langgraph.prebuilt.chat_agent_executor.AgentState for the definition of the
        # default AgentState.
        #
        # For this simple example, we will just use the default AgentState.
        # Create the ReAct agent graph with the tools:
        self.graph = create_react_agent(
            model=self.llm,
            tools=self.tools,
        )
    
    def _create_tools(self) -> List[Any]:
        """Create and return the list of tools for the agent.
        
        Students should implement:
        - Calculator tool from Assignment 1
        - DateTime tool from Assignment 1
        - Weather tool using Tavily search
        
        Returns:
            List: List of tool objects
        """
        # The Annotated type is used to specify the type of the argument as well as add
        # a description to provide more information to the model.
        # NOTE: Annotated is a Python construct and not a LangGraph construct. It's adapted (like function docstring)
        # to pass information to the model.
        @tool
        def calculator(expression: Annotated[str, "The mathematical expression to evaluate"]) -> str:
            """Evaluate a mathematical expression using basic arithmetic operations (+, -, *, /, %, //).
            Examples: '5 + 3', '10 * (2 + 3)', '15 / 3'
            """
            result = Calculator.evaluate_expression(expression)
            if isinstance(result, str) and result.startswith("Error"):
                raise ValueError(result)
            return str(result)

        @tool
        def execute_datetime_code(code: Annotated[str, "Python code to execute for datetime operations"]) -> str:
            """Execute Python code for datetime operations. The code should use datetime or time modules.
            Examples: 
            - 'print(datetime.datetime.now().strftime("%Y-%m-%d"))'
            - 'print(datetime.datetime.now().year)'
            """
            output_buffer = io.StringIO()
            code = f"import datetime\nimport time\n{code}"
            try:
                with contextlib.redirect_stdout(output_buffer):
                    exec(code)
                return output_buffer.getvalue().strip()
            except Exception as e:
                raise ValueError(f"Error executing datetime code: {str(e)}")

        @tool
        def get_weather(location: Annotated[str, "The location to get weather for (city, country)"]) -> str:
            """Get the current weather for a given location using Tavily search.
            Examples: 'New York, USA', 'London, UK', 'Tokyo, Japan'
            """
            search = TavilySearchResults(max_results=3)
            query = f"what is the current weather temperature in {location} right now"
            results = search.invoke(query)
            
            if not results:
                return f"Could not find weather information for {location}"
            
            # We are using the first result only but you could also provide a more complex
            # response to the LLM by processing the results if required.
            return results[0].get("content", f"Could not find weather information for {location}")
        
        return [calculator, execute_datetime_code, get_weather]

    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using the tool-using agent.
        
        Args:
            message: The user's input message
            chat_history: List of previous chat messages
            
        Returns:
            str: The assistant's response
        """
        # Run the graph with the user's message:
        result = self.graph.invoke({"messages": [("user", message)]})
        
        # Return the last assistant message:
        return result["messages"][-1].content if result["messages"] else "No response generated"
