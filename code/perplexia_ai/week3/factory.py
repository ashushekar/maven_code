"""Factory for creating Week 3 implementations.

This module contains factory methods for creating the appropriate chat implementation
based on the selected Week 3 mode.
"""

from enum import Enum
from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.week3.part3 import DeepResearchChat
from perplexia_ai.week3.part2 import AgenticRAGChat
from perplexia_ai.week3.part1 import ToolUsingAgentChat
# If you want to use the MCP version of the tool-using agent,
# uncomment the following line and comment out the line above.
# from perplexia_ai.week3.part1_mcp import ToolUsingAgentChat


class Week3Mode(Enum):
    """Enum for different Week 3 implementation modes."""
    PART1_TOOL_USING_AGENT = "part1"
    PART2_AGENTIC_RAG = "part2"
    PART3_DEEP_RESEARCH = "part3"


def create_chat_implementation(mode: Week3Mode) -> ChatInterface:
    """Create a chat implementation for the specified Week 3 mode.
    
    Args:
        mode: Which Week 3 implementation to use
        
    Returns:
        ChatInterface: The initialized chat implementation
    """
    if mode == Week3Mode.PART1_TOOL_USING_AGENT:
        return ToolUsingAgentChat()
    elif mode == Week3Mode.PART2_AGENTIC_RAG:
        return AgenticRAGChat()
    elif mode == Week3Mode.PART3_DEEP_RESEARCH:
        return DeepResearchChat()
