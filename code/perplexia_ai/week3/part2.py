"""Part 2 - Agentic RAG implementation.

This implementation focuses on:
- Building an Agentic RAG system with dynamic search strategy
- Using LangGraph for controlling the RAG workflow
- Evaluating retrieved information quality
"""

import os.path as osp
from typing import Dict, List, Optional, Any
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END, START, MessagesState

# For document retrieval
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.week3.prompts import (
    DOCUMENT_EVALUATOR_PROMPT,
    DOCUMENT_SYNTHESIZER_PROMPT,
    QUERY_REWRITER_PROMPT,
)
from opik.integrations.langchain import OpikTracer
from langchain_core.prompts import PromptTemplate
from langgraph.prebuilt import ToolNode, tools_condition

# NOTE: Update this to the path of the documents on your machine.
BASE_DIR = "C:/Study/Cohort3/problem_fourth_ai/code/perplexia_ai/docs"

FILE_PATHS = [
    osp.join(BASE_DIR, "2019-annual-performance-report.pdf"),
    osp.join(BASE_DIR, "2020-annual-performance-report.pdf"),
    osp.join(BASE_DIR, "2021-annual-performance-report.pdf"),
    osp.join(BASE_DIR, "2022-annual-performance-report.pdf"),
]

class DocumentEvaluation(BaseModel):
    """Evaluation result for retrieved documents."""
    is_sufficient: bool = Field(description="Whether the documents provide sufficient information")
    feedback: str = Field(description="Feedback about the document quality and what's missing")

class AgenticRAGState(MessagesState):
    # This is the feedback from the evaluation node when the retrieved documents
    # are not sufficient.
    feedback: str
    # Whether the retrieved documents are sufficient. This will be set by the
    # evaluation node.
    is_sufficient: bool
    # Everything else is tracked within the 'messages' of the MessagesState.

class AgenticRAGChat(ChatInterface):
    """Week 3 Part 2 implementation focusing on Agentic RAG."""
    
    def __init__(self):
        pass
    
    def initialize(self) -> None:
        """Initialize components for the Agentic RAG system."""
        # Initialize models
        self.llm = ChatOpenAI(model="gpt-5-nano", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.evaluator_llm = self.llm.with_structured_output(DocumentEvaluation)

        # Load documents and create vector store
        docs = self._load_and_process_documents()
        print(f"Loading {len(docs)} documents into vector store")
        self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
        self.vector_store.add_documents(docs)
        
        # Create tools
        self.tools = self._create_tools()
        
        # Create the graph
        self.graph = self._create_graph()

        # Create Opik Tracer
        self.tracer = OpikTracer(
            graph=self.graph.get_graph(xray=True),
            project_name="agentic-rag-graph"
        )

    # NOTE: IF you are encountering rate limit errors here from OpenAI,
    # use the function from week2 part2 which adds sleep intervals between
    # embedding requests.
    def _load_and_process_documents(self) -> list[Document]:
        """Load and process OPM documents."""
        docs = []
        for file_path in FILE_PATHS:
            # For each document, load the pages and then combine them.
            # Then use RecursiveCharacterTextSplitter to split the document into chunks of 1000 tokens.
            # Then convert the chunks to Document objects with metadata.
            print(f"Loading document from {file_path}")
            loader = PyPDFLoader(file_path)
            page_docs = loader.load()
            # Combine all the page docs and then use RecursiveCharacterTextSplitter
            # to split the document into chunks of 1000 tokens.
            combined_doc = "\n".join([doc.page_content for doc in page_docs])
            # You can experiment with different chunk sizes and overlaps.
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_text(combined_doc)
            # Convert the chunks to Document objects with metadata.
            docs.extend([Document(page_content=chunk, metadata={"source": file_path}) for chunk in chunks])
        # NOTE: You can also do any custom processing on
        # the documents here if needed.
        return docs
    
    def _create_tools(self) -> List[Any]:
        """Create retriever and search tools."""
        # Create retriever tool
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        retriever_tool = create_retriever_tool(
            retriever,
            name="search_opm_docs",
            description=(
                "Search through OPM (Office of Personnel Management) documents from 2019-2022. "
                "Only use this for questions about OPM's operations, policies, and performance "
                "related to the years 2019-2022. Anything else should not invoke this tool."
            )
        )
        
        # Create web search tool with better description
        @tool("web_search")
        def search_web(query: str) -> list[dict]:
            """
            Invoke this tool to search the web for the latest information on any topic.            

            Args:
                query: The search query to look up
                
            Returns:
                List of search results with title, content, and URL
            """
            search = TavilySearchResults(max_results=3)
            return search.invoke(query)
        
        return [retriever_tool, search_web]

    def _generate_query_or_respond(self, state: AgenticRAGState):
        """Call the model to generate a response based on the current state. Given
        the question, it will decide to retrieve information (by generating tool-calls)
        using any of the tools or respond directly to the user.
        """
        print("Generating query or responding...")
        # This prompt can be iterated on to improve the quality of when/what to use the
        # tool calls for.
        prompt = PromptTemplate.from_template(
            """
            You are a helpful assistant that can answer questions about the user's query using the
            provided tools.

            Query: {question}
            """
        )
        # This will be user's original question for the first iteration.
        # For subsequent iterations, it will be the rewritten query.
        question = state["messages"][-1].content
        chain = prompt | self.llm.bind_tools(self.tools)
        # NOTE: We are only providing this node the new/original user query.
        # We are not providing the previously retrieved documents or feedback,
        # because those are taken care of by the query-rewriting node.
        # This is where you can flex your design muscles to try out different approaches.
        response = chain.invoke({"question": question})
        return {"messages": [response]}

    def _evaluate_documents(self, state: AgenticRAGState):
        """Evaluate the documents retrieved from the retriever tool.
        """
        print("Evaluating documents...")
        # This is the original user question:
        user_question = state["messages"][0].content
        retrieved_docs = state["messages"][-1].content
        chain = DOCUMENT_EVALUATOR_PROMPT | self.evaluator_llm
        evaluation = chain.invoke({"question": user_question, "retrieved_docs": retrieved_docs})
        print(f"Evaluation result: {evaluation}")
        return {
            "is_sufficient": evaluation.is_sufficient,
            "feedback": evaluation.feedback
        }

    def _synthesize_answer(self, state: AgenticRAGState):
        print("Synthesizing answer...")
        user_question = state["messages"][0].content
        retrieved_docs = state["messages"][-1].content
        chain = DOCUMENT_SYNTHESIZER_PROMPT | self.llm
        answer = chain.invoke({"question": user_question, "retrieved_docs": retrieved_docs})
        return {"messages": [answer]}

    def _query_rewriter(self, state: AgenticRAGState):
        print("Rewriting query...")
        user_question = state["messages"][0].content
        retrieved_docs = state["messages"][-1].content
        feedback = state["feedback"]
        chain = QUERY_REWRITER_PROMPT | self.llm
        new_query = chain.invoke({
            "question": user_question, 
            "feedback": feedback,
            "retrieved_docs": retrieved_docs
        })
        print(f"Rewritten query: {new_query.content}")
        return {"messages": [new_query]}
    

    def _create_graph(self) -> Any:
        """Create the agentic RAG graph."""
        # Create the graph builder
        graph_builder = StateGraph(AgenticRAGState)
        graph_builder.add_node("generate_query_or_respond", self._generate_query_or_respond)
        graph_builder.add_node("retrieve_documents", ToolNode(self.tools))
        graph_builder.add_node("evaluate_documents", self._evaluate_documents)
        graph_builder.add_node("synthesize_answer", self._synthesize_answer)
        graph_builder.add_node("query_rewriter", self._query_rewriter)

        # Edges:
        graph_builder.add_edge(START, "generate_query_or_respond")
        # If the query did not generate any tool call, we end the workflow
        # and return the last message.
        graph_builder.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,
            {
                "tools": "retrieve_documents",
                END:END,
            },
        )
        graph_builder.add_edge("retrieve_documents", "evaluate_documents")
        graph_builder.add_conditional_edges(
            "evaluate_documents",
            lambda x: "synthesize_answer" if x["is_sufficient"] else "query_rewriter",
            {
                "synthesize_answer": "synthesize_answer",
                "query_rewriter": "query_rewriter",
            },
        )
        graph_builder.add_edge("query_rewriter", "generate_query_or_respond")
        graph_builder.add_edge("synthesize_answer", END)

        return graph_builder.compile()
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using the Agentic RAG system."""
        print("\n=== STARTING NEW QUERY ===")
        print(f"Query: {message}")
        
        # Create initial state
        state = {
            "messages": [HumanMessage(content=message)],
            "retrieved_docs": [],
            "evaluation": None,
            "final_answer": None,
            "next_step": "agent",
            "iteration_count": 0
        }
        
        # Run the workflow
        result = self.graph.invoke(
            state,
            config={
                "callbacks": [self.tracer],
            },
        )
        
        print("\n=== QUERY COMPLETED ===")
        
        # Return the final answer if available, otherwise the last message
        if result.get("final_answer"):
            return result["final_answer"]
        return result["messages"][-1].content
