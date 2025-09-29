"""
LangGraph-Based Decision-Making Head for Circuit Analysis
Integrates Hybrid RAG, Circuit Simulation, and ReAct Framework
"""

import os
import json
import asyncio
from typing import Annotated, TypedDict, List, Dict, Any, Optional, Literal
from datetime import datetime
import logging

# LangGraph and LangChain
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama

# Vector stores and embeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, DirectoryLoader
)
from langchain.schema import Document

# Utilities
import operator
from pydantic import BaseModel, Field
import numpy as np
from dotenv import load_dotenv

# Circuit simulation
import subprocess
import tempfile
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== STATE DEFINITIONS ====================

class CircuitAnalysisState(TypedDict):
    """State for the circuit analysis workflow"""
    # Input
    question: str
    image_description: str
    netlist: str
    refined_netlist: Optional[str]
    
    # Processing state
    messages: Annotated[List[HumanMessage | AIMessage | SystemMessage], operator.add]
    iteration: int
    max_iterations: int
    
    # Retrieval
    retrieved_context: Optional[str]
    retrieval_queries: List[str]
    
    # Simulation
    simulation_attempted: bool
    simulation_results: Optional[Dict[str, Any]]
    simulation_valid: bool
    simulation_error: Optional[str]
    
    # Analysis
    circuit_type: Optional[str]
    complexity_level: Optional[str]
    component_analysis: Optional[Dict[str, Any]]
    
    # Decision making
    next_action: Optional[str]
    confidence_score: float
    reasoning_steps: List[Dict[str, Any]]
    
    # Output
    final_answer: Optional[str]
    answer_generated: bool
    

class SimulationResult(BaseModel):
    """Structured simulation result"""
    success: bool
    dc_analysis: Optional[Dict[str, float]] = None
    ac_analysis: Optional[Dict[str, float]] = None
    transient_analysis: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    output_log: Optional[str] = None


# ==================== HYBRID RAG SYSTEM ====================

class HybridRAGRetriever:
    """
    Production-grade Hybrid RAG system combining:
    - Dense retrieval (vector similarity)
    - Sparse retrieval (BM25)
    - Cross-encoder reranking
    - Circuit-specific context
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize embeddings
        self.embeddings = self._initialize_embeddings()
        
        # Initialize vector stores
        self.vector_store = None
        self.bm25_retriever = None
        
        # Initialize reranker
        self.reranker = self._initialize_reranker()
        
        # Load knowledge base
        self._load_knowledge_base()
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        embedding_type = self.config.get("embedding_type", "huggingface")
        
        if embedding_type == "openai":
            return OpenAIEmbeddings(
                model=self.config.get("embedding_model", "text-embedding-3-small")
            )
        else:
            return HuggingFaceEmbeddings(
                model_name=self.config.get(
                    "embedding_model", 
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
            )
    
    def _initialize_reranker(self):
        """Initialize cross-encoder reranker"""
        model = HuggingFaceCrossEncoder(
            model_name=self.config.get(
                "reranker_model",
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        )
        return CrossEncoderReranker(model=model, top_n=5)
    
    def _load_knowledge_base(self):
        """Load and index knowledge base documents"""
        kb_path = self.config.get("knowledge_base_path", "./circuit_knowledge_base")
        
        if not os.path.exists(kb_path):
            logger.warning(f"Knowledge base path {kb_path} does not exist")
            return
        
        # Load documents
        documents = self._load_documents(kb_path)
        
        if not documents:
            logger.warning("No documents loaded from knowledge base")
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 512),
            chunk_overlap=self.config.get("chunk_overlap", 50),
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        logger.info(f"Loaded {len(documents)} documents, split into {len(chunks)} chunks")
        
        # Create vector store
        persist_dir = self.config.get("persist_directory", "./chroma_db")
        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=persist_dir
        )
        
        # Create BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = self.config.get("top_k", 5)
    
    def _load_documents(self, kb_path: str) -> List[Document]:
        """Load documents from knowledge base directory"""
        documents = []
        
        # Load PDFs
        try:
            pdf_loader = DirectoryLoader(
                kb_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True
            )
            documents.extend(pdf_loader.load())
        except Exception as e:
            logger.warning(f"Error loading PDFs: {e}")
        
        # Load DOCX files
        try:
            docx_loader = DirectoryLoader(
                kb_path,
                glob="**/*.docx",
                loader_cls=Docx2txtLoader,
                show_progress=True
            )
            documents.extend(docx_loader.load())
        except Exception as e:
            logger.warning(f"Error loading DOCX files: {e}")
        
        # Load text files
        try:
            txt_loader = DirectoryLoader(
                kb_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=True
            )
            documents.extend(txt_loader.load())
        except Exception as e:
            logger.warning(f"Error loading text files: {e}")
        
        return documents
    
    def retrieve(
        self, 
        query: str, 
        circuit_context: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> str:
        """
        Perform hybrid retrieval with optional circuit context
        
        Args:
            query: Search query
            circuit_context: Optional circuit-specific context
            top_k: Number of results to return
            
        Returns:
            Formatted context string
        """
        
        if not self.vector_store or not self.bm25_retriever:
            logger.warning("RAG system not initialized, returning empty context")
            return "No knowledge base available."
        
        # Enhance query with circuit context
        enhanced_query = self._enhance_query(query, circuit_context)
        
        # Dense retrieval
        dense_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": top_k * 2}
        )
        
        # Create ensemble retriever
        ensemble_retriever = EnsembleRetriever(
            retrievers=[dense_retriever, self.bm25_retriever],
            weights=[0.7, 0.3]  # Favor dense retrieval
        )
        
        # Add reranking
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.reranker,
            base_retriever=ensemble_retriever
        )
        
        # Retrieve documents
        try:
            docs = compression_retriever.get_relevant_documents(enhanced_query)
            
            # Format context
            context = self._format_context(docs, circuit_context)
            return context
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return "Error retrieving context from knowledge base."
    
    def _enhance_query(
        self, 
        query: str, 
        circuit_context: Optional[Dict[str, Any]]
    ) -> str:
        """Enhance query with circuit-specific context"""
        
        if not circuit_context:
            return query
        
        enhancements = [query]
        
        if circuit_context.get("circuit_type"):
            enhancements.append(f"Circuit type: {circuit_context['circuit_type']}")
        
        if circuit_context.get("components"):
            components = ", ".join(circuit_context["components"])
            enhancements.append(f"Components: {components}")
        
        if circuit_context.get("analysis_type"):
            enhancements.append(f"Analysis: {circuit_context['analysis_type']}")
        
        return " | ".join(enhancements)
    
    def _format_context(
        self, 
        docs: List[Document], 
        circuit_context: Optional[Dict[str, Any]]
    ) -> str:
        """Format retrieved documents into context string"""
        
        context_parts = []
        
        # Add circuit context if available
        if circuit_context:
            context_parts.append("=== CIRCUIT CONTEXT ===")
            if circuit_context.get("circuit_type"):
                context_parts.append(f"Circuit Type: {circuit_context['circuit_type']}")
            if circuit_context.get("components"):
                context_parts.append(f"Components: {', '.join(circuit_context['components'])}")
            context_parts.append("")
        
        # Add retrieved knowledge
        context_parts.append("=== RETRIEVED KNOWLEDGE ===")
        
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content.strip()
            context_parts.append(f"\n[Source {i}] {source}")
            context_parts.append(content)
            context_parts.append("-" * 50)
        
        return "\n".join(context_parts)


# ==================== CIRCUIT SIMULATION TOOLS ====================

class CircuitSimulator:
    """NgSpice-based circuit simulator"""
    
    def __init__(self):
        self._verify_ngspice()
    
    def _verify_ngspice(self):
        """Verify NgSpice is installed"""
        try:
            result = subprocess.run(
                ["ngspice", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info("NgSpice verified successfully")
            else:
                logger.warning("NgSpice verification failed")
        except FileNotFoundError:
            logger.error("NgSpice not found. Please install: https://ngspice.sourceforge.io/")
        except Exception as e:
            logger.error(f"Error verifying NgSpice: {e}")
    
    def simulate(self, netlist: str) -> SimulationResult:
        """
        Simulate a SPICE netlist
        
        Args:
            netlist: SPICE netlist string
            
        Returns:
            SimulationResult with parsed results
        """
        
        # Validate netlist
        validation_errors = self._validate_netlist(netlist)
        if validation_errors:
            return SimulationResult(
                success=False,
                error_message=f"Netlist validation failed: {'; '.join(validation_errors)}"
            )
        
        # Create temporary file for netlist
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cir', delete=False) as f:
            f.write(netlist)
            netlist_path = f.name
        
        try:
            # Run NgSpice simulation
            result = subprocess.run(
                ["ngspice", "-b", netlist_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Parse output
            if result.returncode == 0:
                parsed_results = self._parse_simulation_output(result.stdout)
                return SimulationResult(
                    success=True,
                    dc_analysis=parsed_results.get("dc"),
                    ac_analysis=parsed_results.get("ac"),
                    transient_analysis=parsed_results.get("tran"),
                    output_log=result.stdout
                )
            else:
                return SimulationResult(
                    success=False,
                    error_message=result.stderr or "Simulation failed",
                    output_log=result.stdout
                )
                
        except subprocess.TimeoutExpired:
            return SimulationResult(
                success=False,
                error_message="Simulation timeout (>30s)"
            )
        except Exception as e:
            return SimulationResult(
                success=False,
                error_message=f"Simulation error: {str(e)}"
            )
        finally:
            # Clean up temporary file
            try:
                os.unlink(netlist_path)
            except:
                pass
    
    def _validate_netlist(self, netlist: str) -> List[str]:
        """Validate SPICE netlist syntax"""
        errors = []
        lines = netlist.strip().split('\n')
        
        has_ground = False
        has_component = False
        has_end = False
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('*'):
                continue
            
            # Check for ground node
            if '0' in line.split():
                has_ground = True
            
            # Check for components
            if line[0].upper() in 'RCLDVIQMX':
                has_component = True
            
            # Check for .END
            if line.upper() == '.END':
                has_end = True
        
        if not has_ground:
            errors.append("No ground node (0) found")
        if not has_component:
            errors.append("No circuit components found")
        if not has_end:
            errors.append("Missing .END statement")
        
        return errors
    
    def _parse_simulation_output(self, output: str) -> Dict[str, Any]:
        """Parse NgSpice simulation output"""
        results = {}
        
        # Parse DC analysis (simplified)
        dc_pattern = r'v\((\w+)\)\s*=\s*([-\d.e+]+)'
        dc_matches = re.findall(dc_pattern, output, re.IGNORECASE)
        if dc_matches:
            results["dc"] = {node: float(voltage) for node, voltage in dc_matches}
        
        # Parse AC analysis (simplified)
        # Would need more sophisticated parsing for real AC analysis
        
        # Parse transient analysis (simplified)
        # Would need more sophisticated parsing for real transient analysis
        
        return results


# ==================== LANGGRAPH TOOLS ====================

# Initialize simulator globally
simulator = CircuitSimulator()

@tool
def simulate_circuit(netlist: str) -> Dict[str, Any]:
    """
    Simulate a SPICE circuit netlist using NgSpice.
    
    Args:
        netlist: SPICE netlist string
        
    Returns:
        Dictionary containing simulation results
    """
    result = simulator.simulate(netlist)
    return result.dict()


@tool
def retrieve_knowledge(query: str, circuit_type: str = "general") -> str:
    """
    Retrieve relevant knowledge from the circuit analysis knowledge base.
    
    Args:
        query: Search query describing what information is needed
        circuit_type: Type of circuit (analog, digital, general)
        
    Returns:
        Formatted context string with relevant knowledge
    """
    # This will be injected by the graph
    return query  # Placeholder


@tool
def analyze_components(netlist: str) -> Dict[str, Any]:
    """
    Analyze circuit components from netlist.
    
    Args:
        netlist: SPICE netlist string
        
    Returns:
        Dictionary with component analysis
    """
    lines = netlist.strip().split('\n')
    components = {
        "resistors": [],
        "capacitors": [],
        "inductors": [],
        "voltage_sources": [],
        "current_sources": [],
        "other": []
    }
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('*') or line.startswith('.'):
            continue
        
        component_type = line[0].upper()
        if component_type == 'R':
            components["resistors"].append(line)
        elif component_type == 'C':
            components["capacitors"].append(line)
        elif component_type == 'L':
            components["inductors"].append(line)
        elif component_type == 'V':
            components["voltage_sources"].append(line)
        elif component_type == 'I':
            components["current_sources"].append(line)
        else:
            components["other"].append(line)
    
    return {
        "summary": {k: len(v) for k, v in components.items()},
        "details": components
    }


# ==================== LANGGRAPH WORKFLOW NODES ====================

def initialize_state(state: CircuitAnalysisState) -> CircuitAnalysisState:
    """Initialize the analysis state"""
    logger.info("Initializing circuit analysis state")
    
    state["iteration"] = 0
    state["max_iterations"] = state.get("max_iterations", 10)
    state["simulation_attempted"] = False
    state["answer_generated"] = False
    state["confidence_score"] = 0.0
    state["reasoning_steps"] = []
    state["retrieval_queries"] = []
    
    # Add initial system message
    system_msg = SystemMessage(content="""You are an expert circuit analysis AI assistant.
You help students and engineers understand and solve circuit problems by:
1. Analyzing circuit netlists and diagrams
2. Running simulations to verify behavior
3. Retrieving relevant circuit theory knowledge
4. Providing step-by-step explanations

Always think carefully and show your reasoning.""")
    
    state["messages"] = [system_msg]
    
    return state


def decide_next_action(state: CircuitAnalysisState) -> CircuitAnalysisState:
    """Decide what action to take next based on current state"""
    logger.info(f"Deciding next action (iteration {state['iteration']})")
    
    # Check iteration limit
    if state["iteration"] >= state["max_iterations"]:
        state["next_action"] = "generate_answer"
        return state
    
    # Decision logic
    if not state.get("retrieved_context"):
        state["next_action"] = "retrieve_knowledge"
    elif not state.get("simulation_attempted"):
        state["next_action"] = "simulate_circuit"
    elif state.get("simulation_attempted") and not state.get("answer_generated"):
        state["next_action"] = "generate_answer"
    else:
        state["next_action"] = "end"
    
    logger.info(f"Next action: {state['next_action']}")
    return state


async def retrieve_knowledge_node(state: CircuitAnalysisState, config: Dict[str, Any]) -> CircuitAnalysisState:
    """Retrieve relevant knowledge using Hybrid RAG"""
    logger.info("Retrieving knowledge from RAG system")
    
    # Initialize RAG retriever
    rag_retriever = config.get("rag_retriever")
    if not rag_retriever:
        logger.warning("RAG retriever not configured")
        state["retrieved_context"] = "Knowledge base not available."
        return state
    
    # Prepare circuit context
    circuit_context = {
        "circuit_type": state.get("circuit_type", "unknown"),
        "components": [],  # Would extract from netlist
        "analysis_type": "general"
    }
    
    # Retrieve context
    try:
        context = rag_retriever.retrieve(
            query=state["question"],
            circuit_context=circuit_context,
            top_k=5
        )
        state["retrieved_context"] = context
        state["retrieval_queries"].append(state["question"])
        
        # Add to reasoning steps
        state["reasoning_steps"].append({
            "step": len(state["reasoning_steps"]) + 1,
            "action": "retrieve_knowledge",
            "result": "Successfully retrieved relevant context",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Knowledge retrieval failed: {e}")
        state["retrieved_context"] = f"Error retrieving knowledge: {str(e)}"
    
    state["iteration"] += 1
    return state


async def simulate_circuit_node(state: CircuitAnalysisState) -> CircuitAnalysisState:
    """Simulate the circuit using NgSpice"""
    logger.info("Simulating circuit")
    
    netlist = state.get("refined_netlist") or state.get("netlist")
    
    if not netlist:
        logger.warning("No netlist available for simulation")
        state["simulation_attempted"] = True
        state["simulation_valid"] = False
        state["simulation_error"] = "No netlist provided"
        return state
    
    try:
        # Run simulation
        result = simulator.simulate(netlist)
        
        state["simulation_attempted"] = True
        state["simulation_results"] = result.dict()
        state["simulation_valid"] = result.success
        
        if result.success:
            logger.info("Simulation successful")
            state["reasoning_steps"].append({
                "step": len(state["reasoning_steps"]) + 1,
                "action": "simulate_circuit",
                "result": "Simulation completed successfully",
                "details": result.dict(),
                "timestamp": datetime.now().isoformat()
            })
        else:
            logger.warning(f"Simulation failed: {result.error_message}")
            state["simulation_error"] = result.error_message
            state["reasoning_steps"].append({
                "step": len(state["reasoning_steps"]) + 1,
                "action": "simulate_circuit",
                "result": "Simulation failed",
                "error": result.error_message,
                "timestamp": datetime.now().isoformat()
            })
    
    except Exception as e:
        logger.error(f"Simulation error: {e}")
        state["simulation_attempted"] = True
        state["simulation_valid"] = False
        state["simulation_error"] = str(e)
    
    state["iteration"] += 1
    return state


async def generate_answer_node(state: CircuitAnalysisState, config: Dict[str, Any]) -> CircuitAnalysisState:
    """Generate final answer using LLM with all gathered context"""
    logger.info("Generating final answer")
    
    # Get LLM from config
    llm = config.get("llm")
    if not llm:
        logger.error("LLM not configured")
        state["final_answer"] = "Error: LLM not available"
        state["answer_generated"] = True
        return state
    
    # Construct comprehensive prompt
    prompt_parts = [
        f"Question: {state['question']}",
        f"\nCircuit Description: {state['image_description']}",
    ]
    
    # Add netlist
    if state.get("refined_netlist"):
        prompt_parts.append(f"\nSPICE Netlist:\n{state['refined_netlist']}")
    
    # Add simulation results
    if state.get("simulation_valid") and state.get("simulation_results"):
        sim_results = state["simulation_results"]
        prompt_parts.append(f"\nSimulation Results:")
        prompt_parts.append(json.dumps(sim_results, indent=2))
    elif state.get("simulation_error"):
        prompt_parts.append(f"\nSimulation Error: {state['simulation_error']}")
    
    # Add retrieved context
    if state.get("retrieved_context"):
        prompt_parts.append(f"\n{state['retrieved_context']}")
    
    prompt_parts.append("\nBased on all the information above, provide a comprehensive answer to the question.")
    prompt_parts.append("Include:")
    prompt_parts.append("1. Direct answer to the question")
    prompt_parts.append("2. Step-by-step explanation")
    prompt_parts.append("3. Relevant calculations (if applicable)")
    prompt_parts.append("4. References to circuit theory")
    
    full_prompt = "\n".join(prompt_parts)
    
    try:
        # Generate answer
        messages = [
            SystemMessage(content="You are an expert circuit analysis tutor. Provide clear, educational explanations."),
            HumanMessage(content=full_prompt)
        ]
        
        response = await llm.ainvoke(messages)
        state["final_answer"] = response.content
        state["answer_generated"] = True
        state["confidence_score"] = 0.9 if state.get("simulation_valid") else 0.7
        
        state["reasoning_steps"].append({
            "step": len(state["reasoning_steps"]) + 1,
            "action": "generate_answer",
            "result": "Answer generated successfully",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        state["final_answer"] = f"Error generating answer: {str(e)}"
        state["answer_generated"] = True
    
    return state


def route_decision(state: CircuitAnalysisState) -> Literal["retrieve", "simulate", "generate", "end"]:
    """Route to next node based on decided action"""
    action = state.get("next_action", "end")
    
    if action == "retrieve_knowledge":
        return "retrieve"
    elif action == "simulate_circuit":
        return "simulate"
    elif action == "generate_answer":
        return "generate"
    else:
        return "end"


# ==================== BUILD LANGGRAPH WORKFLOW ====================

def create_circuit_analysis_graph(config: Dict[str, Any]) -> StateGraph:
    """
    Create the LangGraph workflow for circuit analysis
    
    Args:
        config: Configuration dictionary containing:
            - llm: Language model instance
            - rag_retriever: HybridRAGRetriever instance
            
    Returns:
        Compiled StateGraph
    """
    
    # Create graph
    workflow = StateGraph(CircuitAnalysisState)
    
    # Add nodes
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("decide", decide_next_action)
    workflow.add_node("retrieve", lambda s: retrieve_knowledge_node(s, config))
    workflow.add_node("simulate", simulate_circuit_node)
    workflow.add_node("generate", lambda s: generate_answer_node(s, config))
    
    # Set entry point
    workflow.set_entry_point("initialize")
    
    # Add edges
    workflow.add_edge("initialize", "decide")
    
    # Add conditional routing from decide node
    workflow.add_conditional_edges(
        "decide",
        route_decision,
        {
            "retrieve": "retrieve",
            "simulate": "simulate",
            "generate": "generate",
            "end": END
        }
    )
    
    # After each action, go back to decide
    workflow.add_edge("retrieve", "decide")
    workflow.add_edge("simulate", "decide")
    workflow.add_edge("generate", END)
    
    # Compile with checkpointing for memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


# ==================== MAIN INTERFACE ====================

class CircuitAnalysisEngine:
    """
    Main interface for circuit analysis with LangGraph-based Decision-Making Head
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the circuit analysis engine
        
        Args:
            config_path: Path to configuration JSON file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize RAG retriever
        self.rag_retriever = HybridRAGRetriever(self.config.get("rag", {}))
        
        # Create LangGraph workflow
        self.graph = create_circuit_analysis_graph({
            "llm": self.llm,
            "rag_retriever": self.rag_retriever
        })
        
        logger.info("Circuit Analysis Engine initialized successfully")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        
        default_config = {
            "llm": {
                "provider": "openai",  # or "anthropic", "ollama"
                "model": "gpt-4-turbo-preview",
                "temperature": 0.3
            },
            "rag": {
                "embedding_type": "huggingface",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "knowledge_base_path": "./circuit_knowledge_base",
                "persist_directory": "./chroma_db",
                "chunk_size": 512,
            } 
        }
