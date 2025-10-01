"""
LangGraph-Based Decision-Making Head for Circuit Analysis
Integrates Hybrid RAG, Circuit Simulation, and ReAct Framework
Fixed and Enhanced Version
"""

import os
import json
import asyncio
from typing import Annotated, TypedDict, List, Dict, Any, Optional, Literal, Union
from datetime import datetime
import logging

# LangGraph and LangChain
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
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
from pydantic import BaseModel, Field, validator
import numpy as np
from dotenv import load_dotenv

# Circuit simulation
import subprocess
import tempfile
import re
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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
    messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage]], operator.add]
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

    class Config:
        arbitrary_types_allowed = True


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
        self.embeddings = None
        self.vector_store = None
        self.bm25_retriever = None
        self.reranker = None
        
        try:
            # Initialize embeddings
            self.embeddings = self._initialize_embeddings()
            
            # Initialize reranker
            self.reranker = self._initialize_reranker()
            
            # Load knowledge base
            self._load_knowledge_base()
            
            logger.info("HybridRAGRetriever initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG retriever: {e}")
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        embedding_type = self.config.get("embedding_type", "huggingface")
        
        try:
            if embedding_type == "openai":
                model = self.config.get("embedding_model", "text-embedding-3-small")
                return OpenAIEmbeddings(model=model)
            else:
                model_name = self.config.get(
                    "embedding_model", 
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                return HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
    
    def _initialize_reranker(self):
        """Initialize cross-encoder reranker"""
        try:
            model_name = self.config.get(
                "reranker_model",
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            model = HuggingFaceCrossEncoder(model_name=model_name)
            return CrossEncoderReranker(model=model, top_n=5)
        except Exception as e:
            logger.warning(f"Failed to initialize reranker: {e}")
            return None
    
    def _load_knowledge_base(self):
        """Load and index knowledge base documents"""
        kb_path = self.config.get("knowledge_base_path", "./circuit_knowledge_base")
        
        if not os.path.exists(kb_path):
            logger.warning(f"Knowledge base path {kb_path} does not exist. Creating directory...")
            os.makedirs(kb_path, exist_ok=True)
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
        try:
            persist_dir = self.config.get("persist_directory", "./chroma_db")
            os.makedirs(persist_dir, exist_ok=True)
            
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=persist_dir
            )
            logger.info(f"Vector store created with {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            self.vector_store = None
        
        # Create BM25 retriever
        try:
            self.bm25_retriever = BM25Retriever.from_documents(chunks)
            self.bm25_retriever.k = self.config.get("top_k", 5)
            logger.info("BM25 retriever created successfully")
        except Exception as e:
            logger.error(f"Failed to create BM25 retriever: {e}")
            self.bm25_retriever = None
    
    def _load_documents(self, kb_path: str) -> List[Document]:
        """Load documents from knowledge base directory"""
        documents = []
        
        # Load PDFs
        try:
            pdf_loader = DirectoryLoader(
                kb_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=False,
                silent_errors=True
            )
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            logger.info(f"Loaded {len(pdf_docs)} PDF documents")
        except Exception as e:
            logger.warning(f"Error loading PDFs: {e}")
        
        # Load DOCX files
        try:
            docx_loader = DirectoryLoader(
                kb_path,
                glob="**/*.docx",
                loader_cls=Docx2txtLoader,
                show_progress=False,
                silent_errors=True
            )
            docx_docs = docx_loader.load()
            documents.extend(docx_docs)
            logger.info(f"Loaded {len(docx_docs)} DOCX documents")
        except Exception as e:
            logger.warning(f"Error loading DOCX files: {e}")
        
        # Load text files
        try:
            txt_loader = DirectoryLoader(
                kb_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                show_progress=False,
                silent_errors=True
            )
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
            logger.info(f"Loaded {len(txt_docs)} text documents")
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
            logger.warning("RAG system not fully initialized, returning limited context")
            return "Knowledge base not available or not fully initialized."
        
        # Enhance query with circuit context
        enhanced_query = self._enhance_query(query, circuit_context)
        
        try:
            # Dense retrieval
            dense_retriever = self.vector_store.as_retriever(
                search_kwargs={"k": top_k * 2}
            )
            
            # Create ensemble retriever
            ensemble_retriever = EnsembleRetriever(
                retrievers=[dense_retriever, self.bm25_retriever],
                weights=[0.7, 0.3]  # Favor dense retrieval
            )
            
            # Add reranking if available
            if self.reranker:
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=self.reranker,
                    base_retriever=ensemble_retriever
                )
                docs = compression_retriever.get_relevant_documents(enhanced_query)
            else:
                docs = ensemble_retriever.get_relevant_documents(enhanced_query)
            
            # Format context
            context = self._format_context(docs, circuit_context)
            return context
            
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return f"Error retrieving context from knowledge base: {str(e)}"
    
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
            context_parts.append(f"\n[Source {i}] {os.path.basename(source)}")
            context_parts.append(content)
            context_parts.append("-" * 50)
        
        return "\n".join(context_parts)


# ==================== CIRCUIT SIMULATION TOOLS ====================

class CircuitSimulator:
    """NgSpice-based circuit simulator"""
    
    def __init__(self):
        self.ngspice_available = self._verify_ngspice()
    
    def _verify_ngspice(self) -> bool:
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
                return True
            else:
                logger.warning("NgSpice verification failed")
                return False
        except FileNotFoundError:
            logger.error("NgSpice not found. Please install from: https://ngspice.sourceforge.io/")
            return False
        except Exception as e:
            logger.error(f"Error verifying NgSpice: {e}")
            return False
    
    def simulate(self, netlist: str) -> SimulationResult:
        """
        Simulate a SPICE netlist
        
        Args:
            netlist: SPICE netlist string
            
        Returns:
            SimulationResult with parsed results
        """
        
        if not self.ngspice_available:
            return SimulationResult(
                success=False,
                error_message="NgSpice not available. Please install NgSpice."
            )
        
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
                    output_log=result.stdout[:1000]  # Truncate long logs
                )
            else:
                return SimulationResult(
                    success=False,
                    error_message=result.stderr or "Simulation failed",
                    output_log=result.stdout[:1000] if result.stdout else None
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
            except Exception:
                pass
    
    def _validate_netlist(self, netlist: str) -> List[str]:
        """Validate SPICE netlist syntax"""
        errors = []
        
        if not netlist or not netlist.strip():
            errors.append("Empty netlist")
            return errors
        
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
            if line and line[0].upper() in 'RCLDVIQMXBEFGHJKSTUW':
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
        
        try:
            # Parse DC analysis
            dc_pattern = r'v\((\w+)\)\s*=\s*([-\d.e+]+)'
            dc_matches = re.findall(dc_pattern, output, re.IGNORECASE)
            if dc_matches:
                results["dc"] = {node: float(voltage) for node, voltage in dc_matches}
            
            # Parse current results
            current_pattern = r'i\((\w+)\)\s*=\s*([-\d.e+]+)'
            current_matches = re.findall(current_pattern, output, re.IGNORECASE)
            if current_matches:
                if "dc" not in results:
                    results["dc"] = {}
                results["dc"].update({f"i_{source}": float(current) for source, current in current_matches})
            
        except Exception as e:
            logger.warning(f"Error parsing simulation output: {e}")
        
        return results


# ==================== GLOBAL INSTANCES ====================

# These will be initialized by the engine
_simulator_instance = None
_rag_instance = None


def get_simulator() -> CircuitSimulator:
    """Get or create simulator instance"""
    global _simulator_instance
    if _simulator_instance is None:
        _simulator_instance = CircuitSimulator()
    return _simulator_instance


def get_rag_retriever() -> Optional[HybridRAGRetriever]:
    """Get RAG retriever instance"""
    return _rag_instance


# ==================== LANGGRAPH TOOLS ====================

@tool
def simulate_circuit(netlist: str) -> Dict[str, Any]:
    """
    Simulate a SPICE circuit netlist using NgSpice.
    
    Args:
        netlist: SPICE netlist string
        
    Returns:
        Dictionary containing simulation results
    """
    simulator = get_simulator()
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
    rag = get_rag_retriever()
    if rag:
        circuit_context = {"circuit_type": circuit_type}
        return rag.retrieve(query, circuit_context)
    return "Knowledge base not available."


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
        "diodes": [],
        "transistors": [],
        "other": []
    }
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('*') or line.startswith('.'):
            continue
        
        if not line:
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
        elif component_type == 'D':
            components["diodes"].append(line)
        elif component_type in ['Q', 'M']:
            components["transistors"].append(line)
        else:
            components["other"].append(line)
    
    # Calculate summary
    summary = {k: len(v) for k, v in components.items() if v}
    total_components = sum(summary.values())
    
    return {
        "summary": summary,
        "total_components": total_components,
        "details": components,
        "circuit_complexity": "simple" if total_components < 5 else "moderate" if total_components < 15 else "complex"
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

Always think carefully and show your reasoning. Be thorough and educational.""")
    
    if "messages" not in state or not state["messages"]:
        state["messages"] = [system_msg]
    
    return state


def decide_next_action(state: CircuitAnalysisState) -> CircuitAnalysisState:
    """Decide what action to take next based on current state"""
    logger.info(f"Deciding next action (iteration {state['iteration']})")
    
    # Check iteration limit
    if state["iteration"] >= state["max_iterations"]:
        logger.info("Max iterations reached, generating answer")
        state["next_action"] = "generate_answer"
        return state
    
    # Decision logic based on what's been done
    if not state.get("retrieved_context"):
        state["next_action"] = "retrieve_knowledge"
        logger.info("Next action: retrieve_knowledge (no context yet)")
    elif not state.get("simulation_attempted") and state.get("netlist"):
        state["next_action"] = "simulate_circuit"
        logger.info("Next action: simulate_circuit (netlist available, not yet simulated)")
    elif state.get("simulation_attempted") and not state.get("answer_generated"):
        state["next_action"] = "generate_answer"
        logger.info("Next action: generate_answer (have context and simulation)")
    else:
        state["next_action"] = "generate_answer"
        logger.info("Next action: generate_answer (fallback)")
    
    return state


def retrieve_knowledge_node(state: CircuitAnalysisState) -> CircuitAnalysisState:
    """Retrieve relevant knowledge using Hybrid RAG"""
    logger.info("Retrieving knowledge from RAG system")
    
    rag = get_rag_retriever()
    if not rag:
        logger.warning("RAG retriever not available")
        state["retrieved_context"] = "Knowledge base not available."
        state["iteration"] += 1
        return state
    
    # Prepare circuit context
    circuit_context = {
        "circuit_type": state.get("circuit_type", "unknown"),
        "components": [],
        "analysis_type": "general"
    }
    
    # Extract component info if netlist is available
    if state.get("netlist"):
        try:
            comp_analysis = analyze_components.invoke({"netlist": state["netlist"]})
            if comp_analysis and "summary" in comp_analysis:
                circuit_context["components"] = list(comp_analysis["summary"].keys())
        except Exception as e:
            logger.warning(f"Could not analyze components: {e}")
    
    # Retrieve context
    try:
        context = rag.retrieve(
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
        state["reasoning_steps"].append({
            "step": len(state["reasoning_steps"]) + 1,
            "action": "retrieve_knowledge",
            "result": "Failed to retrieve context",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
    
    state["iteration"] += 1
    return state


def simulate_circuit_node(state: CircuitAnalysisState) -> CircuitAnalysisState:
    """Simulate the circuit using NgSpice"""
    logger.info("Simulating circuit")
    
    netlist = state.get("refined_netlist") or state.get("netlist")
    
    if not netlist:
        logger.warning("No netlist available for simulation")
        state["simulation_attempted"] = True
        state["simulation_valid"] = False
        state["simulation_error"] = "No netlist provided"
        state["iteration"] += 1
        return state
    
    try:
        simulator = get_simulator()
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
                "details": {
                    "dc_nodes": len(result.dc_analysis) if result.dc_analysis else 0,
                    "has_ac": result.ac_analysis is not None,
                    "has_transient": result.transient_analysis is not None
                },
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
        state["reasoning_steps"].append({
            "step": len(state["reasoning_steps"]) + 1,
            "action": "simulate_circuit",
            "result": "Simulation exception",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
    
    state["iteration"] += 1
    return state


def generate_answer_node(state: CircuitAnalysisState, llm: Any) -> CircuitAnalysisState:
    """Generate final answer using LLM with all gathered context"""
    logger.info("Generating final answer")
    
    if not llm:
        logger.error("LLM not provided")
        state["final_answer"] = "Error: Language model not available"
        state["answer_generated"] = True
        return state
    
    # Construct comprehensive prompt
    prompt_parts = [
        f"Question: {state['question']}",
        f"\nCircuit Description: {state.get('image_description', 'No description provided')}",
    ]
    
    # Add netlist
    netlist = state.get("refined_netlist") or state.get("netlist")
    if netlist:
        prompt_parts.append(f"\nSPICE Netlist:\n{netlist}")
    
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
    prompt_parts.append("2. Step-by-step explanation with clear reasoning")
    prompt_parts.append("3. Relevant calculations (if applicable)")
    prompt_parts.append("4. References to circuit theory concepts")
    prompt_parts.append("5. Verification using simulation results (if available)")
    
    full_prompt = "\n".join(prompt_parts)
    
    try:
        # Generate answer
        messages = [
            SystemMessage(content="You are an expert circuit analysis tutor. Provide clear, educational explanations with step-by-step reasoning."),
            HumanMessage(content=full_prompt)
        ]
        
        response = llm.invoke(messages)
        
        if hasattr(response, 'content'):
            answer = response.content
        else:
            answer = str(response)
        
        state["final_answer"] = answer
        state["answer_generated"] = True
        state["confidence_score"] = 0.9 if state.get("simulation_valid") else 0.7
        
        state["reasoning_steps"].append({
            "step": len(state["reasoning_steps"]) + 1,
            "action": "generate_answer",
            "result": "Answer generated successfully",
            "confidence": state["confidence_score"],
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info("Answer generated successfully")
        
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        state["final_answer"] = f"Error generating answer: {str(e)}\n\nHowever, here's what I gathered:\n"
        if state.get("simulation_results"):
            state["final_answer"] += f"\nSimulation Results: {json.dumps(state['simulation_results'], indent=2)}"
        if state.get("retrieved_context"):
            state["final_answer"] += f"\n\nRelevant Context:\n{state['retrieved_context'][:500]}..."
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

def create_circuit_analysis_graph(llm: Any, rag_retriever: Optional[HybridRAGRetriever] = None) -> StateGraph:
    """
    Create the LangGraph workflow for circuit analysis
    
    Args:
        llm: Language model instance
        rag_retriever: HybridRAGRetriever instance (optional)
            
    Returns:
        Compiled StateGraph
    """
    
    # Set global RAG instance
    global _rag_instance
    _rag_instance = rag_retriever
    
    # Create graph
    workflow = StateGraph(CircuitAnalysisState)
    
    # Add nodes with proper lambda wrapping to pass llm
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("decide", decide_next_action)
    workflow.add_node("retrieve", retrieve_knowledge_node)
    workflow.add_node("simulate", simulate_circuit_node)
    workflow.add_node("generate", lambda state: generate_answer_node(state, llm))
    
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
    
    # After each action, go back to decide (except generate which goes to END)
    workflow.add_edge("retrieve", "decide")
    workflow.add_edge("simulate", "decide")
    workflow.add_edge("generate", END)
    
    # Compile with checkpointing for memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    logger.info("Circuit analysis graph compiled successfully")
    return app


# ==================== MAIN INTERFACE ====================

class CircuitAnalysisEngine:
    """
    Main interface for circuit analysis with LangGraph-based Decision-Making Head
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the circuit analysis engine
        
        Args:
            config: Configuration dictionary
        """
        # Load configuration
        self.config = config or self._default_config()
        
        # Validate configuration
        self._validate_config()
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Initialize RAG retriever
        self.rag_retriever = None
        try:
            self.rag_retriever = HybridRAGRetriever(self.config.get("rag", {}))
        except Exception as e:
            logger.warning(f"Failed to initialize RAG retriever: {e}")
        
        # Create LangGraph workflow
        self.graph = create_circuit_analysis_graph(
            llm=self.llm,
            rag_retriever=self.rag_retriever
        )
        
        logger.info("Circuit Analysis Engine initialized successfully")
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "llm": {
                "provider": "openai",
                "model": "gpt-4-turbo-preview",
                "temperature": 0.3,
                "max_tokens": 2000
            },
            "rag": {
                "embedding_type": "huggingface",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "knowledge_base_path": "./circuit_knowledge_base",
                "persist_directory": "./chroma_db",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "top_k": 5
            },
            "simulation": {
                "timeout": 30,
                "max_retries": 2
            },
            "workflow": {
                "max_iterations": 10
            }
        }
    
    def _validate_config(self):
        """Validate configuration"""
        required_keys = ["llm", "rag"]
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
        
        # Validate LLM config
        if "provider" not in self.config["llm"]:
            raise ValueError("LLM provider not specified in config")
    
    def _initialize_llm(self):
        """Initialize language model based on configuration"""
        llm_config = self.config["llm"]
        provider = llm_config["provider"].lower()
        model = llm_config.get("model")
        temperature = llm_config.get("temperature", 0.3)
        max_tokens = llm_config.get("max_tokens", 2000)
        
        try:
            if provider == "openai":
                return ChatOpenAI(
                    model=model or "gpt-4-turbo-preview",
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            elif provider == "anthropic":
                return ChatAnthropic(
                    model=model or "claude-3-sonnet-20240229",
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            elif provider == "ollama":
                return ChatOllama(
                    model=model or "llama2",
                    temperature=temperature
                )
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def analyze(
        self,
        question: str,
        netlist: str,
        image_description: str = "",
        circuit_type: Optional[str] = None,
        max_iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze a circuit and answer questions
        
        Args:
            question: Question about the circuit
            netlist: SPICE netlist of the circuit
            image_description: Optional description of circuit diagram
            circuit_type: Optional circuit type hint
            max_iterations: Maximum workflow iterations
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Starting circuit analysis for question: {question[:50]}...")
        
        # Prepare initial state
        initial_state = {
            "question": question,
            "netlist": netlist,
            "image_description": image_description or "No description provided",
            "circuit_type": circuit_type,
            "refined_netlist": None,
            "messages": [],
            "iteration": 0,
            "max_iterations": max_iterations or self.config.get("workflow", {}).get("max_iterations", 10),
            "retrieved_context": None,
            "retrieval_queries": [],
            "simulation_attempted": False,
            "simulation_results": None,
            "simulation_valid": False,
            "simulation_error": None,
            "complexity_level": None,
            "component_analysis": None,
            "next_action": None,
            "confidence_score": 0.0,
            "reasoning_steps": [],
            "final_answer": None,
            "answer_generated": False
        }
        
        try:
            # Run the graph
            config = {"configurable": {"thread_id": f"analysis_{datetime.now().timestamp()}"}}
            final_state = self.graph.invoke(initial_state, config)
            
            # Extract results
            result = {
                "question": question,
                "answer": final_state.get("final_answer", "No answer generated"),
                "confidence_score": final_state.get("confidence_score", 0.0),
                "simulation_results": final_state.get("simulation_results"),
                "simulation_valid": final_state.get("simulation_valid", False),
                "reasoning_steps": final_state.get("reasoning_steps", []),
                "iterations": final_state.get("iteration", 0),
                "retrieved_context": final_state.get("retrieved_context"),
                "success": final_state.get("answer_generated", False)
            }
            
            logger.info(f"Analysis completed in {result['iterations']} iterations")
            return result
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "question": question,
                "answer": f"Analysis failed: {str(e)}",
                "confidence_score": 0.0,
                "simulation_results": None,
                "simulation_valid": False,
                "reasoning_steps": [],
                "iterations": 0,
                "success": False,
                "error": str(e)
            }
    
    async def analyze_async(
        self,
        question: str,
        netlist: str,
        image_description: str = "",
        circuit_type: Optional[str] = None,
        max_iterations: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Async version of analyze method
        
        Args:
            question: Question about the circuit
            netlist: SPICE netlist of the circuit
            image_description: Optional description of circuit diagram
            circuit_type: Optional circuit type hint
            max_iterations: Maximum workflow iterations
            
        Returns:
            Dictionary with analysis results
        """
        # For now, wrap sync version - could be improved with true async graph
        return self.analyze(question, netlist, image_description, circuit_type, max_iterations)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all system components"""
        simulator = get_simulator()
        
        return {
            "llm": {
                "provider": self.config["llm"]["provider"],
                "model": self.config["llm"]["model"],
                "available": self.llm is not None
            },
            "rag": {
                "available": self.rag_retriever is not None,
                "vector_store": self.rag_retriever.vector_store is not None if self.rag_retriever else False,
                "bm25": self.rag_retriever.bm25_retriever is not None if self.rag_retriever else False,
                "reranker": self.rag_retriever.reranker is not None if self.rag_retriever else False
            },
            "simulator": {
                "ngspice_available": simulator.ngspice_available if simulator else False
            },
            "graph": {
                "compiled": self.graph is not None
            }
        }
