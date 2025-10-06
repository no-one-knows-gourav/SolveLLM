"""
LangGraph-Based Circuit Analysis Engine with Incremental RAG Updates
Production-ready implementation with PySpice simulation and Hybrid RAG
"""

import os
import json
import asyncio
import hashlib
import re
import pickle
import tempfile
from typing import Annotated, TypedDict, List, Dict, Any, Optional, Literal, Union
from datetime import datetime
import logging
from pathlib import Path
import operator

# LangGraph and LangChain
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOllama

# Vector stores and embeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Document processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, DirectoryLoader
from langchain.schema import Document

# Utilities
from pydantic import BaseModel
import numpy as np
from dotenv import load_dotenv

# PySpice
try:
    from PySpice.Spice.Netlist import Circuit
    PYSPICE_AVAILABLE = True
except ImportError:
    PYSPICE_AVAILABLE = False

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== STATE & MODELS ====================

class CircuitAnalysisState(TypedDict):
    """State for circuit analysis workflow"""
    question: str
    image_description: str
    netlist: str
    refined_netlist: Optional[str]
    messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage]], operator.add]
    iteration: int
    max_iterations: int
    retrieved_context: Optional[str]
    retrieval_queries: List[str]
    simulation_attempted: bool
    simulation_results: Optional[Dict[str, Any]]
    simulation_valid: bool
    simulation_error: Optional[str]
    circuit_type: Optional[str]
    complexity_level: Optional[str]
    component_analysis: Optional[Dict[str, Any]]
    next_action: Optional[str]
    confidence_score: float
    reasoning_steps: List[Dict[str, Any]]
    final_answer: Optional[str]
    answer_generated: bool


class SimulationResult(BaseModel):
    """Structured simulation result"""
    success: bool
    dc_analysis: Optional[Dict[str, float]] = None
    ac_analysis: Optional[Dict[str, Any]] = None
    transient_analysis: Optional[Dict[str, Any]] = None
    operating_point: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    raw_output: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


# ==================== KNOWLEDGE BASE INDEXER ====================

class KnowledgeBaseIndexer:
    """Manages incremental knowledge base indexing"""
    
    def __init__(self, kb_path: str, persist_dir: str, metadata_file: str = "kb_metadata.pkl"):
        self.kb_path = Path(kb_path)
        self.persist_dir = Path(persist_dir)
        self.metadata_file = self.persist_dir / metadata_file
        self.file_metadata = {}
        
        self.kb_path.mkdir(parents=True, exist_ok=True)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._load_metadata()
    
    def _load_metadata(self):
        """Load existing file metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'rb') as f:
                    self.file_metadata = pickle.load(f)
                logger.info(f"Loaded metadata for {len(self.file_metadata)} files")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
                self.file_metadata = {}
        else:
            logger.info("No existing metadata found")
            self.file_metadata = {}
    
    def _save_metadata(self):
        """Save file metadata"""
        try:
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.file_metadata, f)
            logger.info(f"Saved metadata for {len(self.file_metadata)} files")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _get_file_hash(self, filepath: Path) -> str:
        """Calculate MD5 hash of file"""
        try:
            hasher = hashlib.md5()
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing {filepath}: {e}")
            return ""
    
    def _get_file_info(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Get file information"""
        try:
            stat = filepath.stat()
            return {
                "path": str(filepath),
                "hash": self._get_file_hash(filepath),
                "mtime": stat.st_mtime,
                "size": stat.st_size
            }
        except Exception as e:
            logger.error(f"Error getting file info for {filepath}: {e}")
            return None
    
    def scan_changes(self) -> Dict[str, List[Path]]:
        """Scan for file changes"""
        changes = {"added": [], "modified": [], "deleted": []}
        
        # Get current files
        current_files = {}
        for pattern in ["**/*.pdf", "**/*.docx", "**/*.txt"]:
            for filepath in self.kb_path.glob(pattern):
                if filepath.is_file():
                    current_files[str(filepath)] = filepath
        
        # Check for new/modified files
        for filepath_str, filepath in current_files.items():
            file_info = self._get_file_info(filepath)
            if not file_info:
                continue
            
            if filepath_str not in self.file_metadata:
                changes["added"].append(filepath)
                logger.info(f"New file: {filepath.name}")
            else:
                old_hash = self.file_metadata[filepath_str].get("hash")
                if old_hash != file_info["hash"]:
                    changes["modified"].append(filepath)
                    logger.info(f"Modified file: {filepath.name}")
        
        # Check for deleted files
        for filepath_str in list(self.file_metadata.keys()):
            if filepath_str not in current_files:
                changes["deleted"].append(Path(filepath_str))
                logger.info(f"Deleted file: {Path(filepath_str).name}")
        
        return changes
    
    def update_metadata(self, files: List[Path]):
        """Update metadata for files"""
        for filepath in files:
            file_info = self._get_file_info(filepath)
            if file_info:
                self.file_metadata[str(filepath)] = file_info
    
    def remove_from_metadata(self, files: List[Path]):
        """Remove files from metadata"""
        for filepath in files:
            self.file_metadata.pop(str(filepath), None)
    
    def needs_reindex(self) -> bool:
        """Check if reindexing needed"""
        changes = self.scan_changes()
        return bool(changes["added"] or changes["modified"] or changes["deleted"])
    
    def save(self):
        """Save metadata"""
        self._save_metadata()


# ==================== HYBRID RAG RETRIEVER ====================

class HybridRAGRetriever:
    """Hybrid RAG with incremental updates"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.embeddings = None
        self.vector_store = None
        self.bm25_retriever = None
        self.reranker = None
        self.indexer = None
        self.all_chunks = []
        
        try:
            self.embeddings = self._initialize_embeddings()
            self.reranker = self._initialize_reranker()
            
            kb_path = config.get("knowledge_base_path", "./circuit_knowledge_base")
            persist_dir = config.get("persist_directory", "./chroma_db")
            self.indexer = KnowledgeBaseIndexer(kb_path, persist_dir)
            
            self._load_or_create_knowledge_base()
            logger.info("HybridRAGRetriever initialized")
        except Exception as e:
            logger.error(f"Failed to initialize RAG: {e}")
    
    def _initialize_embeddings(self):
        """Initialize embeddings"""
        embedding_type = self.config.get("embedding_type", "huggingface")
        
        if embedding_type == "openai":
            return OpenAIEmbeddings(
                model=self.config.get("embedding_model", "text-embedding-3-small")
            )
        else:
            return HuggingFaceEmbeddings(
                model_name=self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
    
    def _initialize_reranker(self):
        """Initialize reranker"""
        try:
            model_name = self.config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            model = HuggingFaceCrossEncoder(model_name=model_name)
            return CrossEncoderReranker(model=model, top_n=5)
        except Exception as e:
            logger.warning(f"Failed to initialize reranker: {e}")
            return None
    
    def _chroma_db_exists(self) -> bool:
        """Check if Chroma DB exists"""
        persist_dir = Path(self.config.get("persist_directory", "./chroma_db"))
        
        if not persist_dir.exists():
            return False
        
        chroma_sqlite = persist_dir / "chroma.sqlite3"
        if not chroma_sqlite.exists():
            return False
        
        try:
            test_store = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=self.embeddings
            )
            if test_store._collection.count() > 0:
                logger.info(f"Found existing Chroma DB with {test_store._collection.count()} embeddings")
                return True
        except Exception as e:
            logger.warning(f"Error checking Chroma DB: {e}")
        
        return False
    
    def _load_or_create_knowledge_base(self):
        """Load existing or create new knowledge base"""
        db_exists = self._chroma_db_exists()
        needs_update = self.indexer.needs_reindex()
        
        if db_exists and not needs_update:
            logger.info("Loading existing Chroma DB (no changes)")
            self._load_existing_db()
        elif db_exists and needs_update:
            logger.info("Performing incremental update")
            self._incremental_update()
        else:
            logger.info("Creating new knowledge base")
            self._create_new_db()
    
    def _load_existing_db(self):
        """Load existing database"""
        persist_dir = self.config.get("persist_directory", "./chroma_db")
        
        try:
            self.vector_store = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings
            )
            
            chunks_file = Path(persist_dir) / "bm25_chunks.pkl"
            if chunks_file.exists():
                with open(chunks_file, 'rb') as f:
                    self.all_chunks = pickle.load(f)
                
                self.bm25_retriever = BM25Retriever.from_documents(self.all_chunks)
                self.bm25_retriever.k = self.config.get("top_k", 5)
                logger.info(f"Loaded {len(self.all_chunks)} chunks")
            else:
                logger.warning("BM25 chunks file not found")
        except Exception as e:
            logger.error(f"Error loading DB: {e}")
            self.vector_store = None
            self.bm25_retriever = None
    
    def _create_new_db(self):
        """Create new database"""
        kb_path = self.config.get("knowledge_base_path", "./circuit_knowledge_base")
        
        if not os.path.exists(kb_path):
            os.makedirs(kb_path, exist_ok=True)
            logger.warning(f"Created knowledge base directory: {kb_path}")
            return
        
        documents = self._load_all_documents(kb_path)
        if not documents:
            logger.warning("No documents found")
            return
        
        self._split_and_index_documents(documents)
        
        all_files = []
        for pattern in ["**/*.pdf", "**/*.docx", "**/*.txt"]:
            all_files.extend(Path(kb_path).glob(pattern))
        
        self.indexer.update_metadata(all_files)
        self.indexer.save()
    
    def _incremental_update(self):
        """Perform incremental update"""
        changes = self.indexer.scan_changes()
        self._load_existing_db()
        
        if not self.vector_store:
            logger.warning("Cannot load existing DB, creating new one")
            self._create_new_db()
            return
        
        if changes["deleted"]:
            self._remove_documents(changes["deleted"])
            self.indexer.remove_from_metadata(changes["deleted"])
        
        files_to_add = changes["added"] + changes["modified"]
        if files_to_add:
            if changes["modified"]:
                self._remove_documents(changes["modified"])
            
            documents = self._load_documents_from_paths(files_to_add)
            if documents:
                self._add_documents_to_index(documents)
            
            self.indexer.update_metadata(files_to_add)
        
        self.indexer.save()
        logger.info(f"Update complete: +{len(changes['added'])} ~{len(changes['modified'])} -{len(changes['deleted'])}")
    
    def _remove_documents(self, filepaths: List[Path]):
        """Remove documents from indexes"""
        try:
            ids_to_remove = []
            chunks_to_keep = []
            
            for chunk in self.all_chunks:
                source = chunk.metadata.get("source", "")
                if not any(str(fp) == source for fp in filepaths):
                    chunks_to_keep.append(chunk)
                else:
                    chunk_id = chunk.metadata.get("id")
                    if chunk_id:
                        ids_to_remove.append(chunk_id)
            
            if ids_to_remove and self.vector_store:
                self.vector_store._collection.delete(ids=ids_to_remove)
                logger.info(f"Removed {len(ids_to_remove)} chunks")
            
            self.all_chunks = chunks_to_keep
            
            if self.all_chunks:
                self.bm25_retriever = BM25Retriever.from_documents(self.all_chunks)
                self.bm25_retriever.k = self.config.get("top_k", 5)
            
            self._save_bm25_chunks()
        except Exception as e:
            logger.error(f"Error removing documents: {e}")
    
    def _add_documents_to_index(self, documents: List[Document]):
        """Add documents to index"""
        if not documents:
            return
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 512),
            chunk_overlap=self.config.get("chunk_overlap", 50),
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        new_chunks = text_splitter.split_documents(documents)
        
        for i, chunk in enumerate(new_chunks):
            chunk.metadata["id"] = f"chunk_{len(self.all_chunks) + i}"
        
        if self.vector_store:
            self.vector_store.add_documents(new_chunks)
            logger.info(f"Added {len(new_chunks)} chunks")
        
        self.all_chunks.extend(new_chunks)
        
        self.bm25_retriever = BM25Retriever.from_documents(self.all_chunks)
        self.bm25_retriever.k = self.config.get("top_k", 5)
        
        self._save_bm25_chunks()
        logger.info(f"Total chunks: {len(self.all_chunks)}")
    
    def _save_bm25_chunks(self):
        """Save BM25 chunks"""
        persist_dir = Path(self.config.get("persist_directory", "./chroma_db"))
        chunks_file = persist_dir / "bm25_chunks.pkl"
        
        try:
            with open(chunks_file, 'wb') as f:
                pickle.dump(self.all_chunks, f)
        except Exception as e:
            logger.error(f"Error saving chunks: {e}")
    
    def _split_and_index_documents(self, documents: List[Document]):
        """Split and index documents"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 512),
            chunk_overlap=self.config.get("chunk_overlap", 50),
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        
        for i, chunk in enumerate(chunks):
            chunk.metadata["id"] = f"chunk_{i}"
        
        self.all_chunks = chunks
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        
        try:
            persist_dir = self.config.get("persist_directory", "./chroma_db")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=persist_dir
            )
            logger.info("Vector store created")
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            self.vector_store = None
        
        try:
            self.bm25_retriever = BM25Retriever.from_documents(chunks)
            self.bm25_retriever.k = self.config.get("top_k", 5)
            logger.info("BM25 retriever created")
        except Exception as e:
            logger.error(f"Failed to create BM25 retriever: {e}")
            self.bm25_retriever = None
        
        self._save_bm25_chunks()
    
    def _load_documents_from_paths(self, paths: List[Path]) -> List[Document]:
        """Load documents from paths"""
        documents = []
        
        for filepath in paths:
            if not filepath.exists():
                continue
            
            try:
                suffix = filepath.suffix.lower()
                
                if suffix == '.pdf':
                    loader = PyPDFLoader(str(filepath))
                elif suffix == '.docx':
                    loader = Docx2txtLoader(str(filepath))
                elif suffix == '.txt':
                    loader = TextLoader(str(filepath))
                else:
                    continue
                
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {filepath.name}: {len(docs)} pages")
            except Exception as e:
                logger.warning(f"Error loading {filepath.name}: {e}")
        
        return documents
    
    def _load_all_documents(self, kb_path: str) -> List[Document]:
        """Load all documents from directory"""
        documents = []
        
        for loader_cls, pattern in [(PyPDFLoader, "**/*.pdf"), 
                                     (Docx2txtLoader, "**/*.docx"), 
                                     (TextLoader, "**/*.txt")]:
            try:
                loader = DirectoryLoader(
                    kb_path,
                    glob=pattern,
                    loader_cls=loader_cls,
                    show_progress=False,
                    silent_errors=True
                )
                docs = loader.load()
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} {pattern} files")
            except Exception as e:
                logger.warning(f"Error loading {pattern}: {e}")
        
        return documents
    
    def force_reindex(self):
        """Force complete reindex"""
        logger.info("Forcing complete reindex")
        self.indexer.file_metadata = {}
        self._create_new_db()
    
    def retrieve(self, query: str, circuit_context: Optional[Dict[str, Any]] = None, top_k: int = 5) -> str:
        """Retrieve relevant context"""
        if not self.vector_store or not self.bm25_retriever:
            return "Knowledge base not available"
        
        enhanced_query = self._enhance_query(query, circuit_context)
        
        try:
            dense_retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k * 2})
            ensemble_retriever = EnsembleRetriever(
                retrievers=[dense_retriever, self.bm25_retriever],
                weights=[0.7, 0.3]
            )
            
            if self.reranker:
                retriever = ContextualCompressionRetriever(
                    base_compressor=self.reranker,
                    base_retriever=ensemble_retriever
                )
                docs = retriever.get_relevant_documents(enhanced_query)
            else:
                docs = ensemble_retriever.get_relevant_documents(enhanced_query)
            
            return self._format_context(docs, circuit_context)
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return f"Error retrieving context: {str(e)}"
    
    def _enhance_query(self, query: str, circuit_context: Optional[Dict[str, Any]]) -> str:
        """Enhance query with context"""
        if not circuit_context:
            return query
        
        parts = [query]
        if circuit_context.get("circuit_type"):
            parts.append(f"Circuit type: {circuit_context['circuit_type']}")
        if circuit_context.get("components"):
            parts.append(f"Components: {', '.join(circuit_context['components'])}")
        
        return " | ".join(parts)
    
    def _format_context(self, docs: List[Document], circuit_context: Optional[Dict[str, Any]]) -> str:
        """Format context"""
        parts = []
        
        if circuit_context:
            parts.append("=== CIRCUIT CONTEXT ===")
            if circuit_context.get("circuit_type"):
                parts.append(f"Circuit Type: {circuit_context['circuit_type']}")
            if circuit_context.get("components"):
                parts.append(f"Components: {', '.join(circuit_context['components'])}")
            parts.append("")
        
        parts.append("=== RETRIEVED KNOWLEDGE ===")
        for i, doc in enumerate(docs, 1):
            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            parts.append(f"\n[Source {i}] {source}")
            parts.append(doc.page_content.strip())
            parts.append("-" * 50)
        
        return "\n".join(parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            "total_chunks": len(self.all_chunks),
            "indexed_files": len(self.indexer.file_metadata),
            "vector_store_ready": self.vector_store is not None,
            "bm25_ready": self.bm25_retriever is not None,
            "reranker_ready": self.reranker is not None
        }


# ==================== PYSPICE SIMULATOR ====================

class PySpiceSimulator:
    """Circuit simulator using PySpice"""
    
    def __init__(self):
        self.pyspice_available = PYSPICE_AVAILABLE
        if not PYSPICE_AVAILABLE:
            logger.warning("PySpice not available")
        else:
            logger.info("PySpice simulator initialized")
    
    def simulate(self, netlist: str) -> SimulationResult:
        """Simulate SPICE netlist"""
        if not self.pyspice_available:
            return SimulationResult(
                success=False,
                error_message="PySpice not available. Install with: pip install PySpice"
            )
        
        validation_errors = self._validate_netlist(netlist)
        if validation_errors:
            return SimulationResult(
                success=False,
                error_message=f"Validation failed: {'; '.join(validation_errors)}"
            )
        
        try:
            circuit_info = self._parse_netlist(netlist)
            if circuit_info.get("error"):
                return SimulationResult(success=False, error_message=circuit_info["error"])
            
            circuit = self._create_circuit(circuit_info)
            if not circuit:
                return SimulationResult(success=False, error_message="Failed to create circuit")
            
            return self._run_simulation(circuit, circuit_info)
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            return SimulationResult(success=False, error_message=str(e))
    
    def _validate_netlist(self, netlist: str) -> List[str]:
        """Validate netlist"""
        errors = []
        if not netlist or not netlist.strip():
            return ["Empty netlist"]
        
        lines = netlist.strip().split('\n')
        has_ground = has_component = has_analysis = False
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('*'):
                continue
            
            if '0' in line.split() or 'GND' in line.upper():
                has_ground = True
            if line and line[0].upper() in 'RCLDVIQMXBEFGHJKSTUW':
                has_component = True
            if line.startswith('.') and line.split()[0].upper() in ['.OP', '.DC', '.AC', '.TRAN']:
                has_analysis = True
        
        if not has_ground:
            errors.append("No ground node")
        if not has_component:
            errors.append("No components")
        
        return errors
    
    def _parse_netlist(self, netlist: str) -> Dict[str, Any]:
        """Parse netlist"""
        lines = netlist.strip().split('\n')
        info = {"title": "Circuit", "components": [], "analysis": [], "options": []}
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('*') or line.upper() == '.END':
                continue
            
            if line and line[0].upper() in 'RCLDVIQMXBEFGHJKSTUW':
                info["components"].append(self._parse_component(line))
            elif line.startswith('.'):
                cmd = line.split()[0].upper()
                if cmd in ['.OP', '.DC', '.AC', '.TRAN']:
                    info["analysis"].append({"type": cmd[1:], "command": line})
        
        if not info["analysis"]:
            info["analysis"].append({"type": "OP", "command": ".OP"})
        
        return info
    
    def _parse_component(self, line: str) -> Dict[str, Any]:
        """Parse component line"""
        parts = line.split()
        return {
            "type": line[0].upper(),
            "name": parts[0],
            "nodes": parts[1:3] if len(parts) >= 3 else [],
            "value": ' '.join(parts[3:]) if len(parts) > 3 else "",
            "raw": line
        }
    
    def _create_circuit(self, info: Dict[str, Any]) -> Optional[Circuit]:
        """Create PySpice circuit"""
        try:
            circuit = Circuit(info["title"])
            
            for comp in info["components"]:
                comp_type = comp["type"]
                name = comp["name"][1:]
                nodes = [n if n.upper() != 'GND' else '0' for n in comp.get("nodes", [])]
                value_str = comp.get("value", "")
                
                try:
                    value = self._parse_value(value_str.split()[0] if value_str else "1")
                    
                    if comp_type == 'R':
                        circuit.R(name, nodes[0], nodes[1], value)
                    elif comp_type == 'C':
                        circuit.C(name, nodes[0], nodes[1], value)
                    elif comp_type == 'L':
                        circuit.L(name, nodes[0], nodes[1], value)
                    elif comp_type == 'V':
                        dc_val = self._extract_dc_value(value_str) if 'DC' in value_str.upper() else value
                        circuit.V(name, nodes[0], nodes[1], dc_val)
                    elif comp_type == 'I':
                        dc_val = self._extract_dc_value(value_str) if 'DC' in value_str.upper() else value
                        circuit.I(name, nodes[0], nodes[1], dc_val)
                except Exception as e:
                    logger.warning(f"Error adding {comp['name']}: {e}")
            
            return circuit
        except Exception as e:
            logger.error(f"Error creating circuit: {e}")
            return None
    
    def _parse_value(self, value_str: str) -> float:
        """Parse value with SI prefixes"""
        value_str = value_str.strip().upper()
        multipliers = {
            'T': 1e12, 'G': 1e9, 'MEG': 1e6, 'X': 1e6,
            'K': 1e3, 'M': 1e-3, 'U': 1e-6, 'N': 1e-9,
            'P': 1e-12, 'F': 1e-15
        }
        
        match = re.match(r'([-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)\s*([A-Z]*)', value_str)
        if match:
            number = float(match.group(1))
            suffix = match.group(3)
            for prefix, mult in multipliers.items():
                if suffix.startswith(prefix):
                    return number * mult
            return number
        
        try:
            return float(value_str)
        except:
            return 1.0
    
    def _extract_dc_value(self, value_str: str) -> float:
        """Extract DC value"""
        parts = value_str.upper().split()
        for i, part in enumerate(parts):
            if part == 'DC' and i + 1 < len(parts):
                return self._parse_value(parts[i + 1])
        return 0.0
    
    def _run_simulation(self, circuit: Circuit, info: Dict[str, Any]) -> SimulationResult:
        """Run simulation"""
        try:
            simulator = circuit.simulator(temperature=25, nominal_temperature=25)
            results = SimulationResult(success=True)
            
            for analysis in info["analysis"]:
                analysis_type = analysis["type"]
                
                try:
                    if analysis_type == "OP":
                        analysis_result = simulator.operating_point()
                        op_results = {}
                        for node in analysis_result.nodes.values():
                            node_name = str(node)
                            if node_name != '0':
                                try:
                                    op_results[node_name] = float(node)
                                except:
                                    pass
                        results.operating_point = op_results
                        results.dc_analysis = op_results
                        logger.info(f"OP analysis: {len(op_results)} nodes")
                    
                    elif analysis_type == "DC":
                        cmd_parts = analysis["command"].split()
                        if len(cmd_parts) >= 5:
                            source = cmd_parts[1]
                            start = self._parse_value(cmd_parts[2])
                            stop = self._parse_value(cmd_parts[3])
                            step = self._parse_value(cmd_parts[4])
                            
                            analysis_result = simulator.dc(**{source: slice(start, stop, step)})
                            dc_results = {"sweep_variable": source, "data": {}}
                            
                            for node in analysis_result.nodes.values():
                                node_name = str(node)
                                if node_name != '0':
                                    dc_results["data"][node_name] = list(node)
                            
                            results.dc_analysis = dc_results
                            logger.info("DC sweep analysis complete")
                    
                    elif analysis_type == "AC":
                        cmd_parts = analysis["command"].split()
                        if len(cmd_parts) >= 5:
                            variation = cmd_parts[1]
                            points = int(cmd_parts[2])
                            fstart = self._parse_value(cmd_parts[3])
                            fstop = self._parse_value(cmd_parts[4])
                            
                            analysis_result = simulator.ac(
                                start_frequency=fstart,
                                stop_frequency=fstop,
                                number_of_points=points,
                                variation=variation.lower()
                            )
                            
                            ac_results = {"frequency": list(analysis_result.frequency), "data": {}}
                            for node in analysis_result.nodes.values():
                                node_name = str(node)
                                if node_name != '0':
                                    ac_results["data"][node_name] = {
                                        "magnitude": list(abs(node)),
                                        "phase": list(np.angle(node, deg=True))
                                    }
                            
                            results.ac_analysis = ac_results
                            logger.info("AC analysis complete")
                    
                    elif analysis_type == "TRAN":
                        cmd_parts = analysis["command"].split()
                        if len(cmd_parts) >= 3:
                            tstep = self._parse_value(cmd_parts[1])
                            tstop = self._parse_value(cmd_parts[2])
                            tstart = self._parse_value(cmd_parts[3]) if len(cmd_parts) >= 4 else 0
                            
                            analysis_result = simulator.transient(
                                step_time=tstep,
                                end_time=tstop,
                                start_time=tstart
                            )
                            
                            tran_results = {"time": list(analysis_result.time), "data": {}}
                            for node in analysis_result.nodes.values():
                                node_name = str(node)
                                if node_name != '0':
                                    tran_results["data"][node_name] = list(node)
                            
                            results.transient_analysis = tran_results
                            logger.info("Transient analysis complete")
                
                except Exception as e:
                    logger.error(f"Error in {analysis_type} analysis: {e}")
                    if not results.error_message:
                        results.error_message = f"{analysis_type} failed: {str(e)}"
            
            if not (results.operating_point or results.dc_analysis or results.ac_analysis or results.transient_analysis):
                results.success = False
                results.error_message = "No analysis results"
            
            return results
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            return SimulationResult(success=False, error_message=str(e))


# ==================== GLOBAL INSTANCES ====================

_simulator_instance = None
_rag_instance = None


def get_simulator() -> PySpiceSimulator:
    """Get simulator instance"""
    global _simulator_instance
    if _simulator_instance is None:
        _simulator_instance = PySpiceSimulator()
    return _simulator_instance


def get_rag_retriever() -> Optional[HybridRAGRetriever]:
    """Get RAG instance"""
    return _rag_instance


# ==================== LANGGRAPH TOOLS ====================

@tool
def simulate_circuit(netlist: str) -> Dict[str, Any]:
    """Simulate SPICE netlist"""
    return get_simulator().simulate(netlist).dict()


@tool
def retrieve_knowledge(query: str, circuit_type: str = "general") -> str:
    """Retrieve knowledge from knowledge base"""
    rag = get_rag_retriever()
    if rag:
        return rag.retrieve(query, {"circuit_type": circuit_type})
    return "Knowledge base not available"


@tool
def analyze_components(netlist: str) -> Dict[str, Any]:
    """Analyze circuit components"""
    lines = netlist.strip().split('\n')
    components = {
        "resistors": [], "capacitors": [], "inductors": [],
        "voltage_sources": [], "current_sources": [],
        "diodes": [], "transistors": [], "other": []
    }
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('*') or line.startswith('.'):
            continue
        
        comp_type = line[0].upper()
        if comp_type == 'R':
            components["resistors"].append(line)
        elif comp_type == 'C':
            components["capacitors"].append(line)
        elif comp_type == 'L':
            components["inductors"].append(line)
        elif comp_type == 'V':
            components["voltage_sources"].append(line)
        elif comp_type == 'I':
            components["current_sources"].append(line)
        elif comp_type == 'D':
            components["diodes"].append(line)
        elif comp_type in ['Q', 'M']:
            components["transistors"].append(line)
        else:
            components["other"].append(line)
    
    summary = {k: len(v) for k, v in components.items() if v}
    total = sum(summary.values())
    
    return {
        "summary": summary,
        "total_components": total,
        "details": components,
        "circuit_complexity": "simple" if total < 5 else "moderate" if total < 15 else "complex"
    }


# ==================== WORKFLOW NODES ====================

def initialize_state(state: CircuitAnalysisState) -> CircuitAnalysisState:
    """Initialize state"""
    logger.info("Initializing analysis state")
    
    state["iteration"] = 0
    state["max_iterations"] = state.get("max_iterations", 10)
    state["simulation_attempted"] = False
    state["answer_generated"] = False
    state["confidence_score"] = 0.0
    state["reasoning_steps"] = []
    state["retrieval_queries"] = []
    
    system_msg = SystemMessage(content="""You are an expert circuit analysis AI assistant.
You help students and engineers by:
1. Analyzing circuit netlists and diagrams
2. Running simulations to verify behavior
3. Retrieving relevant circuit theory
4. Providing step-by-step explanations

Be thorough and educational.""")
    
    if "messages" not in state or not state["messages"]:
        state["messages"] = [system_msg]
    
    return state


def decide_next_action(state: CircuitAnalysisState) -> CircuitAnalysisState:
    """Decide next action"""
    logger.info(f"Deciding next action (iteration {state['iteration']})")
    
    if state["iteration"] >= state["max_iterations"]:
        state["next_action"] = "generate_answer"
        return state
    
    if not state.get("retrieved_context"):
        state["next_action"] = "retrieve_knowledge"
    elif not state.get("simulation_attempted") and state.get("netlist"):
        state["next_action"] = "simulate_circuit"
    else:
        state["next_action"] = "generate_answer"
    
    return state


def retrieve_knowledge_node(state: CircuitAnalysisState) -> CircuitAnalysisState:
    """Retrieve knowledge"""
    logger.info("Retrieving knowledge")
    
    rag = get_rag_retriever()
    if not rag:
        state["retrieved_context"] = "Knowledge base not available"
        state["iteration"] += 1
        return state
    
    circuit_context = {
        "circuit_type": state.get("circuit_type", "unknown"),
        "components": [],
        "analysis_type": "general"
    }
    
    if state.get("netlist"):
        try:
            comp_analysis = analyze_components.invoke({"netlist": state["netlist"]})
            if comp_analysis and "summary" in comp_analysis:
                circuit_context["components"] = list(comp_analysis["summary"].keys())
        except Exception as e:
            logger.warning(f"Could not analyze components: {e}")
    
    try:
        context = rag.retrieve(state["question"], circuit_context, top_k=5)
        state["retrieved_context"] = context
        state["retrieval_queries"].append(state["question"])
        
        state["reasoning_steps"].append({
            "step": len(state["reasoning_steps"]) + 1,
            "action": "retrieve_knowledge",
            "result": "Success",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        state["retrieved_context"] = f"Error: {str(e)}"
        state["reasoning_steps"].append({
            "step": len(state["reasoning_steps"]) + 1,
            "action": "retrieve_knowledge",
            "result": "Failed",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })
    
    state["iteration"] += 1
    return state


def simulate_circuit_node(state: CircuitAnalysisState) -> CircuitAnalysisState:
    """Simulate circuit"""
    logger.info("Simulating circuit")
    
    netlist = state.get("refined_netlist") or state.get("netlist")
    
    if not netlist:
        state["simulation_attempted"] = True
        state["simulation_valid"] = False
        state["simulation_error"] = "No netlist"
        state["iteration"] += 1
        return state
    
    try:
        result = get_simulator().simulate(netlist)
        
        state["simulation_attempted"] = True
        state["simulation_results"] = result.dict()
        state["simulation_valid"] = result.success
        
        if result.success:
            state["reasoning_steps"].append({
                "step": len(state["reasoning_steps"]) + 1,
                "action": "simulate_circuit",
                "result": "Success",
                "timestamp": datetime.now().isoformat()
            })
        else:
            state["simulation_error"] = result.error_message
            state["reasoning_steps"].append({
                "step": len(state["reasoning_steps"]) + 1,
                "action": "simulate_circuit",
                "result": "Failed",
                "error": result.error_message,
                "timestamp": datetime.now().isoformat()
            })
    except Exception as e:
        state["simulation_attempted"] = True
        state["simulation_valid"] = False
        state["simulation_error"] = str(e)
    
    state["iteration"] += 1
    return state


def generate_answer_node(state: CircuitAnalysisState, llm: Any) -> CircuitAnalysisState:
    """Generate answer"""
    logger.info("Generating answer")
    
    if not llm:
        state["final_answer"] = "Error: LLM not available"
        state["answer_generated"] = True
        return state
    
    prompt_parts = [
        f"Question: {state['question']}",
        f"\nCircuit Description: {state.get('image_description', 'No description')}",
    ]
    
    netlist = state.get("refined_netlist") or state.get("netlist")
    if netlist:
        prompt_parts.append(f"\nSPICE Netlist:\n{netlist}")
    
    if state.get("simulation_valid") and state.get("simulation_results"):
        sim_results = state["simulation_results"]
        prompt_parts.append("\nSimulation Results (PySpice):")
        
        if sim_results.get("operating_point"):
            prompt_parts.append("\nOperating Point:")
            for node, voltage in sim_results["operating_point"].items():
                prompt_parts.append(f"  Node {node}: {voltage:.6f} V")
        
        if sim_results.get("dc_analysis"):
            prompt_parts.append("\nDC Analysis: Available")
        if sim_results.get("ac_analysis"):
            prompt_parts.append("\nAC Analysis: Available")
        if sim_results.get("transient_analysis"):
            prompt_parts.append("\nTransient Analysis: Available")
    elif state.get("simulation_error"):
        prompt_parts.append(f"\nSimulation Error: {state['simulation_error']}")
    
    if state.get("retrieved_context"):
        prompt_parts.append(f"\n{state['retrieved_context']}")
    
    prompt_parts.append("\n" + "="*60)
    prompt_parts.append("Provide a comprehensive answer including:")
    prompt_parts.append("1. Direct answer")
    prompt_parts.append("2. Step-by-step explanation")
    prompt_parts.append("3. Calculations")
    prompt_parts.append("4. Circuit theory")
    prompt_parts.append("5. Simulation verification (if available)")
    
    full_prompt = "\n".join(prompt_parts)
    
    try:
        messages = [
            SystemMessage(content="You are an expert circuit analysis tutor. Provide clear, educational explanations with detailed reasoning."),
            HumanMessage(content=full_prompt)
        ]
        
        response = llm.invoke(messages)
        answer = response.content if hasattr(response, 'content') else str(response)
        
        state["final_answer"] = answer
        state["answer_generated"] = True
        
        confidence = 0.5
        if state.get("simulation_valid"):
            confidence += 0.3
        if state.get("retrieved_context") and len(state.get("retrieved_context", "")) > 100:
            confidence += 0.2
        
        state["confidence_score"] = min(confidence, 1.0)
        
        state["reasoning_steps"].append({
            "step": len(state["reasoning_steps"]) + 1,
            "action": "generate_answer",
            "result": "Success",
            "confidence": state["confidence_score"],
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Answer generated (confidence: {state['confidence_score']:.2f})")
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        state["final_answer"] = f"Error generating answer: {str(e)}"
        state["answer_generated"] = True
        state["confidence_score"] = 0.3
    
    return state


def route_decision(state: CircuitAnalysisState) -> Literal["retrieve", "simulate", "generate", "end"]:
    """Route to next node"""
    action = state.get("next_action", "end")
    
    if action == "retrieve_knowledge":
        return "retrieve"
    elif action == "simulate_circuit":
        return "simulate"
    elif action == "generate_answer":
        return "generate"
    else:
        return "end"


# ==================== BUILD WORKFLOW ====================

def create_circuit_analysis_graph(llm: Any, rag_retriever: Optional[HybridRAGRetriever] = None) -> StateGraph:
    """Create LangGraph workflow"""
    global _rag_instance
    _rag_instance = rag_retriever
    
    workflow = StateGraph(CircuitAnalysisState)
    
    workflow.add_node("initialize", initialize_state)
    workflow.add_node("decide", decide_next_action)
    workflow.add_node("retrieve", retrieve_knowledge_node)
    workflow.add_node("simulate", simulate_circuit_node)
    workflow.add_node("generate", lambda state: generate_answer_node(state, llm))
    
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "decide")
    
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
    
    workflow.add_edge("retrieve", "decide")
    workflow.add_edge("simulate", "decide")
    workflow.add_edge("generate", END)
    
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    logger.info("Circuit analysis graph compiled")
    return app


# ==================== MAIN ENGINE ====================

class CircuitAnalysisEngine:
    """Main circuit analysis engine"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        self._validate_config()
        
        self.llm = self._initialize_llm()
        
        self.rag_retriever = None
        try:
            self.rag_retriever = HybridRAGRetriever(self.config.get("rag", {}))
        except Exception as e:
            logger.warning(f"Failed to initialize RAG: {e}")
        
        self.graph = create_circuit_analysis_graph(self.llm, self.rag_retriever)
        logger.info("Circuit Analysis Engine initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
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
            "workflow": {"max_iterations": 10}
        }
    
    def _validate_config(self):
        """Validate configuration"""
        if "llm" not in self.config or "provider" not in self.config["llm"]:
            raise ValueError("Invalid LLM configuration")
    
    def _initialize_llm(self):
        """Initialize LLM"""
        llm_config = self.config["llm"]
        provider = llm_config["provider"].lower()
        model = llm_config.get("model")
        temperature = llm_config.get("temperature", 0.3)
        max_tokens = llm_config.get("max_tokens", 2000)
        
        if provider == "openai":
            return ChatOpenAI(model=model or "gpt-4-turbo-preview", temperature=temperature, max_tokens=max_tokens)
        elif provider == "anthropic":
            return ChatAnthropic(model=model or "claude-3-sonnet-20240229", temperature=temperature, max_tokens=max_tokens)
        elif provider == "ollama":
            return ChatOllama(model=model or "llama2", temperature=temperature)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def analyze(self, question: str, netlist: str, image_description: str = "",
                circuit_type: Optional[str] = None, max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """Analyze circuit"""
        logger.info(f"Starting analysis: {question[:50]}...")
        
        initial_state = {
            "question": question,
            "netlist": netlist,
            "image_description": image_description or "No description",
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
            config = {"configurable": {"thread_id": f"analysis_{datetime.now().timestamp()}"}}
            final_state = self.graph.invoke(initial_state, config)
            
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
            
            logger.info(f"Analysis complete in {result['iterations']} iterations")
            return result
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return {
                "question": question,
                "answer": f"Analysis failed: {str(e)}",
                "confidence_score": 0.0,
                "success": False,
                "error": str(e)
            }
    
    async def analyze_async(self, question: str, netlist: str, image_description: str = "",
                           circuit_type: Optional[str] = None, max_iterations: Optional[int] = None) -> Dict[str, Any]:
        """Async version"""
        return self.analyze(question, netlist, image_description, circuit_type, max_iterations)
    
    def force_reindex_knowledge_base(self):
        """Force complete reindex"""
        if self.rag_retriever:
            self.rag_retriever.force_reindex()
        else:
            logger.warning("RAG not available")
    
    def refresh_knowledge_base(self):
        """Check and update knowledge base"""
        if self.rag_retriever and self.rag_retriever.indexer:
            if self.rag_retriever.indexer.needs_reindex():
                logger.info("Changes detected, updating")
                self.rag_retriever._incremental_update()
            else:
                logger.info("No changes detected")
        else:
            logger.warning("RAG not available")
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get KB statistics"""
        if self.rag_retriever:
            return self.rag_retriever.get_stats()
        return {"error": "RAG not available"}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        simulator = get_simulator()
        
        status = {
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
                "type": "PySpice",
                "available": simulator.pyspice_available if simulator else False
            },
            "graph": {"compiled": self.graph is not None}
        }
        
        if self.rag_retriever:
            status["knowledge_base"] = self.get_knowledge_base_stats()
        
        return status


