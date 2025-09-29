import os
import json
import numpy as np
import asyncio
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path

# Core libraries
import torch
import faiss
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    pipeline
)
from sentence_transformers import SentenceTransformer
import spacy

# Search and retrieval
from rank_bm25 import BM25Okapi
import elasticsearch
from elasticsearch import Elasticsearch

# Document processing
import PyPDF2
import docx
from bs4 import BeautifulSoup
import markdown

# Vector database
import chromadb
from chromadb.utils import embedding_functions

# Circuit analysis specific
import spice_parser
import netlistx

# API and async
import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Container for retrieval results"""
    content: str
    metadata: Dict[str, Any]
    score: float
    source: str
    chunk_id: str
    document_id: str

@dataclass
class CircuitContext:
    """Container for circuit-specific context"""
    netlist: str
    simulation_results: Dict[str, Any]
    components: List[Dict[str, Any]]
    circuit_type: str
    complexity_level: str

class HybridRAGSystem:
    """
    Hybrid RAG system combining dense and sparse retrieval for circuit analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self._init_embedding_models()
        self._init_reranking_models()
        self._init_search_engines()
        self._init_vector_stores()
        
        # Initialize circuit-specific components
        self._init_circuit_analyzers()
        
        # Knowledge base paths
        self.knowledge_base_path = Path(config.get("knowledge_base_path", "./knowledge_base"))
        self.processed_docs_path = Path(config.get("processed_docs_path", "./processed_docs"))
        
        # Create directories if they don't exist
        self.knowledge_base_path.mkdir(exist_ok=True)
        self.processed_docs_path.mkdir(exist_ok=True)

    def _init_embedding_models(self):
        """Initialize embedding models for dense retrieval"""
        # Dense embedding model - optimized for technical/scientific content
        self.dense_model_name = self.config.get("dense_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.dense_model = SentenceTransformer(self.dense_model_name)
        
        # Specialized physics/engineering embedding model (if available)
        self.domain_model_name = self.config.get("domain_model", "sentence-transformers/allenai-specter")
        try:
            self.domain_model = SentenceTransformer(self.domain_model_name)
        except:
            logger.warning(f"Domain model {self.domain_model_name} not available, using general model")
            self.domain_model = self.dense_model

    def _init_reranking_models(self):
        """Initialize cross-encoder models for reranking"""
        self.reranker_name = self.config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.reranker = pipeline(
            "text-classification",
            model=self.reranker_name,
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Circuit-specific reranker (if available)
        self.circuit_reranker_name = self.config.get("circuit_reranker", None)
        if self.circuit_reranker_name:
            try:
                self.circuit_reranker = pipeline(
                    "text-classification",
                    model=self.circuit_reranker_name,
                    device=0 if torch.cuda.is_available() else -1
                )
            except:
                logger.warning("Circuit-specific reranker not available")
                self.circuit_reranker = self.reranker

    def _init_search_engines(self):
        """Initialize sparse retrieval engines"""
        # BM25 for local sparse search
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_metadata = []
        
        # Elasticsearch for advanced sparse search (optional)
        es_config = self.config.get("elasticsearch", {})
        if es_config.get("enabled", False):
            try:
                self.es_client = Elasticsearch([es_config.get("host", "localhost:9200")])
                self.es_index = es_config.get("index", "circuit_knowledge")
            except:
                logger.warning("Elasticsearch not available, using BM25 only")
                self.es_client = None

    def _init_vector_stores(self):
        """Initialize vector databases"""
        # ChromaDB for vector storage
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.processed_docs_path / "chroma_db")
        )
        
        # Create collections for different types of content
        self.general_collection = self._get_or_create_collection("general_circuits")
        self.analog_collection = self._get_or_create_collection("analog_circuits")
        self.digital_collection = self._get_or_create_collection("digital_circuits")
        self.spice_collection = self._get_or_create_collection("spice_netlists")

    def _init_circuit_analyzers(self):
        """Initialize circuit-specific analysis tools"""
        # SPICE parser for netlist analysis
        self.spice_parser = spice_parser.SpiceParser()
        
        # Circuit component classifier
        self.component_classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased",  # Replace with circuit-specific model if available
            device=0 if torch.cuda.is_available() else -1
        )

    def _get_or_create_collection(self, name: str):
        """Get or create a ChromaDB collection"""
        try:
            return self.chroma_client.get_collection(name)
        except:
            return self.chroma_client.create_collection(
                name=name,
                embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=self.dense_model_name
                )
            )

    async def ingest_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Ingest and process documents into the knowledge base
        """
        logger.info(f"Ingesting {len(documents)} documents")
        
        for doc in documents:
            try:
                # Extract text content
                content = await self._extract_text_content(doc)
                
                # Chunk the document
                chunks = await self._chunk_document(content, doc)
                
                # Process and store chunks
                await self._process_and_store_chunks(chunks, doc)
                
                logger.info(f"Successfully ingested document: {doc.get('title', 'Unknown')}")
                
            except Exception as e:
                logger.error(f"Error ingesting document {doc.get('title', 'Unknown')}: {e}")

    async def _extract_text_content(self, document: Dict[str, Any]) -> str:
        """Extract text content from various document formats"""
        file_path = document.get('file_path')
        doc_type = document.get('type', '').lower()
        
        if doc_type == 'pdf':
            return self._extract_pdf_text(file_path)
        elif doc_type == 'docx':
            return self._extract_docx_text(file_path)
        elif doc_type == 'html':
            return self._extract_html_text(file_path)
        elif doc_type == 'markdown':
            return self._extract_markdown_text(file_path)
        elif doc_type == 'txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported document type: {doc_type}")

    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF files"""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text

    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX files"""
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])

    def _extract_html_text(self, file_path: str) -> str:
        """Extract text from HTML files"""
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file.read(), 'html.parser')
            return soup.get_text()

    def _extract_markdown_text(self, file_path: str) -> str:
        """Extract text from Markdown files"""
        with open(file_path, 'r', encoding='utf-8') as file:
            html = markdown.markdown(file.read())
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()

    async def _chunk_document(self, content: str, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk document content using various strategies
        """
        chunk_size = self.config.get("chunk_size", 512)
        chunk_overlap = self.config.get("chunk_overlap", 50)
        
        # Circuit-specific chunking: preserve circuit descriptions and equations
        chunks = []
        
        # Split by sections first (if headers are detected)
        sections = self._split_by_sections(content)
        
        for i, section in enumerate(sections):
            # Further split large sections
            if len(section) > chunk_size * 2:
                sub_chunks = self._sliding_window_chunking(section, chunk_size, chunk_overlap)
                for j, sub_chunk in enumerate(sub_chunks):
                    chunks.append({
                        'content': sub_chunk,
                        'metadata': {
                            **document,
                            'chunk_id': f"{document.get('id', 'unknown')}_{i}_{j}",
                            'section_id': i,
                            'sub_chunk_id': j,
                            'chunk_type': 'sub_section'
                        }
                    })
            else:
                chunks.append({
                    'content': section,
                    'metadata': {
                        **document,
                        'chunk_id': f"{document.get('id', 'unknown')}_{i}",
                        'section_id': i,
                        'chunk_type': 'section'
                    }
                })
        
        return chunks

    def _split_by_sections(self, content: str) -> List[str]:
        """Split content by sections/headers"""
        # Simple implementation - can be enhanced with NLP-based section detection
        import re
        
        # Look for common section patterns
        section_patterns = [
            r'\n#+\s+(.+)\n',  # Markdown headers
            r'\n\d+\.\s+(.+)\n',  # Numbered sections
            r'\n[A-Z][A-Za-z\s]+:\n',  # Colon-ended headers
            r'\n\n(.{1,100})\n={3,}\n',  # Underlined headers
        ]
        
        sections = []
        current_section = ""
        
        lines = content.split('\n')
        
        for line in lines:
            # Check if this line is a header
            is_header = False
            for pattern in section_patterns:
                if re.match(pattern, '\n' + line + '\n'):
                    is_header = True
                    break
            
            if is_header and current_section.strip():
                sections.append(current_section.strip())
                current_section = line + '\n'
            else:
                current_section += line + '\n'
        
        if current_section.strip():
            sections.append(current_section.strip())
        
        return sections if sections else [content]

    def _sliding_window_chunking(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Sliding window chunking with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunks.append(' '.join(chunk_words))
            
            if i + chunk_size >= len(words):
                break
        
        return chunks

    async def _process_and_store_chunks(self, chunks: List[Dict[str, Any]], document: Dict[str, Any]) -> None:
        """Process and store chunks in vector stores and search indices"""
        
        for chunk in chunks:
            content = chunk['content']
            metadata = chunk['metadata']
            
            # Determine circuit type for appropriate collection
            circuit_type = self._classify_circuit_type(content)
            collection = self._get_collection_by_type(circuit_type)
            
            # Generate embeddings
            embedding = self.dense_model.encode(content)
            
            # Store in ChromaDB
            collection.add(
                embeddings=[embedding.tolist()],
                documents=[content],
                metadatas=[metadata],
                ids=[metadata['chunk_id']]
            )
            
            # Add to BM25 index
            self.bm25_corpus.append(content.split())
            self.bm25_metadata.append(metadata)
            
            # Store in Elasticsearch if available
            if hasattr(self, 'es_client') and self.es_client:
                await self._index_to_elasticsearch(content, metadata)

    def _classify_circuit_type(self, content: str) -> str:
        """Classify the type of circuit content"""
        content_lower = content.lower()
        
        # Simple keyword-based classification
        if any(keyword in content_lower for keyword in ['op-amp', 'amplifier', 'analog', 'voltage', 'current']):
            return 'analog'
        elif any(keyword in content_lower for keyword in ['digital', 'logic', 'gate', 'flip-flop', 'counter']):
            return 'digital'
        elif any(keyword in content_lower for keyword in ['spice', 'netlist', '.model', '.subckt']):
            return 'spice'
        else:
            return 'general'

    def _get_collection_by_type(self, circuit_type: str):
        """Get appropriate collection based on circuit type"""
        if circuit_type == 'analog':
            return self.analog_collection
        elif circuit_type == 'digital':
            return self.digital_collection
        elif circuit_type == 'spice':
            return self.spice_collection
        else:
            return self.general_collection

    async def _index_to_elasticsearch(self, content: str, metadata: Dict[str, Any]) -> None:
        """Index content to Elasticsearch"""
        if not self.es_client:
            return
        
        doc = {
            'content': content,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            self.es_client.index(
                index=self.es_index,
                id=metadata['chunk_id'],
                body=doc
            )
        except Exception as e:
            logger.warning(f"Failed to index to Elasticsearch: {e}")

    def _rebuild_bm25_index(self):
        """Rebuild BM25 index after adding documents"""
        if self.bm25_corpus:
            self.bm25_index = BM25Okapi(self.bm25_corpus)

    async def hybrid_retrieve(
        self, 
        query: str, 
        circuit_context: Optional[CircuitContext] = None,
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval combining dense and sparse search
        """
        
        # Enhance query with circuit context if available
        enhanced_query = self._enhance_query_with_context(query, circuit_context)
        
        # Perform dense retrieval
        dense_results = await self._dense_retrieve(enhanced_query, top_k * 2)
        
        # Perform sparse retrieval
        sparse_results = await self._sparse_retrieve(enhanced_query, top_k * 2)
        
        # Combine and rerank results
        combined_results = self._combine_results(
            dense_results, sparse_results, dense_weight, sparse_weight
        )
        
        # Rerank using cross-encoder
        reranked_results = await self._rerank_results(enhanced_query, combined_results, top_k)
        
        return reranked_results

    def _enhance_query_with_context(self, query: str, circuit_context: Optional[CircuitContext]) -> str:
        """Enhance query with circuit-specific context"""
        if not circuit_context:
            return query
        
        enhanced_parts = [query]
        
        # Add circuit type information
        if circuit_context.circuit_type:
            enhanced_parts.append(f"Circuit type: {circuit_context.circuit_type}")
        
        # Add component information
        if circuit_context.components:
            component_names = [comp.get('name', comp.get('type', '')) for comp in circuit_context.components]
            enhanced_parts.append(f"Components: {', '.join(component_names)}")
        
        # Add complexity level
        if circuit_context.complexity_level:
            enhanced_parts.append(f"Complexity: {circuit_context.complexity_level}")
        
        return " | ".join(enhanced_parts)

    async def _dense_retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Dense retrieval using vector similarity"""
        results = []
        
        # Query embedding
        query_embedding = self.dense_model.encode(query)
        
        # Search in all collections
        collections = [
            ('general', self.general_collection),
            ('analog', self.analog_collection),
            ('digital', self.digital_collection),
            ('spice', self.spice_collection)
        ]
        
        for collection_name, collection in collections:
            try:
                # Query the collection
                collection_results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=min(top_k // 2, 50)  # Distribute across collections
                )
                
                # Convert to RetrievalResult objects
                if collection_results['documents']:
                    for i in range(len(collection_results['documents'][0])):
                        results.append(RetrievalResult(
                            content=collection_results['documents'][0][i],
                            metadata=collection_results['metadatas'][0][i],
                            score=1 - collection_results['distances'][0][i],  # Convert distance to similarity
                            source=f"dense_{collection_name}",
                            chunk_id=collection_results['ids'][0][i],
                            document_id=collection_results['metadatas'][0][i].get('id', 'unknown')
                        ))
                        
            except Exception as e:
                logger.warning(f"Error querying {collection_name} collection: {e}")
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def _sparse_retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Sparse retrieval using BM25 and/or Elasticsearch"""
        results = []
        
        # BM25 search
        if self.bm25_index:
            query_tokens = query.split()
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top scoring documents
            top_indices = np.argsort(bm25_scores)[::-1][:top_k]
            
            for idx in top_indices:
                if idx < len(self.bm25_metadata) and bm25_scores[idx] > 0:
                    results.append(RetrievalResult(
                        content=" ".join(self.bm25_corpus[idx]),
                        metadata=self.bm25_metadata[idx],
                        score=bm25_scores[idx],
                        source="sparse_bm25",
                        chunk_id=self.bm25_metadata[idx]['chunk_id'],
                        document_id=self.bm25_metadata[idx].get('id', 'unknown')
                    ))
        
        # Elasticsearch search (if available)
        if hasattr(self, 'es_client') and self.es_client:
            es_results = await self._elasticsearch_search(query, top_k)
            results.extend(es_results)
        
        # Sort by score and return top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def _elasticsearch_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Search using Elasticsearch"""
        if not self.es_client:
            return []
        
        try:
            # Construct Elasticsearch query
            es_query = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content^2", "metadata.title", "metadata.description"],
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                },
                "size": top_k,
                "sort": ["_score"]
            }
            
            response = self.es_client.search(
                index=self.es_index,
                body=es_query
            )
            
            results = []
            for hit in response['hits']['hits']:
                results.append(RetrievalResult(
                    content=hit['_source']['content'],
                    metadata=hit['_source']['metadata'],
                    score=hit['_score'],
                    source="sparse_elasticsearch",
                    chunk_id=hit['_id'],
                    document_id=hit['_source']['metadata'].get('id', 'unknown')
                ))
            
            return results
            
        except Exception as e:
            logger.warning(f"Elasticsearch search failed: {e}")
            return []

    def _combine_results(
        self, 
        dense_results: List[RetrievalResult], 
        sparse_results: List[RetrievalResult],
        dense_weight: float,
        sparse_weight: float
    ) -> List[RetrievalResult]:
        """Combine dense and sparse retrieval results"""
        
        # Normalize scores
        if dense_results:
            max_dense_score = max(r.score for r in dense_results)
            for result in dense_results:
                result.score = (result.score / max_dense_score) * dense_weight
        
        if sparse_results:
            max_sparse_score = max(r.score for r in sparse_results)
            for result in sparse_results:
                result.score = (result.score / max_sparse_score) * sparse_weight
        
        # Combine results, handling duplicates
        combined = {}
        
        for result in dense_results + sparse_results:
            chunk_id = result.chunk_id
            if chunk_id in combined:
                # Combine scores for duplicate chunks
                combined[chunk_id].score += result.score
                combined[chunk_id].source += f"+{result.source}"
            else:
                combined[chunk_id] = result
        
        return list(combined.values())

    async def _rerank_results(
        self, 
        query: str, 
        results: List[RetrievalResult], 
        top_k: int
    ) -> List[RetrievalResult]:
        """Rerank results using cross-encoder"""
        
        if not results:
            return []
        
        # Prepare query-document pairs for reranking
        pairs = [(query, result.content) for result in results]
        
        try:
            # Use the reranker to score pairs
            scores = []
            for query_text, doc_text in pairs:
                # Truncate if too long
                input_text = f"{query_text} [SEP] {doc_text}"[:512]
                score = self.reranker(input_text)
                
                # Extract score (assuming binary classification with positive class)
                if isinstance(score, list):
                    score = score[0]['score'] if score[0]['label'] == 'POSITIVE' else 1 - score[0]['score']
                else:
                    score = score['score'] if score['label'] == 'POSITIVE' else 1 - score['score']
                
                scores.append(score)
            
            # Update result scores with reranker scores
            for i, result in enumerate(results):
                result.score = scores[i]
            
            # Sort by new scores
            results.sort(key=lambda x: x.score, reverse=True)
            
        except Exception as e:
            logger.warning(f"Reranking failed, using original scores: {e}")
            # Fall back to original scores
            results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]

    def format_context_for_llm(
        self, 
        retrieval_results: List[RetrievalResult],
        circuit_context: Optional[CircuitContext] = None
    ) -> str:
        """Format retrieval results and circuit context for LLM consumption"""
        
        context_parts = []
        
        # Add circuit context if available
        if circuit_context:
            context_parts.append("CIRCUIT ANALYSIS CONTEXT:")
            context_parts.append(f"Circuit Type: {circuit_context.circuit_type}")
            context_parts.append(f"Complexity: {circuit_context.complexity_level}")
            
            if circuit_context.netlist:
                context_parts.append(f"SPICE Netlist:\n{circuit_context.netlist}")
            
            if circuit_context.simulation_results:
                context_parts.append(f"Simulation Results: {json.dumps(circuit_context.simulation_results, indent=2)}")
            
            if circuit_context.components:
                context_parts.append(f"Components: {[comp.get('name', comp.get('type')) for comp in circuit_context.components]}")
            
            context_parts.append("="*50)
        
        # Add retrieved context
        context_parts.append("RETRIEVED KNOWLEDGE:")
        
        for i, result in enumerate(retrieval_results):
            context_parts.append(f"\n[Source {i+1}] ({result.source}, Score: {result.score:.3f})")
            context_parts.append(f"Document: {result.metadata.get('title', 'Unknown')}")
            context_parts.append(f"Content: {result.content}")
            context_parts.append("-" * 30)
        
        return "\n".join(context_parts)

    async def query_knowledge_base(
        self, 
        question: str,
        circuit_context: Optional[CircuitContext] = None,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Main interface for querying the knowledge base
        """
        
        # Perform hybrid retrieval
        results = await self.hybrid_retrieve(
            query=question,
            circuit_context=circuit_context,
            top_k=top_k
        )
        
        # Format context for LLM
        formatted_context = self.format_context_for_llm(results, circuit_context)
        
        return {
            'formatted_context': formatted_context,
            'retrieval_results': results,
            'circuit_context': circuit_context,
            'query': question,
            'timestamp': datetime.now().isoformat()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base"""
        stats = {
            'total_documents_bm25': len(self.bm25_corpus),
            'collections': {}
        }
        
        # Get ChromaDB collection stats
        collections = [
            ('general', self.general_collection),
            ('analog', self.analog_collection),
            ('digital', self.digital_collection),
            ('spice', self.spice_collection)
        ]
        
        for name, collection in collections:
            try:
                count = collection.count()
                stats['collections'][name] = count
            except:
                stats['collections'][name] = 0
        
        return stats

# Example usage and configuration
def create_circuit_rag_system():
    """Factory function to create a configured RAG system"""
    
    config = {
        "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
        "domain_model": "sentence-transformers/allenai-specter",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "knowledge_base_path": "./circuit_knowledge_base",
        "processed_docs_path": "./processed_circuit_docs",
        "elasticsearch": {
            "enabled": False,  # Set to True if you have Elasticsearch running
            "host": "localhost:9200",
            "index": "circuit_knowledge"
        }
    }
    
    return HybridRAGSystem(config)
