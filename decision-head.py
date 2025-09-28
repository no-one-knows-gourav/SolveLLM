import os
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import re

# LLM and AI libraries
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import openai  # For GPT models if using OpenAI API
from anthropic import Anthropic  # For Claude models if using Anthropic API

# Circuit simulation and analysis
import PySpice
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Import our RAG system
from circuit_analyzer_rag import HybridRAGSystem, CircuitContext, RetrievalResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of actions the decision-making head can take"""
    RETRIEVE = "retrieve"
    SIMULATE = "simulate" 
    ANALYZE = "analyze"
    REASON = "reason"
    VALIDATE = "validate"
    GENERATE_RESPONSE = "generate_response"

class ValidationResult(Enum):
    """Simulation validation results"""
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"

@dataclass
class Action:
    """Represents an action in the ReAct framework"""
    action_type: ActionType
    parameters: Dict[str, Any]
    timestamp: datetime
    reasoning: str

@dataclass
class Observation:
    """Represents an observation from an action"""
    action_id: str
    result: Any
    success: bool
    error_message: Optional[str]
    metadata: Dict[str, Any]

@dataclass
class ReActStep:
    """Single step in ReAct reasoning"""
    step_id: int
    thought: str
    action: Action
    observation: Observation

class DecisionMakingHead:
    """
    ReAct-based decision making system for circuit analysis
    Integrates RAG retrieval, simulation, and reasoning
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rag_system = None
        self.simulator = CircuitSimulator()
        self.llm_client = None
        self.steps_history: List[ReActStep] = []
        
        # Initialize components
        self._initialize_llm()
        self._initialize_rag_system()
    
    def _initialize_llm(self):
        """Initialize the LLM for reasoning"""
        llm_config = self.config.get("llm", {})
        llm_type = llm_config.get("type", "huggingface")
        
        if llm_type == "openai":
            openai.api_key = llm_config.get("api_key")
            self.llm_client = "openai"
        elif llm_type == "anthropic":
            self.llm_client = Anthropic(api_key=llm_config.get("api_key"))
        elif llm_type == "huggingface":
            model_name = llm_config.get("model_name", "microsoft/DialoGPT-large")
            self.llm_pipeline = pipeline(
                "text-generation",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1
            )
            self.llm_client = "huggingface"
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
    
    def _initialize_rag_system(self):
        """Initialize the RAG system"""
        rag_config = self.config.get("rag", {})
        
        # Import and initialize RAG system
        from circuit_analyzer_rag import create_circuit_rag_system
        self.rag_system = create_circuit_rag_system()
    
    async def process_query(
        self,
        question: str,
        image_description: str,
        netlist: str,
        refined_netlist: Optional[str] = None,
        max_steps: int = 10
    ) -> Dict[str, Any]:
        """
        Main processing function using ReAct framework
        """
        
        # Initialize processing context
        context = {
            "original_question": question,
            "image_description": image_description,
            "initial_netlist": netlist,
            "refined_netlist": refined_netlist or netlist,
            "simulation_results": None,
            "validation_result": ValidationResult.UNCERTAIN,
            "retrieved_context": None,
            "final_answer": None
        }
        
        self.steps_history = []
        current_netlist = refined_netlist or netlist
        
        # Start ReAct reasoning loop
        for step_num in range(max_steps):
            
            # Generate thought based on current context
            thought = await self._generate_thought(context, step_num)
            
            # Decide on next action
            action = await self._decide_action(context, thought)
            
            # Execute action and get observation
            observation = await self._execute_action(action, context)
            
            # Create ReAct step
            react_step = ReActStep(
                step_id=step_num,
                thought=thought,
                action=action,
                observation=observation
            )
            self.steps_history.append(react_step)
            
            # Update context based on observation
            context = await self._update_context(context, action, observation)
            
            # Check if we should terminate
            if await self._should_terminate(context, action, observation):
                break
        
        # Generate final response
        final_answer = await self._generate_final_response(context)
        context["final_answer"] = final_answer
        
        return {
            "answer": final_answer,
            "context": context,
            "reasoning_steps": self.steps_history,
            "metadata": {
                "total_steps": len(self.steps_history),
                "simulation_successful": context.get("validation_result") == ValidationResult.VALID,
                "knowledge_retrieved": context.get("retrieved_context") is not None
            }
        }
    
    async def _generate_thought(self, context: Dict[str, Any], step_num: int) -> str:
        """Generate reasoning thought for current step"""
        
        # Construct prompt for thought generation
        prompt = self._construct_thought_prompt(context, step_num)
        
        # Generate thought using LLM
        thought = await self._call_llm(prompt, max_tokens=150)
        
        return thought
    
    def _construct_thought_prompt(self, context: Dict[str, Any], step_num: int) -> str:
        """Construct prompt for thought generation"""
        
        prompt_parts = [
            "You are an expert circuit analysis system using step-by-step reasoning.",
            f"Step {step_num + 1}: Analyze the current situation and decide what to do next.",
            "",
            f"Question: {context['original_question']}",
            f"Circuit Description: {context['image_description']}",
        ]
        
        if context.get("refined_netlist"):
            prompt_parts.append(f"Available Netlist:\n{context['refined_netlist']}")
        
        if context.get("simulation_results"):
            prompt_parts.append(f"Previous Simulation Results: {json.dumps(context['simulation_results'], indent=2)}")
        
        if context.get("retrieved_context"):
            prompt_parts.append(f"Retrieved Knowledge: Available")
        
        # Add previous steps context
        if self.steps_history:
            prompt_parts.append("\nPrevious reasoning steps:")
            for step in self.steps_history[-3:]:  # Last 3 steps for context
                prompt_parts.append(f"- {step.thought}")
                prompt_parts.append(f"  Action: {step.action.action_type.value}")
                prompt_parts.append(f"  Result: {'Success' if step.observation.success else 'Failed'}")
        
        prompt_parts.extend([
            "",
            "Think about what you need to do next. Consider:",
            "1. Do you need to retrieve relevant knowledge about this circuit type?",
            "2. Should you simulate the circuit to verify behavior?",
            "3. Do you need to analyze the simulation results?",
            "4. Are you ready to provide a final answer?",
            "",
            "Thought:"
        ])
        
        return "\n".join(prompt_parts)
    
    async def _decide_action(self, context: Dict[str, Any], thought: str) -> Action:
        """Decide on the next action based on current thought and context"""
        
        # Simple rule-based action selection (can be enhanced with LLM-based selection)
        
        # If no knowledge retrieved yet, retrieve first
        if not context.get("retrieved_context"):
            return Action(
                action_type=ActionType.RETRIEVE,
                parameters={"query": context["original_question"]},
                timestamp=datetime.now(),
                reasoning=thought
            )
        
        # If netlist available but not simulated, simulate
        if context.get("refined_netlist") and not context.get("simulation_results"):
            return Action(
                action_type=ActionType.SIMULATE,
                parameters={"netlist": context["refined_netlist"]},
                timestamp=datetime.now(),
                reasoning=thought
            )
        
        # If simulation results available but not validated, validate
        if context.get("simulation_results") and context.get("validation_result") == ValidationResult.UNCERTAIN:
            return Action(
                action_type=ActionType.VALIDATE,
                parameters={"simulation_results": context["simulation_results"]},
                timestamp=datetime.now(),
                reasoning=thought
            )
        
        # If everything is available, generate final response
        return Action(
            action_type=ActionType.GENERATE_RESPONSE,
            parameters={
                "context": context,
                "question": context["original_question"]
            },
            timestamp=datetime.now(),
            reasoning=thought
        )
    
    async def _execute_action(self, action: Action, context: Dict[str, Any]) -> Observation:
        """Execute the chosen action and return observation"""
        
        try:
            if action.action_type == ActionType.RETRIEVE:
                result = await self._execute_retrieve_action(action.parameters, context)
            elif action.action_type == ActionType.SIMULATE:
                result = await self._execute_simulate_action(action.parameters)
            elif action.action_type == ActionType.VALIDATE:
                result = await self._execute_validate_action(action.parameters)
            elif action.action_type == ActionType.GENERATE_RESPONSE:
                result = await self._execute_generate_response_action(action.parameters)
            else:
                raise ValueError(f"Unknown action type: {action.action_type}")
            
            return Observation(
                action_id=f"{action.action_type.value}_{datetime.now().isoformat()}",
                result=result,
                success=True,
                error_message=None,
                metadata={"action_type": action.action_type.value}
            )
            
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return Observation(
                action_id=f"{action.action_type.value}_{datetime.now().isoformat()}",
                result=None,
                success=False,
                error_message=str(e),
                metadata={"action_type": action.action_type.value}
            )
    
    async def _execute_retrieve_action(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute knowledge retrieval action"""
        
        query = parameters["query"]
        
        # Create circuit context if available
        circuit_context = None
        if context.get("refined_netlist"):
            circuit_context = CircuitContext(
                netlist=context["refined_netlist"],
                simulation_results=context.get("simulation_results", {}),
                components=[],  # Would extract from netlist
                circuit_type="unknown",  # Would classify
                complexity_level="unknown"  # Would assess
            )
        
        # Query RAG system
        rag_result = await self.rag_system.query_knowledge_base(
            question=query,
            circuit_context=circuit_context,
            top_k=5
        )
        
        return rag_result
    
    async def _execute_simulate_action(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute circuit simulation action"""
        
        netlist = parameters["netlist"]
        
        # Simulate using our circuit simulator
        simulation_result = self.simulator.simulate_netlist(netlist)
        
        return simulation_result
    
    async def _execute_validate_action(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute simulation validation action"""
        
        simulation_results = parameters["simulation_results"]
        
        # Simple validation logic (can be enhanced)
        if simulation_results.get("success", False):
            results = simulation_results.get("results", {})
            
            # Check if results contain meaningful data
            if any(results.values()):
                return {
                    "validation_result": ValidationResult.VALID,
                    "confidence": 0.8,
                    "reasoning": "Simulation completed successfully with meaningful results"
                }
            else:
                return {
                    "validation_result": ValidationResult.UNCERTAIN,
                    "confidence": 0.5,
                    "reasoning": "Simulation completed but results are unclear"
                }
        else:
            return {
                "validation_result": ValidationResult.INVALID,
                "confidence": 0.9,
                "reasoning": f"Simulation failed: {simulation_results.get('error', 'Unknown error')}"
            }
    
    async def _execute_generate_response_action(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute final response generation action"""
        
        context = parameters["context"]
        question = parameters["question"]
        
        # Construct comprehensive prompt for final answer generation
        prompt = self._construct_final_answer_prompt(context, question)
        
        # Generate final answer
        final_answer = await self._call_llm(prompt, max_tokens=500)
        
        return {"final_answer": final_answer}
    
    def _construct_final_answer_prompt(self, context: Dict[str, Any], question: str) -> str:
        """Construct prompt for final answer generation"""
        
        prompt_parts = [
            "You are an expert circuit analysis system. Provide a comprehensive answer to the user's question.",
            "",
            f"Question: {question}",
            f"Circuit Description: {context.get('image_description', 'N/A')}",
        ]
        
        # Add simulation results if valid
        if context.get("validation_result") == ValidationResult.VALID:
            simulation_results = context.get("simulation_results", {})
            prompt_parts.extend([
                "",
                "Simulation Results:",
                json.dumps(simulation_results.get("results", {}), indent=2)
            ])
        
        # Add retrieved knowledge context
        if context.get("retrieved_context"):
            retrieved_context = context["retrieved_context"]
            prompt_parts.extend([
                "",
                "Relevant Knowledge:",
                retrieved_context.get("formatted_context", "")
            ])
        
        # Add netlist if available
        if context.get("refined_netlist"):
            prompt_parts.extend([
                "",
                "Circuit Netlist:",
                context["refined_netlist"]
            ])
        
        prompt_parts.extend([
            "",
            "Instructions:",
            "1. Analyze the circuit based on the available information",
            "2. Use simulation results to support your analysis (if available and valid)",
            "3. Reference relevant theoretical knowledge from the knowledge base",
            "4. Provide step-by-step reasoning for your conclusions",
            "5. Include numerical calculations where applicable",
            "6. Explain any assumptions made",
            "",
            "Answer:"
        ])
        
        return "\n".join(prompt_parts)
    
    async def _update_context(
        self, 
        context: Dict[str, Any], 
        action: Action, 
        observation: Observation
    ) -> Dict[str, Any]:
        """Update context based on action and observation"""
        
        if observation.success:
            if action.action_type == ActionType.RETRIEVE:
                context["retrieved_context"] = observation.result
            elif action.action_type == ActionType.SIMULATE:
                context["simulation_results"] = observation.result
            elif action.action_type == ActionType.VALIDATE:
                context["validation_result"] = observation.result.get("validation_result", ValidationResult.UNCERTAIN)
            elif action.action_type == ActionType.GENERATE_RESPONSE:
                context["final_answer"] = observation.result.get("final_answer")
        
        return context
    
    async def _should_terminate(
        self, 
        context: Dict[str, Any], 
        action: Action, 
        observation: Observation
    ) -> bool:
        """Decide whether to terminate the ReAct loop"""
        
        # Terminate if final response is generated
        if action.action_type == ActionType.GENERATE_RESPONSE and observation.success:
            return True
        
        # Terminate if we have enough information and validation is complete
        if (context.get("retrieved_context") and 
            context.get("simulation_results") and 
            context.get("validation_result") != ValidationResult.UNCERTAIN):
            return True
        
        return False
    
    async def _generate_final_response(self, context: Dict[str, Any]) -> str:
        """Generate final response based on accumulated context"""
        
        if context.get("final_answer"):
            return context["final_answer"]
        
        # Fallback response generation
        response_parts = [
            f"Based on my analysis of the circuit: {context.get('image_description', 'Unknown circuit')}"
        ]
        
        if context.get("simulation_results"):
            response_parts.append(f"The simulation results indicate: {context['simulation_results']}")
        
        if context.get("retrieved_context"):
            response_parts.append("According to the relevant circuit theory...")
        
        response_parts.append(f"Therefore, to answer your question: {context['original_question']}")
        response_parts.append("Further analysis may be needed for a complete understanding.")
        
        return " ".join(response_parts)
    
    async def _call_llm(self, prompt: str, max_tokens: int = 150) -> str:
        """Call the configured LLM with the given prompt"""
        
        try:
            if self.llm_client == "openai":
                response = await openai.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.3
                )
                return response.choices[0].message.content.strip()
            
            elif isinstance(self.llm_client, Anthropic):
                response = await self.llm_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.3
                )
                return response.content[0].text.strip()
            
            elif self.llm_client == "huggingface":
                response = self.llm_pipeline(
                    prompt,
                    max_length=len(prompt.split()) + max_tokens,
                    num_return_sequences=1,
                    temperature=0.3,
                    do_sample=True
                )
                return response[0]['generated_text'][len(prompt):].strip()
            
            else:
                raise ValueError("No valid LLM client configured")
                
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"[LLM Error: {str(e)}]"

def create_decision_making_head(config: Dict[str, Any]) -> DecisionMakingHead:
    """Factory function to create a configured decision making head"""
    
    default_config = {
        "llm": {
            "type": "huggingface",  # or "openai" or "anthropic"
            "model_name": "microsoft/DialoGPT-large",
            "api_key": None  # Required for OpenAI/Anthropic
        },
        "rag": {
            # RAG system configuration
        }
    }
    
    # Merge with provided config
    final_config = {**default_config, **config}
    
    return DecisionMakingHead(final_config)
