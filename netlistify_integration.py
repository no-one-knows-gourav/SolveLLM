"""
Complete Circuit Analysis Pipeline
Integrates Netlistify (Net-Head) with LangGraph Decision-Making Head
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import subprocess
from datetime import datetime

# Import decision-making head
from langgraph_decision_head import CircuitAnalysisEngine, setup_knowledge_base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetlistifyWrapper:
    """
    Wrapper for Netlistify repository to generate SPICE netlists from circuit images
    """
    
    def __init__(self, netlistify_path: str):
        """
        Initialize Netlistify wrapper
        
        Args:
            netlistify_path: Path to Netlistify repository root
        """
        self.netlistify_path = Path(netlistify_path)
        self._verify_installation()
    
    def _verify_installation(self):
        """Verify Netlistify is properly installed"""
        if not self.netlistify_path.exists():
            raise ValueError(f"Netlistify path does not exist: {self.netlistify_path}")
        
        # Check for required files
        required_files = ["main.py", "requirements.txt"]  # Adjust based on actual Netlistify structure
        
        for file in required_files:
            if not (self.netlistify_path / file).exists():
                logger.warning(f"Expected file not found: {file}")
        
        logger.info("Netlistify installation verified")
    
    def generate_netlist(
        self, 
        image_path: str,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate SPICE netlist from circuit image using Netlistify
        
        Args:
            image_path: Path to circuit schematic image
            output_path: Optional path to save netlist
            
        Returns:
            Dictionary containing netlist and metadata
        """
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        logger.info(f"Generating netlist for: {image_path}")
        
        try:
            # Run Netlistify
            # Adjust command based on actual Netlistify API
            cmd = [
                sys.executable,
                str(self.netlistify_path / "main.py"),
                "--image", image_path
            ]
            
            if output_path:
                cmd.extend(["--output", output_path])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.netlistify_path)
            )
            
            if result.returncode != 0:
                logger.error(f"Netlistify failed: {result.stderr}")
                return {
                    "success": False,
                    "error": result.stderr,
                    "netlist": None,
                    "components": []
                }
            
            # Parse output
            netlist = self._parse_netlistify_output(result.stdout, output_path)
            components = self._extract_components_from_netlist(netlist)
            
            return {
                "success": True,
                "netlist": netlist,
                "components": components,
                "image_path": image_path,
                "metadata": {
                    "generation_time": datetime.now().isoformat(),
                    "component_count": len(components)
                }
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Netlistify execution timeout")
            return {
                "success": False,
                "error": "Netlist generation timeout",
                "netlist": None,
                "components": []
            }
        except Exception as e:
            logger.error(f"Netlistify error: {e}")
            return {
                "success": False,
                "error": str(e),
                "netlist": None,
                "components": []
            }
    
    def _parse_netlistify_output(
        self, 
        stdout: str, 
        output_path: Optional[str]
    ) -> str:
        """Parse Netlistify output to extract netlist"""
        
        # If output was saved to file
        if output_path and os.path.exists(output_path):
            with open(output_path, 'r') as f:
                return f.read()
        
        # Otherwise parse from stdout
        # Adjust based on actual Netlistify output format
        return stdout
    
    def _extract_components_from_netlist(self, netlist: str) -> List[Dict[str, Any]]:
        """Extract component information from netlist"""
        
        components = []
        
        for line in netlist.split('\n'):
            line = line.strip()
            if not line or line.startswith('*') or line.startswith('.'):
                continue
            
            parts = line.split()
            if len(parts) < 2:
                continue
            
            component_id = parts[0]
            component_type = self._classify_component(component_id[0])
            
            components.append({
                "id": component_id,
                "type": component_type,
                "line": line
            })
        
        return components
    
    def _classify_component(self, prefix: str) -> str:
        """Classify component type from SPICE prefix"""
        prefix = prefix.upper()
        
        component_map = {
            'R': 'resistor',
            'C': 'capacitor',
            'L': 'inductor',
            'V': 'voltage_source',
            'I': 'current_source',
            'D': 'diode',
            'Q': 'transistor',
            'M': 'mosfet',
            'X': 'subcircuit',
            'E': 'vcvs',
            'G': 'vccs',
            'F': 'cccs',
            'H': 'ccvs'
        }
        
        return component_map.get(prefix, 'unknown')


class CompleteCircuitAnalyzer:
    """
    Complete circuit analysis pipeline combining:
    1. Netlistify (Net-Head) for netlist generation
    2. LangGraph Decision-Making Head for analysis
    """
    
    def __init__(
        self,
        netlistify_path: str,
        config_path: Optional[str] = None
    ):
        """
        Initialize complete analyzer
        
        Args:
            netlistify_path: Path to Netlistify repository
            config_path: Path to configuration file
        """
        
        # Initialize Net-Head (Netlistify)
        self.netlistify = NetlistifyWrapper(netlistify_path)
        
        # Initialize Decision-Making Head
        self.decision_head = CircuitAnalysisEngine(config_path)
        
        logger.info("Complete Circuit Analyzer initialized")
    
    async def analyze_from_image(
        self,
        image_path: str,
        question: str,
        image_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete analysis pipeline from circuit image to answer
        
        Args:
            image_path: Path to circuit schematic image
            question: User's question about the circuit
            image_description: Optional manual description
            
        Returns:
            Complete analysis results
        """
        
        logger.info("="*50)
        logger.info("STARTING COMPLETE CIRCUIT ANALYSIS")
        logger.info("="*50)
        
        start_time = datetime.now()
        
        # Step 1: Generate netlist using Netlistify
        logger.info("Step 1: Generating netlist with Netlistify...")
        netlist_result = self.netlistify.generate_netlist(image_path)
        
        if not netlist_result["success"]:
            return {
                "success": False,
                "error": f"Netlist generation failed: {netlist_result['error']}",
                "stage": "netlist_generation"
            }
        
        netlist = netlist_result["netlist"]
        components = netlist_result["components"]
        
        logger.info(f"✓ Netlist generated with {len(components)} components")
        
        # Step 2: Generate circuit description if not provided
        if not image_description:
            image_description = self._generate_circuit_description(components)
        
        logger.info(f"Circuit description: {image_description}")
        
        # Step 3: Run Decision-Making Head analysis
        logger.info("Step 2: Running Decision-Making Head analysis...")
        
        analysis_result = await self.decision_head.analyze(
            question=question,
            image_description=image_description,
            netlist=netlist
        )
        
        if not analysis_result["success"]:
            return {
                "success": False,
                "error": f"Analysis failed: {analysis_result.get('error')}",
                "stage": "decision_head_analysis",
                "netlist": netlist
            }
        
        logger.info("✓ Analysis completed")
        
        # Step 4: Compile comprehensive results
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        comprehensive_result = {
            "success": True,
            "question": question,
            "answer": analysis_result["answer"],
            "confidence": analysis_result["confidence"],
            
            # Net-Head results
            "net_head": {
                "netlist": netlist,
                "components": components,
                "component_count": len(components)
            },
            
            # Decision-Head results
            "decision_head": {
                "reasoning_steps": analysis_result["reasoning_steps"],
                "simulation_results": analysis_result["analysis"]["simulation_results"],
                "simulation_valid": analysis_result["analysis"]["simulation_valid"],
                "context_retrieved": analysis_result["analysis"]["retrieved_context_available"]
            },
            
            # Metadata
            "metadata": {
                "image_path": image_path,
                "circuit_description": image_description,
                "processing_time_seconds": processing_time,
                "timestamp": datetime.now().isoformat(),
                "iterations": analysis_result["metadata"]["iterations"]
            }
        }
        
        logger.info("="*50)
        logger.info(f"ANALYSIS COMPLETE ({processing_time:.2f}s)")
        logger.info("="*50)
        
        return comprehensive_result
    
    def analyze_from_image_sync(
        self,
        image_path: str,
        question: str,
        image_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Synchronous wrapper for analyze_from_image"""
        return asyncio.run(
            self.analyze_from_image(image_path, question, image_description)
        )
    
    async def batch_analyze(
        self,
        tasks: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Batch analyze multiple circuits
        
        Args:
            tasks: List of dicts with 'image_path', 'question', 'description'
            
        Returns:
            List of analysis results
        """
        
        results = []
        
        for i, task in enumerate(tasks, 1):
            logger.info(f"\nProcessing task {i}/{len(tasks)}")
            
            result = await self.analyze_from_image(
                image_path=task["image_path"],
                question=task["question"],
                image_description=task.get("description")
            )
            
            results.append({
                "task_id": i,
                "result": result
            })
        
        return results
    
    def _generate_circuit_description(self, components: List[Dict[str, Any]]) -> str:
        """Generate natural language description from components"""
        
        if not components:
            return "Unknown circuit configuration"
        
        # Count component types
        type_counts = {}
        for comp in components:
            comp_type = comp["type"]
            type_counts[comp_type] = type_counts.get(comp_type, 0) + 1
        
        # Build description
        parts = ["Circuit with"]
        
        for comp_type, count in type_counts.items():
            if count == 1:
                parts.append(f"1 {comp_type}")
            else:
                parts.append(f"{count} {comp_type}s")
        
        # Infer circuit type
        if "subcircuit" in type_counts or any("op" in c["id"].lower() for c in components):
            parts.append("(Op-amp based analog circuit)")
        elif "transistor" in type_counts or "mosfet" in type_counts:
            parts.append("(Transistor-based circuit)")
        elif "voltage_source" in type_counts and "resistor" in type_counts:
            parts.append("(Resistive circuit)")
        
        return ", ".join(parts)


# ==================== COMMAND-LINE INTERFACE ====================

def create_cli():
    """Create command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Complete Circuit Analysis System"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to circuit schematic image"
    )
    
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question about the circuit"
    )
    
    parser.add_argument(
        "--description",
        type=str,
        default=None,
        help="Optional circuit description"
    )
    
    parser.add_argument(
        "--netlistify-path",
        type=str,
        required=True,
        help="Path to Netlistify repository"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON file for results"
    )
    
    return parser


async def main_cli():
    """Main CLI function"""
    parser = create_cli()
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CompleteCircuitAnalyzer(
        netlistify_path=args.netlistify_path,
        config_path=args.config
    )
    
    # Run analysis
    result = await analyzer.analyze_from_image(
        image_path=args.image,
        question=args.question,
        image_description=args.description
    )
    
    # Print results
    print("\n" + "="*70)
    print("CIRCUIT ANALYSIS RESULTS")
    print("="*70)
    
    if result["success"]:
        print(f"\nImage: {result['metadata']['image_path']}")
        print(f"Question: {result['question']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nConfidence: {result['confidence']:.1%}")
        print(f"Processing Time: {result['metadata']['processing_time_seconds']:.2f}s")
        
        print(f"\nComponents Detected: {result['net_head']['component_count']}")
        
        if result['decision_head']['simulation_valid']:
            print("Simulation: Valid")
        else:
            print("Simulation: Failed or not attempted")
        
        print(f"\nKnowledge Retrieved: {'Yes' if result['decision_head']['context_retrieved'] else 'No'}")
        
        print(f"\nReasoning Steps: {len(result['decision_head']['reasoning_steps'])}")
        for step in result['decision_head']['reasoning_steps']:
            print(f"  • Step {step['step']}: {step['action']} - {step['result']}")
    else:
        print(f"\nError: {result['error']}")
        print(f"Stage: {result.get('stage', 'unknown')}")
    
    print("\n" + "="*70)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output}")


# ==================== EXAMPLE USAGE ====================

async def example_usage():
    """Example demonstrating the complete pipeline"""
    
    # Setup (run once)
    # setup_knowledge_base(
    #     knowledge_base_path="./circuit_knowledge_base",
    #     documents_to_add=[
    #         "./docs/circuit_fundamentals.pdf",
    #         "./docs/op_amp_guide.pdf"
    #     ]
    # )
    
    # Initialize analyzer
    analyzer = CompleteCircuitAnalyzer(
        netlistify_path="/path/to/netlistify",  # Replace with actual path
        config_path="config.json"
    )
    
    # Single analysis
    result = await analyzer.analyze_from_image(
        image_path="examples/inverting_amp.jpg",
        question="What is the voltage gain of this amplifier?"
    )
    
    print(f"Answer: {result['answer']}")
    
    # Batch analysis
    tasks = [
        {
            "image_path": "examples/circuit1.jpg",
            "question": "What is the output voltage?",
            "description": "Simple voltage divider"
        },
        {
            "image_path": "examples/circuit2.jpg",
            "question": "Calculate the cutoff frequency",
            "description": "RC low-pass filter"
        }
    ]
    
    batch_results = await analyzer.batch_analyze(tasks)
    
    for task_result in batch_results:
        print(f"\nTask {task_result['task_id']}:")
        print(f"Answer: {task_result['result']['answer']}")


if __name__ == "__main__":
    # Run CLI
    asyncio.run(main_cli())
