import os
import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

# Computer Vision and ML
import torch
import torchvision
from ultralytics import YOLO
from transformers import pipeline, AutoProcessor, AutoModel

# Import our custom modules
from circuit_analyzer_rag import HybridRAGSystem, CircuitContext, create_circuit_rag_system
from decision_making_head import DecisionMakingHead, create_decision_making_head

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetHead:
    """
    Net-Head component for converting circuit images to SPICE netlists
    Implements YOLO + ResNet + DETR pipeline based on Netlistify architecture
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize the computer vision models"""
        
        # YOLO for component detection
        yolo_model_path = self.config.get("yolo_model_path", "yolov8n.pt")
        self.component_detector = YOLO(yolo_model_path)
        
        # ResNet for circuit orientation (simplified implementation)
        self.orientation_classifier = pipeline(
            "image-classification",
            model="microsoft/resnet-50",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # DETR for connectivity extraction (using a general object detection model as placeholder)
        self.connectivity_extractor = pipeline(
            "object-detection",
            model="facebook/detr-resnet-50",
            device=0 if torch.cuda.is_available() else -1
        )
        
        # Circuit component classifier
        self.component_classifier = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224",
            device=0 if torch.cuda.is_available() else -1
        )
    
    def process_circuit_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process circuit schematic image and generate SPICE netlist
        
        Args:
            image_path: Path to the circuit schematic image
            
        Returns:
            Dictionary containing netlist and component information
        """
        
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            image_cv = cv2.imread(image_path)
            
            # Step 1: Component Detection using YOLO
            components = self._detect_components(image, image_cv)
            
            # Step 2: Circuit Orientation Determination
            orientation = self._determine_orientation(image)
            
            # Step 3: Connectivity Extraction using DETR
            connections = self._extract_connectivity(image, components)
            
            # Step 4: Generate SPICE Netlist
            netlist = self._generate_netlist(components, connections, orientation)
            
            return {
                "success": True,
                "netlist": netlist,
                "components": components,
                "connections": connections,
                "orientation": orientation,
                "metadata": {
                    "image_path": image_path,
                    "component_count": len(components),
                    "connection_count": len(connections)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing circuit image: {e}")
            return {
                "success": False,
                "error": str(e),
                "netlist": None,
                "components": [],
                "connections": [],
                "orientation": None
            }
    
    def _detect_components(self, image: Image.Image, image_cv: np.ndarray) -> List[Dict[str, Any]]:
        """Detect circuit components using YOLO"""
        
        # Run YOLO detection
        results = self.component_detector(image)
        
        components = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Classify component type (simplified)
                    component_region = image.crop((int(x1), int(y1), int(x2), int(y2)))
                    component_type = self._classify_component(component_region)
                    
                    components.append({
                        "id": f"C{i+1}",
                        "type": component_type,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "confidence": float(confidence),
                        "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                        "value": self._estimate_component_value(component_type)
                    })
        
        return components
    
    def _classify_component(self, component_image: Image.Image) -> str:
        """Classify the type of circuit component"""
        
        try:
            # Use image classifier to determine component type
            result = self.component_classifier(component_image)
            
            # Map classifier output to circuit component types (simplified)
            component_mapping = {
                "resistor": ["resistor", "resistance", "ohm"],
                "capacitor": ["capacitor", "cap", "farad"],
                "inductor": ["inductor", "coil", "henry"],
                "voltage_source": ["battery", "voltage", "source"],
                "current_source": ["current", "source"],
                "diode": ["diode", "led"],
                "transistor": ["transistor", "bjt", "mosfet"],
                "op_amp": ["amplifier", "op-amp", "operational"]
            }
            
            # Simple keyword matching (would need proper circuit-specific model)
            predicted_label = result[0]["label"].lower()
            
            for component_type, keywords in component_mapping.items():
                if any(keyword in predicted_label for keyword in keywords):
                    return component_type
            
            return "unknown"
            
        except Exception as e:
            logger.warning(f"Component classification failed: {e}")
            return "unknown"
    
    def _estimate_component_value(self, component_type: str) -> str:
        """Estimate component value (simplified implementation)"""
        
        # Default values for common components
        default_values = {
            "resistor": "1K",
            "capacitor": "1uF", 
            "inductor": "1mH",
            "voltage_source": "5V",
            "current_source": "1mA",
            "diode": "1N4148",
            "transistor": "2N2222",
            "op_amp": "LM741"
        }
        
        return default_values.get(component_type, "1")
    
    def _determine_orientation(self, image: Image.Image) -> Dict[str, Any]:
        """Determine circuit orientation and layout"""
        
        # Simplified orientation analysis
        width, height = image.size
        aspect_ratio = width / height
        
        return {
            "aspect_ratio": aspect_ratio,
            "orientation": "landscape" if aspect_ratio > 1 else "portrait",
            "rotation": 0,  # Would implement rotation detection
            "layout_type": "standard"
        }
    
    def _extract_connectivity(self, image: Image.Image, components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract connectivity between components using DETR"""
        
        # Simplified connectivity extraction
        # In a full implementation, this would use advanced computer vision
        # to detect wires and connection points
        
        connections = []
        
        # Simple heuristic: connect nearby components
        for i, comp1 in enumerate(components):
            for j, comp2 in enumerate(components[i+1:], i+1):
                # Calculate distance between component centers
                dist = np.sqrt(
                    (comp1["center"][0] - comp2["center"][0])**2 + 
                    (comp1["center"][1] - comp2["center"][1])**2
                )
                
                # If components are close, assume they're connected
                if dist < 100:  # Threshold would be calibrated
                    connections.append({
                        "from": comp1["id"],
                        "to": comp2["id"],
                        "type": "wire",
                        "nodes": [f"n{len(connections)+1}"]
                    })
        
        return connections
    
    def _generate_netlist(
        self, 
        components: List[Dict[str, Any]], 
        connections: List[Dict[str, Any]], 
        orientation: Dict[str, Any]
    ) -> str:
        """Generate SPICE netlist from detected components and connections"""
        
        netlist_lines = [
            "* Generated SPICE Netlist",
            f"* Components: {len(components)}, Connections: {len(connections)}",
            ""
        ]
        
        # Node assignment (simplified)
        node_counter = 1
        component_nodes = {}
        
        # Process components
        for component in components:
            comp_id = component["id"]
            comp_type = component["type"]
            comp_value = component["value"]
            
            # Assign nodes to component (simplified 2-terminal components)
            if comp_type in ["resistor", "capacitor", "inductor", "diode"]:
                node1 = node_counter
                node2 = node_counter + 1
                node_counter += 2
                
                component_nodes[comp_id] = [node1, node2]
                
                # Generate SPICE line
                if comp_type == "resistor":
                    netlist_lines.append(f"R{comp_id[1:]} {node1} {node2} {comp_value}")
                elif comp_type == "capacitor":
                    netlist_lines.append(f"C{comp_id[1:]} {node1} {node2} {comp_value}")
                elif comp_type == "inductor":
                    netlist_lines.append(f"L{comp_id[1:]} {node1} {node2} {comp_value}")
                elif comp_type == "diode":
                    netlist_lines.append(f"D{comp_id[1:]} {node1} {node2} {comp_value}")
            
            elif comp_type == "voltage_source":
                node1 = node_counter
                node2 = 0  # Ground
                node_counter += 1
                
                component_nodes[comp_id] = [node1, node2]
                netlist_lines.append(f"V{comp_id[1:]} {node1} {node2} DC {comp_value}")
            
            elif comp_type == "op_amp":
                # Op-amp has multiple nodes
                nodes = list(range(node_counter, node_counter + 5))
                node_counter += 5
                
                component_nodes[comp_id] = nodes
                netlist_lines.append(f"X{comp_id[1:]} {' '.join(map(str, nodes))} {comp_value}")
        
        # Add end statement
        netlist_lines.extend(["", ".END"])
        
        return "\n".join(netlist_lines)

class RefinementHead:
    """
    Refinement-Head component for improving generated netlists using MLLM
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize MLLM for refinement
        self.mllm = pipeline(
            "text-generation",
            model=config.get("mllm_model", "microsoft/DialoGPT-large"),
            device=0 if torch.cuda.is_available() else -1
        )
    
    async def refine_netlist(
        self, 
        initial_netlist: str, 
        circuit_description: str, 
        user_prompt: str
    ) -> str:
        """
        Refine the initial netlist using MLLM
        
        Args:
            initial_netlist: Initial SPICE netlist from Net-Head
            circuit_description: Description of the circuit from image analysis
            user_prompt: User's question/prompt about the circuit
            
        Returns:
            Refined SPICE netlist
        """
        
        # Construct refinement prompt
        refinement_prompt = self._construct_refinement_prompt(
            initial_netlist, circuit_description, user_prompt
        )
        
        try:
            # Generate refined netlist using MLLM
            response = self.mllm(
                refinement_prompt,
                max_length=len(refinement_prompt.split()) + 200,
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True
            )
            
            # Extract refined netlist from response
            generated_text = response[0]["generated_text"]
            refined_netlist = self._extract_netlist_from_response(generated_text)
            
            return refined_netlist if refined_netlist else initial_netlist
            
        except Exception as e:
            logger.error(f"Netlist refinement failed: {e}")
            return initial_netlist
    
    def _construct_refinement_prompt(
        self, 
        netlist: str, 
        description: str, 
        user_prompt: str
    ) -> str:
        """Construct prompt for netlist refinement"""
        
        prompt = f"""
You are an expert SPICE circuit analyst. Please review and refine the following netlist.

User Question: {user_prompt}
Circuit Description: {description}

Initial Netlist:
{netlist}

Please refine this netlist by:
1. Correcting any syntax errors
2. Adding missing components that should be present
3. Fixing node connections based on typical circuit patterns  
4. Adding appropriate component values
5. Ensuring the circuit can answer the user's question

Refined Netlist:
"""
        return prompt
    
    def _extract_netlist_from_response(self, response: str) -> Optional[str]:
        """Extract SPICE netlist from MLLM response"""
        
        # Look for netlist patterns in the response
        lines = response.split('\n')
        netlist_lines = []
        in_netlist = False
        
        for line in lines:
            line = line.strip()
            
            # Start of netlist indicators
            if any(indicator in line.lower() for indicator in ['refined netlist:', 'netlist:', '* ']):
                in_netlist = True
                if line.startswith('*'):
                    netlist_lines.append(line)
                continue
            
            # End of netlist indicators  
            if in_netlist and (line == '.END' or line == '.end'):
                netlist_lines.append(line)
                break
            
            # Netlist content
            if in_netlist and line:
                # Check if line looks like SPICE syntax
                if (line.startswith(('R', 'C', 'L', 'V', 'I', 'D', 'Q', 'X', '*', '.')) or
                    line.upper() in ['.END', '.AC', '.DC', '.TRAN']):
                    netlist_lines.append(line)
        
        return '\n'.join(netlist_lines) if netlist_lines else None

class CircuitAnalyzerSystem:
    """
    Complete LLM-based Circuit Analyzer System
    Integrates Net-Head, Refinement-Head, RAG System, and Decision-Making Head
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize all components
        self.net_head = NetHead(config.get("net_head", {}))
        self.refinement_head = RefinementHead(config.get("refinement_head", {}))
        self.rag_system = create_circuit_rag_system()
        self.decision_head = create_decision_making_head(config.get("decision_head", {}))
        
        # Initialize question-diagram separator (YOLO-based)
        self.question_separator = YOLO("yolov8n.pt")  # Would use specialized model
    
    async def analyze_circuit(
        self, 
        image_path: str, 
        user_question: str = None
    ) -> Dict[str, Any]:
        """
        Complete circuit analysis pipeline
        
        Args:
            image_path: Path to circuit image containing diagram and question
            user_question: Optional explicit user question
            
        Returns:
            Comprehensive analysis results
        """
        
        logger.info(f"Starting circuit analysis for: {image_path}")
        
        try:
            # Step 1: Separate question from diagram using YOLO
            separation_result = await self._separate_question_and_diagram(image_path)
            
            circuit_image_path = separation_result["diagram_path"]
            extracted_question = separation_result.get("question_text", "")
            
            # Use provided question or extracted question
            final_question = user_question or extracted_question
            if not final_question:
                final_question = "Analyze this circuit and explain its behavior."
            
            # Step 2: Process circuit diagram with Net-Head
            logger.info("Processing circuit diagram with Net-Head...")
            net_head_result = self.net_head.process_circuit_image(circuit_image_path)
            
            if not net_head_result["success"]:
                return {
                    "success": False,
                    "error": f"Net-Head processing failed: {net_head_result['error']}",
                    "answer": None
                }
            
            initial_netlist = net_head_result["netlist"]
            circuit_description = self._generate_circuit_description(net_head_result)
            
            # Step 3: Refine netlist with Refinement-Head
            logger.info("Refining netlist with Refinement-Head...")
            refined_netlist = await self.refinement_head.refine_netlist(
                initial_netlist=initial_netlist,
                circuit_description=circuit_description,
                user_prompt=final_question
            )
            
            # Step 4: Process with Decision-Making Head using ReAct framework
            logger.info("Processing with Decision-Making Head...")
            decision_result = await self.decision_head.process_query(
                question=final_question,
                image_description=circuit_description,
                netlist=initial_netlist,
                refined_netlist=refined_netlist
            )
            
            # Compile comprehensive results
            result = {
                "success": True,
                "answer": decision_result["answer"],
                "question": final_question,
                "circuit_description": circuit_description,
                "analysis_components": {
                    "initial_netlist": initial_netlist,
                    "refined_netlist": refined_netlist,
                    "detected_components": net_head_result.get("components", []),
                    "simulation_results": decision_result["context"].get("simulation_results"),
                    "knowledge_context": decision_result["context"].get("retrieved_context")
                },
                "reasoning_steps": decision_result["reasoning_steps"],
                "metadata": {
                    "total_reasoning_steps": decision_result["metadata"]["total_steps"],
                    "simulation_successful": decision_result["metadata"]["simulation_successful"],
                    "knowledge_retrieved": decision_result["metadata"]["knowledge_retrieved"],
                    "component_count": len(net_head_result.get("components", [])),
                    "image_path": image_path
                }
            }
            
            logger.info("Circuit analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Circuit analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "Sorry, I encountered an error while analyzing the circuit. Please try again."
            }
    
    async def _separate_question_and_diagram(self, image_path: str) -> Dict[str, Any]:
        """Separate question text from circuit diagram using YOLO"""
        
        # Simplified implementation - would use specialized YOLO model
        # trained to detect text regions vs circuit diagram regions
        
        image = Image.open(image_path)
        
        # For now, assume the entire image is the circuit diagram
        # and no separate question text region
        
        return {
            "diagram_path": image_path,
            "question_text": "",  # Would extract using OCR if text region detected
            "question_bbox": None,
            "diagram_bbox": None
        }
    
    def _generate_circuit_description(self, net_head_result: Dict[str, Any]) -> str:
        """Generate natural language description of the circuit"""
        
        components = net_head_result.get("components", [])
        
        if not components:
            return "Unknown circuit configuration"
        
        # Count component types
        component_counts = {}
        for comp in components:
            comp_type = comp["type"]
            component_counts[comp_type] = component_counts.get(comp_type, 0) + 1
        
        # Generate description
        description_parts = ["Circuit contains:"]
        
        for comp_type, count in component_counts.items():
            if count == 1:
                description_parts.append(f"1 {comp_type}")
            else:
                description_parts.append(f"{count} {comp_type}s")
        
        # Add circuit type inference
        if "op_amp" in component_counts:
            description_parts.append("(Op-amp based analog circuit)")
        elif any(comp in component_counts for comp in ["transistor", "diode"]):
            description_parts.append("(Transistor-based circuit)")
        elif "voltage_source" in component_counts and "resistor" in component_counts:
            description_parts.append("(Basic resistive circuit)")
        
        return " ".join(description_parts)
    
    async def batch_analyze_circuits(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple circuits in batch"""
        
        results = []
        
        for image_path in image_paths:
            try:
                result = await self.analyze_circuit(image_path)
                results.append({
                    "image_path": image_path,
                    "result": result
                })
            except Exception as e:
                results.append({
                    "image_path": image_path,
                    "result": {
                        "success": False,
                        "error": str(e),
                        "answer": None
                    }
                })
        
        return results
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        
        return {
            "rag_system_stats": self.rag_system.get_statistics(),
            "net_head_config": self.config.get("net_head", {}),
            "refinement_head_config": self.config.get("refinement_head", {}),
            "decision_head_config": self.config.get("decision_head", {}),
            "system_status": "operational"
        }

def create_circuit_analyzer_system(config_path: Optional[str] = None) -> CircuitAnalyzerSystem:
    """
    Factory function to create a complete circuit analyzer system
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Configured CircuitAnalyzerSystem instance
    """
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            "net_head": {
                "yolo_model_path": "yolov8n.pt",
                "component_threshold": 0.5
            },
            "refinement_head": {
                "mllm_model": "microsoft/DialoGPT-large"
            },
            "decision_head": {
                "llm": {
                    "type": "huggingface",
                    "model_name": "microsoft/DialoGPT-large"
                }
            },
            "rag": {
                "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
                "chunk_size": 512,
                "top_k": 5
            }
        }
    
    return CircuitAnalyzerSystem(config)

# Example usage and testing
async def main():
    """Example usage of the complete circuit analyzer system"""
    
    # Initialize the system
    analyzer = create_circuit_analyzer_system()
    
    # Example circuit analysis
    image_path = "example_circuit.jpg"  # Replace with actual image path
    user_question = "What is the voltage gain of this amplifier?"
    
    try:
        result = await analyzer.analyze_circuit(image_path, user_question)
        
        if result["success"]:
            print("=== Circuit Analysis Results ===")
            print(f"Question: {result['question']}")
            print(f"Answer: {result['answer']}")
            print(f"\nCircuit Description: {result['circuit_description']}")
            print(f"\nComponents Detected: {len(result['metadata']['component_count'])}")
            print(f"Reasoning Steps: {result['metadata']['total_reasoning_steps']}")
            print(f"Simulation Successful: {result['metadata']['simulation_successful']}")
            
            # Print reasoning steps
            print("\n=== Reasoning Process ===")
            for i, step in enumerate(result['reasoning_steps']):
                print(f"Step {i+1}: {step.thought}")
                print(f"  Action: {step.action.action_type.value}")
                print(f"  Success: {step.observation.success}")
                print()
        else:
            print(f"Analysis failed: {result['error']}")
    
    except FileNotFoundError:
        print("Example image not found. Please provide a valid circuit image path.")
    except Exception as e:
        print(f"Error: {e}")
    
    # Print system statistics
    print("\n=== System Statistics ===")
    stats = analyzer.get_system_statistics()
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
