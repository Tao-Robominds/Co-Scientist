"""
Base agent module for the Co-Scientist system.
"""
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from pydantic import BaseModel, create_model

import openai
from openai.types.chat import ChatCompletion

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Agent")

T = TypeVar('T', bound=BaseModel)

class Agent:
    """
    Base agent class for the Co-Scientist system.
    
    Each agent represents a specialized AI assistant with a specific role
    in the research workflow.
    """
    
    def __init__(
        self,
        name: str,
        instructions: str,
        model: str = "gpt-4o",
        tools: Optional[List[Dict[str, Any]]] = None,
        output_type: Optional[Type[BaseModel]] = None,
    ):
        """
        Initialize an agent.
        
        Args:
            name: Name of the agent
            instructions: System instructions for the agent
            model: OpenAI model to use
            tools: List of tools available to the agent
            output_type: Pydantic model type to validate and structure the output
        """
        self.name = name
        self.instructions = instructions
        self.model = model
        self.tools = tools or []
        self.output_type = output_type
        self.client = openai.OpenAI()
        
    async def run(self, input_text: str) -> Union[str, BaseModel]:
        """
        Run the agent on the given input text.
        
        Args:
            input_text: The input text to process
            
        Returns:
            Either a string response or a structured Pydantic model
        """
        # Prepare the system message
        system_message = self.instructions
        
        # If we're using a structured output type, add JSON instruction
        if self.output_type:
            schema_text = ""
            try:
                schema = self.output_type.model_json_schema()
                schema_text = f"\n\nHere is the expected JSON schema:\n{json.dumps(schema, indent=2)}"
            except Exception as e:
                logger.error(f"Error getting schema: {e}")
            
            system_message = f"{system_message}\n\nYou must respond with a JSON object that conforms to the required format.{schema_text}"
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": input_text}
        ]
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        }
        
        if self.tools:
            # Format tools properly for the OpenAI API
            formatted_tools = []
            for tool in self.tools:
                # Check if it's already in the correct format
                if "type" in tool and tool["type"] == "function" and "function" in tool:
                    formatted_tools.append(tool)
                else:
                    # If it's a simplified tool like {"type": "web_search"}, convert it
                    if "type" in tool and tool["type"] == "web_search":
                        formatted_tools.append({
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "description": "Search the web for information",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "query": {
                                            "type": "string",
                                            "description": "The search query"
                                        }
                                    },
                                    "required": ["query"]
                                }
                            }
                        })
            
            kwargs["tools"] = formatted_tools
        
        if self.output_type:
            # Use 'json_object' as the type
            kwargs["response_format"] = {"type": "json_object"}
        
        # Run the API call in a separate thread since it's synchronous
        try:
            logger.info(f"Calling OpenAI API with model: {self.model}")
            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                **kwargs
            )
            response = completion.choices[0].message.content
            logger.info(f"Got API response (length: {len(response) if response else 0})")
            
            if not response:
                logger.error("Empty response from OpenAI API")
                return None
            
            if self.output_type and response:
                try:
                    # Parse the response as JSON
                    logger.info("Parsing response as JSON")
                    data = json.loads(response)
                    
                    # Create an instance of the output type
                    logger.info(f"Validating response as {self.output_type.__name__}")
                    result = self.output_type.model_validate(data)
                    logger.info("Validation successful")
                    return result
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error: {e}, Response: {response[:100]}...")
                    # If JSON parsing fails, try to extract JSON from the response
                    try:
                        # Try to find JSON within the response
                        logger.info("Attempting to extract JSON from response")
                        start_idx = response.find("{")
                        end_idx = response.rfind("}")
                        if start_idx >= 0 and end_idx > start_idx:
                            json_str = response[start_idx:end_idx+1]
                            data = json.loads(json_str)
                            result = self.output_type.model_validate(data)
                            logger.info("Successfully extracted and validated JSON")
                            return result
                    except Exception as inner_e:
                        logger.error(f"Failed to extract JSON: {inner_e}")
                    
                    return response
                except Exception as e:
                    logger.error(f"Validation error: {e}")
                    # Return the original JSON data if validation fails
                    return response
        except Exception as e:
            logger.error(f"API call error: {e}")
            return None
        
        return response
    
    def _create_safe_schema(self, model_class: Type[BaseModel]) -> Dict[str, Any]:
        """
        Create a safe JSON schema without default values to comply with OpenAI's requirements.
        
        Args:
            model_class: The Pydantic model class
            
        Returns:
            A JSON schema compatible with OpenAI's API
        """
        # Get the original schema
        schema = model_class.model_json_schema()
        
        # Define a recursive function to clean defaults
        def remove_defaults(obj: Dict[str, Any]) -> None:
            if not isinstance(obj, dict):
                return
                
            # Remove default at current level
            if "default" in obj:
                del obj["default"]
            
            # Process properties
            if "properties" in obj:
                for prop_name, prop_value in obj["properties"].items():
                    if isinstance(prop_value, dict):
                        remove_defaults(prop_value)
            
            # Process array items
            if "items" in obj and isinstance(obj["items"], dict):
                remove_defaults(obj["items"])
            
            # Process nested objects
            for key, value in list(obj.items()):
                if isinstance(value, dict):
                    remove_defaults(value)
        
        # Clean the schema
        remove_defaults(schema)
        return schema 