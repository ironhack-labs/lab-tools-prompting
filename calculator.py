import os
import sys
from langchain_community.llms import Ollama
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import Dict, Any, Optional, TypedDict
from langchain_core.tools import render_text_description
import time

def check_ollama_connection():
    """Check if Ollama is running and accessible"""
    try:
        model = Ollama(model="phi")
        # Try a simple completion to test connection
        model.invoke("test")
        return True
    except Exception as e:
        print("\nError connecting to Ollama. Please ensure:")
        print("1. WSL2 is installed and running")
        print("2. Ubuntu is installed in WSL2")
        print("3. Ollama is installed in Ubuntu")
        print("4. Ollama server is running ('ollama serve' in Ubuntu terminal)")
        print(f"\nError details: {str(e)}")
        return False

# Define our calculator tools
@tool
def multiply(x: float, y: float) -> float:
    """Multiply two numbers together."""
    return x * y

@tool
def add(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y

def setup_chain():
    """Set up the LangChain processing chain"""
    # Create tools list
    tools = [multiply, add]
    
    # Initialize Ollama model with retry logic
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            model = Ollama(model="phi")
            break
        except Exception as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to initialize Ollama after {max_retries} attempts")
            print(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    # Create the prompt template
    rendered_tools = render_text_description(tools)
    system_prompt = f"""
    You are an assistant that has access to the following set of tools.
    Here are the names and descriptions for each tool:

    {rendered_tools}

    Given the user input, return the name and input of the tool to use.
    Return your response as a JSON blob with 'name' and 'arguments' keys.
    """

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("user", "{input}")]
    )

    # Create the chain
    return (
        prompt 
        | model 
        | JsonOutputParser() 
        | RunnablePassthrough.assign(output=invoke_tool)
    )

class ToolCallRequest(TypedDict):
    name: str
    arguments: Dict[str, Any]

def invoke_tool(
    tool_call_request: ToolCallRequest,
    config: Optional[Dict] = None
):
    """Invoke the requested tool with given arguments"""
    tools = [multiply, add]
    tool_name_to_tool = {tool.name: tool for tool in tools}
    name = tool_call_request["name"]
    requested_tool = tool_name_to_tool[name]
    return requested_tool.invoke(tool_call_request["arguments"], config=config)

def main():
    """Main function to run the calculator"""
    if not check_ollama_connection():
        sys.exit(1)
        
    print("Initializing calculator...")
    chain = setup_chain()
    print("Calculator ready!")
    print("\nEnter calculations (e.g., 'what's 5 times 3' or 'what's 10 plus 20')")
    print("Type 'exit' to quit")
    
    while True:
        try:
            user_input = input("\nEnter calculation: ").strip().lower()
            if user_input == 'exit':
                break
                
            result = chain.invoke({"input": user_input})
            print(f"Result: {result['output']}")
            
        except KeyboardInterrupt:
            print("\nExiting calculator...")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again with a different calculation.")

if __name__ == "__main__":
    main()