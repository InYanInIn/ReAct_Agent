from datetime import datetime

from langchain.tools import tool
from langchain_core.messages import HumanMessage

from agent.model import get_model
from agent.utils import run_in_sandbox

llm = get_model()


class SandboxExecutor:
    """Tool for executing code in a sandbox environment."""

    @staticmethod
    @tool("sandbox")
    def execute_code(code: str) -> dict:
        """Executes Python code safely in sandbox and returns result."""
        print("S" + str(datetime.now().strftime("%H:%M:%S")))
        return run_in_sandbox(code)
