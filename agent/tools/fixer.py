from datetime import datetime

from langchain.tools import tool
from langchain_core.messages import HumanMessage

from agent.model import get_model
from agent.utils import extract_python_code

llm = get_model()


class CodeFixer:
    """Tool for generating code fixes based on diagnosis."""

    @staticmethod
    @tool("code_fixer")
    def generate_fix(source_code: str, diagnosis: str) -> str:
        """Generate fixed code based on diagnosis and original code."""
        print("F" + str(datetime.now().strftime("%H:%M:%S")))
        prompt = f"""
    You are a careful code fixer. Given the original code and an analysis/diagnosis, produce a corrected, minimal, and runnable Python implementation that:

    - Preserves original function names and signatures.
    - Changes only what is necessary to fix the identified issues.
    - Keeps docstrings and examples (update only if necessary).
    - Follows PEP8 and uses clear variable names.
    - Includes necessary imports and is self-contained.
    - Is ready to be executed by unit tests.


    INPUT:
    ORIGINAL CODE:
    ```python
    {source_code}
    ANALYSIS / DIAGNOSIS:
    {diagnosis}

    CONSTRAINTS:
    - Do NOT add new features or change the API.
    - Do NOT include any tests, example usage, demo code, or sanity prints.
    - **Do NOT include Example usage**.
    - Do NOT include any `if __name__ == "__main__"` blocks.
    - Do NOT include comments unless they were part of the original docstrings.
    - Return ONLY the complete Python source file as plain text. No markdown fences, no explanations, no extra lines before/after the file.

    OUTPUT: the complete corrected Python source code (as plain text).
    """

        response = llm.invoke([HumanMessage(content=prompt)])

        return extract_python_code(response).strip()
