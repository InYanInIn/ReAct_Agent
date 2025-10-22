import time
from datetime import datetime

from langchain.tools import tool
from langchain_core.messages import HumanMessage

from agent.model import get_model
from agent.utils import output_post_processing

llm = get_model()


class CodeAnalyzer:
    """Tool for analyzing code and diagnosing issues."""

    @staticmethod
    @tool("analyzer")
    def analyze_code(code: str, error_context: str = "") -> str:
        """Analyzes Python code and returns its structure."""
        print("A"+str(datetime.now().strftime("%H:%M:%S")))

        prompt = f"""
    You are a senior code reviewer and bug detective. Deeply analyze the provided Python code and produce a precise, actionable diagnosis.

    INPUT:
    CODE:
    ```python
    {code}
    ERROR / CONTEXT:
    {error_context if error_context != "" else "No specific error context provided."}

    OUTPUT FORMAT (strictly follow this structure):

    1. REQUIREMENTS (explicit and implicit)

    Short summary (1-2 sentences) describing what the function is supposed to do.

    Bullet list of all explicit requirements (from docstring, examples, tests).

    Bullet list of implicit requirements and assumptions (types, edge cases, performance expectations).

    2. COMPLIANCE CHECK (mapping requirements → status)
    For each requirement above produce:

    Requirement: ...

    Status: [OK | MISSING | INCORRECT]

    Evidence: code lines / behavior that justify the status.

    3. ROOT-CAUSE DIAGNOSIS (one or more items)
    For each failing requirement give:

    Symptom (what goes wrong)

    Root cause (precise code location and why it leads to the symptom)

    Repro steps or example input that demonstrates the bug

    4. ISSUE CLASSIFICATION (one line per issue)

    Severity: [CRITICAL | MAJOR | MINOR | POTENTIAL]

    Short rationale.

    5. FIX SUGGESTION (for each issue)

    Exact change to code (small diff style or short code snippet)

    Why this fix resolves the root cause

    Any tradeoffs or side effects

    6. QUICK CHECKLIST (pass/fail)

    Minimal tests or assertions to verify the fix

    7. SUMMARY

    Overall correctness estimate (0..100)

    Top 3 action items (ordered by priority)

    Be concise but exhaustive. Reference code by line snippet or small quoted fragments. If something is ambiguous, state the assumption you make and proceed with the analysis under that assumption.
    """
        # print(analysis_prompt)

        response = llm.invoke([HumanMessage(content=prompt)])
        # print(diagnosis)

        return output_post_processing(response)
