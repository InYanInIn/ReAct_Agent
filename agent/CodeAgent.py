from typing import Annotated

from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langchain.tools import tool

from agent.ReActAgent import ReActAgent
from tools import run_in_sandbox, extract_python_code, output_post_processing
from model import get_model

llm = get_model()


class State(dict):
    source_code: str
    error_description: str
    action_history: Annotated[list, add_messages]
    current_diagnosis: str
    proposed_fix: str
    verification_results: str
    attempt_count: int
    status: str


@tool("sandbox")
def sandbox_tool(code: str) -> dict:
    """Executes Python code safely in sandbox and returns result."""
    return run_in_sandbox(state['proposed_fix'])


@tool("analyzer")
def analyzer_tool(code: str) -> str:
    """Analyzes Python code and returns its structure."""

    analysis_prompt = f"""
You are a senior code reviewer and bug detective. Deeply analyze the provided Python code and produce a precise, actionable diagnosis.

INPUT:
CODE:
```python
{code}
ERROR / CONTEXT:
{state.get('error_description', 'No specific error provided.')}

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

    diagnosis = llm.invoke([HumanMessage(content=analysis_prompt)])
    # print(diagnosis)

    diagnosis = output_post_processing(diagnosis)

    return diagnosis


@tool("code_fixer")
def generate_code_fix_tool(source_code: str) -> str:
    """Generate fixed code based on diagnosis and original code."""
    prompt = fix_prompt = f"""
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
{state.get('current_diagnosis', 'No diagnosis provided.')}

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
    code = extract_python_code(response)

    return code.strip()


@tool("evaluator")
def evaluate_fix_quality_tool(original_code: str) -> str:
    """Evaluate if the fix is successful and determine next steps."""

    prompt = f"""
    CODE QUALITY EVALUATION MISSION:

    ORIGINAL CODE (with issues):
    ```python
    {state['source_code']}
    ```

    FIXED CODE (proposed solution):
    ```python
    {state['proposed_fix']}
    ```

    VALIDATION RESULTS:
    {state['verification_results']}

    **COMPREHENSIVE EVALUATION CHECKLIST:**

    1. **REQUIREMENTS FULFILLMENT ANALYSIS**
       - Compare fixed code against ALL original requirements from comments/docstrings
       - Verify no stated functionality is missing or incomplete
       - Check if implicit requirements are addressed

    2. **ISSUE RESOLUTION VERIFICATION**
       - Does the fixed code address ALL issues mentioned in the original problem?
       - Are there any remaining symptoms of the original bugs?
       - Check edge cases and boundary conditions

    3. **CODE QUALITY ASSESSMENT**
       - Does the fix maintain or improve code readability?
       - Are there any code smells or anti-patterns introduced?
       - Check for proper error handling and robustness

    4. **LOGICAL CORRECTNESS VALIDATION**
       - Trace through the fixed logic step by step
       - Verify variable usage and state management
       - Check for off-by-one errors, infinite loops, etc.

    5. **COMPARATIVE ANALYSIS**
       - What specific changes were made from original to fixed?
       - Do these changes directly address the root causes?
       - Are the changes minimal and focused?

    **EVALUATION DECISION MATRIX:**

    Return "SUCCESS" only if:
    ✅ All original requirements are properly implemented
    ✅ No remaining bugs from original issues
    ✅ Code follows Python best practices
    ✅ No new functionality beyond requirements added

    Return "NEEDS_BETTER_FIX" if:
    🔴 Original issues are partially fixed but not completely
    🔴 Code has syntax errors or runtime issues
    🔴 Requirements are still not fully met

    Return "NEEDS_BETTER_ANALYSIS" if:
    🟡 Fix addresses wrong problems
    🟡 New issues are introduced that weren't in original
    🟡 The core problem was misunderstood

    Return "FAILED" if:
    ⚫ Code is completely broken or makes no sense
    ⚫ Multiple failed attempts with no progress

    **CRITICAL THINKING:**
    - Don't just rely on verification results - actually read and understand both code versions
    - Look for "silent bugs" that don't cause crashes but produce wrong results
    - Consider if the fix would pass code review in a professional setting

    Return ONLY ONE WORD: SUCCESS, NEEDS_BETTER_FIX, NEEDS_BETTER_ANALYSIS, FAILED
    """
    output = output_post_processing(llm.invoke([HumanMessage(content=prompt)]))

    return output


tools_map = {
    "analyzer": analyzer_tool,
    "code_fixer": generate_code_fix_tool,
    "sandbox": sandbox_tool,
    "evaluator": evaluate_fix_quality_tool
}

graph_builder = StateGraph(State)

react_agent = ReActAgent(tools_map=tools_map)

graph_builder.add_node("analyzer", analyzer_tool)
graph_builder.add_node("code_fixer", generate_code_fix_tool)
graph_builder.add_node("sandbox", sandbox_tool)
graph_builder.add_node("evaluator", evaluate_fix_quality_tool)

graph_builder.add_node("react_agent", react_agent)
graph_builder.add_edge(START, "react_agent")
graph_builder.add_edge("react_agent", "analyzer")
graph_builder.add_edge("react_agent", "code_fixer")
graph_builder.add_edge("react_agent", "sandbox")
graph_builder.add_edge("react_agent", "evaluator")
graph_builder.add_edge("react_agent", END)

graph = graph_builder.compile()
state = State(
    source_code="",
    error_description="",
    action_history=[],
    current_diagnosis="",
    proposed_fix="",
    verification_results="",
    attempt_count=0,
    status=""
)


def get_fixed_code(source_code: str, error_description: str = "") -> str:
    global state
    new_state = State(
        source_code=source_code,
        error_description=error_description,
        action_history=[],
        current_diagnosis="",
        proposed_fix="",
        verification_results="",
        attempt_count=0,
        status=""
    )
    state = react_agent.step(new_state)
    for _ in range(10):
        state = react_agent.step(state)
        if state.get("status") in ("SUCCESS", "FAILED"):
            break
    return state["proposed_fix"]
