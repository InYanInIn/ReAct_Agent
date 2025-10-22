from datetime import datetime

from langchain.tools import tool
from langchain_core.messages import HumanMessage

from agent.model import get_model
from agent.utils import output_post_processing

llm = get_model()


class CodeEvaluator:
    """Tool for evaluating the quality of code fixes."""

    @staticmethod
    @tool("evaluator")
    def evaluate_fix(original_code: str, fixed_code: str, verification_results: str) -> str:
        """Evaluate if the fix is successful and determine next steps."""
        print("E" + str(datetime.now().strftime("%H:%M:%S")))
        prompt = f"""
        CODE QUALITY EVALUATION MISSION:

        ORIGINAL CODE (with issues):
        ```python
        {original_code}
        ```

        FIXED CODE (proposed solution):
        ```python
        {fixed_code}
        ```

        VALIDATION RESULTS:
        {verification_results}

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

        output = llm.invoke([HumanMessage(content=prompt)])

        return output_post_processing(output)

