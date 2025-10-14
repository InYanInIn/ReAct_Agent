from typing import Dict, Callable, Tuple, Optional, Any
from tools import save_history_to_logs

MAX_ATTEMPTS = 20
LOG_PATH = "./logs/agent_logs.json"

class ReActAgent:
    def __init__(self, tools_map: Dict[str, Callable], max_attempts: int = MAX_ATTEMPTS):
        """
        tools_map: mapping from tool name (str) to callable tool function.
        Example: {"analyzer": analyzer_tool, "code_fixer": generate_code_fix_tool, ...}
        """
        self.tools = tools_map
        self.max_attempts = max_attempts

    def __call__(self, state):
        """
        Make the instance callable so StateGraph accepts it as a node.
        This simply forwards to the step() method.
        """
        return self.step(state)

    def reason(self, state: dict) -> str:
        """
        Return a short token describing the next high-level intent.
        Possible returns: "analyze", "fix", "sandbox", "evaluate", "finish", "give_up"
        """
        # If we've already reached a terminal status, finish immediately.
        if state.get("status") in ("SUCCESS", "FAILED"):
            return "finish"

        # Safety: limit attempts to avoid infinite loops.
        if state.get("attempt_count", 0) >= self.max_attempts:
            return "give_up"

        # If there is no diagnosis, do analysis first.
        if not state.get("current_diagnosis"):
            return "analyze"

        # If we have a diagnosis but no proposed fix yet, ask the fixer to generate one.
        if state.get("current_diagnosis") and not state.get("proposed_fix"):
            return "fix"

        # If we have a proposed fix but haven't run it in sandbox, execute it.
        if state.get("proposed_fix") and not state.get("verification_results"):
            return "sandbox"

        # If we have verification results but haven't evaluated them with the evaluator tool,
        # proceed to evaluation (the evaluator will decide SUCCESS/NEEDS_BETTER_*).
        if state.get("verification_results") and state.get("status") not in ("SUCCESS", "FAILED"):
            return "evaluate"

        # Fallback: analyze again (maybe new info appeared).
        return "analyze"

    def select_action(self, thought: str, state: dict) -> Tuple[Optional[str], Any]:
        """
        Given a thought token and current state, return (tool_name, action_input).
        action_input may be a single object or a tuple/dict depending on the tool signature.
        If tool_name is None -> no action to perform (terminal).
        """
        # Map high-level intent to exactly the tool names you registered with @tool(...)
        if thought == "analyze":
            # analyzer_tool expects source code string (signature: analyzer_tool(code: str))
            src = state.get("source_code", "")
            return "analyzer", src

        if thought == "fix":
            # code_fixer expects (source_code, diagnosis)
            src = state.get("source_code", "")
            diag = state.get("current_diagnosis", "")
            # return "code_fixer", diag
            return "code_fixer", src

        if thought == "sandbox":
            # sandbox_tool expects code string to run
            fixed = state.get("proposed_fix", "")
            return "sandbox", fixed

        if thought == "evaluate":
            # evaluator expects original_code, fixed_code, verification_results
            orig = state.get("source_code", "")
            fixed = state.get("proposed_fix", "")
            verification = state.get("verification_results", "")
            return "evaluator", orig

        if thought == "finish":
            return None, None

        if thought == "give_up":
            return None, None

        # default no-op
        return None, None

    def step(self, state: dict) -> dict:
        """
        Perform one ReAct iteration. Returns updated state.
        This method:
          - decides next intent (reason)
          - maps to tool (select_action)
          - calls the tool and records the observation
          - updates state fields (current_diagnosis / proposed_fix / verification_results / status / attempt_count)
          - appends a short entry to action_history (if present)
        """
        thought = self.reason(state)

        # Terminal conditions
        if thought in ("finish", "give_up"):
            if thought == "give_up":
                state["status"] = "FAILED"
            # do not call any tool
            save_history_to_logs(LOG_PATH, state.get("action_history", []))
            return state

        tool_name, action_input = self.select_action(thought, state)
        if tool_name is None:
            save_history_to_logs(LOG_PATH, state.get("action_history", []))
            return state  # nothing to do

        # Lookup the callable tool
        tool_callable = self.tools.get(tool_name)
        if tool_callable is None:
            # Tool not available: mark as failed
            state["status"] = "FAILED"
            save_history_to_logs(LOG_PATH, state.get("action_history", []))
            return state

        # print(f"ReActAgent: invoking tool '{tool_name}' with input: {action_input}")

        # Prepare inputs according to whether we returned tuple or single arg
        try:
            observation = tool_callable(action_input)
        except Exception as e:
            # If tool raised, capture the exception as observation so we can analyze it
            observation = {"tool_error": str(e)}

        # print(f"ReActAgent: tool '{tool_name}' returned observation: {observation}")

        # Update attempt counter
        state["attempt_count"] = state.get("attempt_count", 0) + 1

        # Record action history if the field exists
        history = state.get("action_history")
        entry = {
            "action": tool_name,
            "input": action_input,
            "observation": observation
        }
        if isinstance(history, list):
            history.append(entry)
        else:
            state["action_history"] = [entry]

        # Update specific state fields depending on which tool ran:
        # - analyzer -> update current_diagnosis
        # - code_fixer -> update proposed_fix
        # - sandbox -> update verification_results
        # - evaluator -> update status (SUCCESS / NEEDS_BETTER_FIX / NEEDS_BETTER_ANALYSIS / FAILED)
        if tool_name == "analyzer":
            # analyzer is expected to return a diagnosis string (LLM text)
            state["current_diagnosis"] = observation

        elif tool_name == "code_fixer":
            # code_fixer should return the fixed code string
            state["proposed_fix"] = observation

            # Reset verification/results when new fix is proposed
            state.pop("verification_results", None)
            state.pop("status", None)

        elif tool_name == "sandbox":
            # sandbox returns a dict with run outcome, tests, stdout, exceptions, etc.
            state["verification_results"] = observation

        elif tool_name == "evaluator":
            # evaluator returns a short token like "SUCCESS" or "NEEDS_BETTER_FIX"
            eval_result = observation.strip() if isinstance(observation, str) else str(observation)
            # Normalize common outputs
            eval_result = eval_result.upper()
            if eval_result not in ("SUCCESS", "NEEDS_BETTER_FIX", "NEEDS_BETTER_ANALYSIS", "FAILED"):
                # print(f"Warning: evaluator returned unexpected result: {eval_result}")
                # Try to extract one of the expected tokens if LLM added extra text
                for token in ("SUCCESS", "NEEDS_BETTER_FIX", "NEEDS_BETTER_ANALYSIS", "FAILED"):
                    if token in eval_result:
                        eval_result = token
                        save_history_to_logs(LOG_PATH, state.get("action_history", []))
                        break
                else:
                    # If nothing matched, mark as FAILED to be safe
                    eval_result = "FAILED"

            state["status"] = eval_result

            # If evaluator indicates we need better analysis, clear diagnosis so analyzer runs again
            if eval_result == "NEEDS_BETTER_ANALYSIS":
                state["current_diagnosis"] = None
            # If evaluator requires a better fix, clear proposed_fix so code_fixer runs again
            if eval_result == "NEEDS_BETTER_FIX":
                state["proposed_fix"] = None

        # Return updated state to caller
        return state
