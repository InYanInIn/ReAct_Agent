from typing import Dict, Callable, Tuple, Optional, Any

from agent.state import DebugState
from agent.tools import get_tools
from agent.utils import save_history_to_logs

MAX_ATTEMPTS = 20
LOG_PATH = "./logs/agent_logs.json"


class ReActAgent:
    def __init__(self, max_attempts: int = MAX_ATTEMPTS):
        """
        tools_map: mapping from tool name (str) to callable tool function.
        Example: {"analyzer": analyzer_tool, "code_fixer": generate_code_fix_tool, ...}
        """
        self.tools = get_tools()
        self.max_attempts = max_attempts
        self._setup_action_map()

    def _setup_action_map(self):
        """Define how each action maps to tools and their inputs"""
        self.action_map = {
            "analyze": {
                "tool": "analyzer",
                "inputs": lambda state: {"code": state.source_code, "error_context": state.error_description}
            },
            "fix": {
                "tool": "code_fixer",
                "inputs": lambda state: {"source_code": state.source_code, "diagnosis": state.current_diagnosis}
            },
            "sandbox": {
                "tool": "sandbox",
                "inputs": lambda state: {"code": state.proposed_fix}
            },
            "evaluate": {
                "tool": "evaluator",
                "inputs": lambda state: {"original_code": state.source_code, "fixed_code": state.proposed_fix, "verification_results": state.verification_results}
            },
            "finish": {
                "tool": None,
                "inputs": lambda state: None
            },
            "give_up": {
                "tool": None,
                "inputs": lambda state: None
            }
        }

    def __call__(self, state: DebugState) -> DebugState:
        """
        Make the instance callable so StateGraph accepts it as a node.
        This simply forwards to the step() method.
        """
        return self.step(state)

    def reason(self, state: DebugState) -> str:
        """
        Return a short token describing the next high-level intent.
        Possible returns: "analyze", "fix", "sandbox", "evaluate", "finish", "give_up"
        """

        if state.is_terminal():
            return "finish"

        if not state.should_continue(self.max_attempts):
            return "give_up"

        # Decision logic
        if not state.current_diagnosis:
            return "analyze"
        elif not state.proposed_fix:
            return "fix"
        elif not state.verification_results:
            return "sandbox"
        elif not state.status:
            return "evaluate"
        else:
            return "analyze"

    def select_action(self, thought: str, state: DebugState) -> Tuple[Optional[str], Any]:
        """
        Given a thought token and current state, return (tool_name, action_input).
        action_input may be a single object or a tuple/dict depending on the tool signature.
        If tool_name is None -> no action to perform (terminal).
        """
        action_config = self.action_map.get(thought)
        if not action_config:
            return None, None

        tool_name = action_config["tool"]
        inputs_func = action_config["inputs"]
        inputs = inputs_func(state)

        return tool_name, inputs

    def _call_tool(self, tool_name: str, inputs: Any) -> Any:
        """Safely call the tool and return its output."""
        tool = self.tools.get(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found.")

        try:
            if isinstance(inputs, dict):
                return tool.invoke(inputs)
            elif inputs is not None:
                return tool.invoke(inputs)
            else:
                return tool.invoke()
        except Exception as e:
            return {"tool_error": str(e)}

    def _update_state_from_tool(self, state: DebugState, tool_name: str, observation: Any):
        """Update the DebugState based on the tool used and its observation"""
        tool_handlers = {
            "analyzer": self._handle_analyzer_result,
            "code_fixer": self._handle_fixer_result,
            "sandbox": self._handle_sandbox_result,
            "evaluator": self._handle_ealuator_result
        }

        handler = tool_handlers.get(tool_name)
        if handler:
            handler(state, observation)

    def _handle_analyzer_result(self, state: DebugState, observation: Any):
        """Handle analyzer tool results"""
        state.current_diagnosis = observation

    def _handle_fixer_result(self, state: DebugState, observation: Any):
        """Handle code fixer tool results"""
        state.proposed_fix = observation
        # Reset verification/results when new fix is proposed
        state.verification_results = ""
        state.status = ""

    def _handle_sandbox_result(self, state: DebugState, observation: Any):
        """Handle sandbox tool results"""
        state.verification_results = str(observation)

    def _handle_ealuator_result(self, state: DebugState, observation: Any):
        """Handle evaluator tool results"""
        eval_result = self._normalize_evaluator_output(observation)
        state.status = eval_result

        if eval_result == "NEEDS_BETTER_ANALYSIS":
            state.current_diagnosis = ""
        elif eval_result == "NEEDS_BETTER_FIX":
            state.proposed_fix = ""

    def _normalize_evaluator_output(self, observation: Any) -> str:
        """Normalize evaluator output to expected tokens"""
        eval_result = observation.strip() if isinstance(observation, str) else str(observation)
        eval_result = eval_result.upper()

        valid_statuses = ("SUCCESS", "NEEDS_BETTER_FIX", "NEEDS_BETTER_ANALYSIS", "FAILED")
        if eval_result in valid_statuses:
            return eval_result
        else:
            for token in valid_statuses:
                if token in eval_result:
                    return token

        return "FAILED"

    def step(self, state: DebugState) -> DebugState:
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
                state.status = "FAILED"
            # do not call any tool
            save_history_to_logs(LOG_PATH, state.action_history)
            return state

        tool_name, action_input = self.select_action(thought, state)
        if not tool_name:
            save_history_to_logs(LOG_PATH, state.action_history)
            return state  # nothing to do

        observation = self._call_tool(tool_name, action_input)

        state.attempt_count += 1
        state.record_action(tool_name, action_input, observation)
        self._update_state_from_tool(state, tool_name, observation)

        return state
