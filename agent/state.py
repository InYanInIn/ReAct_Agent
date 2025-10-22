from dataclasses import dataclass, field
from typing import Annotated, Any
from langgraph.graph.message import add_messages

MAX_ATTEMPTS = 20

@dataclass
class DebugState:
    """Represents the state of the code debugging/fixing process."""
    source_code: str = ""
    error_description: str = ""
    action_history: Annotated[list, add_messages] = field(default_factory=list)
    current_diagnosis: str = ""
    proposed_fix: str = ""
    verification_results: str = ""
    attempt_count: int = 0
    status: str = ""
    last_action: str = ""

    def reset_fix_attempt(self):
        """Resets the proposed fix and related fields for a new attempt."""
        self.proposed_fix = ""
        self.verification_results = ""
        self.status = ""
        self.attempt_count = 0

    def record_action(self, action: str, input_data: Any, observation: Any):
        """Records an action taken during the debugging process."""
        # entry = {
        #     "action": action,
        #     "input": input_data,
        #     "observation": observation,
        #     "timestamp": len(self.action_history) + 1
        # }
        content = f"Action: {action}\nInput: {input_data}\nObservation: {observation}"

        self.action_history.append(("assistant", content))
        self.last_action = action

    def is_terminal(self) -> bool:
        """Checks if the debugging process has reached a terminal state."""
        return self.status in ("SUCCESS", "FAILED")

    def should_continue(self, max_attempts: int = MAX_ATTEMPTS) -> bool:
        """Determines if further actions should be taken."""
        return not self.is_terminal() and self.attempt_count < max_attempts
