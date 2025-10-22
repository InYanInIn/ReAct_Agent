from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from langchain.tools import tool

from agent.react_agent import ReActAgent
from agent.state import DebugState
from agent.utils import run_in_sandbox, extract_python_code, output_post_processing
from agent.model import get_model

class Workflow:
    """Orchestrates the debugging workflow."""
    def __init__(self):
        self.agent = ReActAgent()
        self.graph = self._build_graph()

    def _build_graph(self):
        gpaph_builder = StateGraph(DebugState)

        gpaph_builder.add_node("react_agent", self.agent)

        gpaph_builder.add_edge(START, "react_agent")

        gpaph_builder.add_conditional_edges(
            "react_agent",
            self._should_continue,
            {
                "continue": "react_agent",
                "end": END,
            }
        )
        return gpaph_builder.compile()

    def _should_continue(self, state: DebugState) -> str:
        """Determine whether to continue or end the workflow."""
        if state.is_terminal() or not state.should_continue(self.agent.max_attempts):
            return "end"
        return "continue"

    def debug_code(self, source_code: str, error_description: str = "") -> DebugState:
        """Main method to debug and fix the provided source code."""
        initial_state = DebugState(
            source_code=source_code,
            error_description=error_description
        )
        final_state = self.graph.invoke(initial_state)
        return final_state


def get_fixed_code(source_code: str, error_description: str = "") -> str:
    """Utility function to run the workflow and get the fixed code."""
    workflow = Workflow()
    final_state = workflow.debug_code(source_code, error_description)
    return final_state.get("proposed_fix")
