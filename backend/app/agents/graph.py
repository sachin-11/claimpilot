"""Graph builder.

    START -> intake -> supervisor -+-> coverage --+
                                   +-> risk ------+--> supervisor -> decision -> approval
                                   +-> decision (missing documents)               |
                                                                         interrupt() ... resume
                                                                                  |
                                                                      finalize -> END
    any node --(failure)--> error -> END

`supervisor` is re-entered after every worker; coverage and risk run in the same super-step, so
the next supervisor visit is the join. Nodes carry a RetryPolicy for transient errors. A node
that still fails after its retries raises out of the graph and the runner takes the error path
(runner._fail_run); permanent errors hop to the `error` node inside the graph."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.deps import AgentDeps
from app.agents.nodes.approval import approval
from app.agents.nodes.base import guarded, retry_policy
from app.agents.nodes.coverage import make_coverage_node
from app.agents.nodes.decision import make_decision_node
from app.agents.nodes.error import make_error_node
from app.agents.nodes.finalize import make_finalize_node
from app.agents.nodes.intake import make_intake_node
from app.agents.nodes.risk import make_risk_node
from app.agents.state import ClaimState
from app.agents.supervisor import make_supervisor_node


def build_graph(deps: AgentDeps, checkpointer: Any = None):  # type: ignore[no-untyped-def]
    retry = retry_policy(deps.settings)
    graph = StateGraph(ClaimState)

    def add(name: str, fn: Any, *, retries: bool = True, ends: bool = False) -> None:
        graph.add_node(
            name,
            guarded(name, fn),
            retry_policy=retry if retries else None,
            destinations=("error", END) if ends else ("error",),  # drawing only
        )

    add("intake", make_intake_node(deps))
    add("coverage", make_coverage_node(deps))
    add("risk", make_risk_node(deps))
    add("decision", make_decision_node(deps))
    # No retries around the interrupt: it is a pause, not a failure.
    add("approval", approval, retries=False)
    add("finalize", make_finalize_node(deps), ends=True)
    graph.add_node(
        "supervisor", make_supervisor_node(deps), destinations=("coverage", "risk", "decision", END)
    )
    graph.add_node("error", make_error_node(deps))

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "supervisor")
    graph.add_edge("coverage", "supervisor")
    graph.add_edge("risk", "supervisor")
    graph.add_edge("decision", "approval")
    graph.add_edge("approval", "finalize")
    graph.add_edge("finalize", END)
    graph.add_edge("error", END)
    return graph.compile(checkpointer=checkpointer)
