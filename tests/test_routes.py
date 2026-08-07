"""路由函数单测（文档第 4 节）。"""

from app.graph.routes import (
    route_after_approval,
    route_after_output_validation,
    route_after_validation,
)


def test_route_after_validation_errors_goto_failure():
    state = {"errors": [{"node": "validate_plan", "message": "bad"}]}
    assert route_after_validation(state) == "generate_failure_report"


def test_route_after_validation_approval_required():
    state = {"errors": [], "approval_required": True}
    assert route_after_validation(state) == "human_approval"


def test_route_after_validation_proceed():
    state = {"errors": [], "approval_required": False}
    assert route_after_validation(state) == "build_tasks"


def test_route_after_approval_approved():
    assert route_after_approval({"approval_decision": "approved"}) == "build_tasks"


def test_route_after_approval_rejected():
    assert route_after_approval({"approval_decision": "rejected"}) == "generate_failure_report"
    assert route_after_approval({}) == "generate_failure_report"


def test_route_after_output_has_candidates():
    state = {"model_results": [{"status": "success", "answer": "ok"}, {"status": "failed", "answer": ""}]}
    assert route_after_output_validation(state) == "score_results"


def test_route_after_output_no_candidates_all_failed():
    state = {"model_results": [{"status": "failed", "answer": ""}]}
    assert route_after_output_validation(state) == "generate_failure_report"


def test_route_after_output_empty_answer_not_candidate():
    state = {"model_results": [{"status": "success", "answer": "   "}]}
    assert route_after_output_validation(state) == "generate_failure_report"
