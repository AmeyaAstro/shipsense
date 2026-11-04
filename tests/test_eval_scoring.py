"""The eval scorer itself must be right, or every reported number is wrong."""
from run_eval import load_questions, score

Q = {"must_include": [["25%", "25 percent"], ["credit"]], "expected_doc": "03_refund.md",
     "required_tools": ["lookup_shipment"]}


def test_correct_needs_every_group():
    assert score(Q, "They get 25 PERCENT as credit.", [], [], [])["correct"]
    assert not score(Q, "They get 25% back.", [], [], [])["correct"]  # 'credit' group missing


def test_only_valid_citations_count():
    src = ["03_refund.md § Credit amounts"]
    assert score(Q, "x [03_refund.md § Credit amounts]", src, [], [])["cites_expected_doc"]
    assert not score(Q, "x [03_refund.md § Credit amounts]", [], [], [])["cites_expected_doc"]  # never retrieved


def test_required_tools():
    assert score(Q, "", [], ["search_policy", "lookup_shipment"], [])["required_tools_used"]
    assert not score(Q, "", [], ["search_policy"], [])["required_tools_used"]


def test_question_file_is_well_formed():
    qs = load_questions()
    assert len(qs) == 25 and len({q["id"] for q in qs}) == 25
    assert all(q["must_include"] and all(q["must_include"]) for q in qs)
