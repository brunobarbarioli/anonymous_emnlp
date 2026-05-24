from agents.multi_agent_orchestrator import (
    _normalize_model_alignment_payload,
    _normalize_model_claims,
    _normalize_model_robustness_checks,
    _parse_model_json,
)


def test_parse_model_json_accepts_fenced_json():
    payload = _parse_model_json(
        '```json\n{"checks": [{"name": "A", "summary": "Check A"}]}\n```'
    )

    assert payload["checks"][0]["name"] == "A"


def test_model_claim_normalization_does_not_create_fallback_claims():
    claims = _normalize_model_claims(
        {"main_results": []},
        {"headline_table_selection": [{"item_id": "Table1"}, {"item_id": "Table2"}]},
    )

    assert claims == []


def test_model_claim_normalization_uses_only_model_text():
    claims = _normalize_model_claims(
        {
            "main_results": [
                {
                    "claim_text": "The treatment increased take-up in the main experiment.",
                    "mapped_tables": ["Table1"],
                }
            ]
        },
        {"headline_table_selection": [{"item_id": "Table1"}, {"item_id": "Table2"}]},
    )

    assert len(claims) == 1
    assert claims[0]["source"] == "model"
    assert claims[0]["claim_text"] == "The treatment increased take-up in the main experiment."


def test_alignment_normalization_uses_model_record_descriptions():
    payload = _normalize_model_alignment_payload(
        {
            "misalignment_records": [
                {
                    "issue": "Different clustering level",
                    "manuscript_location": "Table 2 notes",
                    "code_location": "analysis.do:42",
                    "what_differs": "The manuscript clusters by school; code clusters by district.",
                    "why_it_matters": "Standard errors change.",
                    "severity": "medium",
                }
            ]
        },
        raw_response="{}",
    )

    assert payload["findings"][0]["model_generated"] is True
    assert "Different clustering level" in payload["findings"][0]["message"]
    assert "analysis.do:42" in payload["findings"][0]["message"]


def test_robustness_normalization_does_not_pad_templates():
    checks = _normalize_model_robustness_checks(
        {
            "checks": [
                {
                    "name": "Alternative cluster level",
                    "summary": "Recompute inference with the other cluster level used in the package.",
                    "category": "inference",
                    "subcategory": "cluster_level",
                }
            ]
        }
    )

    assert len(checks) == 1
    assert checks[0]["model_generated"] is True
    assert checks[0]["name"] == "Alternative cluster level"
