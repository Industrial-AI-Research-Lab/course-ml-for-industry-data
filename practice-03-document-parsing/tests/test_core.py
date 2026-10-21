"""Fast tests without network: metrics, tables, grading, store, triage, replay with images, data."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from plant_assistant import DATA_DIR, DOCS_DIR
from plant_assistant.docs import load_gt_pages, load_json, load_questions, triage
from plant_assistant.image_tasks import merge_tile_answers, tiles
from plant_assistant.images import vision_message
from plant_assistant.llm import CallLogger, ChatResponse, ReplayClient, loggable_messages, request_hash
from plant_assistant.metrics import cer, normalize_tag, number_scores, numbers, plain, repeated_share, tag_scores
from plant_assistant.qa import answer_on_page, canonical_unit, grade, paired, wilson
from plant_assistant.store import _continues, chunk_regulation, structure_score, title_metadata
from plant_assistant.tables import row_recovered, rows_of, table_score, to_html, to_markdown_as_parsed


# --- metrics -----------------------------------------------------------------------------------
def test_plain_removes_markup():
    assert plain("**4.3** <td>2,8</td> | x |\n|---|") == "4.3 2,8 x"


def test_cer_zero_and_one_edit():
    assert cer("4,5 мм/с", "4,5 мм/с") == 0
    assert cer("4,5 мм/с", "4,6 мм/с") == pytest.approx(1 / 8)


def test_numbers_keep_notation():
    assert numbers("не позднее 25 000 часов, 4,5 мм/с, п. 4.3.3") == ["25 000", "4,5", "4.3"]


def test_number_scores_detect_rewrite():
    s = number_scores("граница 4,5 и 7,1", "граница 4.5 и 7,1")
    assert s["numbers_exact"] == 0.5 and s["numbers_value"] == 1.0


def test_repeated_share():
    line = "<td></td><td></td> a long repeated table line here"
    assert repeated_share("\n".join([line] * 4)) == pytest.approx(0.75)


def test_tags_normalized():
    assert normalize_tag("Р-101") == "P-101"  # Cyrillic Р
    assert normalize_tag("pt 101") == "PT-101"
    s = tag_scores(["P-101", "PT-101"], ["P-101", "PT101", "FV-999"])
    assert s["correct"] == 2 and s["invented"] == ["FV-999"]


# --- tables ------------------------------------------------------------------------------------
CELLS = load_json("tables.json")["R-12_rev3/table3"]["cells"]
ROWS = load_json("tables.json")["R-12_rev3/table3"]["rows"]


def test_html_keeps_all_rows():
    assert table_score(to_html(CELLS), ROWS)["recovered"] == 5


def test_markdown_shifts_spanned_row():
    md = to_markdown_as_parsed(CELLS, 2)
    assert "| II | 4,5 | 7,1 |  |" in md


def test_rows_of_plain_text():
    rows = rows_of("Насосы I 2,8 4,5\nII 4,5 7,1")
    assert row_recovered(rows, "II", ["4,5", "7,1"])
    assert not row_recovered(rows, "II", ["4,5", "7,2"])


# --- grading -----------------------------------------------------------------------------------
Q = {"value": "4,5", "unit": "mm/s"}


@pytest.mark.parametrize("answer,ok,reason", [
    ({"value": "4,5", "unit": "мм/с", "not_stated": False}, True, "ok"),
    ({"value": "4.5", "unit": "mm/s", "not_stated": False}, True, "ok"),
    ({"value": "45", "unit": "мм/с", "not_stated": False}, False, "wrong value"),
    ({"value": "4,5", "unit": "мм", "not_stated": False}, False, "wrong unit"),
    ({"value": None, "unit": None, "not_stated": True}, False, "said not stated"),
    (None, False, "invalid answer"),
])
def test_grade(answer, ok, reason):
    assert grade(Q, answer) == (ok, reason)


def test_grade_negative():
    assert grade({"value": None, "unit": None}, {"value": None, "not_stated": True}) == (True, "ok")
    assert grade({"value": None, "unit": None}, {"value": "60", "not_stated": False}) == (False, "invented a value")


def test_units():
    assert canonical_unit("об/мин") == "rpm" and canonical_unit("m3/h") == "m³/h" and canonical_unit("°С") == "°C"


def test_answer_on_page():
    assert answer_on_page({"value": "25 000"}, "не позднее 25000 часов")
    assert not answer_on_page({"value": "2,8"}, "Насосы 2 8 4,5")


def test_wilson_and_paired():
    lo, hi = wilson(40, 50)
    assert lo < 0.8 < hi
    assert paired([True] * 40 + [False] * 10, [False] * 40 + [False] * 10)["verdict"] == "better"
    assert paired([True, False] * 25, [False, True] * 25)["verdict"] == "not different"


# --- store -------------------------------------------------------------------------------------
def gt_pages(file: str) -> list[str]:
    pages = load_gt_pages()
    return [pages[(file, i)] for i in range(1, 1 + sum(1 for f, _ in pages if f == file))]


def test_title_metadata():
    m = title_metadata(gt_pages("R-12_rev3.pdf")[0])
    assert m["doc_id"] == "R-12" and m["revision"] == 3 and m["effective_date"] == "2025-03-14"
    assert m["supersedes"] == 2 and m["title"] == "Вибрационный контроль насосного оборудования"


def test_clause_numbering_rule():
    assert _continues("4.3.4", "4.3.3") and _continues("5", "4.5.2")
    assert not _continues("720", "4.4.1") and not _continues("5", "5.2")


def test_heading_rule_and_scope():
    pages = ["РЕГЛАМЕНТ R-12\nВибрационный контроль\nРедакция 3\nОбласть действия: Насосная станция № 2 "
             "Владелец документа: Служба главного механика",
             "1 Область применения\n1.1 Настоящий регламент устанавливает порядок вибрационного контроля\n"
             "центробежных насосов.\n4.3 Нормы вибрации"]
    meta, chunks = chunk_regulation(pages, "x.pdf")
    assert meta["scope"] == "Насосная станция № 2"
    kinds = {c.clause: c.chunk_type for c in chunks}
    assert kinds == {"1": "section", "1.1": "clause"}  # 4.3 does not continue the numbering after 1.1


def test_structure_from_text_layer():
    _, chunks = chunk_regulation(gt_pages("R-12_rev3.pdf"), "R-12_rev3.pdf")
    gt = [c["clause"] for c in load_json("clauses.json")["R-12_rev3"]]
    assert structure_score(chunks, gt)["share"] == 1.0
    c433 = next(c for c in chunks if c.clause == "4.3.3")
    assert c433.parent_id == "R-12-r3-4.3" and "24 часов" in c433.text
    assert c433.section_path == ["R-12 ред. 3", "4 Вибрационный контроль", "4.3 Нормы вибрации"]


# --- triage and replay -------------------------------------------------------------------------
def test_triage():
    assert triage(DOCS_DIR / "R-12_rev3.pdf").kind == "digital"
    assert triage(DOCS_DIR / "R-12_rev3_scan.pdf").kind == "scan"
    assert triage(DOCS_DIR / "PID-R12-001.png").kind == "image"


def test_images_are_hashed_in_logs(tmp_path):
    img = Image.new("L", (40, 20), 255)
    msgs = [vision_message("read", img)]
    logged = loggable_messages(msgs)
    url = logged[0]["content"][1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;sha256:") and len(url) < 60
    other = [vision_message("read", Image.new("L", (40, 20), 0))]
    assert request_hash("m", msgs, None) != request_hash("m", other, None)
    log = CallLogger(tmp_path / "log.jsonl")
    resp = ChatResponse(content="OK", finish_reason="stop", model="m", profile="p", request_hash=request_hash("m", msgs, None))
    log.log(resp, msgs, None, {}, {})
    replay = ReplayClient(tmp_path / "log.jsonl")
    assert replay.chat(msgs).content == "OK"


# --- drawings ----------------------------------------------------------------------------------
def test_tiles_overlap_and_cover():
    parts = tiles(Image.new("L", (1000, 600), 255), rows=2, cols=2, overlap=0.15)
    assert [p.size for p in parts] == [(575, 345)] * 4  # 500 + 75 overlap, 300 + 45 overlap


def test_merge_tiles_title_block_from_corner():
    answers = [{"tags": ["P-212A", "PID-R12-001"], "drawing_number": "PID-R12-001", "revision": None, "date": None},
               {"tags": ["p 212a", "TT-213"], "drawing_number": "PID-R12-003", "revision": None, "date": None},
               None,
               {"tags": ["E-213"], "drawing_number": "PID-R12-002", "revision": "2", "date": "21.08.2024"}]
    merged = merge_tile_answers(answers)
    assert merged["tags"] == ["P-212A", "PID-R12-001", "TT-213", "E-213"]
    assert merged["drawing_number"] == "PID-R12-002" and merged["revision"] == "2"


def test_pid_samples():
    samples = load_json("pid.json")
    assert set(samples) == {"clean", "dense", "dense_scan"}
    assert len(samples["clean"]["tags"]) == 12 and len(samples["dense"]["tags"]) == 30
    for s in samples.values():
        assert (DATA_DIR / s["file"]).exists()
        assert len({normalize_tag(t) for t in s["tags"]}) == len(s["tags"])


# --- data --------------------------------------------------------------------------------------
def test_questions():
    qs = load_questions()
    assert len(qs) == 50 and sum(q["value"] is None for q in qs) == 8
    pages = load_gt_pages()
    for q in qs:
        if q["value"] is not None:
            assert answer_on_page(q, pages[(q["file"], q["page"])]), q["id"]


def test_answer_page_of_table3():
    q = next(q for q in load_questions() if q["id"] == "t03")
    assert q["page"] == load_json("tables.json")["R-12_rev3/table3"]["page"]
    assert json.loads(json.dumps(q))["value"] == "4,5"


def test_parse_dots_layout_full_and_cut():
    from plant_assistant.readers import parse_layout

    full = '[{"bbox": [1, 2, 3, 4], "category": "Text", "text": "4.3.3 \u0432 24 \u0447"}, {"bbox": [5, 6, 7, 8], "category": "Picture"}]'
    assert [b["category"] for b in parse_layout(full)] == ["Text", "Picture"]
    cut = full[:-30]  # the answer stopped at the token limit
    assert [b["category"] for b in parse_layout(cut)] == ["Text"]
