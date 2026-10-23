"""Build the reference results of practice 3 (lecturer, once, after scripts/run_readers.py).

    uv run python scripts/build_reference.py            # everything
    uv run python scripts/build_reference.py --skip-qa  # no model calls for the mini-eval

It needs the course model (profile "course") for the mini-eval, the table representations and the images;
the local model (Ollama) and the lecturer's key for the frontier model are used for the images only.
Every model call is logged to data/replay/*.jsonl, so the notebook can replay it without a model.

Writes: reference/*.csv, reference/results.md, reference/store/ (the chunk store: input of practice 4).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from plant_assistant import DATA_DIR, DOCS_DIR, REFERENCE_DIR, REPLAY_DIR  # noqa: E402
from plant_assistant.config import get_profile, load_env  # noqa: E402
from plant_assistant.docs import load_json, load_questions  # noqa: E402
from plant_assistant.evaluate import (READER_NAMES, page_table, qa_pairs, qa_summary, reader_pages,  # noqa: E402
                                      reading_summary, structure_summary, table_summary)
from plant_assistant.image_tasks import run_image_task, run_pid_tiles  # noqa: E402
from plant_assistant.llm import CallLogger, LLMClient, ReplayClient  # noqa: E402
from plant_assistant.qa import RESPONSE_FORMAT, attribute, grade, messages, parse_answer, run_qa  # noqa: E402
from plant_assistant.readers import available_readers  # noqa: E402
from plant_assistant.store import build_store  # noqa: E402
from plant_assistant.tables import to_html, to_json_records, to_markdown_as_parsed  # noqa: E402

ORDER = ["pymupdf", "tesseract", "easyocr", "docling", "vlm:dots-ocr", "vlm:local", "vlm:course", "vlm:gpt-6.1-sol"]
PAIRS = [("vlm:course", "docling"), ("docling", "easyocr"), ("vlm:course", "easyocr"), ("easyocr", "pymupdf"),
         ("vlm:dots-ocr", "docling"), ("vlm:dots-ocr", "vlm:course"), ("vlm:course", "vlm:local"),
         ("vlm:gpt-6.1-sol", "vlm:course"), ("digital", "vlm:course"), ("tesseract", "easyocr")]
REPRESENTATION_QUESTIONS = [
    ("What is the inspection limit of vibration velocity for class II machines?", "4,5", "mm/s"),
    ("What is the stop limit of vibration velocity for class II machines?", "7,1", "mm/s"),
    ("What is the inspection limit of vibration velocity for class IV machines?", "11,2", "mm/s"),
]


class ReplayOrLive:
    """Saved answers where they exist; a live call (appended to the same replay file) for a new request.
    Adding a reader then costs only the new calls."""

    def __init__(self, profile: str, log: Path):
        self.replay = ReplayClient(log, strict=False)
        self.live = LLMClient(get_profile(profile), logger=CallLogger(log))
        self.profile_name, self.model = self.live.profile_name, self.live.model

    def chat(self, messages, response_format=None, meta=None):
        r = self.replay.chat(messages, response_format, meta)
        return r if r.error != "ReplayMiss" else self.live.chat(messages, response_format, meta)


def client(profile: str, log_name: str, replay: bool = False):
    """A live client that writes a fresh replay file, or (replay=True) the saved answers of an earlier run
    plus live calls for requests that are not saved yet."""
    log = REPLAY_DIR / log_name
    if replay and log.exists():
        return ReplayOrLive(profile, log)
    log.unlink(missing_ok=True)
    return LLMClient(get_profile(profile), logger=CallLogger(log))


def representations() -> dict[str, str]:
    t = load_json("tables.json")["R-12_rev3/table3"]
    caption = t["caption"]
    return {"markdown (merged cells lost)": caption + "\n\n" + to_markdown_as_parsed(t["cells"], t["header_rows"]),
            "html (rowspan, colspan)": caption + "\n\n" + to_html(t["cells"]),
            "json records": caption + "\n\n" + to_json_records(t["cells"], t["header_rows"])}


def run_representations(c) -> pd.DataFrame:
    rows = []
    for name, text in representations().items():
        for question, value, unit in REPRESENTATION_QUESTIONS:
            r = c.chat(messages(question, text), response_format=RESPONSE_FORMAT,
                       meta={"task": "representation", "representation": name})
            ans = parse_answer(r.content)
            ok, reason = grade({"value": value, "unit": unit}, ans)
            rows.append({"profile": c.profile_name, "representation": name, "question": question, "expected": value,
                         "answer": (ans or {}).get("value"), "correct": ok, "reason": reason,
                         "prompt_tokens": r.prompt_tokens})
    return pd.DataFrame(rows)


def md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    return df.to_markdown(floatfmt=floatfmt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-qa", action="store_true")
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--replay", action="store_true",
                    help="reuse saved model answers; call the model only for requests that are not saved yet")
    args = ap.parse_args()
    load_env()
    REFERENCE_DIR.mkdir(exist_ok=True)
    readers = [r for r in ORDER if r in available_readers()]
    print("readers:", readers)

    # 1. reading metrics, tables, structure (no model calls)
    pages = page_table(readers)
    pages.to_csv(REFERENCE_DIR / "reading_pages.csv", index=False)
    reading = reading_summary(pages)
    reading.to_csv(REFERENCE_DIR / "reading.csv")
    tables = table_summary(["digital"] + readers)
    tables.to_csv(REFERENCE_DIR / "tables.csv")
    structure = structure_summary(["digital"] + readers)
    structure.to_csv(REFERENCE_DIR / "structure.csv")

    # 2. mini-eval: 50 questions x readers, course model answers, code grades
    questions = load_questions()
    qa_path = REFERENCE_DIR / "qa_answers.jsonl"
    if not args.skip_qa:
        c = client("course", "qa_course.jsonl", args.replay)
        results = []
        for reader in ["digital"] + readers:
            t0 = time.perf_counter()
            res = run_qa(reader, reader_pages(reader), questions, c, concurrency=8)
            results += res
            print(f"qa {reader:18s} {sum(r['correct'] for r in res)}/{len(res)} in {time.perf_counter() - t0:.0f}s")
        attribute(results)
        with open(qa_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        rep = pd.concat([run_representations(client(p, f"representations_{p}.jsonl", args.replay))
                         for p in ("course", "local")])
        rep.to_csv(REFERENCE_DIR / "representations.csv", index=False)
    results = [json.loads(x) for x in qa_path.read_text(encoding="utf-8").splitlines()]
    qa = qa_summary(results)
    qa.to_csv(REFERENCE_DIR / "qa.csv")
    pairs = qa_pairs(results, PAIRS)
    pairs.to_csv(REFERENCE_DIR / "qa_pairs.csv", index=False)
    rep = pd.read_csv(REFERENCE_DIR / "representations.csv")

    # 3. a VLM on a drawing and on a photo
    img_path = REFERENCE_DIR / "images.jsonl"
    if not args.skip_images:
        out = []
        samples, truth_np = load_json("pid.json"), load_json("nameplate.json")
        for profile in ("local", "course", "gpt-6.1-sol"):
            if get_profile(profile).missing_settings():  # no key here: only the answers saved earlier
                c = ReplayClient(REPLAY_DIR / f"images_{profile}.jsonl", strict=False)
                print(f"images {profile}: no key, saved answers only")
            else:
                c = client(profile, f"images_{profile}.jsonl", args.replay)
            runs = []
            for name, truth in samples.items():
                runs.append({"sample": name, **run_image_task(c, "pid", DATA_DIR / truth["file"], truth)})
                if name != "clean":  # small print: the same sheet in 2 x 2 tiles
                    runs.append({"sample": name, **run_pid_tiles(c, DATA_DIR / truth["file"], truth)})
            runs.append({"sample": "nameplate",
                         **run_image_task(c, "nameplate", DOCS_DIR / "nameplate_P-101.jpg", truth_np)})
            missed = [f"{r['sample']}/{r['task']}" for r in runs if "ReplayMiss" in (r["error"] or "")]
            if missed:
                print(f"images {profile}: not run yet, run again with the key: {', '.join(missed)}")
            out += [r for r in runs if "ReplayMiss" not in (r["error"] or "")]
            print(f"images {profile}: done")
        img_path.write_text("\n".join(json.dumps(o, ensure_ascii=False) for o in out) + "\n", encoding="utf-8")
    images = [json.loads(x) for x in img_path.read_text(encoding="utf-8").splitlines()]
    img_rows = []
    for o in images:
        s = o["score"]
        if o["task"] in ("pid", "pid_tiles"):
            detail = (f"tags {s['correct']}/{s['found']} correct, precision {s['precision']:.0%}, "
                      f"recall {s['recall']:.0%}; invented: {', '.join(s['invented']) or '—'}; "
                      f"missed: {', '.join(s['missed']) or '—'}; title block {s['title_block']}/3")
        else:
            wrong = [f"{k}={v['got']}" for k, v in s["fields"].items() if not v["ok"]]
            detail = f"fields {s['accuracy']:.0%}; wrong: {', '.join(wrong) or '—'}"
        img_rows.append({"profile": o["profile"], "sample": o.get("sample", o["task"]), "task": o["task"],
                         "result": detail, "latency_s": o["latency_s"], "prompt_tokens": o["prompt_tokens"]})
    img_df = pd.DataFrame(img_rows)
    img_df.to_csv(REFERENCE_DIR / "images.csv", index=False)

    # 4. the chunk store
    store = build_store(folder=REFERENCE_DIR / "store")
    print("store:", store)

    # 5. results.md
    rep_summary = rep.groupby(["profile", "representation"])["correct"].sum().unstack(0).astype(int)
    text = f"""# Practice 3: reference results

Generated by `scripts/build_reference.py`. Documents are synthetic; numbers are illustrative.
Scanned set: R-12 revision 3 (6 pages), datasheet PS-101 (Russian), datasheet DS-P-205 (Chinese); 8 pages.

## 1. Reading the scans

CER after removing Markdown/HTML markup; numbers kept as written and by value; length ratio to the
original; repeated long lines; seconds per page on the lecturer's machine (CPU for OCR and Docling,
course server and RTX 4090 for VLMs).

{md(reading)}

## 2. Tables: rows recovered (key cell + numbers in order)

{md(tables)}

## 3. Structure: clause tree rebuilt from the scanned regulation

{md(structure)}

## 4. Mini-eval: 50 questions, answers by the course model, checked by code

{md(qa, ".2f")}

Paired comparison on the same questions (95% bootstrap interval):

{pairs.to_markdown(index=False)}

## 5. One table in three representations: correct answers out of {len(REPRESENTATION_QUESTIONS)}

{rep_summary.to_markdown()}

{rep[["profile", "representation", "question", "expected", "answer", "correct"]].to_markdown(index=False)}

## 6. VLM on a drawing and on a nameplate photo

{img_df.to_markdown(index=False)}

## 7. Chunk store

{json.dumps(store, ensure_ascii=False, indent=1)}
"""
    (REFERENCE_DIR / "results.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
