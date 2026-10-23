"""Build notebooks/practice_03_document_parsing.ipynb from the cells below.

The notebook is generated so that its source stays readable in code review.
Edit the cells here, then run:  uv run python scripts/build_notebook.py
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "practice_03_document_parsing.ipynb"

cells: list[tuple[str, str]] = []


def md(text: str) -> None:
    cells.append(("md", text.strip("\n")))


def code(text: str) -> None:
    cells.append(("code", text.strip("\n")))


# ---------------------------------------------------------------------------------------------
md(r"""
# Practice 3. Reading industrial documents: six readers, one regulation

**ML in Industry 2026 · Topic 3: Industrial documents**

## Goal

An engineer asks: *"After the bearing change, pump P-101 vibrates at 4.8 mm/s. Is this normal?"*
The answer is in Table 3 of regulation R-12, clause 4.3, and the plant archive has R-12 only as a **scan**.
Before any model can answer, something has to *read* that page. Today we measure how well different
readers do it, and we build the chunk store that practice 4 (RAG) will search.

```
scan of R-12 ──► reader ──► text, tables, clauses ──► chunks with metadata ──► store/  (input of practice 4)
                  │
   PyMuPDF · EasyOCR (or Tesseract) · Docling · dots.ocr (document VLM, replay)
   · a general VLM: Qwen 3.8 27B, Qwen3.5 4B, a frontier model (replay)
```

## Steps and what you should see

| Step | What we do | Expected result |
|------|------------|-----------------|
| 1 | Setup and smoke test | the model answers text **and** reads an image |
| 2 | The archive and triage | 2 digital PDFs, 3 scans, 2 images |
| 3 | One page, six readers | the same Table 3 page read six ways; the text layer of a scan is empty; a document VLM returns boxes |
| 4 | The whole scanned set | CER, numbers, length, repetitions, seconds per page |
| 5 | Tables | rows recovered per reader; one table in Markdown, HTML and JSON |
| 6 | A VLM on a drawing and a nameplate | tags with precision and recall on a clean fragment, a dense sheet and its scan; tiles; nameplate fields |
| 7 | Structure, chunks, metadata, revisions | a chunk store with two revisions of R-12 |
| 8 | Mini-eval end-to-end | 50 questions, accuracy with interval, paired comparison, where answers were lost |

## Success criteria

- `store/chunks.jsonl` has all documents; every chunk has `doc_id`, `revision` (for R-12), `clause`, `page`.
- Your reading table (step 4) and your mini-eval table (step 8) are close to `reference/results.md`;
  the paired comparison gives the same verdicts.

## Live or replay

Every model call goes through one OpenAI-compatible client. If your model answers, the notebook calls it
**live**. If not, it **replays** the lecturer's saved answers for exactly the same requests.
Heavy readers (EasyOCR, Docling) do not run in the notebook: their outputs are in `data/parsed/`, and
`scripts/run_readers.py` reproduces them on your machine. dots.ocr and the frontier model run on the
lecturer's key: you always see their saved answers.

## Common problems

| Problem | What to do |
|---------|-----------|
| `Mode: REPLAY` but you have the course token | check `COURSE_LLM_BASE_URL` and `COURSE_LLM_API_KEY` in `.env`; run `uv run python -m plant_assistant.smoke_test course` |
| the smoke test passes text but fails the image | the model is not a VLM; for Ollama check `ollama show <model>` (needs "vision") |
| `ReplayMiss` | you changed a prompt, a page render or a saved reader output: replay works only for unchanged requests |
| Tesseract: `TesseractNotFoundError` or no `rus` | install Tesseract and the Russian language data (README, "Readers on your machine") |
| Docling or EasyOCR download weights on the first run | expected; run `scripts/run_readers.py` once before the session, or use the saved outputs |
| Docling takes minutes per document on CPU | expected (about 30 s per page on the lecturer's CPU); use `data/parsed/docling.jsonl` |
| a VLM answer is empty, `finish_reason = length` | thinking mode is on and used all tokens: keep it off for transcription |
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 0. Where we are going

The final tables of this practice, measured by the lecturer. At the end you will know where every number
comes from.
""")
code(r"""
import pandas as pd
from IPython.display import HTML, Markdown, display

from plant_assistant import REFERENCE_DIR

pd.set_option("display.max_colwidth", 200)
pd.set_option("display.float_format", "{:.3f}".format)
display(Markdown("**Reading the scans** (CER: lower is better; numbers: share kept)"))
display(pd.read_csv(REFERENCE_DIR / "reading.csv", index_col=0))
display(Markdown("**Mini-eval: 50 questions about the scans** (same model answers, only the reader changes)"))
display(pd.read_csv(REFERENCE_DIR / "qa.csv", index_col=0))
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 1. Setup and smoke test

Profiles are in `models.toml`, secrets in `.env` (as in practice 1). All profiles of this practice are VLMs:
the course Qwen 3.8 27B, the local Qwen3.5 4B in Ollama, and the lecturer's frontier model (replay only).
""")
code(r"""
import json
import os

from plant_assistant import DOCS_DIR, REPLAY_DIR, RUNS_DIR
from plant_assistant.config import get_profile, list_profiles, load_env
from plant_assistant.images import vision_message
from plant_assistant.llm import CallLogger, LLMClient, ReplayClient
from plant_assistant.smoke_test import test_image

load_env()
PROFILE = os.environ.get("PRACTICE_PROFILE", "course")  # course or local
print("PROFILE =", PROFILE)
pd.DataFrame([{"profile": k, "description": v} for k, v in list_profiles().items()])
""")
code(r"""
def model_reads_images(profile_name: str) -> bool:
    # One text call and one image call. No network retries: if the model is not there, we want to know now.
    try:
        client = LLMClient(get_profile(profile_name), max_network_retries=0)
    except ValueError as exc:  # profile not configured (no URL or key)
        print(f"{profile_name}: {exc}")
        return False
    seen = client.chat([vision_message("What is written in the image? Answer with the text only.", test_image())])
    print(f"{profile_name}: {seen.content.strip()[:30]!r} ({seen.latency_s:.1f}s)" if seen.ok else seen.error)
    return seen.ok and "417" in seen.content


LIVE = model_reads_images(PROFILE)
print("Mode:", f"LIVE calls to {PROFILE}" if LIVE else "REPLAY of the lecturer's saved answers (course model)")


def client_for(task: str):
    # Live model if it answers, otherwise the lecturer's saved answers of the course model for this task.
    if LIVE:
        return LLMClient(get_profile(PROFILE), logger=CallLogger(RUNS_DIR / f"my-{PROFILE}" / f"{task}.jsonl"))
    return ReplayClient(REPLAY_DIR / f"{task}_course.jsonl")
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 2. The archive and triage

What the plant has: two born-digital revisions of R-12, a scan of revision 3 (the paper copy in the
control room), two scanned pump datasheets (one from a Chinese supplier), a P&ID fragment and a photo of a
nameplate. All documents are **synthetic**: R-12, the plant and the numbers are invented for the course.

Triage decides the route of every file: a born-digital PDF goes to the text layer, a scan needs recognition.

> **Predict, then run.** How many of the 7 files will be detected as scans?
""")
code(r"""
from plant_assistant.docs import load_registry, triage_archive

registry = pd.DataFrame(load_registry())
tri = pd.DataFrame([t.__dict__ for t in triage_archive()])
registry.merge(tri[["file", "chars_per_page", "images", "kind"]], on="file")
""")
md(r"""
Two signals are enough here: characters in the text layer and pictures on the page. A real archive also has
**mixed** files (a scan with a poor OCR layer, a digital PDF with scanned attachments): there triage is per page.
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 3. One page, six readers

Page 4 of the scanned R-12: Table 3 with the vibration limits, merged cells, the page is slightly rotated.
""")
code(r"""
from IPython.display import Image as IPImage

from plant_assistant.images import jpeg_bytes, render_page


def show_jpeg(img) -> None:
    display(IPImage(jpeg_bytes(img, quality=80)))  # a scan as JPEG keeps the notebook small


SCAN = DOCS_DIR / "R-12_rev3_scan.pdf"
PAGE = 4
page_img = render_page(SCAN, PAGE, dpi=110)
show_jpeg(page_img.crop((0, 0, page_img.width, int(page_img.height * 0.32))))
""")
md(r"""
**Reader 1: PyMuPDF**, the text layer. It runs in milliseconds and is perfect on born-digital PDFs.
""")
code(r"""
from plant_assistant.readers import read_pymupdf

text_layer = read_pymupdf(SCAN)[PAGE - 1]
print(f"characters on page {PAGE}: {len(text_layer['text'].strip())}  ({text_layer['seconds'] * 1000:.1f} ms)")
""")
md(r"""
Zero characters: a scan has no text layer. If triage had sent this file to PyMuPDF, the system would index an
empty page **without any error**. This is the first silent loss of the pipeline.

> **Poll.** Which reader will have the lowest CER on the scanned pages? (a) EasyOCR (b) Docling (c) a VLM (d) about the same

**Readers 2 and 3: a classic OCR engine and a document parser.** Their outputs are saved in `data/parsed/`
(EasyOCR and Docling need PyTorch and model weights; `scripts/run_readers.py` reproduces them).
""")
code(r"""
from plant_assistant.evaluate import READER_NAMES
from plant_assistant.readers import available_readers, load_outputs

READERS = [r for r in ["pymupdf", "tesseract", "easyocr", "docling", "vlm:dots-ocr", "vlm:local", "vlm:course",
                       "vlm:gpt-6.1-sol"]
           if r in available_readers()]
print("saved readers:", READERS)


def show_side_by_side(texts: dict, n_lines: int = 30) -> None:
    cols = "".join(
        f"<td style='vertical-align:top'><b>{name}</b><pre style='font-size:11px;white-space:pre-wrap'>"
        + "\n".join(t.splitlines()[:n_lines]).replace("<", "&lt;") + "</pre></td>" for name, t in texts.items())
    display(HTML(f"<table><tr>{cols}</tr></table>"))


ocr_readers = [r for r in ("tesseract", "easyocr", "docling") if r in READERS]
show_side_by_side({READER_NAMES[r]: load_outputs(r)[("R-12_rev3_scan.pdf", PAGE)]["text"] for r in ocr_readers})
""")
md(r"""
Look at Table 3. Where are the classes I, II, III? What happened to `2,8`? In which order are clauses 4.4.1 and
4.4.2? Every error here is **silent**: nothing crashes, the text just becomes a little wrong.

**Reader 4: a VLM.** The course Qwen 3.8 27B gets the page image and the prompt below. This call runs live.
""")
code(r"""
from plant_assistant.readers import TRANSCRIBE_PROMPT, read_vlm

print(TRANSCRIBE_PROMPT)
vlm_page = read_vlm(SCAN, client_for("transcribe"), pages=[PAGE], reader_name=f"vlm:{PROFILE}")[0]
print(f"\n--- {vlm_page['reader']}: {vlm_page['seconds']:.1f}s, {vlm_page.get('completion_tokens')} output tokens ---\n")
print(vlm_page["text"][:1800])
""")
code(r"""
from plant_assistant.docs import load_gt_pages
from plant_assistant.metrics import page_scores

gt = load_gt_pages()
truth = gt[("R-12_rev3_scan.pdf", PAGE)]
one_page = {r: load_outputs(r)[("R-12_rev3_scan.pdf", PAGE)]["text"] for r in READERS}
one_page["vlm (your run)"] = vlm_page["text"]
pd.DataFrame({r: page_scores(truth, t) for r, t in one_page.items()}).T[
    ["cer", "numbers", "numbers_exact", "numbers_value", "length_ratio", "repeated"]]
""")
md(r"""
- **CER** after removing the Markdown and HTML markup. It also counts a different order of table cells as errors.
- **numbers_exact / numbers_value**: share of the numbers of the page that the reader kept as written (`4,5`) and
  by value (`4.5` also counts). A gap between them means the reader *rewrote* numbers instead of reading them.
- **length_ratio, repeated**: the traces of the typical VLM failures: missing parts, invented parts, repetition loops.

**Reader 5: a document VLM.** dots.ocr (rednote hi lab) is a small model trained only to parse pages. With its own
layout prompt it returns every block of the page with a **category** and a **box**, and the text of the block
(tables as HTML). A general VLM gives text only. The box is what rule 1 of lecture 1, "show the source", needs:
a chunk can point to its place on the page. dots.ocr runs on the lecturer's server, so you see its saved answers.
""")
code(r"""
from plant_assistant.images import draw_layout
from plant_assistant.readers import DOTS_LAYOUT_PROMPT

dots = load_outputs("vlm:dots-ocr")[("R-12_rev3_scan.pdf", PAGE)]
print(f"{len(dots['layout'])} blocks, {dots['seconds']:.0f}s:", [b["category"] for b in dots["layout"]][:8], "...")
boxed = draw_layout(render_page(SCAN, PAGE, dpi=110), dots["layout"], dpi=110)
show_jpeg(boxed.crop((0, 0, boxed.width, int(boxed.height * 0.62))))
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 4. The whole scanned set

8 pages: R-12 revision 3 (6 pages), the datasheet of P-101 (Russian and English) and the datasheet of P-205
(Chinese). One of the R-12 pages is a "copy of a copy": faint, rotated, noisy.
""")
code(r"""
from plant_assistant.evaluate import page_table, reading_summary

pages = page_table(READERS)
reading_summary(pages)
""")
code(r"""
# CER per page: where does each reader break?
cer_by_page = pages.pivot_table(index=["doc", "page"], columns="reader", values="cer").round(3)
cer_by_page.style.background_gradient(axis=None, cmap="Reds", vmin=0, vmax=0.6).format("{:.3f}")
""")
md(r"""
> **Chat.** Look at the P-205 datasheet (Chinese). Did any VLM change a number or a unit? Type yes or no.
""")
code(r"""
p205 = ("DS-P-205_scan.pdf", 1)
show_side_by_side({READER_NAMES[r]: load_outputs(r)[p205]["text"] for r in ("easyocr", "vlm:local", "vlm:course")
                   if r in READERS}, n_lines=40)
pages[pages["doc"] == "P-205 (zh)"][["reader", "cer", "numbers_exact", "numbers_value", "length_ratio"]]
""")
md(r"""
The OCR engine was configured for Russian and English, like most Russian setups. On Chinese it does not fail
loudly: it returns confident garbage for the labels and correct digits. Note the tag: `P-205` may come back with
a **Cyrillic** `Р`. A VLM reads the labels, but check that it did not *translate* them or change the units.
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 5. Tables

In industrial documents the answer usually sits in a table. A table row is *recovered* when one row of the
output has the key cell (class `II`, tag `K-301`) and the right numbers in the right order.
""")
code(r"""
from plant_assistant.evaluate import table_summary

table_summary(["digital"] + READERS)
""")
md(r"""
The first row is the **text layer of the born-digital original**: every character is right (CER 0), and still
no table row is recovered, because every cell is on its own line. Text is not structure.

**One table in three representations.** We give the model Table 3 as Markdown (as many converters write it:
the merged cells are lost and the row of class II shifts to the left), as HTML (rowspan, colspan) and as JSON
records, and ask three questions about it.

> **Chat.** Which representation will give the correct answer for class II? Type md, html or json.
""")
code(r"""
from plant_assistant.docs import load_json
from plant_assistant.tables import to_html, to_json_records, to_markdown_as_parsed

t3 = load_json("tables.json")["R-12_rev3/table3"]
REPS = {"markdown (merged cells lost)": to_markdown_as_parsed(t3["cells"], 2),
        "html (rowspan, colspan)": to_html(t3["cells"]),
        "json records": to_json_records(t3["cells"], 2)}
print(REPS["markdown (merged cells lost)"])
""")
code(r"""
from plant_assistant.qa import RESPONSE_FORMAT, grade, messages, parse_answer

REP_QUESTIONS = [("What is the inspection limit of vibration velocity for class II machines?", "4,5", "mm/s"),
                 ("What is the stop limit of vibration velocity for class II machines?", "7,1", "mm/s"),
                 ("What is the inspection limit of vibration velocity for class IV machines?", "11,2", "mm/s")]
rep_client = client_for("representations")
rows = []
for name, table in REPS.items():
    for question, value, unit in REP_QUESTIONS:
        r = rep_client.chat(messages(question, t3["caption"] + "\n\n" + table), response_format=RESPONSE_FORMAT,
                            meta={"task": "representation", "representation": name})
        ans = parse_answer(r.content)
        ok, reason = grade({"value": value, "unit": unit}, ans)
        rows.append({"representation": name, "question": question[:60], "expected": value,
                     "answer": (ans or {}).get("value"), "correct": ok, "prompt tokens": r.prompt_tokens})
pd.DataFrame(rows)
""")
md(r"""
Two lessons. (1) The representation of a table is a design decision: HTML keeps merged cells and costs more
tokens, JSON records are best for parameter tables (one row = one item of equipment, a direct source for the
tools of topic 5). (2) A strong model can sometimes repair a broken table from the context: do not count on it,
measure it on your tables.
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 6. A VLM on a drawing and on a nameplate

What is realistic in 2026 without your own labelled drawings: **tags and the title block**. What is not:
the topology, which element is connected to which.

> **Chat.** The drawing has 12 equipment and instrument tags. How many will the VLM find? Type a number.
""")
code(r"""
from PIL import Image

pid_img = Image.open(DOCS_DIR / "PID-R12-001.png")  # the clean fragment of sheet 1
pid_img.resize((pid_img.width // 2, pid_img.height // 2))
""")
code(r"""
from plant_assistant import DATA_DIR
from plant_assistant.image_tasks import run_image_task, run_pid_tiles

pid_samples = load_json("pid.json")  # three samples: the clean fragment, a dense sheet, its scanned copy


def show_pid(result: dict) -> None:
    s = result["score"]
    print(f"{result['task']:9s} precision {s['precision']:.0%}, recall {s['recall']:.0%}, "
          f"title block {s['title_block']}/3, {result['prompt_tokens']} prompt tokens")
    print("   invented:", ", ".join(s["invented"]) or "—", "| missed:", ", ".join(s["missed"]) or "—")


pid = run_image_task(client_for("images"), "pid", DATA_DIR / pid_samples["clean"]["file"], pid_samples["clean"])
print("found:", sorted(pid["answer"]["tags"]))
print("title block:", {k: pid["answer"][k] for k in ("drawing_number", "revision", "date")})
show_pid(pid)
""")
md(r"""
Instrument bubbles show the letters and the number on two lines (`PT` over `101`): the model has to join them.
Tags are compared after normalization by code (case, spaces, Cyrillic look-alikes), as in practice 1.

**A real sheet is not a clean fragment.** Sheet 2 of the same plant: 30 tags in small print, a spare pair
`P-212A/B`, long function codes (`LSHH`, `PDIT`), vertical tags, a tag crossed by a signal line, and text that
is *not* a tag: line numbers, references to other drawings, a revision table with three dates.
Then its paper copy from the archive: scanned at a lower resolution, folded, stamped.

> **Chat.** Precision and recall on the scanned copy: above or below 90%? Which one drops more?
""")
code(r"""
dense = Image.open(DATA_DIR / pid_samples["dense_scan"]["file"])
show_jpeg(dense.resize((dense.width // 2, dense.height // 2)))
""")
code(r"""
for name in ("dense", "dense_scan"):
    print(f"--- {name}: {pid_samples[name]['about']}")
    show_pid(run_image_task(client_for("images"), "pid", DATA_DIR / pid_samples[name]["file"], pid_samples[name]))
""")
md(r"""
Look at the invented tags: most are not random, they are *plausible* tags (`PT-213` next to `TT-213`) or
other text read as a tag (`PID-R12-001` is the number of another drawing). Precision catches them; a quick look
at the list does not.

**Tiles.** Small print is lost when the model shrinks the whole sheet. Cut it into 2 × 2 tiles with overlap,
read every tile, merge the tags. The title block is read from the bottom right tile only: the other tiles
also "see" a drawing number, on the off-sheet connectors.
""")
code(r"""
show_pid(run_pid_tiles(client_for("images"), DATA_DIR / pid_samples["dense_scan"]["file"], pid_samples["dense_scan"]))
""")
md(r"""
Four tiles cost about four calls. Whether they pay off is a measurement, not a rule: compare recall, precision and
tokens with the whole sheet in the reference table below (in the lecturer's run tiles helped the local model
on the scan and hurt it on the sheet itself).

**Nameplate.** The Pydantic approach from practice 1, applied to a photo with glare and perspective.
""")
code(r"""
plate = Image.open(DOCS_DIR / "nameplate_P-101.jpg")
show_jpeg(plate.resize((plate.width // 2, plate.height // 2)))
np_result = run_image_task(client_for("images"), "nameplate", DOCS_DIR / "nameplate_P-101.jpg", load_json("nameplate.json"))
pd.DataFrame(np_result["score"]["fields"]).T
""")
code(r"""
# The lecturer's runs: the local 4B model, the course model and a frontier model
pd.read_csv(REFERENCE_DIR / "images.csv")
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 7. Structure, chunks, metadata, revisions

The store is the input of practice 4, so every chunk must be **citable**: document, revision, clause, page.
Steps (all rules, no model): metadata from the title page; running headers and footers removed; clause
numbers at the start of a line rebuild the tree (a number counts only if it continues the numbering, so
`720 часов` at the start of a wrapped line is not a clause); one chunk per clause; a table is one chunk
with its caption; headings become breadcrumbs and parents.

> **Poll.** How many chunks will clause 4.3 (with its sub-clauses and Table 3) produce? (a) 1 (b) 4 (c) 6
""")
code(r"""
from plant_assistant.readers import read_pymupdf_tables
from plant_assistant.store import chunk_regulation

r12 = read_pymupdf_tables(DOCS_DIR / "R-12_rev3.pdf")  # born-digital: text layer + table finder, no OCR
meta, chunks = chunk_regulation([p["text"] for p in r12], "R-12_rev3.pdf", "pymupdf+tables", r12[0]["version"])
print(json.dumps(meta, ensure_ascii=False))
c43 = [c for c in chunks if (c.clause or "").startswith("4.3")]
pd.DataFrame([{"chunk_id": c.chunk_id, "type": c.chunk_type, "clause": c.clause, "page": c.page,
               "parent": c.parent_id, "breadcrumbs": c.breadcrumbs(), "text": c.text[:80]} for c in c43])
""")
code(r"""
table_chunk = next(c for c in c43 if c.chunk_type == "table")
print(table_chunk.text)
""")
md(r"""
The table finder of PyMuPDF brought the rows back (compare with step 5: the plain text layer had 0 rows).
It copied the merged cell "Насосы" into both of its rows, which is good for search, and the merged header
became `Граница виброскорости мм/с ,` plus an empty `Col4`. A table is one chunk with its caption, so the
model always sees the whole table. One more domain rule: the finder also takes the two-column section 3 for
a table; we keep only tables with a caption "Таблица N —" above them (see `readers.read_pymupdf_tables`).

**The same tree from the scans.** How much of the clause structure survives each reader?
""")
code(r"""
from plant_assistant.evaluate import structure_summary

structure_summary(["digital"] + READERS)
""")
md(r"""
**Build the store.** Both revisions of R-12 (born-digital: text layer + table finder) and both datasheets
(scans, as read by the course VLM).
The older revision stays in the store with `status = superseded` and `superseded_by = 3`: questions like
"what changed" need it, and topic 4 filters it out when it answers about the current rules.
""")
code(r"""
from plant_assistant import STORE_DIR
from plant_assistant.store import build_store

summary = build_store(scan_reader="vlm:course", folder=STORE_DIR)
print(summary)
docs = pd.read_json(STORE_DIR / "documents.jsonl", lines=True)
docs[["revision", "superseded_by"]] = docs[["revision", "superseded_by"]].astype("Int64")
docs[["doc_id", "revision", "effective_date", "status", "superseded_by", "source_file", "parser", "content_hash"]]
""")
code(r"""
store = pd.read_json(STORE_DIR / "chunks.jsonl", lines=True)
both = store[(store["chunk_type"] == "table") & store["text"].str.contains("Таблица 3")]
for _, ch in both.iterrows():
    row2 = [line for line in ch["text"].splitlines() if "|II|" in line.replace(" ", "")]
    print(f"revision {int(ch['revision'])} ({ch['status']}): {row2}")
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 8. Mini-eval end-to-end

50 questions about the scanned documents, built by the rules of lecture 2: each comes from a known clause,
table row or datasheet line; the answer is a number with a unit; 8 questions are negative ("not stated").
For each question the reader's text of the page with the answer goes to the model (search is topic 4),
the model answers in JSON, and **code** checks the number and the unit. The same model answers for every
reader, so a difference between readers is a difference in what they read.

`digital` is the ceiling: the text layer of the born-digital original.
""")
code(r"""
from plant_assistant.docs import load_questions

questions = load_questions()
print(pd.Series([q["slice"] for q in questions]).value_counts().to_dict())
pd.DataFrame(questions)[["id", "question", "value", "unit", "where", "slice"]].head(8)
""")
md(r"""
> **Predict, then run.** Docling against EasyOCR on these scans: better, worse or not different?

Answers come from the lecturer's run of the course model (replay, seconds). With the course token you can
run it live: set `QA_LIVE = True` (about 350 calls, 2–3 minutes).
""")
code(r"""
from plant_assistant.evaluate import qa_pairs, qa_summary, reader_pages
from plant_assistant.qa import run_qa

QA_LIVE = False
qa_client = client_for("qa") if QA_LIVE else ReplayClient(REPLAY_DIR / "qa_course.jsonl")
results = []
for reader in ["digital"] + READERS:
    results += run_qa(reader, reader_pages(reader), questions, qa_client, concurrency=8 if QA_LIVE else 1)
qa_summary(results)
""")
code(r"""
PAIRS = [("vlm:course", "docling"), ("docling", "easyocr"), ("vlm:course", "easyocr"), ("easyocr", "pymupdf"),
         ("vlm:dots-ocr", "docling"), ("vlm:dots-ocr", "vlm:course"), ("vlm:course", "vlm:local"),
         ("vlm:gpt-6.1-sol", "vlm:course"), ("digital", "vlm:course")]
qa_pairs(results, PAIRS)
""")
md(r"""
- An interval of ±10–14 points on 50 questions is normal: two strong readers will often be **not different**.
- `not_stated` questions are answered "correctly" even by the empty text layer: a negative question checks that
  the model does not invent, not that the reader reads. Look at the slices, not only at the total.
- Slices of 5–14 questions are shown as counts, not percentages.

**Where was the answer lost?** We already have the answer with the ground truth text: the `digital` row.
If the model answered a question right on the born-digital text and wrong on a reader's text, the reader lost
it (`reading`). If it was wrong on the born-digital text too, the model is the failing link (`model`).
`number_on_page` shows whether the right number was still on the reader's page: often it is, but in the wrong
row of a broken table.
""")
code(r"""
from plant_assistant.qa import attribute

res = pd.DataFrame(attribute([dict(r) for r in results]))
wrong = res[~res["correct"] & res["reader"].isin(["easyocr", "docling", "vlm:course"])]
wrong[["reader", "id", "slice", "question", "expected", "value", "unit", "reason", "number_on_page", "lost_at"]]
""")

# ---------------------------------------------------------------------------------------------
md(r"""
## 9. Results and next steps

**Check yourself:** `store/` is built; your tables in steps 4 and 8 match `reference/results.md` (live runs may
differ by a question or two: that is why we use intervals).

**What you built:** triage, six readers with one interface, reading metrics beyond CER (numbers, length,
repetitions), table scores and representations, a VLM on a drawing and a photo, structural chunks with
metadata and revisions, and an end-to-end mini-eval with paired comparison and the step where an answer was lost.

**Typical mistakes** (check before you trust a number):
- a scan was sent to the text layer: empty pages, no error;
- Tesseract without the `rus` language data, or a parser whose OCR language was not set;
- a VLM answer wrapped in a ```` ```markdown ```` block (strip it, as `readers.strip_fence` does);
- a VLM "fixed" `4,8` to `4.8` or `мм/с` to `mm/s`: compare numbers by value, but count such rewrites;
- low CER taken as "no invented text": look at the length ratio and the numbers;
- a tag reader judged on one clean fragment: measure it on a dense sheet and on a scan, as in step 6;
- a table cut into pieces by chunking, or a clause number broken by OCR (`4.З.2`) that breaks the tree;
- "reader X is better" from one run or from two averages without a paired comparison.

**Homework (optional, see the README for the full list):**
1. Preprocessing: deskew and binarize the scans before EasyOCR or Tesseract; measure CER per page before and after.
2. "CER on what matters": CER only on numbers, units and tags. Does the ranking of readers change?
3. Routing: the cheapest reader for every page, the VLM only where a check fails (empty text, few numbers,
   a table not recovered). Accuracy and cost against "VLM on every page".
4. Your own document: an open technical document, 5 questions with answers, the mandatory readers, accuracy.

**Next practice (topic 4):** `store/` becomes a search index; the questions of step 8 become a retrieval eval.
""")


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [nbf.v4.new_markdown_cell(t) if kind == "md" else nbf.v4.new_code_cell(t) for kind, t in cells]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    build()
