# Practice 3. Reading industrial documents: six readers, one regulation

Part of the course **ML in Industry 2026** (topic 3: industrial documents). The project is the
**Plant Engineer Assistant**. After this practice it reads scanned regulations and datasheets, measures how
much each reader loses, and keeps the result as a chunk store with metadata: the input of practice 4 (RAG).

The running example comes from lectures 1–3: *"Pump P-101 vibrates at 4.8 mm/s after a bearing change.
Is this normal?"* The limit is in Table 3 of regulation R-12, clause 4.3, and the plant has R-12 as a scan.

Everything you need is in this folder: synthetic documents, ground truth, the saved outputs of heavy readers,
the lecturer's saved model answers (replay files) and reference results.

## What is inside

| Path | What it is |
|------|------------|
| `data/docs/` | the plant archive: R-12 rev. 2 and rev. 3 (born-digital), a scan of rev. 3, two scanned datasheets (P-101 in Russian, P-205 in Chinese), a P&ID fragment, a nameplate photo |
| `data/documents.csv` | registry of the archive |
| `data/drawings/` | two hard samples for the tag reader, not in the archive: a dense P&ID sheet in small print and its scanned paper copy |
| `data/ground_truth/` | page texts of the born-digital originals, tables, clause tree, 50 questions, P&ID tags of the three drawing samples, nameplate fields; `twins/` are the born-digital originals of the scanned datasheets |
| `data/parsed/<reader>.jsonl` | saved outputs of every reader on the 8 scanned pages, with seconds per page |
| `data/replay/*.jsonl` | the lecturer's model calls: page transcription (incl. dots.ocr layout), mini-eval answers, table representations, images |
| `reference/` | reference tables (`results.md`, `*.csv`), the reference chunk store `reference/store/`, the executed notebook |
| `scripts/` | `generate_documents.py` (documents and ground truth), `run_readers.py`, `build_reference.py`, `build_notebook.py` |
| `src/plant_assistant/` | `readers.py`, `metrics.py`, `tables.py`, `store.py`, `qa.py`, `image_tasks.py`, the client from practice 1 |

All documents are **synthetic** and marked as teaching material. R-12, the plant, the pumps and Table 3
are invented; Table 3 is in the style of ISO 20816 but its values are not from the standard.

## Reproduce the practice (30–60 minutes)

### 1. Install

Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/) (or plain `pip`).

```bash
git clone https://github.com/Industrial-AI-Research-Lab/course-ml-for-industry-data.git
cd course-ml-for-industry-data/practice-03-document-parsing
uv sync --extra notebook --extra dev
uv run pytest        # 28 tests, no network
```

This is enough for the notebook: heavy readers are replayed from `data/parsed/`.

### 2. Choose a model

| Profile | Model | What you need |
|---------|-------|---------------|
| `course` (default) | Qwen 3.8 27B, multimodal, course server | URL and token from the lecturers |
| `local` | Qwen3.5 4B, Q4, vision, via Ollama | `ollama pull qwen3.5:4b-q4_K_M` (3.3 GB), 8 GB RAM |
| none of these | replay of the lecturer's runs | nothing: the notebook switches to replay automatically |

```bash
cp .env.example .env          # fill in COURSE_LLM_* 
uv run python -m plant_assistant.smoke_test course     # text and an image; ends with OK
uv run python -m plant_assistant.smoke_test local
```

### 3. Run the notebook

```bash
uv run jupyter lab notebooks/practice_03_document_parsing.ipynb
```

Run all cells from top to bottom. `PRACTICE_PROFILE=local` switches the live calls to Ollama.
Live calls in the notebook: one page transcription, 9 questions about table representations,
four images (three P&ID samples and the nameplate) and one scan in 4 tiles.
The mini-eval (350 answers) is replayed; set `QA_LIVE = True` in step 8 to run it live with the course model.

### 4. Readers on your machine (mandatory: Docling and one OCR engine)

```bash
uv sync --all-extras                                   # + Docling, EasyOCR (PyTorch, ~2 GB)
uv run python scripts/run_readers.py --readers pymupdf easyocr docling
uv run python scripts/run_readers.py --readers tesseract   # if Tesseract is installed
uv run python scripts/run_readers.py --readers vlm:course   # 8 pages, ~1 minute
```

Your outputs overwrite `data/parsed/<reader>.jsonl` (git shows the difference: compare with the lecturer's run).
The first run of EasyOCR and Docling downloads model weights. On a CPU expect about 20 s per page for EasyOCR
and Docling. **Tesseract** is optional: install it with your package manager (`apt install tesseract-ocr
tesseract-ocr-rus`, `brew install tesseract tesseract-lang`, or the Windows installer from UB Mannheim) and
check `tesseract --list-langs` shows `rus`.

## Expected results

See `reference/results.md`. In short (lecturer's run, 8 scanned pages, 50 questions):

- the text layer of a scan is empty: PyMuPDF answers only the "not stated" questions;
- a classic OCR engine keeps most numbers but breaks tables, roman numerals and Chinese labels;
- Docling with default settings reorders clauses and drops text on the faint page;
- VLMs read the scans almost like the born-digital text; a 4B local VLM is close to the 27B course model;
- the document VLM dots.ocr reads as well as the general VLMs and also returns a box and a category for every block
  (the base for "show the source"); it runs on the lecturer's key, so you get its saved outputs;
- paired comparison: some differences are clear, some are "not different" on 50 questions;
- P&ID tags: every model reads the clean fragment without errors; on the dense sheet and especially on its scan
  the local 4B model misses tags and invents plausible ones, and tiles raise recall on the scan but not precision.

Live numbers may differ by a page or a question from the reference: models and OCR engines are not fully
deterministic, and your CPU or GPU changes the seconds per page.

## Success criteria

- `store/chunks.jsonl` and `store/documents.jsonl` exist; every R-12 chunk has `doc_id`, `revision`, `clause`, `page`;
  revision 2 has `status = superseded`, `superseded_by = 3`.
- Your reading table and mini-eval table are close to `reference/results.md`; the paired verdicts are the same.

## Optional tasks (for better understanding)

Each is a section you add to the notebook or a script; a team can take any of them into its project.

1. **Preprocessing.** Deskew and binarize the scans before EasyOCR or Tesseract; CER per page before and after.
   Expected: a large gain on the rotated, faint page, almost none on clean pages.
2. **CER on what matters.** CER only on numbers, units and tags. Does the ranking of readers change?
3. **Routing.** The cheapest reader for every page and the VLM only where a check fails (empty text, few
   numbers, a table not recovered). Accuracy and cost against "VLM on every page".
4. **Diff of revisions.** Find the changed clauses and the changed limit between R-12 rev. 2 and rev. 3 in the store;
   write the list of changes into the document metadata.
5. **Metadata by an LLM.** Extract number, revision, date and scope of R-12 with the schema approach of practice 1,
   from the scan; compare with the rules in `store.title_metadata`.
6. **Chunking ablation.** Fixed 500 tokens / by clause / by clause with whole tables and parents, against
   the 50 questions with the chunk found by the clause. Where is the difference?
7. **VLM against OCR+LLM on a table.** Send the Table 3 page to the VLM as an image with the question,
   against OCR → HTML → LLM. Several runs; cost and time per page.
8. **Tiling a drawing.** Step 6 uses 2 × 2 tiles with 15% overlap. Try 3 × 3 and other overlaps on the dense
   sheet and its scan: recall, precision and tokens against the whole image. Add a rule by code that drops
   drawing numbers and line numbers from the tag list: how much precision does it win back?
9. **Another parser.** Marker or MinerU: add it as a reader, measure, compare in pairs with Docling.
10. **Your own document.** An open technical document with a permissive licence, 5 questions with answers,
    the mandatory readers, accuracy. This is the most useful step for your project.
11. **A rare language: VLM against the engine.** P-205 datasheet: Tesseract with `chi_sim`, the local VLM, the
    course VLM. CER, numbers and units, length ratio: did the VLM translate the labels?
