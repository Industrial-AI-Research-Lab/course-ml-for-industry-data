# ML in Industry 2026: practices

Practical assignments of the course **Machine Learning Methods for Industrial Data Processing**
(master's programme "Big Data and Machine Learning", 2026). The focus of the year is **LLM-based systems
for industrial data**: how to build them, how to measure their quality and how to run them in production.

Every practice is a self-contained folder: code, a Jupyter notebook, synthetic data, the lecturer's saved
model answers (replay files) and reference results. You can run each practice in three ways:

- with a **local open-weight model** through [Ollama](https://ollama.com/download) on your laptop;
- with the **course model** (Qwen 3.8 27B on the course server; URL and token come from the lecturers);
- with **no model at all**: the notebook replays the lecturer's saved answers for exactly the same requests.

The practices grow one demo project, the **Plant Engineer Assistant**: an LLM system that helps a plant
engineer work with maintenance logs, regulations and equipment data.

## Practices

| # | Practice | Topic | What you build | Folder |
|---|----------|-------|----------------|--------|
| 1 | Structured extraction from maintenance logs | LLM-based systems in industry | typed schema, constrained output, business rules, retry, rules baseline, model comparison, call logs | [practice-01-structured-extraction/](practice-01-structured-extraction/) |

Practices for topics 2–7 (evaluation, industrial documents, RAG, tools and agents, LLMOps, safety and
security) will appear here after the corresponding lectures.

## How to start

```bash
git clone https://github.com/Industrial-AI-Research-Lab/course-ml-for-industry-data.git
cd course-ml-for-industry-data/practice-01-structured-extraction
```

Then follow the README of the practice folder: it lists the requirements, the model options
(including what to do on a weak laptop) and the expected results step by step.

## Conventions

- Python 3.11–3.13, dependencies pinned in `pyproject.toml` and `uv.lock`; [uv](https://docs.astral.sh/uv/) is recommended, plain `pip` works too.
- Models are called through one OpenAI-compatible client with named profiles in `models.toml`. Secrets live only in `.env` (never committed).
- Data is synthetic and ships with the repository. Numbers measured on it are illustrative.
- Everything a student needs works from Russia without a VPN or a foreign card. Frontier APIs appear only as replay files.
