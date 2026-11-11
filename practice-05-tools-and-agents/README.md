# Practice 5. Tools, workflows and agents

Investigate a change in production quality with three read-only tools.
Compare a LangGraph workflow with an agent that chooses its tools.
Allow 60 to 90 minutes. All data is synthetic. No model training is needed.

## Start

Use Python 3.12 and uv. Run from this practice folder.

```bash
uv sync --locked --all-extras
```

Copy `.env.example` to `.env` in this folder. Enter the course URL, token and model.
Keep the token out of the notebook. If you already have an env file elsewhere,
set `COURSE_ENV_FILE` to its path before starting Jupyter.

```bash
uv run --locked jupyter lab notebooks/practice_05_tools_and_agents.ipynb
```

Live mode is the default and uses the course model with native tool calling.
Select the practice environment as the notebook kernel. Read and run one section
at a time with Shift+Enter. Responses can differ between runs.

The notebook includes source views, step controls and three written checkpoints.
Keep your answers in the "Your answer" cells and save a personal copy.
Buttons need a running kernel in both live and replay modes.

## Work offline

After installing dependencies, the supplied examples can run without keys or model
downloads. Start Jupyter with replay selected explicitly.

```bash
PRACTICE_MODE=replay uv run --offline --locked jupyter lab notebooks/practice_05_tools_and_agents.ipynb
```

Tools and graphs execute locally. Only model responses come from recordings.
A changed question, tool schema or observation raises `ReplayMiss`. Restore the
supplied example or use live mode. There is no automatic live fallback.
Use `reference/practice_05_executed.ipynb` to compare with a completed replay run.

## Check your work

```bash
uv run --offline --locked pytest -q
uv run --offline --locked python scripts/execute_notebook.py
```

The checks select replay, remove model credentials and block network connections.
The second command saves your executed copy in `runs/practice_05_executed.ipynb`.
Tests check calculations, graph branches and replay behavior. They do not certify
every sentence of a model response. Read its claims and sources yourself.

The tools are in `src/quality_assistant/tools.py`, the graphs in `systems.py`
and the notebook controls in `ui.py`. The two CSV files in `data/` are the inputs.
`scripts/generate_data.py` shows how the synthetic records were created.

Finish with a short explanation of when your task needs a workflow or an agent.
An optional extension adds one read-only calculation tool. Try it in live mode
with questions where it is useful, unnecessary or lacks data.
