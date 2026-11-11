"""Native LangChain messages with exact-request replay of course model responses."""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import messages_from_dict
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from .tools import ROOT

REPLAY = ROOT / 'data/replay/course.jsonl'


class ReplayMiss(ValueError):
    pass


def canonical(messages):
    """Keep all model-visible content and tool links, omit transport metadata."""
    rows = []
    for m in messages:
        row = dict(type=m.type, content=m.content)
        for key in ('tool_calls', 'tool_call_id', 'name'):
            value = getattr(m, key, None)
            if value:
                row[key] = value
        rows.append(row)
    return rows


class CourseModel(BaseChatModel):
    mode: Literal['replay', 'live'] = 'live'
    model_id: str = 'iairlab/qwen3.8-27b'
    replay_path: Path = REPLAY
    tool_schemas: list[dict] = Field(default_factory=list)
    choice: Any = None
    calls: list[dict] = Field(default_factory=list, exclude=True)

    @property
    def _llm_type(self):
        return 'course-exact-replay' if self.mode == 'replay' else 'course-live'

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        # Both the explicit graph and create_agent use this method.
        return self.model_copy(update=dict(tool_schemas=[convert_to_openai_tool(t) for t in tools],
                                           choice=tool_choice))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        request = dict(model=self.model_id, messages=canonical(messages), tools=self.tool_schemas,
                       tool_choice=self.choice, stop=stop)
        digest = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        if self.mode == 'replay':
            entries = [json.loads(line) for line in self.replay_path.read_text().splitlines()]
            saved = next((r for r in entries if r['request_hash'] == digest), None)
            if saved is None:
                raise ReplayMiss('No recording matches this request. Restore the supplied question and tools, '
                                 'or use live mode. No network request was made.')
            message = messages_from_dict([saved['response']])[0]
            self.calls.append(dict(mode='replay', model=self.model_id, recorded_seconds=saved['seconds'],
                                   usage=saved['usage'], request_hash=digest))
        else:
            from dotenv import load_dotenv
            from langchain_openai import ChatOpenAI
            # Only live mode reads credentials. COURSE_ENV_FILE can select an existing env file.
            load_dotenv(os.environ.get('COURSE_ENV_FILE', str(ROOT / '.env')), override=False)
            url = os.getenv('COURSE_LLM_BASE_URL') or os.getenv('LITE_LLM_URL')
            token = os.getenv('COURSE_LLM_API_KEY') or os.getenv('LITE_LLM_TOKEN_QWEN')
            if not url or not token:
                raise ValueError('Live mode needs COURSE_LLM_BASE_URL and COURSE_LLM_API_KEY.')
            client = ChatOpenAI(model=self.model_id, base_url=url, api_key=token, temperature=0,
                                max_tokens=1400, timeout=90, max_retries=0,
                                extra_body={'chat_template_kwargs': {'enable_thinking': False}})
            if self.tool_schemas:
                options = dict(tool_choice=self.choice) if self.choice else {}
                client = client.bind_tools(self.tool_schemas, **options)
            started = time.perf_counter()
            message = client.invoke(messages, stop=stop)
            seconds = round(time.perf_counter() - started, 3)
            usage = message.usage_metadata or {}
            self.calls.append(dict(mode='live', model=self.model_id, recorded_seconds=seconds,
                                   usage=usage, request_hash=digest))
        return ChatResult(generations=[ChatGeneration(message=message)])


def make_model(mode=None):
    mode = mode or os.getenv('PRACTICE_MODE', 'live')
    if mode not in ('replay', 'live'):
        raise ValueError('Choose replay or live.')
    if mode != 'replay':
        from dotenv import load_dotenv
        load_dotenv(os.environ.get('COURSE_ENV_FILE', str(ROOT / '.env')), override=False)
    manifest = ROOT / 'data/replay/manifest.json'
    model_id = json.loads(manifest.read_text())['model'] if mode == 'replay' and manifest.exists() else \
        os.getenv('COURSE_LLM_MODEL', 'iairlab/qwen3.8-27b')
    return CourseModel(mode=mode, model_id=model_id)
