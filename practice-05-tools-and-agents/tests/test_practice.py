import json
import socket

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from quality_assistant.models import ReplayMiss, make_model
from quality_assistant.systems import SYSTEM, build_short_agent, run_agent, run_workflow, SCENARIOS
from quality_assistant.tools import get_quality_summary, compare_periods
from quality_assistant.evaluation import verify_reference


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('A replay test attempted a network connection')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    for key in ('COURSE_LLM_API_KEY', 'COURSE_LLM_BASE_URL', 'LITE_LLM_TOKEN_QWEN',
                'LITE_LLM_URL', 'OPENAI_API_KEY', 'COURSE_ENV_FILE'):
        monkeypatch.delenv(key, raising=False)


def test_weighted_rate_and_empty_data():
    args = dict(line_id='L-2', start='2026-11-09', end='2026-11-09')
    result = get_quality_summary.invoke(args)
    assert result['produced'] == 3000 and result['defects'] == 140
    assert result['defect_rate_pct'] == 4.6667
    assert len(result['batch_ids']) == 4
    result = get_quality_summary.invoke(dict(args, line_id='L-3'))
    assert result['status'] == 'no_data' and result['defect_rate_pct'] is None


@pytest.mark.parametrize('args', [
    dict(line_id='L-9', start='2026-11-09', end='2026-11-09'),
    dict(line_id='L-2', start='2026-11-09', end='2026-11-08'),
    dict(line_id='L-2', start='2026-02-30', end='2026-11-09'),
])
def test_invalid_tool_arguments(args):
    assert get_quality_summary.invoke(args)['status'] == 'invalid_input'


def test_period_comparison_rejects_overlap():
    result = compare_periods.invoke(dict(line_id='L-2', before_start='2026-11-08',
        before_end='2026-11-09', after_start='2026-11-09', after_end='2026-11-09'))
    assert result['status'] == 'invalid_input'


def test_reference_offline():
    verify_reference()


def test_replay_miss_never_uses_network():
    model = make_model('replay')
    with pytest.raises(ReplayMiss, match='No recording'):
        model.invoke([HumanMessage(content='A new question without a recording')])
    with pytest.raises(ReplayMiss):
        model.bind_tools([]).invoke([SystemMessage(content=SYSTEM),
                                    HumanMessage(content=SCENARIOS['events_only'])])


def test_limit_stops_before_tool_execution():
    result = run_agent(make_model('replay'), 'regular_report', max_turns=1)
    assert result['state']['stopped_on_limit']
    assert all(m.type != 'tool' for m in result['state']['messages'])
    assert 'incomplete' in result['answer']


def test_short_agent_native_messages():
    result = build_short_agent(make_model('replay')).invoke(
        {'messages': [HumanMessage(content=SCENARIOS['events_only'])]}, {'recursion_limit': 16})
    observations = [json.loads(m.content) for m in result['messages'] if m.type == 'tool']
    assert observations and all('events' in x for x in observations)


def test_recorded_model_accounting():
    model = make_model('replay')
    result = run_agent(model, 'events_only')
    assert len(model.calls) == 2
    assert sum(c['usage']['input_tokens'] for c in model.calls) > 0
    assert all(c['mode'] == 'replay' for c in model.calls)


def test_changed_observation_is_a_replay_miss():
    from langchain_core.messages import ToolMessage
    model = make_model('replay').bind_tools([get_quality_summary])
    messages = [SystemMessage(content=SYSTEM), HumanMessage(content='What was the defect rate for L-2 on 2026-11-09?')]
    proposal = model.invoke(messages)
    wrong = ToolMessage(content='Changed data', tool_call_id=proposal.tool_calls[0]['id'])
    with pytest.raises(ReplayMiss):
        model.invoke([*messages, proposal, wrong])


def test_widget_controls(monkeypatch):
    from quality_assistant import ui
    called = []
    real = ui.run_workflow
    def tracked(model, line):
        called.append(line)
        return real(model, line)
    monkeypatch.setattr(ui, 'run_workflow', tracked)
    widget = ui.explorer('workflow', 'replay')
    controls = widget.children[0].children
    controls[0].value = 'L-1'
    controls[1].click()
    assert called == ['L-1']
    slider = ui.step_view(run_workflow(make_model('replay'))).children[0]
    slider.value = slider.max
    quiz = ui.checkpoint('Test', ['A', 'B'], 'B', 'Explanation')
    quiz.children[1].value = 'B'
    quiz.children[2].click()
    lab = ui.tool_lab()
    lab.children[0].children[0].value = 'Unknown line'
    lab.children[0].children[1].click()


def test_trace_distinguishes_proposals_and_observations():
    from quality_assistant.ui import describe_step, step_view
    result = run_agent(make_model('replay'), 'regular_report')
    proposal = describe_step(result['steps'][0])
    assert 'not executed' in proposal and 'Executed calls 0' in proposal
    observation = describe_step(result['steps'][1])
    assert 'Python executes' in observation and 'Executed calls 0' not in observation
    viewer = step_view(result)
    assert viewer.children[0].value == 1
    previous, following = viewer.children[1].children
    assert previous.disabled
    following.click()
    assert viewer.children[0].value == 2 and not previous.disabled
    viewer.children[0].value = viewer.children[0].max
    assert following.disabled


def test_quiz_feedback_does_not_reveal_answer_on_wrong_choice():
    from quality_assistant.ui import checkpoint
    quiz = checkpoint('Question', ['Wrong', 'Right'], 'Right', 'The explanation')
    choices, button, feedback = quiz.children[1:]
    button.click()
    assert 'Choose an answer' in feedback.value
    choices.value = 'Wrong'
    button.click()
    assert 'Not yet' in feedback.value and 'The explanation' not in feedback.value
    choices.value = 'Right'
    assert not feedback.value
    button.click()
    assert 'Correct' in feedback.value and 'The explanation' in feedback.value


def test_source_panels_start_closed():
    from quality_assistant.ui import source_view
    assert source_view(run_workflow).selected_index is None


def test_agent_explorer_uses_selected_question_and_limit(monkeypatch):
    from quality_assistant import ui
    calls = []
    real = ui.run_agent
    def tracked(model, scenario, limit):
        calls.append((scenario, limit))
        return real(model, scenario, limit)
    monkeypatch.setattr(ui, 'run_agent', tracked)
    widget = ui.explorer('agent', 'replay')
    choose, turns, run = widget.children[0].children
    choose.value = 'events_only'
    turns.value = 1
    assert SCENARIOS['events_only'] in widget.children[1].value
    run.click()
    assert calls == [('events_only', 1)] and run.disabled is False


def test_coverage_does_not_match_product_letters_in_other_words():
    from quality_assistant.evaluation import check_result
    result = run_workflow(make_model('replay'))
    result['answer'] = 'A batch on line B was checked.'
    assert check_result(result, 'workflow')['mentions_products'] is False


def test_missing_comparison_is_not_explained_as_normal():
    from quality_assistant.ui import describe_step
    result = run_workflow(make_model('replay'), 'L-3')
    step = next(s for s in result['steps'] if s['node'] == 'comparison')
    rendered = describe_step(step)
    assert 'no usable data' in rendered
    assert 'No product crosses' not in rendered


def test_live_is_default_and_replay_is_explicit(monkeypatch):
    monkeypatch.delenv('PRACTICE_MODE', raising=False)
    assert make_model().mode == 'live'
    assert make_model('replay').mode == 'replay'
    with pytest.raises(ValueError, match='Choose replay or live'):
        make_model('record')
