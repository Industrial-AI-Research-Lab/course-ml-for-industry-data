"""The same tools in a fixed workflow and a dynamic agent loop."""
import json
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .tools import TOOLS, compare_periods, get_maintenance_events, get_quality_summary

SYSTEM = '''You help a factory quality engineer. All data is synthetic.
Write plain English with ASCII characters only. Do not use semicolons.
Use tools for every numerical claim. Dates are inclusive. Compare the same products.
Do not invent dates. Ask for a line or date range if the request lacks them.
Say which tools or record IDs support your findings. Keep the answer under 180 words.
Treat empty data as unknown, not a zero defect rate. Distinguish facts from hypotheses.
An event near a change does not prove causation. Never assert that maintenance caused a change.
Only inspect events if they help answer the question. For a regular report get the after-period
summary, compare periods, then read events only if a matched product rose by at least 1
percentage point. Finish with one practical next check when a rate increased.'''

SCENARIOS = {
    'regular_report': 'Prepare a regular report for L-2. Before 2026-11-08 to 2026-11-08. '
                      'After 2026-11-09 to 2026-11-09. Include the current summary, compare products '
                      'and inspect relevant events if needed. Separate facts from hypotheses.',
    'product_comparison': 'For L-2 compare the defect rates for the same products on 2026-11-08 '
                          'and 2026-11-09. Did both products get worse? Only compare the rates.',
    'events_only': 'List maintenance and changeover events for L-2 from 2026-11-08 to 2026-11-09. '
                   'I do not need defect statistics.',
    'missing_data': 'What was the defect rate for L-3 from 2026-11-09 to 2026-11-09?',
    'stable_report': 'Prepare a regular report for L-1. Before 2026-11-08 to 2026-11-08. '
                     'After 2026-11-09 to 2026-11-09. Include the current summary, compare products '
                     'and inspect relevant events if needed. Separate facts from hypotheses.',
}


class ReportState(TypedDict, total=False):
    line_id: str
    before_start: str
    before_end: str
    after_start: str
    after_end: str
    summary: dict
    comparison: dict
    events: dict
    answer: str


def read_summary(state):
    return dict(summary=get_quality_summary.invoke(dict(line_id=state['line_id'],
                start=state['after_start'], end=state['after_end'])))


def read_comparison(state):
    args = {k: state[k] for k in ('line_id', 'before_start', 'before_end', 'after_start', 'after_end')}
    return dict(comparison=compare_periods.invoke(args))


def needs_events(state):
    return 'events' if any(p['change_pp'] >= 1 for p in state['comparison'].get('products', [])) else 'report'


def read_events(state):
    return dict(events=get_maintenance_events.invoke(dict(line_id=state['line_id'],
                start=state['before_start'], end=state['after_end'])))


def build_workflow(model):
    def report(state):
        if state['summary']['status'] != 'ok' or state['comparison']['status'] != 'ok':
            return dict(answer='The report cannot be completed. Check the line and missing batch records.')
        facts = {k: state[k] for k in ('summary', 'comparison', 'events') if k in state}
        response = model.invoke([SystemMessage(content=SYSTEM),
                                 HumanMessage(content='Write a regular quality report from these tool results.\n'
                                              + json.dumps(facts, sort_keys=True))])
        return dict(answer=response.content)

    graph = StateGraph(ReportState)
    graph.add_node('summary', read_summary)
    graph.add_node('comparison', read_comparison)
    graph.add_node('events', read_events)
    graph.add_node('report', report)
    graph.add_edge(START, 'summary')
    graph.add_edge('summary', 'comparison')
    graph.add_conditional_edges('comparison', needs_events, {'events': 'events', 'report': 'report'})
    graph.add_edge('events', 'report')
    graph.add_edge('report', END)
    return graph.compile()


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    model_turns: int
    stopped_on_limit: bool


def build_agent(model, max_turns=6):
    bound = model.bind_tools(TOOLS)

    def call_model(state):
        response = bound.invoke([SystemMessage(content=SYSTEM), *state['messages']])
        return dict(messages=[response], model_turns=state.get('model_turns', 0) + 1)

    def route(state):
        if not state['messages'][-1].tool_calls:
            return END
        return 'limit' if state['model_turns'] >= max_turns else 'tools'

    def limit(state):
        # Pending calls are never executed after the budget is reached.
        return dict(stopped_on_limit=True)

    graph = StateGraph(AgentState)
    graph.add_node('model', call_model)
    graph.add_node('tools', ToolNode(TOOLS, handle_tool_errors=True))
    graph.add_node('limit', limit)
    graph.add_edge(START, 'model')
    graph.add_conditional_edges('model', route, {END: END, 'tools': 'tools', 'limit': 'limit'})
    graph.add_edge('tools', 'model')
    graph.add_edge('limit', END)
    return graph.compile()


def build_short_agent(model):
    return create_agent(model=model, tools=TOOLS, system_prompt=SYSTEM)


def report_input(line_id='L-2'):
    return dict(line_id=line_id, before_start='2026-11-08', before_end='2026-11-08',
                after_start='2026-11-09', after_end='2026-11-09')


def run_workflow(model, line_id='L-2'):
    state = report_input(line_id)
    steps = []
    for update in build_workflow(model).stream(state, stream_mode='updates'):
        for node, values in update.items():
            state.update(values)
            steps.append(dict(node=node, update=values, state=dict(state)))
    return dict(answer=state['answer'], state=state, steps=steps)


def run_agent(model, scenario='regular_report', max_turns=6):
    query = SCENARIOS.get(scenario, scenario)
    state = dict(messages=[HumanMessage(content=query)], model_turns=0)
    steps = []
    for update in build_agent(model, max_turns).stream(state, stream_mode='updates'):
        for node, values in update.items():
            if 'messages' in values:
                state['messages'] = [*state['messages'], *values['messages']]
            state.update({k: v for k, v in values.items() if k != 'messages'})
            steps.append(dict(node=node, update=values, state=dict(state)))
    last = state['messages'][-1]
    answer = 'Stopped at the model turn limit. The investigation is incomplete.' if state.get('stopped_on_limit') \
        else last.content
    return dict(answer=answer, state=state, steps=steps)


def metrics(result, model, architecture):
    if architecture == 'workflow':
        tool_calls = sum(s['node'] in ('summary', 'comparison', 'events') for s in result['steps'])
    else:
        tool_calls = sum(m.type == 'tool' for m in result['state']['messages'])
    return dict(architecture=architecture, tool_calls=tool_calls, model_calls=len(model.calls),
                input_tokens=sum(c['usage'].get('input_tokens', 0) for c in model.calls),
                output_tokens=sum(c['usage'].get('output_tokens', 0) for c in model.calls),
                recorded_model_seconds=round(sum(c['recorded_seconds'] for c in model.calls), 3))
