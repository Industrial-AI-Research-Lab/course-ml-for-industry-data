"""Small notebook controls with ordinary Python functions underneath."""
import html
import inspect
import json

import ipywidgets as W
import pandas as pd
from IPython.display import HTML, display

from .models import ReplayMiss, make_model
from .systems import SCENARIOS, run_agent, run_workflow


def source_view(*functions):
    tabs = W.Accordion(children=[W.HTML('<pre>' + html.escape(inspect.getsource(f)) + '</pre>')
                                  for f in functions], selected_index=None)
    for i, f in enumerate(functions):
        tabs.set_title(i, f.__name__)
    return tabs


def readable(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump()
    raise TypeError(type(value).__name__)


def text_block(text):
    return ''.join('<p>' + html.escape(p).replace('\n', '<br>') + '</p>'
                   for p in str(text).split('\n\n') if p)


def raw_panel(title, value):
    return '<details><summary>' + title + '</summary><pre>' + html.escape(
        json.dumps(value, default=readable, indent=2)) + '</pre></details>'


def describe_result(value):
    if value.get('status') == 'invalid_input':
        return text_block('Invalid input. ' + value['error'])
    if value.get('status') == 'no_data':
        return text_block('No matching batch data. The defect rate is unknown, not zero.')
    if 'defect_rate_pct' in value:
        return text_block(f"{value['line_id']} has {value['defects']} defective items out of "
                          f"{value['produced']}. The weighted rate is {value['defect_rate_pct']} percent.")
    if 'before' in value:
        return ''.join(text_block(f"Product {p['product_id']}. Before {p['before']['defect_rate_pct']} percent. "
                                  f"After {p['after']['defect_rate_pct']} percent. Change {p['change_pp']} percentage points.")
                       for p in value['products'])
    if 'events' in value:
        return ''.join(text_block(f"{e['event_id']} on {e['date']}. {e['event']}") for e in value['events']) \
            or text_block('No matching events are recorded. This does not prove that no work took place.')
    return ''


def describe_step(step):
    node, update, state = step['node'], step['update'], step['state']
    if node == 'model':
        message = update['messages'][-1]
        if message.tool_calls:
            body = text_block('The model proposes these calls. They have not executed at this step.')
            for call in message.tool_calls:
                body += '<h4>' + html.escape(call['name']) + '</h4><pre>' + html.escape(
                    json.dumps(call['args'], indent=2)) + '</pre>'
        else:
            body = text_block('The model returns a final answer. No new tool is requested.') + text_block(message.content)
    elif node == 'tools':
        body = text_block('Python executes the proposed calls and returns observations to the model.')
        for message in update['messages']:
            body += '<h4>' + html.escape(message.name or 'Tool result') + '</h4>'
            try:
                body += describe_result(json.loads(message.content))
            except (ValueError, TypeError):
                body += text_block(message.content)
    elif node == 'limit':
        body = text_block('The turn budget is exhausted. Pending calls are not executed. The task is incomplete.')
    elif node == 'report':
        body = text_block(update['answer'])
    else:
        value = update[{'summary': 'summary', 'comparison': 'comparison', 'events': 'events'}[node]]
        body = describe_result(value)
        if node == 'comparison':
            event_needed = any(p['change_pp'] >= 1 for p in value.get('products', []))
            if value.get('status') != 'ok':
                body += text_block('The comparison has no usable data. The workflow passes this gap to the report.')
            else:
                body += text_block('The reporting rule selects events next.' if event_needed else
                                   'No product crosses the reporting threshold. The workflow goes to report.')
    messages = state.get('messages', [])
    if messages:
        proposed = sum(len(getattr(m, 'tool_calls', [])) for m in messages)
        executed = sum(m.type == 'tool' for m in messages)
        body += text_block(f"Model turns {state.get('model_turns', 0)}. Proposed calls {proposed}. Executed calls {executed}.")
    return body


def step_view(result):
    steps = result['steps']
    choose = W.IntSlider(value=1, min=1, max=len(steps), description='Step', continuous_update=False)
    previous = W.Button(description='Previous')
    following = W.Button(description='Next')
    output = W.HTML()

    def show(change=None):
        step = steps[choose.value - 1]
        previous.disabled = choose.value == 1
        following.disabled = choose.value == len(steps)
        output.value = '<h4>Step ' + str(choose.value) + ' of ' + str(len(steps)) + '. ' + \
            html.escape(step['node']) + '</h4>' + describe_step(step) + \
            raw_panel('Raw node update', step['update']) + raw_panel('Full state after this step', step['state'])
    choose.observe(show, names='value')
    previous.on_click(lambda _: setattr(choose, 'value', max(1, choose.value - 1)))
    following.on_click(lambda _: setattr(choose, 'value', min(len(steps), choose.value + 1)))
    show()
    return W.VBox([choose, W.HBox([previous, following]), output])


SCENARIO_LABELS = {
    'regular_report': 'Regular report on L-2',
    'product_comparison': 'Compare products on L-2',
    'events_only': 'Read events only',
    'missing_data': 'No batches on L-3',
    'stable_report': 'Stable report on L-1',
}


def explorer(kind='workflow', mode=None):
    options = [('Increase on L-2', 'L-2'), ('Stable L-1', 'L-1'), ('No data on L-3', 'L-3')] \
        if kind == 'workflow' else [(SCENARIO_LABELS[k], k) for k in SCENARIOS]
    choose = W.Dropdown(options=options, description='Scenario', layout=W.Layout(width='360px'))
    turns = W.Dropdown(options=[1, 2, 6], value=6, description='Turn limit')
    run = W.Button(description='Run', button_style='primary')
    preview = W.HTML()
    output = W.Output()

    def changed(_=None):
        prompt = ('Prepare the regular report for ' + choose.value + '. Compare November 8 with November 9, 2026.') \
            if kind == 'workflow' else SCENARIOS[choose.value]
        preview.value = '<p><strong>Question</strong></p>' + text_block(prompt) + text_block(
            'Predict the next steps, then click Run. Changed settings apply to a new run.')
        with output:
            output.clear_output(wait=False)
    choose.observe(changed, names='value')
    turns.observe(changed, names='value')

    def execute(_=None):
        run.disabled = True
        with output:
            output.clear_output(wait=True)
            try:
                model = make_model(mode)
                result = run_workflow(model, choose.value) if kind == 'workflow' else \
                    run_agent(model, choose.value, turns.value)
                display(step_view(result))
                display(W.HTML(raw_panel('Final answer from this run', result['answer'])))
            except (ReplayMiss, ValueError) as exc:
                print(str(exc))
            finally:
                run.disabled = False
    run.on_click(execute)
    controls = [choose, run] if kind == 'workflow' else [choose, turns, run]
    changed()
    return W.VBox([W.HBox(controls), preview, output])


def checkpoint(question, choices, correct, explanation):
    answer = W.RadioButtons(options=choices, value=None, layout=W.Layout(width='auto'))
    check = W.Button(description='Check my answer')
    output = W.HTML()

    def feedback(_):
        if answer.value is None:
            output.value = text_block('Choose an answer first.')
        elif answer.value == correct:
            output.value = text_block('Correct. ' + explanation)
        else:
            output.value = text_block('Not yet. Find the step that supports your choice, then try again.')
    answer.observe(lambda _: setattr(output, 'value', ''), names='value')
    check.on_click(feedback)
    return W.VBox([W.HTML('<p>'+html.escape(question)+'</p>'), answer, check, output])


def rates_view(comparison):
    rows = []
    for p in comparison['products']:
        for period in ('before', 'after'):
            value = p[period]['defect_rate_pct']
            rows.append('<tr><td>'+html.escape(p['product_id'])+'</td><td>'+period+'</td><td>'
                        +f'<meter min="0" max="10" value="{value}"></meter> {value:.1f}%</td></tr>')
    display(HTML('<h4>Defect rate by product and period</h4><p>Each bar spans 0 to 10 percent. '
                 'Source is the synthetic batch data for L-2 on November 8 and 9, 2026.</p>'
                 '<table><tr><th>Product</th><th>Period</th><th>Defect rate</th></tr>'
                 + ''.join(rows) + '</table>'))


def tool_lab():
    from .tools import get_quality_summary
    scenarios = {
        'Valid data': dict(line_id='L-2', start='2026-11-09', end='2026-11-09'),
        'No batches': dict(line_id='L-3', start='2026-11-09', end='2026-11-09'),
        'Unknown line': dict(line_id='L-9', start='2026-11-09', end='2026-11-09'),
        'Reversed dates': dict(line_id='L-2', start='2026-11-09', end='2026-11-08'),
    }
    choose = W.Dropdown(options=list(scenarios), description='Input')
    run = W.Button(description='Run tool')
    output = W.Output()

    def execute(_=None):
        with output:
            output.clear_output(wait=True)
            args = scenarios[choose.value]
            print(json.dumps(args, indent=2))
            result = get_quality_summary.invoke(args)
            display(HTML(describe_result(result)))
            display(HTML(raw_panel('Full tool result', result)))
    run.on_click(execute)
    return W.VBox([W.HBox([choose, run]), output])


def graph_view(graph, kind='workflow'):
    """Render the real graph locally. No diagram service is contacted."""
    positions = ({'__start__': (45, 65), 'summary': (175, 65), 'comparison': (315, 65),
                  'events': (455, 135), 'report': (595, 65), '__end__': (735, 65)}
                 if kind == 'workflow' else
                 {'__start__': (45, 65), 'model': (205, 65), 'tools': (385, 65),
                  'limit': (385, 155), '__end__': (585, 155)})
    parts = ['<svg viewBox="0 0 800 210" width="100%" role="img" '
             'aria-label="Execution graph"><rect width="800" height="210" fill="white"/>',
             '<defs><marker id="tip-'+kind+'" markerWidth="8" markerHeight="8" refX="7" refY="3" '
             'orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#555"/></marker></defs>']
    actual = graph.get_graph()
    for edge in actual.edges:
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        dash = ' stroke-dasharray="5 4"' if edge.conditional else ''
        if edge.source == 'tools' and edge.target == 'model':
            path = f'M {x1} {y1-20} Q {(x1+x2)/2} -10 {x2} {y2-20}'
        else:
            path = f'M {x1+45} {y1} L {x2-48} {y2}'
        parts.append(f'<path d="{path}" fill="none" stroke="#555"{dash} '
                     f'marker-end="url(#tip-{kind})"/>')
    for name, (x, y) in positions.items():
        label = name.strip('_').capitalize()
        parts.append(f'<rect x="{x-46}" y="{y-19}" width="92" height="38" rx="4" '
                     'fill="#f4f4f4" stroke="#555"/>')
        parts.append(f'<text x="{x}" y="{y+5}" text-anchor="middle" font-size="13" '
                     f'font-family="sans-serif" fill="#111">{label}</text>')
    parts.append('</svg><p>Dashed edges select a branch from the current state.</p>')
    return HTML(''.join(parts))


def report_comparison(workflow, agent):
    # Ordinary HTML remains readable in static notebook previews too.
    return HTML('<table width="100%"><thead><tr><th>Workflow report</th><th>Agent report</th></tr></thead>'
                '<tbody><tr><td width="50%" valign="top">' + text_block(workflow['answer']) +
                '</td><td width="50%" valign="top">' + text_block(agent['answer']) + '</td></tr></tbody></table>')
