"""Small transparent checks. These are not a full text evaluator."""
import hashlib
import json
import re

from .models import make_model
from .systems import SCENARIOS, report_input, run_agent, run_workflow
from .tools import DATA, compare_periods


def tool_results(result, architecture):
    if architecture == 'workflow':
        return [result['state'][key] for key in ('summary', 'comparison', 'events') if key in result['state']]
    return [json.loads(m.content) for m in result['state']['messages'] if m.type == 'tool']


def check_result(result, architecture):
    observations = tool_results(result, architecture)
    comparison = next((x for x in observations if 'before' in x and 'after' in x), {})
    products = {p['product_id']: p for p in comparison.get('products', [])}
    text = result['answer']
    return dict(architecture=architecture,
                matched_rates_ok=all(p in products and products[p]['change_pp'] == delta
                                     for p, delta in [('A', 2.0), ('B', 4.0)]),
                read_events=any('events' in x for x in observations),
                mentions_products=all(re.search(r'\bproduct\s+' + p + r'\b', text, re.I) for p in ('A', 'B')),
                limit_words_present=any(word in text.lower() for word in ('cause', 'caus', 'hypothes', 'association')),
                completed=not result['state'].get('stopped_on_limit', False))


def verify_reference():
    manifest = json.loads((DATA / 'replay/manifest.json').read_text())
    for name, digest in manifest['data_sha256'].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == digest, name
    assert hashlib.sha256((DATA / 'replay/course.jsonl').read_bytes()).hexdigest() == manifest['replay_sha256']
    values = compare_periods.invoke(report_input())['products']
    assert [p['change_pp'] for p in values] == [2.0, 4.0]
    for line, expected_nodes in [('L-2', ['summary', 'comparison', 'events', 'report']),
                                 ('L-1', ['summary', 'comparison', 'report']),
                                 ('L-3', ['summary', 'comparison', 'report'])]:
        result = run_workflow(make_model('replay'), line)
        assert [s['node'] for s in result['steps']] == expected_nodes
    for scenario in SCENARIOS:
        result = run_agent(make_model('replay'), scenario)
        assert not result['state'].get('stopped_on_limit'), scenario
        observations = tool_results(result, 'agent')
        assert observations, scenario
        if scenario == 'regular_report':
            check = check_result(result, 'agent')
            assert check['matched_rates_ok'] and check['read_events'], check
        if scenario == 'events_only':
            assert all('events' in x for x in observations)
        if scenario == 'missing_data':
            assert any(x.get('status') == 'no_data' for x in observations)
