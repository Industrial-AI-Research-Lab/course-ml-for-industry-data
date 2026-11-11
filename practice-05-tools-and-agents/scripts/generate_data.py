"""Generate the small synthetic dataset without randomness."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = []
for line, day, product, counts in [
    ('L-2', '2026-11-08', 'A', [(100, 1), (900, 9)]),
    ('L-2', '2026-11-08', 'B', [(1000, 40), (1000, 40)]),
    ('L-2', '2026-11-09', 'A', [(100, 10), (1900, 50)]),
    ('L-2', '2026-11-09', 'B', [(500, 40), (500, 40)]),
    ('L-1', '2026-11-08', 'A', [(1000, 20)]),
    ('L-1', '2026-11-09', 'A', [(1000, 20)]),
]:
    for produced, defects in counts:
        rows.append(dict(batch_id=f'B-{len(rows)+1:03d}', date=day, line_id=line,
                         product_id=product, produced=produced, defects=defects))
events = [
    dict(event_id='E-001', date='2026-11-08', line_id='L-2', event='Routine inspection. No fault recorded.'),
    dict(event_id='E-002', date='2026-11-09', line_id='L-2', event='Sealing unit adjusted before production.'),
    dict(event_id='E-003', date='2026-11-09', line_id='L-2', event='New film roll introduced. Supplier unchanged.'),
    dict(event_id='E-004', date='2026-11-08', line_id='L-1', event='Routine inspection. No fault recorded.'),
]
for name, data in [('batches', rows), ('events', events)]:
    path = ROOT / 'data' / f'{name}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(data)
