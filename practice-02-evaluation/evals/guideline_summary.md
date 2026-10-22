# Labelling guideline for defect summaries

Label each summary `FAIL` if it says something that the log entry does not support, otherwise `PASS`. For every `FAIL`, write the rule and a short note.

## FAIL rules

| Rule | Meaning | Example |
|------|---------|---------|
| `cause` | A cause or causal link that the entry does not state, such as "causing", "due to", "indicating". | Entry "repair after failure, unit stopped, plugged strainer", summary "Strainer plugged causing unit failure" |
| `added_fact` | A part, location, type of work or symptom that the entry does not state. | Entry "засор найден, заменён фильтр", summary "Blockage found in filter" |
| `contradiction` | The summary says the opposite of the entry. | Entry "обнаружен подтёк у уплотнения", summary "no defect found during inspection" |
| `term` | A wrong technical term that changes the meaning. | "седло" translated as "saddle" instead of "valve seat" |

## PASS cases

- A paraphrase or translation, such as "течь по прокладке" → "Gasket leak".
- A shorter summary that leaves details out.
- The equipment type from the registry, such as "fan F-230" when the registry says F-230 is a fan.
- A summary that describes the work instead of the defect, such as "filter replaced".
- Format problems (Russian text, more than 15 words, the string "null"). Code checks catch them.

## Disputed cases

| Summary | Entry | Decision |
|---------|-------|----------|
| "Bearing noise indicating bearing wear" | "гул подшипника, замена подшипника" | `FAIL` (`cause`), the entry does not state wear |
| "Blockage found in unit P-202A requiring repair" | "repair after failure ... blockage found" | `PASS`, the repair is stated |
| "Corrosion requiring gasket replacement" | "corroison gasket replaced" | `FAIL` (`cause`), no link is stated |
