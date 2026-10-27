# Practice 4 — reference results

Lecturer's run on the course model `iairlab/qwen3.8-27b` (thinking off), replayed. Search and filters are deterministic; model answers may differ by a question or two in a live run (compare with the intervals).

- Store: 16 documents, 47 pages, 440 chunks (110 of them rows of the long valve table); index version `corpus-cc3cd299df.parser-a53367.emb-none`.
- Retrieval set: 138 questions; by slice: {'base': 50, 'language': 20, 'applicability': 12, 'point_in_time': 10, 'incident': 3, 'what_changed': 4, 'multi_evidence': 8, 'cross_doc': 3, 'conflict': 4, 'table_row': 8, 'tag_variant': 6, 'not_in_documents': 10}.
- Dense search (M1, M2) is **not measured yet**: the course server has no embedding model and this machine had no Ollama. The code path and the replay format are ready (README, 'Dense search').

## Ablation table

Every row adds one change to the row named in `vs` and is compared with it in pairs on the same questions (exact McNemar test, practice 2). Intervals: Wilson, 95%. recall@5 and evidence recall exclude the 10+2 questions whose right answer is "not in the documents".

| config   | n    | recall@5                                    | evidence recall         | end-to-end              | correct + verified      | MRR   | abstain ok   | conflict ok   | vs   | e2e: fixed / broke   | e2e verdict                          | evidence verdict                            |
|:---------|:-----|:--------------------------------------------|:------------------------|:------------------------|:------------------------|:------|:-------------|:--------------|:-----|:---------------------|:-------------------------------------|:--------------------------------------------|
| S0       | 138  | 60/118 = 51% [42%–60%]                      | 53/118 = 45% [36%–54%]  | 75/138 = 54% [46%–62%]  | 49/138 = 36% [28%–44%]  | 0.26  | 10/10        | 0/4           |      |                      |                                      |                                             |
| S1       | 138  | 64/118 = 54% [45%–63%]                      | 57/118 = 48% [39%–57%]  | 80/138 = 58% [50%–66%]  | 53/138 = 38% [31%–47%]  | 0.27  | 10/10        | 0/4           | S0   | +5 / -0              | no detectable difference (p = 0.062) | no detectable difference (+4/-0, p = 0.125) |
| S2       | 138  | 65/118 = 55% [46%–64%]                      | 55/118 = 47% [38%–56%]  | 77/138 = 56% [47%–64%]  | 52/138 = 38% [30%–46%]  | 0.309 | 10/10        | 0/4           | S1   | +6 / -9              | no detectable difference (p = 0.607) | no detectable difference (+3/-5, p = 0.727) |
| S3       | 138  | 78/118 = 66% [57%–74%]                      | 68/118 = 58% [49%–66%]  | 97/138 = 70% [62%–77%]  | 54/138 = 39% [31%–47%]  | 0.352 | 10/10        | 0/4           | S2   | +21 / -1             | B is better (p = 0.000)              | B is better (+14/-1, p = 0.001)             |
| S4       | 138  | 86/118 = 73% [64%–80%]                      | 79/118 = 67% [58%–75%]  | 104/138 = 75% [68%–82%] | 104/138 = 75% [68%–82%] | 0.546 | 10/10        | 0/4           | S3   | +7 / -0              | B is better (p = 0.016)              | B is better (+11/-0, p = 0.001)             |
| S5       | 138  | 86/118 = 73% [64%–80%]                      | 93/118 = 79% [71%–85%]  | 117/138 = 85% [78%–90%] | 117/138 = 85% [78%–90%] | 0.546 | 10/10        | 0/4           | S4   | +13 / -0             | B is better (p = 0.000)              | B is better (+14/-0, p = 0.000)             |
| S6       | 138  | 86/118 = 73% [64%–80%]                      | 93/118 = 79% [71%–85%]  | 115/138 = 83% [76%–89%] | 114/138 = 83% [75%–88%] | 0.546 | 10/10        | 0/4           | S5   | +0 / -2              | no detectable difference (p = 0.500) | no detectable difference (+0/-0, p = 1.000) |
| M1t      | 20   | 8/18 = 44% [25%–66%]                        | 8/18 = 44% [25%–66%]    | 12/20 = 60% [39%–78%]   | 4/20 = 20% [8%–42%]     | 0.233 | —            | —             | S0   | +8 / -1              | B is better (p = 0.039)              | no detectable difference (+7/-2, p = 0.180) |
| M3@S0    | 138  | 90/118 = 76% [68%–83%]                      | 87/118 = 74% [65%–81%]  | 101/138 = 73% [65%–80%] | 56/138 = 41% [33%–49%]  | 0.51  | 10/10        | 0/4           | S0   | +30 / -4             | B is better (p = 0.000)              | B is better (+36/-2, p = 0.000)             |
| M3@S6    | 138  | 100/118 = 85% [77%–90%]                     | 107/118 = 91% [84%–95%] | 124/138 = 90% [84%–94%] | 123/138 = 89% [83%–93%] | 0.826 | 10/10        | 0/4           | S6   | +11 / -2             | B is better (p = 0.022)              | B is better (+14/-0, p = 0.000)             |
| M1@S0    | TODO | TODO: no embedding vectors yet (see README) |                         |                         |                         |       |              |               | S0   |                      |                                      |                                             |
| M2@S0    | TODO | TODO: no embedding vectors yet (see README) |                         |                         |                         |       |              |               | S0   |                      |                                      |                                             |
| M1@S6    | TODO | TODO: no embedding vectors yet (see README) |                         |                         |                         |       |              |               | S6   |                      |                                      |                                             |
| M2@S6    | TODO | TODO: no embedding vectors yet (see README) |                         |                         |                         |       |              |               | S6   |                      |                                      |                                             |

## End-to-end by slice (counts, not percentages)

| config   | base   | language   | applicability   | point_in_time   | incident   | what_changed   | multi_evidence   | cross_doc   | conflict   | table_row   | tag_variant   | not_in_documents   |
|:---------|:-------|:-----------|:----------------|:----------------|:-----------|:---------------|:-----------------|:------------|:-----------|:------------|:--------------|:-------------------|
| S0       | 43/50  | 5/20       | 2/12            | 4/10            | 1/3        | 1/4            | 3/8              | 0/3         | 0/4        | 6/8         | 0/6           | 10/10              |
| S1       | 43/50  | 5/20       | 2/12            | 4/10            | 1/3        | 1/4            | 3/8              | 0/3         | 0/4        | 6/8         | 5/6           | 10/10              |
| S2       | 38/50  | 5/20       | 4/12            | 4/10            | 0/3        | 1/4            | 3/8              | 0/3         | 0/4        | 8/8         | 4/6           | 10/10              |
| S3       | 44/50  | 10/20      | 9/12            | 5/10            | 2/3        | 1/4            | 3/8              | 0/3         | 0/4        | 8/8         | 5/6           | 10/10              |
| S4       | 44/50  | 10/20      | 11/12           | 9/10            | 3/3        | 1/4            | 3/8              | 0/3         | 0/4        | 8/8         | 5/6           | 10/10              |
| S5       | 44/50  | 11/20      | 12/12           | 10/10           | 3/3        | 3/4            | 8/8              | 3/3         | 0/4        | 8/8         | 5/6           | 10/10              |
| S6       | 44/50  | 11/20      | 12/12           | 10/10           | 2/3        | 3/4            | 8/8              | 2/3         | 0/4        | 8/8         | 5/6           | 10/10              |
| M1t      | —      | 12/20      | —               | —               | —          | —              | —                | —           | —          | —           | —             | —                  |
| M3@S0    | 47/50  | 10/20      | 9/12            | 6/10            | 2/3        | 3/4            | 3/8              | 1/3         | 0/4        | 8/8         | 2/6           | 10/10              |
| M3@S6    | 49/50  | 13/20      | 11/12           | 10/10           | 3/3        | 3/4            | 8/8              | 3/3         | 0/4        | 8/8         | 6/6           | 10/10              |

## recall@5 by slice

| config   | base   | language   | applicability   | point_in_time   | incident   | what_changed   | multi_evidence   | cross_doc   | conflict   | table_row   | tag_variant   | not_in_documents   |
|:---------|:-------|:-----------|:----------------|:----------------|:-----------|:---------------|:-----------------|:------------|:-----------|:------------|:--------------|:-------------------|
| S0       | 34/42  | 3/18       | 3/12            | 7/10            | 2/3        | 1/4            | 4/8              | 0/3         | 0/4        | 6/8         | 0/6           | —                  |
| S1       | 34/42  | 3/18       | 3/12            | 7/10            | 2/3        | 1/4            | 4/8              | 0/3         | 0/4        | 6/8         | 4/6           | —                  |
| S2       | 30/42  | 3/18       | 5/12            | 8/10            | 2/3        | 1/4            | 4/8              | 0/3         | 0/4        | 8/8         | 4/6           | —                  |
| S3       | 34/42  | 6/18       | 9/12            | 9/10            | 3/3        | 1/4            | 4/8              | 0/3         | 0/4        | 8/8         | 4/6           | —                  |
| S4       | 37/42  | 8/18       | 11/12           | 9/10            | 3/3        | 1/4            | 4/8              | 0/3         | 0/4        | 8/8         | 5/6           | —                  |
| S5       | 37/42  | 8/18       | 11/12           | 9/10            | 3/3        | 1/4            | 4/8              | 0/3         | 0/4        | 8/8         | 5/6           | —                  |
| S6       | 37/42  | 8/18       | 11/12           | 9/10            | 3/3        | 1/4            | 4/8              | 0/3         | 0/4        | 8/8         | 5/6           | —                  |
| M1t      | —      | 8/18       | —               | —               | —          | —              | —                | —           | —          | —           | —             | —                  |
| M3@S0    | 42/42  | 7/18       | 12/12           | 10/10           | 2/3        | 3/4            | 3/8              | 1/3         | 0/4        | 8/8         | 2/6           | —                  |
| M3@S6    | 42/42  | 11/18      | 12/12           | 10/10           | 3/3        | 3/4            | 4/8              | 1/3         | 0/4        | 8/8         | 6/6           | —                  |

## Evidence recall by slice

| config   | base   | language   | applicability   | point_in_time   | incident   | what_changed   | multi_evidence   | cross_doc   | conflict   | table_row   | tag_variant   | not_in_documents   |
|:---------|:-------|:-----------|:----------------|:----------------|:-----------|:---------------|:-----------------|:------------|:-----------|:------------|:--------------|:-------------------|
| S0       | 34/42  | 3/18       | 2/12            | 5/10            | 2/3        | 1/4            | 0/8              | 0/3         | 0/4        | 6/8         | 0/6           | —                  |
| S1       | 34/42  | 3/18       | 2/12            | 5/10            | 2/3        | 1/4            | 0/8              | 0/3         | 0/4        | 6/8         | 4/6           | —                  |
| S2       | 30/42  | 3/18       | 2/12            | 5/10            | 2/3        | 1/4            | 0/8              | 0/3         | 0/4        | 8/8         | 4/6           | —                  |
| S3       | 34/42  | 6/18       | 6/12            | 6/10            | 3/3        | 1/4            | 0/8              | 0/3         | 0/4        | 8/8         | 4/6           | —                  |
| S4       | 37/42  | 8/18       | 10/12           | 7/10            | 3/3        | 1/4            | 0/8              | 0/3         | 0/4        | 8/8         | 5/6           | —                  |
| S5       | 37/42  | 9/18       | 11/12           | 8/10            | 3/3        | 3/4            | 7/8              | 2/3         | 0/4        | 8/8         | 5/6           | —                  |
| S6       | 37/42  | 9/18       | 11/12           | 8/10            | 3/3        | 3/4            | 7/8              | 2/3         | 0/4        | 8/8         | 5/6           | —                  |
| M1t      | —      | 8/18       | —               | —               | —          | —              | —                | —           | —          | —           | —             | —                  |
| M3@S0    | 42/42  | 7/18       | 12/12           | 10/10           | 1/3        | 3/4            | 1/8              | 1/3         | 0/4        | 8/8         | 2/6           | —                  |
| M3@S6    | 42/42  | 11/18      | 12/12           | 10/10           | 3/3        | 4/4            | 8/8              | 3/3         | 0/4        | 8/8         | 6/6           | —                  |

## Where answers are lost (wrong answers by link)

| config   |   reading |   search |   assembly |   model |
|:---------|----------:|---------:|-----------:|--------:|
| S0       |         0 |       56 |          2 |       5 |
| S6       |         0 |       20 |          1 |       2 |

Reading: the value is in no chunk; search: no chunk with the answer in the context; assembly: the answer chunk is there, a required support or action chunk is not; model: everything was in the context. Practice 3 ceiling with the known page: 98%.

## The P-101 question at S6

Question (asked in Russian): После замены подшипника насос P-101 вибрирует 4,8 мм/с. Это норма? Что делать и в какой срок?

```json
{
 "status": "answered",
 "answer": "The vibration level of 4.8 mm/s exceeds the inspection limit (4.5 mm/s) for a Class II pump but is below the stop limit (7.1 mm/s). Therefore, it is not normal. Actions required: 1) The patrolman must report the exceedance to the shift supervisor within 1 hour. 2) The shift supervisor must create a high-priority inspection request in CMMS. 3) The equipment must be inspected within 24 hours. 4) Vibration measurements must be repeated every shift until the inspection is completed. 5) After inspection, the mechanic must identify the cause (e.g., misalignment, bearing wear) and record it in the CMMS request.",
 "value": "4,5",
 "unit": "мм/с",
 "claims": [
  {
   "text": "Pump P-101 is a Class II machine.",
   "sources": [
    "S4"
   ]
  },
  {
   "text": "The inspection limit for Class II pumps is 4.5 mm/s and the stop limit is 7.1 mm/s.",
   "sources": [
    "S4"
   ]
  },
  {
   "text": "The measured vibration of 4.8 mm/s exceeds the inspection limit of 4.5 mm/s.",
   "sources": [
    "S4"
   ]
  },
  {
   "text": "If the inspection limit is exceeded, the equipment must be inspected within 24 hours and measurements repeated every shift until inspection.",
   "sources": [
    "S8"
   ]
  },
  {
   "text": "The patrolman must report the exceedance to the shift supervisor within 1 hour.",
   "sources": [
    "S19"
   ]
  },
  {
   "text": "The shift supervisor must create a high-priority CMMS request for inspection.",
   "sources": [
    "S9"
   ]
  },
  {
   "text": "The mechanic must specify the cause of the exceedance in the CMMS request after inspection.",
   "sources": [
    "S20"
   ]
  }
 ],
 "citations": [
  {
   "source": "S4",
   "doc_id": "R-12",
   "revision": 3,
   "amendment": null,
   "clause": null,
   "table_no": "3",
   "row_key": "Насосы|II",
   "page": 4
  },
  {
   "source": "S8",
   "doc_id": "R-12",
   "revision": 3,
   "amendment": null,
   "clause": "4.3.3",
   "table_no": null,
   "row_key": null,
   "page": 3
  },
  {
   "source": "S19",
   "doc_id": "R-12",
   "revision": 3,
   "amendment": null,
   "clause": "6.1",
   "table_no": null,
   "row_key": null,
   "page": 5
  },
  {
   "source": "S9",
   "doc_id": "R-12",
   "revision": 3,
   "amendment": null,
   "clause": "6.2",
   "table_no": null,
   "row_key": null,
   "page": 5
  },
  {
   "source": "S20",
   "doc_id": "R-12",
   "revision": 3,
   "amendment": null,
   "clause": "6.3",
   "table_no": null,
   "row_key": null,
   "page": 5
  }
 ],
 "sources": []
}
```

Checks: verified = True, exists = True, in force = True, supports = True. Citations: R-12-r3-table1 (exists True, in force True, supports True); R-12-r3-4.3.3 (exists True, in force True, supports True); R-12-r3-6.1 (exists True, in force True, supports False); R-12-r3-6.2 (exists True, in force True, supports False); R-12-r3-6.3 (exists True, in force True, supports False)

The audit record is in `audit_p101_S6.json`.

## The contract on a conflict (demo: both sides of the P-205 conflict put into the context by hand)

Status: `conflict`; answer: The sources provide different vibration limits for pump P-205 (Class II). Source [S1] sets the inspection limit at 4.5 mm/s and the stop limit at 7.1 mm/s. Source [S2] sets the alarm limit at 7.1 mm/s (with a 7-day deadline to find the cause) and the trip limit at 11.2 mm/s. The values for the alarm/inspection threshold and the stop/trip threshold differ between the plant regulation and the equipment manual.

## Regression gate S6 against S5 (practice 2 rule)

117 -> 115 right of 138, fixed 0, broke 2 (i01, k02), p = 0.5: **PASS**

## Five S0 failures for task 1 of the lecture

See `s0_failures.md`: m01 (search), g01 (search), v03 (search), m06 (assembly), i01 (model).
