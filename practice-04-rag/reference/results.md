# Practice 4 — reference results

Reference run on the course model `iairlab/qwen3.8-27b` (thinking off), replayed. Search and filters are deterministic; model answers may differ by a question or two in a live run (compare with the intervals).

- Store: 16 documents, 47 pages, 440 chunks (110 of them rows of the long valve table); index version `corpus-5719c6dc76.parser-a53367.emb-none`.
- Retrieval set: 143 questions; by slice: {'base': 50, 'language': 20, 'applicability': 12, 'point_in_time': 10, 'incident': 3, 'what_changed': 4, 'multi_evidence': 8, 'cross_doc': 3, 'conflict': 9, 'table_row': 8, 'tag_variant': 6, 'not_in_documents': 10}.
- Mechanisms (the course server): dense search with `iairlab/e5-mistral-7b-instruct` (4096-dim; questions with the instruction, documents without; vectors in `data/replay/vectors/e5-mistral.*`), hybrid = RRF of the BM25 and dense top-20, the LLM reranker (M3, course Qwen, listwise) and the cross-encoder `iairlab/bge-reranker-v2-m3` (M4, `/rerank`), both over the top-20 of the row they are added to.

## Evaluation limits

End-to-end is the code-grader score: one value and its unit, the presence of required numbers in a multi-value answer, or an expected status. It does not grade the full explanation or recommended action. Citation checks test sources, dates and numeric presence; they do not establish support for every claim. The small synthetic holdout and unadjusted paired tests are exploratory. See README, 'What the evaluation does and does not prove'.

## Ablation table

Every row adds one change to the row named in `vs` and is compared with it in pairs on the same questions (exact McNemar test, practice 2). Intervals: Wilson, 95%. recall@5 and evidence recall exclude the questions whose right answer is "not in the documents" (no evidence). From S2b on, the search result is the top-5 chunks plus up to 2 rows of the row collection; documents that the registry links to the equipment (S3+) are not search: they count in evidence recall, not in recall@5.

| config   |   n | recall@5                | evidence recall         | end-to-end              | correct + verified      |   MRR | abstain ok   | conflict ok   | false refusals                | wrong or invented   | right abstentions   |   invalid |   fallbacks |   leaks | vs   | e2e: fixed / broke   | e2e verdict                          | evidence verdict                             |
|:---------|----:|:------------------------|:------------------------|:------------------------|:------------------------|------:|:-------------|:--------------|:------------------------------|:--------------------|:--------------------|----------:|------------:|--------:|:-----|:---------------------|:-------------------------------------|:---------------------------------------------|
| S0       | 143 | 61/123 = 50% [41%–58%]  | 54/123 = 44% [35%–53%]  | 73/143 = 51% [43%–59%]  | 47/143 = 33% [26%–41%]  | 0.247 | 10/10        | 0/9           | 42/123 = 34%                  | 28 (1 invented)     | 19/20 = 95%         |         0 |           0 |       0 |      |                      |                                      |                                              |
| S1       | 143 | 65/123 = 53% [44%–61%]  | 58/123 = 47% [39%–56%]  | 78/143 = 55% [46%–62%]  | 51/143 = 36% [28%–44%]  | 0.257 | 10/10        | 0/9           | 37/123 = 30%                  | 28 (1 invented)     | 19/20 = 95%         |         0 |           0 |       0 | S0   | +5 / -0              | no detectable difference (p = 0.062) | no detectable difference (+4/-0, p = 0.125)  |
| S2       | 143 | 69/123 = 56% [47%–65%]  | 59/123 = 48% [39%–57%]  | 77/143 = 54% [46%–62%]  | 52/143 = 36% [29%–45%]  | 0.309 | 10/10        | 0/9           | 36/123 = 29%                  | 30 (1 invented)     | 19/20 = 95%         |         0 |           0 |       0 | S1   | +6 / -7              | no detectable difference (p = 1.000) | no detectable difference (+4/-3, p = 1.000)  |
| S2b      | 143 | 67/123 = 54% [46%–63%]  | 60/123 = 49% [40%–58%]  | 80/143 = 56% [48%–64%]  | 51/143 = 36% [28%–44%]  | 0.235 | 10/10        | 0/9           | 35/123 = 28%                  | 27 (1 invented)     | 19/20 = 95%         |         1 |           0 |       0 | S1   | +3 / -1              | no detectable difference (p = 0.625) | no detectable difference (+2/-0, p = 0.500)  |
| S3       | 143 | 82/123 = 67% [58%–74%]  | 84/123 = 68% [60%–76%]  | 102/143 = 71% [63%–78%] | 62/143 = 43% [36%–52%]  | 0.287 | 10/10        | 0/9           | 19/123 = 15%                  | 22 (1 invented)     | 19/20 = 95%         |         0 |           0 |       0 | S2b  | +24 / -2             | B is better (p = 0.000)              | B is better (+24/-0, p = 0.000)              |
| S4       | 143 | 93/123 = 76% [67%–82%]  | 94/123 = 76% [68%–83%]  | 109/143 = 76% [69%–82%] | 109/143 = 76% [69%–82%] | 0.472 | 10/10        | 0/9           | 18/123 = 15%                  | 16 (1 invented)     | 19/20 = 95%         |         0 |           0 |       0 | S3   | +7 / -0              | B is better (p = 0.016)              | B is better (+10/-0, p = 0.002)              |
| S5       | 143 | 93/123 = 76% [67%–82%]  | 108/123 = 88% [81%–92%] | 123/143 = 86% [79%–91%] | 123/143 = 86% [79%–91%] | 0.472 | 10/10        | 0/9           | 11/123 = 9%                   | 9                   | 20/20 = 100%        |         0 |           0 |       0 | S4   | +14 / -0             | B is better (p = 0.000)              | B is better (+14/-0, p = 0.000)              |
| S6       | 143 | 93/123 = 76% [67%–82%]  | 108/123 = 88% [81%–92%] | 129/143 = 90% [84%–94%] | 128/143 = 90% [83%–94%] | 0.472 | 10/10        | 7/9           | 11/123 = 9%                   | 3                   | 20/20 = 100%        |         0 |           0 |       0 | S5   | +7 / -1              | no detectable difference (p = 0.070) | no detectable difference (+0/-0, p = 1.000)  |
| M1t      |  20 | 9/18 = 50% [29%–71%]    | 9/18 = 50% [29%–71%]    | 13/20 = 65% [43%–82%]   | 5/20 = 25% [11%–47%]    | 0.236 | —            | —             | 6/18 = 33%                    | 1                   | 2/2 = 100%          |         0 |           0 |       0 | S0   | +9 / -1              | B is better (p = 0.021)              | no detectable difference (+8/-2, p = 0.109)  |
| M1@S0    | 143 | 84/123 = 68% [60%–76%]  | 76/123 = 62% [53%–70%]  | 90/143 = 63% [55%–70%]  | 57/143 = 40% [32%–48%]  | 0.4   | 8/10         | 0/9           | 26/123 = 21%                  | 27 (5 invented)     | 15/20 = 75%         |         0 |           0 |       0 | S0   | +32 / -15            | B is better (p = 0.019)              | B is better (+34/-12, p = 0.002)             |
| M2@S0    | 143 | 77/123 = 63% [54%–71%]  | 65/123 = 53% [44%–61%]  | 86/143 = 60% [52%–68%]  | 49/143 = 34% [27%–42%]  | 0.348 | 10/10        | 0/9           | 25/123 = 20%                  | 32 (3 invented)     | 17/20 = 85%         |         0 |           0 |       0 | S0   | +23 / -10            | B is better (p = 0.035)              | no detectable difference (+19/-8, p = 0.052) |
| M3@S0    | 143 | 93/123 = 76% [67%–82%]  | 90/123 = 73% [65%–80%]  | 101/143 = 71% [63%–77%] | 58/143 = 41% [33%–49%]  | 0.499 | 10/10        | 0/9           | 15/123 = 12%                  | 27 (3 invented)     | 17/20 = 85%         |         0 |           0 |       0 | S0   | +32 / -4             | B is better (p = 0.000)              | B is better (+39/-3, p = 0.000)              |
| M4@S0    | 143 | 90/123 = 73% [65%–80%]  | 80/123 = 65% [56%–73%]  | 94/143 = 66% [58%–73%]  | 59/143 = 41% [34%–49%]  | 0.478 | 10/10        | 0/9           | 22/123 = 18%                  | 27 (3 invented)     | 17/20 = 85%         |         0 |           0 |       0 | S0   | +27 / -6             | B is better (p = 0.000)              | B is better (+30/-4, p = 0.000)              |
| M1@S6    | 143 | 100/123 = 81% [74%–87%] | 115/123 = 93% [88%–97%] | 138/143 = 97% [92%–98%] | 137/143 = 96% [91%–98%] | 0.63  | 10/10        | 9/9           | 4/123 = 3% (1 needs a person) | 1                   | 20/20 = 100%        |         0 |           0 |       0 | S6   | +13 / -4             | B is better (p = 0.049)              | no detectable difference (+14/-7, p = 0.189) |
| M2@S6    | 143 | 99/123 = 80% [73%–87%]  | 113/123 = 92% [86%–96%] | 133/143 = 93% [88%–96%] | 132/143 = 92% [87%–96%] | 0.611 | 10/10        | 7/9           | 6/123 = 5%                    | 4 (1 invented)      | 19/20 = 95%         |         0 |           0 |       0 | S6   | +5 / -1              | no detectable difference (p = 0.219) | no detectable difference (+7/-2, p = 0.180)  |
| M3@S6    | 143 | 104/123 = 85% [77%–90%] | 115/123 = 93% [88%–97%] | 133/143 = 93% [88%–96%] | 132/143 = 92% [87%–96%] | 0.776 | 10/10        | 8/9           | 7/123 = 6%                    | 3 (1 invented)      | 19/20 = 95%         |         0 |           0 |       0 | S6   | +5 / -1              | no detectable difference (p = 0.219) | B is better (+7/-0, p = 0.016)               |
| M4@S6    | 143 | 103/123 = 84% [76%–89%] | 114/123 = 93% [87%–96%] | 132/143 = 92% [87%–96%] | 131/143 = 92% [86%–95%] | 0.681 | 10/10        | 7/9           | 8/123 = 7% (1 needs a person) | 3                   | 20/20 = 100%        |         0 |           0 |       0 | S6   | +4 / -1              | no detectable difference (p = 0.375) | B is better (+6/-0, p = 0.031)               |

## Mechanisms against plant knowledge

Each mechanism at two points: instead of plant knowledge (on S0) and on top of it (on S6), compared in pairs with S0 or S6. End-to-end by slice in counts; the holdout column is end-to-end on the 27 role questions.

| config   | vs   | recall@5   | end-to-end   | e2e: fixed / broke   | e2e verdict                          | language recall@5   | language   | point_in_time   | applicability   | conflict   | table_row   | tag_variant   | multi_evidence   | holdout     |
|:---------|:-----|:-----------|:-------------|:---------------------|:-------------------------------------|:--------------------|:-----------|:----------------|:----------------|:-----------|:------------|:--------------|:-----------------|:------------|
| S0       |      | 50%        | 51%          |                      |                                      | 3/18                | 5/20       | 4/10            | 2/12            | 0/9        | 6/8         | 0/6           | 3/8              | 15/27 = 56% |
| M1@S0    | S0   | 68%        | 63%          | +32 / -15            | B is better (p = 0.019)              | 15/18               | 17/20      | 3/10            | 3/12            | 0/9        | 8/8         | 6/6           | 2/8              | 16/27 = 59% |
| M2@S0    | S0   | 63%        | 60%          | +23 / -10            | B is better (p = 0.035)              | 5/18                | 8/20       | 7/10            | 3/12            | 0/9        | 8/8         | 3/6           | 2/8              | 18/27 = 67% |
| M3@S0    | S0   | 76%        | 71%          | +32 / -4             | B is better (p = 0.000)              | 7/18                | 10/20      | 7/10            | 8/12            | 0/9        | 8/8         | 2/6           | 3/8              | 23/27 = 85% |
| M4@S0    | S0   | 73%        | 66%          | +27 / -6             | B is better (p = 0.000)              | 7/18                | 10/20      | 5/10            | 6/12            | 0/9        | 8/8         | 2/6           | 3/8              | 22/27 = 81% |
| S6       | S5   | 76%        | 90%          | +7 / -1              | no detectable difference (p = 0.070) | 8/18                | 11/20      | 10/10           | 12/12           | 7/9        | 8/8         | 6/6           | 7/8              | 21/27 = 78% |
| M1@S6    | S6   | 81%        | 97%          | +13 / -4             | B is better (p = 0.049)              | 18/18               | 20/20      | 10/10           | 10/12           | 9/9        | 7/8         | 5/6           | 8/8              | 25/27 = 93% |
| M2@S6    | S6   | 80%        | 93%          | +5 / -1              | no detectable difference (p = 0.219) | 11/18               | 14/20      | 10/10           | 12/12           | 7/9        | 8/8         | 6/6           | 8/8              | 23/27 = 85% |
| M3@S6    | S6   | 85%        | 93%          | +5 / -1              | no detectable difference (p = 0.219) | 11/18               | 13/20      | 10/10           | 12/12           | 8/9        | 8/8         | 6/6           | 8/8              | 24/27 = 89% |
| M4@S6    | S6   | 84%        | 92%          | +4 / -1              | no detectable difference (p = 0.375) | 11/18               | 13/20      | 10/10           | 11/12           | 7/9        | 8/8         | 6/6           | 8/8              | 22/27 = 81% |

## The answer contract: v1, v2, v3 (and v2 in thinking mode)

Search and context are the same in every version: evidence recall does not change; only the answer step does.

- **v1** (`S6:v1`, `M*@S6:v1`, the first reference run): the model copies citation objects (document, revision, clause, table, row, page) from the source headers; no limits. Long answers ran out of the 3000 tokens.
- **v2** (`S6:v2`, `M*@S6:v2`): a claim cites the **ids of the chunks** in the context, and the JSON schema makes the id an `enum` of exactly those ids (built per question); at most 5 claims, at most 2 sources per claim; code fills the citation fields from the index. An answer cut by the token limit (3000, unchanged) or not of the schema is asked once more with a short instruction; if it fails again, the answer is `needs_human` with no value (a fallback).
- **v3** (`S6` and every row built on it): v2 plus an internal field `check` first in the schema (at most 300 characters: do the sources agree, is the number in the cited text, is the answer in the documents at all), logged and never shown or graded, and `maxLength` 600 on `answer`. A leak detector in code (markers such as "Wait", "Let me", "Actually", or an answer cut at the limit) asks once more, then falls back to `needs_human`. The internal field records a brief consistency check. It is not evidence of the model's reasoning or of answer correctness; the version comparison measures observed outputs only.
- **v2 in thinking mode** (`S6:think`, profile `course-thinking`): the reasoning goes to its own channel (not logged in the replay); the server applies the JSON schema to the content after it (checked).

Every S6-based row (S6, M1@S6..M4@S6) is measured with v3; its v1 and v2 answers stay in the replay as `X:v1` and `X:v2` rows.

Guided decoding (vLLM behind LiteLLM) enforces the enum, `maxItems` and `maxLength`: asked to cite an id that is not in the context, or to write 9 claims with 4 sources each, or a long essay, the model returned only context ids, 5 claims, 2 sources per claim, strings cut at the limit. Code checks it anyway: 0 schema violations in all v2 and v3 answers.

Main set (143 questions), new against old on the same questions; s / q and tokens are medians and means per question, all chat calls (a retry counts; tokens out of the thinking row include the reasoning):

| new vs old           | invalid   | leaks   | fallbacks   | end-to-end old   | end-to-end new   | new vs old: fixed / broke   | verdict                              | false refusals   | wrong or invented               | verified                      | evidence recall   | s / q        | chat calls / q   | tokens in / q   | tokens out / q   |
|:---------------------|:----------|:--------|:------------|:-----------------|:-----------------|:----------------------------|:-------------------------------------|:-----------------|:--------------------------------|:------------------------------|:------------------|:-------------|:-----------------|:----------------|:-----------------|
| S6:v2 vs S6:v1       | 0 → 0     | 0 → 0   | 0 → 0       | 130/143 = 91%    | 128/143 = 90%    | +0 / -2                     | no detectable difference (p = 0.500) | 12/123 → 12/123  | 1 → 3 (1 invented)              | 128/143 = 90% → 127/143 = 89% | 108/123 = 88%     | 4.77 → 4.42  | 1.0 → 1.0        | 3036 → 3225     | 374 → 214        |
| M1@S6:v2 vs M1@S6:v1 | 5 → 0     | 0 → 0   | 0 → 0       | 133/143 = 93%    | 136/143 = 95%    | +5 / -2                     | no detectable difference (p = 0.453) | 4/123 → 4/123    | 1 → 3 (1 invented)              | 130/143 = 91% → 136/143 = 95% | 115/123 = 93%     | 5.51 → 4.99  | 1.0 → 1.0        | 3386 → 3603     | 468 → 215        |
| M2@S6:v2 vs M2@S6:v1 | 5 → 0     | 0 → 0   | 0 → 0       | 130/143 = 91%    | 133/143 = 93%    | +5 / -2                     | no detectable difference (p = 0.453) | 6/123 → 6/123    | 2 → 4 (1 invented)              | 127/143 = 89% → 133/143 = 93% | 113/123 = 92%     | 5.43 → 5.2   | 1.0 → 1.0        | 3384 → 3601     | 483 → 221        |
| M3@S6:v2 vs M3@S6:v1 | 3 → 0     | 0 → 0   | 0 → 0       | 130/143 = 91%    | 131/143 = 92%    | +3 / -2                     | no detectable difference (p = 1.000) | 7/123 → 7/123    | 3 (1 invented) → 5 (2 invented) | 128/143 = 90% → 130/143 = 91% | 115/123 = 93%     | 6.36 → 5.66  | 2.0 → 2.0        | 4534 → 4726     | 486 → 261        |
| M4@S6:v2 vs M4@S6:v1 | 4 → 0     | 0 → 0   | 0 → 0       | 129/143 = 90%    | 132/143 = 92%    | +4 / -1                     | no detectable difference (p = 0.375) | 8/123 → 8/123    | 2 → 3                           | 127/143 = 89% → 132/143 = 92% | 114/123 = 93%     | 5.16 → 4.86  | 1.0 → 1.0        | 3295 → 3503     | 449 → 217        |
| S6 vs S6:v2          | 0 → 0     | 0 → 0   | 0 → 0       | 128/143 = 90%    | 129/143 = 90%    | +2 / -1                     | no detectable difference (p = 1.000) | 12/123 → 11/123  | 3 (1 invented) → 3              | 127/143 = 89% → 128/143 = 90% | 108/123 = 88%     | 4.42 → 5.79  | 1.0 → 1.0        | 3225 → 3325     | 214 → 266        |
| M1@S6 vs M1@S6:v2    | 0 → 0     | 0 → 0   | 0 → 0       | 136/143 = 95%    | 138/143 = 97%    | +2 / -0                     | no detectable difference (p = 0.500) | 4/123 → 4/123    | 3 (1 invented) → 1              | 136/143 = 95% → 137/143 = 96% | 115/123 = 93%     | 4.99 → 6.01  | 1.0 → 1.0        | 3603 → 3703     | 215 → 272        |
| M2@S6 vs M2@S6:v2    | 0 → 0     | 0 → 0   | 0 → 0       | 133/143 = 93%    | 133/143 = 93%    | +0 / -0                     | no detectable difference (p = 1.000) | 6/123 → 6/123    | 4 (1 invented) → 4 (1 invented) | 133/143 = 93% → 132/143 = 92% | 113/123 = 92%     | 5.2 → 6.87   | 1.0 → 1.01       | 3601 → 3726     | 221 → 297        |
| M3@S6 vs M3@S6:v2    | 0 → 0     | 0 → 0   | 0 → 0       | 131/143 = 92%    | 133/143 = 93%    | +2 / -0                     | no detectable difference (p = 0.500) | 7/123 → 7/123    | 5 (2 invented) → 3 (1 invented) | 130/143 = 91% → 132/143 = 92% | 115/123 = 93%     | 5.66 → 6.73  | 2.0 → 2.0        | 4726 → 4826     | 261 → 316        |
| M4@S6 vs M4@S6:v2    | 0 → 0     | 0 → 0   | 0 → 0       | 132/143 = 92%    | 132/143 = 92%    | +0 / -0                     | no detectable difference (p = 1.000) | 8/123 → 8/123    | 3 → 3                           | 132/143 = 92% → 131/143 = 92% | 114/123 = 93%     | 4.86 → 5.87  | 1.0 → 1.0        | 3503 → 3603     | 217 → 278        |
| S6:think vs S6:v2    | 0 → 0     | 0 → 0   | 0 → 1       | 128/143 = 90%    | 123/143 = 86%    | +1 / -6                     | no detectable difference (p = 0.125) | 12/123 → 14/123  | 3 (1 invented) → 6 (1 invented) | 127/143 = 89% → 123/143 = 86% | 108/123 = 88%     | 4.42 → 17.78 | 1.0 → 1.03       | 3225 → 3396     | 214 → 2022       |

End-to-end by slice:

| config   | multi_evidence   | what_changed   | conflict   | language   | point_in_time   |
|:---------|:-----------------|:---------------|:-----------|:-----------|:----------------|
| S6:v1    | 7/8              | 3/4            | 8/9        | 11/20      | 10/10           |
| S6:v2    | 7/8              | 3/4            | 7/9        | 11/20      | 10/10           |
| M1@S6:v1 | 7/8              | 3/4            | 9/9        | 20/20      | 10/10           |
| M1@S6:v2 | 8/8              | 3/4            | 8/9        | 20/20      | 10/10           |
| M2@S6:v1 | 8/8              | 3/4            | 8/9        | 14/20      | 10/10           |
| M2@S6:v2 | 8/8              | 3/4            | 7/9        | 14/20      | 10/10           |
| M3@S6:v1 | 8/8              | 3/4            | 8/9        | 13/20      | 10/10           |
| M3@S6:v2 | 8/8              | 3/4            | 7/9        | 13/20      | 10/10           |
| M4@S6:v1 | 7/8              | 3/4            | 8/9        | 12/20      | 10/10           |
| M4@S6:v2 | 8/8              | 3/4            | 7/9        | 13/20      | 10/10           |
| S6       | 7/8              | 2/4            | 7/9        | 11/20      | 10/10           |
| M1@S6    | 8/8              | 3/4            | 9/9        | 20/20      | 10/10           |
| M2@S6    | 8/8              | 3/4            | 7/9        | 14/20      | 10/10           |
| M3@S6    | 8/8              | 3/4            | 8/9        | 13/20      | 10/10           |
| M4@S6    | 8/8              | 3/4            | 7/9        | 13/20      | 10/10           |
| S6:think | 4/8              | 2/4            | 7/9        | 11/20      | 10/10           |

Questions to watch (m01: the lecture question; i01: the incident; f03: a conflict answered correctly by v1; n04: "not in the documents"); `check` is the internal field of v3: logged for the audit, never shown to the engineer:

| row      | question   | status           | value   | right   | verified   | retried   | fallback   |   leaks | check (internal)                                                                                                                                                 |
|:---------|:-----------|:-----------------|:--------|:--------|:-----------|:----------|:-----------|--------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| S6:v1    | m01        | not_in_documents |         | False   | False      | False     | False      |       0 |                                                                                                                                                                  |
| S6:v1    | i01        | answered         | 5,0     | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| S6:v1    | f03        | conflict         |         | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| S6:v1    | n04        | not_in_documents |         | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| S6:v2    | m01        | not_in_documents |         | False   | False      | False     | False      |       0 |                                                                                                                                                                  |
| S6:v2    | i01        | answered         | 5,0     | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| S6:v2    | f03        | answered         | 6,0     | False   | False      | False     | False      |       0 |                                                                                                                                                                  |
| S6:v2    | n04        | answered         | 0,05    | False   | False      | False     | False      |       0 |                                                                                                                                                                  |
| S6       | m01        | not_in_documents |         | False   | False      | False     | False      |       0 | The question asks if 4.8 mm/s vibration is normal for pump P-101 (Class II) and what to do. Source PS-101 identifies P-101 as Class II. Source R-12-r3-4.4.2 sta |
| S6       | i01        | answered         | 5,0     | True    | True       | False     | False      |       0 | P-101 is Class II. Table 3 (R-12) sets inspection limit at 5.0 mm/s. Measured 4.8 mm/s is below the limit. Clause 4.3.3 requires inspection within 24h only if t |
| S6       | f03        | answered         | 4,5     | False   | False      | False     | False      |       0 | P-205 is Class II (R-12 Table A.1). R-12 Table 3 sets limits for Pumps Class II: Inspection 4.5 mm/s, Stop 7.1 mm/s. Measured 6.0 mm/s is > 4.5 and < 7.1. OM-P- |
| S6       | n04        | not_in_documents |         | True    | True       | False     | False      |       0 | The question asks for the permissible shaft runout (biение) after repair. Source [R-12-r3-7.2] specifies a tolerance for 'radial displacement' (radialnoe smeshc |
| M1@S6:v1 | m01        | invalid          |         | False   | False      | False     | False      |       0 |                                                                                                                                                                  |
| M1@S6:v1 | i01        | invalid          |         | False   | False      | False     | False      |       0 |                                                                                                                                                                  |
| M1@S6:v1 | f03        | conflict         | 6,0     | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M1@S6:v1 | n04        | not_in_documents |         | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M1@S6:v2 | m01        | answered         | 4,5     | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M1@S6:v2 | i01        | answered         | 5,0     | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M1@S6:v2 | f03        | answered         | 4,5     | False   | False      | False     | False      |       0 |                                                                                                                                                                  |
| M1@S6:v2 | n04        | answered         | 0,05    | False   | False      | False     | False      |       0 |                                                                                                                                                                  |
| M1@S6    | m01        | answered         | 4,5     | True    | True       | False     | False      |       0 | P-101 is Class II (PS-101). R-12 Table 3 sets limits for Class II pumps: inspection 4.5 mm/s, stop 7.1 mm/s. Measured 4.8 mm/s exceeds inspection limit but is b |
| M1@S6    | i01        | answered         | 5,0     | True    | True       | False     | False      |       0 | P-101 is Class II. R-12 (in force 2025-02-10) Table 3 sets inspection limit at 5.0 mm/s. Measured 4.8 mm/s is below the limit. Clause 4.3.3 requires inspection  |
| M1@S6    | f03        | conflict         |         | True    | True       | False     | False      |       0 | P-205 is a pump, class II. R-12 (plant reg) Table 3: inspection limit 4.5 mm/s, stop limit 7.1 mm/s. OM-P-205 (manual) Table 2: normal up to 7.1 mm/s, alarm >7. |
| M1@S6    | n04        | not_in_documents |         | True    | True       | False     | False      |       0 | The question asks for the permissible shaft runout (biение) after repair. Source [R-12-r3-7.2] specifies a tolerance for 'radial displacement' (radial offset/ce |
| M2@S6:v1 | m01        | answered         | 4,5     | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M2@S6:v1 | i01        | answered         | 5,0     | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M2@S6:v1 | f03        | conflict         |         | True    | False      | False     | False      |       0 |                                                                                                                                                                  |
| M2@S6:v1 | n04        | not_in_documents |         | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M2@S6:v2 | m01        | answered         | 4,5     | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M2@S6:v2 | i01        | answered         | 5,0     | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M2@S6:v2 | f03        | answered         | 4,5     | False   | False      | False     | False      |       0 |                                                                                                                                                                  |
| M2@S6:v2 | n04        | answered         | 0,05    | False   | False      | False     | False      |       0 |                                                                                                                                                                  |
| M2@S6    | m01        | answered         | 4,5     | True    | True       | False     | False      |       0 | P-101 is Class II (PS-101). R-12 Table 3 sets limits for Class II pumps: 4.5 mm/s (inspection) and 7.1 mm/s (stop). Measured 4.8 mm/s exceeds inspection limit b |
| M2@S6    | i01        | answered         | 5,0     | True    | True       | False     | False      |       0 | P-101 is Class II. R-12 Table 3 sets inspection limit at 5.0 mm/s. Measured 4.8 mm/s is below the limit. Clause 4.3.3 requires inspection only upon exceeding th |
| M2@S6    | f03        | answered         | 4,5     | False   | False      | False     | False      |       0 | P-205 is Class II (R-12 Table A.1). R-12 Table 3 sets limits for Class II pumps: inspection 4.5 mm/s, stop 7.1 mm/s. Measured 6.0 mm/s is between these limits.  |
| M2@S6    | n04        | answered         | 0,05    | False   | False      | False     | False      |       0 | The question asks for 'shaft beat' (biение) tolerance. Source [R-12-r3-7.2] specifies a tolerance for 'radial displacement' (radialnoe smeshchenie) of 0.05 mm d |
| M3@S6:v1 | m01        | answered         | 4,5     | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M3@S6:v1 | i01        | answered         | 5,0     | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M3@S6:v1 | f03        | conflict         |         | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M3@S6:v1 | n04        | not_in_documents |         | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M3@S6:v2 | m01        | answered         | 4,5     | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M3@S6:v2 | i01        | answered         | 5,0     | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M3@S6:v2 | f03        | answered         | 4,5     | False   | False      | False     | False      |       0 |                                                                                                                                                                  |
| M3@S6:v2 | n04        | answered         | 0,05    | False   | False      | False     | False      |       0 |                                                                                                                                                                  |
| M3@S6    | m01        | answered         | 4,5     | True    | True       | False     | False      |       0 | P-101 is Class II (R-12-r3-table3, PS-101). Vibration 4.8 mm/s exceeds the 'inspection' limit of 4.5 mm/s but is below the 'stop' limit of 7.1 mm/s (R-12-r3-tab |
| M3@S6    | i01        | answered         | 5,0     | True    | True       | False     | False      |       0 | P-101 is Class II (R-12-r2-table3, PS-101). R-12-r2-table1 sets inspection limit at 5.0 mm/s. Measured 4.8 mm/s is below 5.0 mm/s. Clause 4.3.3 requires inspect |
| M3@S6    | f03        | conflict         |         | True    | True       | False     | False      |       0 | P-205 is Class II (37 kW). R-12 (plant reg) limits: 4.5 mm/s (inspect), 7.1 mm/s (stop). OM-P-205 (manual) limits: 7.1 mm/s (alarm), 11.2 mm/s (trip). 6.0 mm/s  |
| M3@S6    | n04        | answered         | 0,05    | False   | False      | False     | False      |       0 | The question asks for shaft runout (biение) tolerance. Source [R-12-r3-7.2] specifies a radial displacement (centring) tolerance of 0.05 mm after repair. It doe |
| M4@S6:v1 | m01        | answered         | 4,5     | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M4@S6:v1 | i01        | answered         |         | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M4@S6:v1 | f03        | conflict         |         | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M4@S6:v1 | n04        | not_in_documents |         | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M4@S6:v2 | m01        | answered         | 4,5     | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M4@S6:v2 | i01        | answered         | 5,0     | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M4@S6:v2 | f03        | answered         | 4,5     | False   | False      | False     | False      |       0 |                                                                                                                                                                  |
| M4@S6:v2 | n04        | not_in_documents |         | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| M4@S6    | m01        | answered         | 4,5     | True    | True       | False     | False      |       0 | P-101 is Class II (Table A.1, PS-101). Table 3 limits for Class II: Inspection 4.5 mm/s, Stop 7.1 mm/s. Measured 4.8 mm/s exceeds Inspection limit but is below  |
| M4@S6    | i01        | answered         | 5,0     | True    | True       | False     | False      |       0 | P-101 is Class II pump at Station 2. R-12 (rev 2) was in force on 2025-02-10. Table 3 sets inspection limit for Class II pumps at 5.0 mm/s. Measured 4.8 mm/s is |
| M4@S6    | f03        | answered         | 4,5     | False   | False      | False     | False      |       0 | P-205 is a pump, class II (R-12 Table A.1). R-12 (in force 2025-03-14) sets limits for class II: inspection 4.5 mm/s, stop 7.1 mm/s. OM-P-205 (in force 2023-05- |
| M4@S6    | n04        | not_in_documents |         | True    | True       | False     | False      |       0 | The question asks for the permissible shaft runout (biение) after repair. Source [R-12-r3-7.2] specifies a tolerance for 'radial displacement' (radial shift/ali |
| S6:think | m01        | not_in_documents |         | False   | False      | False     | False      |       0 |                                                                                                                                                                  |
| S6:think | i01        | answered         | 5,0     | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| S6:think | f03        | conflict         |         | True    | True       | False     | False      |       0 |                                                                                                                                                                  |
| S6:think | n04        | answered         | 0,05    | False   | False      | False     | False      |       0 |                                                                                                                                                                  |

Holdout (27):

| new vs old           | invalid   | leaks   | fallbacks   | end-to-end old   | end-to-end new   | new vs old: fixed / broke   | false refusals   | wrong or invented               |
|:---------------------|:----------|:--------|:------------|:-----------------|:-----------------|:----------------------------|:-----------------|:--------------------------------|
| S6:v2 vs S6:v1       | 0 → 0     | 0 → 0   | 0 → 0       | 21/27 = 78%      | 21/27 = 78%      | +0 / -0                     | 3/23 → 4/23      | 3 (1 invented) → 2 (1 invented) |
| M1@S6:v2 vs M1@S6:v1 | 1 → 0     | 0 → 0   | 0 → 0       | 22/27 = 81%      | 24/27 = 89%      | +2 / -0                     | 2/23 → 2/23      | 2 (2 invented) → 1 (1 invented) |
| M2@S6:v2 vs M2@S6:v1 | 1 → 0     | 0 → 0   | 0 → 0       | 21/27 = 78%      | 22/27 = 81%      | +1 / -0                     | 2/23 → 3/23      | 3 (1 invented) → 2 (1 invented) |
| M3@S6:v2 vs M3@S6:v1 | 0 → 0     | 0 → 0   | 0 → 0       | 23/27 = 85%      | 23/27 = 85%      | +0 / -0                     | 2/23 → 2/23      | 2 (1 invented) → 2 (1 invented) |
| M4@S6:v2 vs M4@S6:v1 | 0 → 0     | 0 → 0   | 0 → 0       | 22/27 = 81%      | 22/27 = 81%      | +0 / -0                     | 3/23 → 3/23      | 2 (1 invented) → 2 (1 invented) |
| S6 vs S6:v2          | 0 → 0     | 0 → 0   | 0 → 0       | 21/27 = 78%      | 21/27 = 78%      | +0 / -0                     | 4/23 → 4/23      | 2 (1 invented) → 2 (1 invented) |
| M1@S6 vs M1@S6:v2    | 0 → 0     | 0 → 0   | 0 → 0       | 24/27 = 89%      | 25/27 = 93%      | +1 / -0                     | 2/23 → 2/23      | 1 (1 invented) → 0              |
| M2@S6 vs M2@S6:v2    | 0 → 0     | 0 → 0   | 0 → 0       | 22/27 = 81%      | 23/27 = 85%      | +1 / -0                     | 3/23 → 3/23      | 2 (1 invented) → 1              |
| M3@S6 vs M3@S6:v2    | 0 → 0     | 0 → 0   | 0 → 0       | 23/27 = 85%      | 24/27 = 89%      | +1 / -0                     | 2/23 → 2/23      | 2 (1 invented) → 1              |
| M4@S6 vs M4@S6:v2    | 0 → 0     | 0 → 0   | 0 → 0       | 22/27 = 81%      | 22/27 = 81%      | +0 / -0                     | 3/23 → 3/23      | 2 (1 invented) → 2 (1 invented) |
| S6:think vs S6:v2    | 0 → 0     | 0 → 0   | 0 → 1       | 21/27 = 78%      | 21/27 = 78%      | +1 / -1                     | 4/23 → 4/23      | 2 (1 invented) → 2              |

Needs a person (6): see the table in the next section (the main rows are v3; `:v1`, `:v2`, `S6:think` are the other versions).

## How the rows fail: refusing against inventing

The columns `false refusals` .. `invalid` split the wrong answers by kind (every question is exactly one of: right, false refusal, wrong or invented, invalid). A false refusal is "not in the documents" or "needs a person" on a question whose answer (or conflict) is in the documents: out of the 123 such questions. Wrong or invented: a value that is not the expected one, one side of a conflict given as the answer, or a value where the right answer is "not in the documents". Right abstentions: out of the 20 questions whose answer is not in the documents ("needs a person" is accepted there: no value is given). Invalid: no answer of the schema.

False refusals add manual work. Wrong or invented values can mislead decisions. S0 fails both ways; S6 almost only by refusing; the reranker alone (M3@S0) refuses less than S0 but keeps giving wrong values and invents values on "not in the documents" questions.

## Latency and cost per question

Median seconds per question = search (measured on the reference machine) + model latency of every call (as measured when the answer was made; replay keeps it). Calls and tokens are per question, all calls counted (an uncached system).

| config   |   median s / question |   p90 s |   search ms (median) |   model calls / question |   chat calls / question |   prompt tokens / question |   completion tokens / question |   cross-encoder tokens / question |
|:---------|----------------------:|--------:|---------------------:|-------------------------:|------------------------:|---------------------------:|-------------------------------:|----------------------------------:|
| S0       |                  1.46 |    2.28 |                  2.9 |                     1    |                    1    |                        843 |                             84 |                                 0 |
| S1       |                  1.46 |    2.28 |                  3   |                     1    |                    1    |                        864 |                             84 |                                 0 |
| S2       |                  1.4  |    2.12 |                  0.3 |                     1    |                    1    |                        727 |                             84 |                                 0 |
| S2b      |                  1.48 |    2.13 |                  4.2 |                     1    |                    1    |                        839 |                            106 |                                 0 |
| S3       |                  1.68 |    2.24 |                  0.5 |                     1    |                    1    |                       1083 |                             84 |                                 0 |
| S4       |                  1.76 |    2.6  |                  0.5 |                     1    |                    1    |                       1135 |                             83 |                                 0 |
| S5       |                  3.05 |    4.06 |                  0.7 |                     1    |                    1    |                       2811 |                             88 |                                 0 |
| S6       |                  5.79 |    8.72 |                  3.7 |                     1    |                    1    |                       3325 |                            266 |                                 0 |
| M1t      |                  1.93 |    2.52 |                  0.3 |                     2    |                    2    |                        769 |                            102 |                                 0 |
| M1@S0    |                  1.97 |    3.28 |                  0.9 |                     2    |                    1    |                       1298 |                             85 |                                 0 |
| M2@S0    |                  1.9  |    2.6  |                  2   |                     2    |                    1    |                       1073 |                             84 |                                 0 |
| M3@S0    |                  3.52 |    4.43 |                  0.3 |                     2    |                    2    |                       2895 |                            136 |                                 0 |
| M4@S0    |                  1.99 |    2.7  |                  1.3 |                     2    |                    1    |                       1341 |                             87 |                              1751 |
| M1@S6    |                  6.01 |    8.7  |                  3.4 |                     2    |                    1    |                       3703 |                            272 |                                 0 |
| M2@S6    |                  6.87 |   12.25 |                  5.8 |                     2.01 |                    1.01 |                       3726 |                            297 |                                 0 |
| M3@S6    |                  6.73 |    8.76 |                  4.3 |                     2    |                    2    |                       4826 |                            316 |                                 0 |
| M4@S6    |                  5.87 |    7.75 |                  4.1 |                     2    |                    1    |                       3603 |                            278 |                              1486 |
| S6:v1    |                  4.77 |    8.9  |                  0.7 |                     1    |                    1    |                       3036 |                            374 |                                 0 |
| M1@S6:v1 |                  5.51 |    8.82 |                  3.2 |                     2    |                    1    |                       3386 |                            468 |                                 0 |
| M2@S6:v1 |                  5.43 |    9.99 |                  5.5 |                     2    |                    1    |                       3384 |                            483 |                                 0 |
| M3@S6:v1 |                  6.36 |   11.28 |                  0.7 |                     2    |                    2    |                       4534 |                            486 |                                 0 |
| M4@S6:v1 |                  5.16 |    9.45 |                  4.4 |                     2    |                    1    |                       3295 |                            449 |                              1486 |
| S6:v2    |                  4.42 |    6.28 |                  3.7 |                     1    |                    1    |                       3225 |                            214 |                                 0 |
| M1@S6:v2 |                  4.99 |    6.8  |                  3.2 |                     2    |                    1    |                       3603 |                            215 |                                 0 |
| M2@S6:v2 |                  5.2  |    6.93 |                  5.7 |                     2    |                    1    |                       3601 |                            221 |                                 0 |
| M3@S6:v2 |                  5.66 |    7.84 |                  4.3 |                     2    |                    2    |                       4726 |                            261 |                                 0 |
| M4@S6:v2 |                  4.86 |    7.38 |                  4.4 |                     2    |                    1    |                       3503 |                            217 |                              1486 |
| S6:think |                 17.78 |   59.66 |                  3.6 |                     1.03 |                    1.03 |                       3396 |                           2022 |                                 0 |

## Mixed rows (S2) against a separate row collection (S2b): the questions S2 broke

The first reference run (rows mixed with the clauses) broke these questions; True = right.

| case_id   | S1    | S2    | S2b   | S3    |
|:----------|:------|:------|:------|:------|
| d01       | False | False | False | True  |
| d02       | False | False | False | True  |
| d04       | True  | False | True  | True  |
| z01       | True  | False | True  | True  |
| z02       | True  | False | True  | True  |
| z04       | True  | False | True  | True  |
| z05       | True  | False | True  | True  |
| i02       | True  | False | True  | False |
| g01       | True  | False | True  | True  |

## Holdout: questions written by role, not by slice

27 questions drafted by `holdout (iairlab/gpt-oss-120b)` from the documents and four role descriptions (operator, mechanic, reliability engineer, auditor), filtered by code (evidence resolves, value in the evidence, lexical leak <= 0.6, no near-copy of the retrieval set) and by hand: 27 of 75 drafts kept. The ladder was never shown to the drafter.

| config   |   n | recall@5              | evidence recall       | end-to-end            | correct + verified    |   MRR | false refusals   | wrong or invented   | right abstentions            |   invalid |   fallbacks |   leaks | vs   | e2e: fixed / broke   | e2e verdict                          | evidence verdict                            |
|:---------|----:|:----------------------|:----------------------|:----------------------|:----------------------|------:|:-----------------|:--------------------|:-----------------------------|----------:|------------:|--------:|:-----|:---------------------|:-------------------------------------|:--------------------------------------------|
| S0       |  27 | 13/23 = 57% [37%–74%] | 10/23 = 43% [26%–63%] | 15/27 = 56% [37%–72%] | 8/27 = 30% [16%–48%]  | 0.303 | 8/23 = 35%       | 4 (1 invented)      | 3/4 = 75%                    |         0 |           0 |       0 |      |                      |                                      |                                             |
| S1       |  27 | 13/23 = 57% [37%–74%] | 10/23 = 43% [26%–63%] | 15/27 = 56% [37%–72%] | 8/27 = 30% [16%–48%]  | 0.303 | 8/23 = 35%       | 4 (1 invented)      | 3/4 = 75%                    |         0 |           0 |       0 | S0   | +0 / -0              | no detectable difference (p = 1.000) | no detectable difference (+0/-0, p = 1.000) |
| S2       |  27 | 15/23 = 65% [45%–81%] | 10/23 = 43% [26%–63%] | 15/27 = 56% [37%–72%] | 6/27 = 22% [11%–41%]  | 0.313 | 4/23 = 17%       | 8 (1 invented)      | 3/4 = 75%                    |         0 |           0 |       0 | S1   | +1 / -1              | no detectable difference (p = 1.000) | no detectable difference (+1/-1, p = 1.000) |
| S2b      |  27 | 12/23 = 52% [33%–71%] | 9/23 = 39% [22%–59%]  | 13/27 = 48% [31%–66%] | 6/27 = 22% [11%–41%]  | 0.241 | 8/23 = 35%       | 6 (1 invented)      | 3/4 = 75%                    |         0 |           0 |       0 | S1   | +0 / -2              | no detectable difference (p = 0.500) | no detectable difference (+0/-1, p = 1.000) |
| S3       |  27 | 15/23 = 65% [45%–81%] | 12/23 = 52% [33%–71%] | 20/27 = 74% [55%–87%] | 9/27 = 33% [19%–52%]  | 0.333 | 4/23 = 17%       | 3 (2 invented)      | 2/4 = 50%                    |         0 |           0 |       0 | S2b  | +8 / -1              | B is better (p = 0.039)              | no detectable difference (+3/-0, p = 0.250) |
| S4       |  27 | 17/23 = 74% [54%–87%] | 17/23 = 74% [54%–87%] | 19/27 = 70% [52%–84%] | 19/27 = 70% [52%–84%] | 0.443 | 4/23 = 17%       | 4 (2 invented)      | 2/4 = 50%                    |         0 |           0 |       0 | S3   | +0 / -1              | no detectable difference (p = 1.000) | no detectable difference (+6/-1, p = 0.125) |
| S5       |  27 | 17/23 = 74% [54%–87%] | 18/23 = 78% [58%–90%] | 20/27 = 74% [55%–87%] | 20/27 = 74% [55%–87%] | 0.443 | 3/23 = 13%       | 4 (2 invented)      | 2/4 = 50%                    |         0 |           0 |       0 | S4   | +1 / -0              | no detectable difference (p = 1.000) | no detectable difference (+1/-0, p = 1.000) |
| S6       |  27 | 17/23 = 74% [54%–87%] | 18/23 = 78% [58%–90%] | 21/27 = 78% [59%–89%] | 21/27 = 78% [59%–89%] | 0.443 | 4/23 = 17%       | 2 (1 invented)      | 3/4 = 75% (1 needs a person) |         0 |           0 |       0 | S5   | +1 / -0              | no detectable difference (p = 1.000) | no detectable difference (+0/-0, p = 1.000) |
| M1@S0    |  27 | 16/23 = 70% [49%–84%] | 13/23 = 57% [37%–74%] | 16/27 = 59% [41%–75%] | 11/27 = 41% [25%–59%] | 0.433 | 6/23 = 26%       | 5 (1 invented)      | 3/4 = 75%                    |         0 |           0 |       0 | S0   | +3 / -2              | no detectable difference (p = 1.000) | no detectable difference (+4/-1, p = 0.375) |
| M2@S0    |  27 | 15/23 = 65% [45%–81%] | 12/23 = 52% [33%–71%] | 18/27 = 67% [48%–81%] | 9/27 = 33% [19%–52%]  | 0.443 | 4/23 = 17%       | 5 (2 invented)      | 2/4 = 50%                    |         0 |           0 |       0 | S0   | +5 / -2              | no detectable difference (p = 0.453) | no detectable difference (+3/-1, p = 0.625) |
| M3@S0    |  27 | 20/23 = 87% [68%–95%] | 20/23 = 87% [68%–95%] | 23/27 = 85% [68%–94%] | 13/27 = 48% [31%–66%] | 0.591 | 2/23 = 9%        | 2 (1 invented)      | 3/4 = 75%                    |         0 |           0 |       0 | S0   | +8 / -0              | B is better (p = 0.008)              | B is better (+10/-0, p = 0.002)             |
| M4@S0    |  27 | 20/23 = 87% [68%–95%] | 19/23 = 83% [63%–93%] | 22/27 = 81% [63%–92%] | 14/27 = 52% [34%–69%] | 0.486 | 2/23 = 9%        | 3 (1 invented)      | 3/4 = 75%                    |         0 |           0 |       0 | S0   | +7 / -0              | B is better (p = 0.016)              | B is better (+9/-0, p = 0.004)              |
| M1@S6    |  27 | 17/23 = 74% [54%–87%] | 18/23 = 78% [58%–90%] | 25/27 = 93% [77%–98%] | 25/27 = 93% [77%–98%] | 0.624 | 2/23 = 9%        | 0                   | 4/4 = 100%                   |         0 |           0 |       0 | S6   | +4 / -0              | no detectable difference (p = 0.125) | no detectable difference (+3/-3, p = 1.000) |
| M2@S6    |  27 | 18/23 = 78% [58%–90%] | 17/23 = 74% [54%–87%] | 23/27 = 85% [68%–94%] | 23/27 = 85% [68%–94%] | 0.581 | 3/23 = 13%       | 1                   | 4/4 = 100%                   |         0 |           0 |       0 | S6   | +2 / -0              | no detectable difference (p = 0.500) | no detectable difference (+1/-2, p = 1.000) |
| M3@S6    |  27 | 19/23 = 83% [63%–93%] | 20/23 = 87% [68%–95%] | 24/27 = 89% [72%–96%] | 24/27 = 89% [72%–96%] | 0.804 | 2/23 = 9%        | 1                   | 4/4 = 100%                   |         0 |           0 |       0 | S6   | +3 / -0              | no detectable difference (p = 0.250) | no detectable difference (+2/-0, p = 0.500) |
| M4@S6    |  27 | 19/23 = 83% [63%–93%] | 19/23 = 83% [63%–93%] | 22/27 = 81% [63%–92%] | 22/27 = 81% [63%–92%] | 0.696 | 3/23 = 13%       | 2 (1 invented)      | 3/4 = 75% (1 needs a person) |         0 |           0 |       0 | S6   | +1 / -0              | no detectable difference (p = 1.000) | no detectable difference (+1/-0, p = 1.000) |

## The status "needs a person": its own small set

6 questions (`data/needs_human.jsonl`, slice `needs_human`, not in the retrieval set): the norm is in the documents, but the answer depends on equipment data that no document and no registry card has (a pump without power and class, a tag that matches two pumps, no tag, a spare pump without a station, a grease named only in a missing datasheet, a measurement in one direction of three). Right: only `needs_human`. `not_in_documents` does not meet the expected status; a value means the system chose a class, a station or a measurement that nobody gave it. S5 has no status field: it shows what the contract adds. Cells: status (value).

| question            | S5               | S6               | M1@S6          | M2@S6          | M3@S6            | M4@S6            | S6:v1            | M1@S6:v1    | M2@S6:v1    | M3@S6:v1    | M4@S6:v1    | S6:v2            | M1@S6:v2    | M2@S6:v2       | M3@S6:v2       | M4@S6:v2       | S6:think         |
|:--------------------|:-----------------|:-----------------|:---------------|:---------------|:-----------------|:-----------------|:-----------------|:------------|:------------|:------------|:------------|:-----------------|:------------|:---------------|:---------------|:---------------|:-----------------|
| nh01                | answered (2,8)   | needs_human      | needs_human    | needs_human    | needs_human      | needs_human      | invalid (length) | needs_human | needs_human | needs_human | needs_human | needs_human      | needs_human | needs_human    | needs_human    | needs_human    | needs_human      |
| nh02                | not_in_documents | needs_human      | needs_human    | needs_human    | needs_human      | needs_human      | needs_human      | needs_human | needs_human | needs_human | needs_human | needs_human      | needs_human | needs_human    | needs_human    | needs_human    | needs_human      |
| nh03                | not_in_documents | not_in_documents | needs_human    | needs_human    | needs_human      | needs_human      | not_in_documents | needs_human | needs_human | needs_human | needs_human | not_in_documents | needs_human | needs_human    | needs_human    | needs_human    | not_in_documents |
| nh04                | answered (4,5)   | answered (4,5)   | needs_human    | conflict       | conflict         | conflict         | answered (4,5)   | needs_human | conflict    | conflict    | conflict    | answered (4,5)   | needs_human | conflict       | conflict       | conflict       | answered (4,5)   |
| nh05                | not_in_documents | needs_human      | needs_human    | needs_human    | not_in_documents | not_in_documents | needs_human      | needs_human | needs_human | needs_human | needs_human | needs_human      | needs_human | needs_human    | needs_human    | needs_human    | needs_human      |
| nh06                | answered (4,5)   | answered (4,5)   | answered (4,5) | answered (4,2) | answered (4,5)   | answered (4,5)   | needs_human      | needs_human | needs_human | needs_human | needs_human | answered (4,5)   | needs_human | answered (4,5) | answered (4,5) | answered (4,5) | needs_human      |
| right (needs_human) | 0/6              | 3/6              | 5/6            | 4/6            | 3/6              | 3/6              | 3/6              | 6/6         | 5/6         | 5/6         | 5/6         | 3/6              | 6/6         | 4/6            | 4/6            | 4/6            | 4/6              |
| gave a value        | 3                | 2                | 1              | 1              | 1                | 1                | 1                | 0           | 0           | 0           | 0           | 2                | 0           | 1              | 1              | 1              | 1                |

Notes: nh01: registry: station 2, no power, no class, no datasheet; not in Table A.1. Class I: above 2,8, inspect within 24 h; class II: normal; nh02: no P-220 in the registry: P-220A (11 kW, class I) and P-220B (37 kW, class II). A: above 4,5, stop; B: inspect within 24 h; nh03: no tag: station 2 has class I and class II pumps. Class I: inspect; class II: normal; nh04: spare pump, class II, station not assigned: R-12 (station 2) 4,5, normal; R-14 (station 3) 3,5, inspect. Not a conflict: only one of the two applies, and the registry does not say which; nh05: R-12 7.1: the grease named in the datasheet; P-215 has no datasheet in the archive; nh06: R-12 4.3.1: the result is the largest of three directions; two are missing, so 4,2 < 4,5 proves nothing. Needs a person: measure again. The status is arguable (a careful 'answered' that asks to re-measure is also safe); graded by status.

`needs_human` on the retrieval set. On a "not in the documents" question it is accepted (grader rule, unchanged): S6:v1: x01; M1@S6:v1: x01; M3@S6:v1: x01; M4@S6:v2: x01; S6:think: x04. On a question with an answer it is a false refusal (counted in `false refusals`): M1@S6: p12; M4@S6: p12; S6:v1: k02; M1@S6:v1: p07, p12; M4@S6:v1: p12; S6:v2: k02; M1@S6:v2: p07, p12; M4@S6:v2: p12; S6:think: m05, k02, f07.

## End-to-end by slice (counts, not percentages)

| config   | base   | language   | applicability   | point_in_time   | incident   | what_changed   | multi_evidence   | cross_doc   | conflict   | table_row   | tag_variant   | not_in_documents   |
|:---------|:-------|:-----------|:----------------|:----------------|:-----------|:---------------|:-----------------|:------------|:-----------|:------------|:--------------|:-------------------|
| S0       | 41/50  | 5/20       | 2/12            | 4/10            | 1/3        | 1/4            | 3/8              | 0/3         | 0/9        | 6/8         | 0/6           | 10/10              |
| S1       | 41/50  | 5/20       | 2/12            | 4/10            | 1/3        | 1/4            | 3/8              | 0/3         | 0/9        | 6/8         | 5/6           | 10/10              |
| S2       | 38/50  | 5/20       | 4/12            | 4/10            | 0/3        | 1/4            | 3/8              | 0/3         | 0/9        | 8/8         | 4/6           | 10/10              |
| S2b      | 41/50  | 5/20       | 1/12            | 5/10            | 1/3        | 1/4            | 3/8              | 0/3         | 0/9        | 8/8         | 5/6           | 10/10              |
| S3       | 49/50  | 10/20      | 8/12            | 6/10            | 1/3        | 1/4            | 3/8              | 0/3         | 0/9        | 8/8         | 6/6           | 10/10              |
| S4       | 49/50  | 10/20      | 10/12           | 9/10            | 3/3        | 1/4            | 3/8              | 0/3         | 0/9        | 8/8         | 6/6           | 10/10              |
| S5       | 50/50  | 11/20      | 12/12           | 10/10           | 3/3        | 3/4            | 7/8              | 3/3         | 0/9        | 8/8         | 6/6           | 10/10              |
| S6       | 50/50  | 11/20      | 12/12           | 10/10           | 3/3        | 2/4            | 7/8              | 3/3         | 7/9        | 8/8         | 6/6           | 10/10              |
| M1t      | —      | 13/20      | —               | —               | —          | —              | —                | —           | —          | —           | —             | —                  |
| M1@S0    | 41/50  | 17/20      | 3/12            | 3/10            | 1/3        | 1/4            | 2/8              | 0/3         | 0/9        | 8/8         | 6/6           | 8/10               |
| M2@S0    | 43/50  | 8/20       | 3/12            | 7/10            | 1/3        | 1/4            | 2/8              | 0/3         | 0/9        | 8/8         | 3/6           | 10/10              |
| M3@S0    | 47/50  | 10/20      | 8/12            | 7/10            | 2/3        | 3/4            | 3/8              | 1/3         | 0/9        | 8/8         | 2/6           | 10/10              |
| M4@S0    | 45/50  | 10/20      | 6/12            | 5/10            | 2/3        | 3/4            | 3/8              | 0/3         | 0/9        | 8/8         | 2/6           | 10/10              |
| M1@S6    | 50/50  | 20/20      | 10/12           | 10/10           | 3/3        | 3/4            | 8/8              | 3/3         | 9/9        | 7/8         | 5/6           | 10/10              |
| M2@S6    | 49/50  | 14/20      | 12/12           | 10/10           | 3/3        | 3/4            | 8/8              | 3/3         | 7/9        | 8/8         | 6/6           | 10/10              |
| M3@S6    | 49/50  | 13/20      | 12/12           | 10/10           | 3/3        | 3/4            | 8/8              | 3/3         | 8/9        | 8/8         | 6/6           | 10/10              |
| M4@S6    | 50/50  | 13/20      | 11/12           | 10/10           | 3/3        | 3/4            | 8/8              | 3/3         | 7/9        | 8/8         | 6/6           | 10/10              |

## recall@5 by slice

| config   | base   | language   | applicability   | point_in_time   | incident   | what_changed   | multi_evidence   | cross_doc   | conflict   | table_row   | tag_variant   | not_in_documents   |
|:---------|:-------|:-----------|:----------------|:----------------|:-----------|:---------------|:-----------------|:------------|:-----------|:------------|:--------------|:-------------------|
| S0       | 32/42  | 3/18       | 3/12            | 7/10            | 2/3        | 1/4            | 4/8              | 0/3         | 3/9        | 6/8         | 0/6           | —                  |
| S1       | 32/42  | 3/18       | 3/12            | 7/10            | 2/3        | 1/4            | 4/8              | 0/3         | 3/9        | 6/8         | 4/6           | —                  |
| S2       | 31/42  | 3/18       | 5/12            | 8/10            | 2/3        | 1/4            | 4/8              | 0/3         | 3/9        | 8/8         | 4/6           | —                  |
| S2b      | 32/42  | 3/18       | 3/12            | 7/10            | 2/3        | 1/4            | 4/8              | 0/3         | 3/9        | 8/8         | 4/6           | —                  |
| S3       | 37/42  | 6/18       | 7/12            | 8/10            | 2/3        | 1/4            | 4/8              | 0/3         | 4/9        | 8/8         | 5/6           | —                  |
| S4       | 41/42  | 8/18       | 10/12           | 9/10            | 3/3        | 1/4            | 4/8              | 0/3         | 4/9        | 8/8         | 5/6           | —                  |
| S5       | 41/42  | 8/18       | 10/12           | 9/10            | 3/3        | 1/4            | 4/8              | 0/3         | 4/9        | 8/8         | 5/6           | —                  |
| S6       | 41/42  | 8/18       | 10/12           | 9/10            | 3/3        | 1/4            | 4/8              | 0/3         | 4/9        | 8/8         | 5/6           | —                  |
| M1t      | —      | 9/18       | —               | —               | —          | —              | —                | —           | —          | —           | —             | —                  |
| M1@S0    | 33/42  | 15/18      | 5/12            | 8/10            | 1/3        | 2/4            | 2/8              | 0/3         | 4/9        | 8/8         | 6/6           | —                  |
| M2@S0    | 36/42  | 5/18       | 6/12            | 10/10           | 1/3        | 1/4            | 4/8              | 0/3         | 4/9        | 8/8         | 2/6           | —                  |
| M3@S0    | 42/42  | 7/18       | 11/12           | 10/10           | 2/3        | 3/4            | 3/8              | 1/3         | 4/9        | 8/8         | 2/6           | —                  |
| M4@S0    | 40/42  | 7/18       | 10/12           | 10/10           | 2/3        | 3/4            | 4/8              | 0/3         | 4/9        | 8/8         | 2/6           | —                  |
| M1@S6    | 38/42  | 18/18      | 11/12           | 10/10           | 1/3        | 3/4            | 3/8              | 0/3         | 5/9        | 6/8         | 5/6           | —                  |
| M2@S6    | 38/42  | 11/18      | 12/12           | 10/10           | 3/3        | 2/4            | 4/8              | 1/3         | 4/9        | 8/8         | 6/6           | —                  |
| M3@S6    | 42/42  | 11/18      | 12/12           | 10/10           | 3/3        | 3/4            | 4/8              | 1/3         | 4/9        | 8/8         | 6/6           | —                  |
| M4@S6    | 42/42  | 11/18      | 12/12           | 10/10           | 3/3        | 3/4            | 4/8              | 0/3         | 4/9        | 8/8         | 6/6           | —                  |

## Evidence recall by slice

| config   | base   | language   | applicability   | point_in_time   | incident   | what_changed   | multi_evidence   | cross_doc   | conflict   | table_row   | tag_variant   | not_in_documents   |
|:---------|:-------|:-----------|:----------------|:----------------|:-----------|:---------------|:-----------------|:------------|:-----------|:------------|:--------------|:-------------------|
| S0       | 32/42  | 3/18       | 2/12            | 5/10            | 2/3        | 1/4            | 0/8              | 0/3         | 3/9        | 6/8         | 0/6           | —                  |
| S1       | 32/42  | 3/18       | 2/12            | 5/10            | 2/3        | 1/4            | 0/8              | 0/3         | 3/9        | 6/8         | 4/6           | —                  |
| S2       | 31/42  | 3/18       | 2/12            | 5/10            | 2/3        | 1/4            | 0/8              | 0/3         | 3/9        | 8/8         | 4/6           | —                  |
| S2b      | 32/42  | 3/18       | 2/12            | 5/10            | 2/3        | 1/4            | 0/8              | 0/3         | 3/9        | 8/8         | 4/6           | —                  |
| S3       | 40/42  | 7/18       | 6/12            | 8/10            | 2/3        | 1/4            | 0/8              | 0/3         | 6/9        | 8/8         | 6/6           | —                  |
| S4       | 42/42  | 8/18       | 9/12            | 9/10            | 3/3        | 1/4            | 0/8              | 0/3         | 8/9        | 8/8         | 6/6           | —                  |
| S5       | 42/42  | 9/18       | 11/12           | 10/10           | 3/3        | 3/4            | 6/8              | 2/3         | 8/9        | 8/8         | 6/6           | —                  |
| S6       | 42/42  | 9/18       | 11/12           | 10/10           | 3/3        | 3/4            | 6/8              | 2/3         | 8/9        | 8/8         | 6/6           | —                  |
| M1t      | —      | 9/18       | —               | —               | —          | —              | —                | —           | —          | —           | —             | —                  |
| M1@S0    | 33/42  | 15/18      | 1/12            | 6/10            | 1/3        | 2/4            | 0/8              | 0/3         | 4/9        | 8/8         | 6/6           | —                  |
| M2@S0    | 36/42  | 5/18       | 1/12            | 7/10            | 1/3        | 1/4            | 0/8              | 0/3         | 4/9        | 8/8         | 2/6           | —                  |
| M3@S0    | 42/42  | 7/18       | 11/12           | 10/10           | 1/3        | 3/4            | 1/8              | 1/3         | 4/9        | 8/8         | 2/6           | —                  |
| M4@S0    | 40/42  | 7/18       | 7/12            | 8/10            | 1/3        | 3/4            | 0/8              | 0/3         | 4/9        | 8/8         | 2/6           | —                  |
| M1@S6    | 42/42  | 18/18      | 7/12            | 10/10           | 3/3        | 4/4            | 7/8              | 3/3         | 9/9        | 7/8         | 5/6           | —                  |
| M2@S6    | 42/42  | 12/18      | 9/12            | 10/10           | 3/3        | 4/4            | 8/8              | 3/3         | 8/9        | 8/8         | 6/6           | —                  |
| M3@S6    | 42/42  | 11/18      | 12/12           | 10/10           | 3/3        | 4/4            | 8/8              | 3/3         | 8/9        | 8/8         | 6/6           | —                  |
| M4@S6    | 42/42  | 11/18      | 11/12           | 10/10           | 3/3        | 4/4            | 8/8              | 3/3         | 8/9        | 8/8         | 6/6           | —                  |

## Where answers are lost (wrong answers by link)

| config   |   reading |   search |   assembly |   model |
|:---------|----------:|---------:|-----------:|--------:|
| S0       |         0 |       60 |          2 |       8 |
| S6       |         0 |       12 |          0 |       2 |

Reading: the value is in no chunk; search: no chunk with the answer in the context; assembly: the answer chunk is there, a required support or action chunk is not; model: everything was in the context. Practice 3 ceiling with the known page: 98%.

## The P-101 question

S6 (BM25): status `not_in_documents`; top-5 R-12 rev. 3 cl. 4.4.2, R-15 rev. 1 cl. 4.3, R-15 rev. 1 cl. 5.1, R-15 rev. 1 cl. 4.1, FS-07 rev. 2 Table 1. Table 3 is at rank 6: the evidence pack has nothing to start from. M3@S6 (reranker over the top-20):

Question (asked in Russian): После замены подшипника насос P-101 вибрирует 4,8 мм/с. Это норма? Что делать и в какой срок?

```json
{
 "status": "answered",
 "answer": "No, 4.8 mm/s exceeds the inspection limit of 4.5 mm/s for Class II pumps. You must inspect the equipment within 24 hours and create a 'high' priority CMMS request. The request must be accepted within 4 hours and completed within 3 days.",
 "value": "4,5",
 "unit": "мм/с",
 "claims": [
  {
   "text": "P-101 is a Class II pump with an inspection vibration limit of 4.5 mm/s.",
   "sources": [
    "R-12-r3-table3",
    "R-12-r3-table1"
   ]
  },
  {
   "text": "Vibration of 4.8 mm/s exceeds the inspection limit, requiring inspection within 24 hours.",
   "sources": [
    "R-12-r3-4.3.3"
   ]
  },
  {
   "text": "A 'high' priority CMMS request must be created for the inspection.",
   "sources": [
    "R-12-r3-6.2"
   ]
  },
  {
   "text": "The 'high' priority request must be accepted within 4 hours and completed within 3 days.",
   "sources": [
    "R-20-r2-table1"
   ]
  }
 ],
 "citations": [
  {
   "source": "R-12-r3-table3",
   "doc_id": "R-12",
   "revision": 3,
   "amendment": null,
   "clause": null,
   "table_no": "А.1",
   "row_key": null,
   "page": 6
  },
  {
   "source": "R-12-r3-table1",
   "doc_id": "R-12",
   "revision": 3,
   "amendment": null,
   "clause": null,
   "table_no": "3",
   "row_key": null,
   "page": 4
  },
  {
   "source": "R-12-r3-4.3.3",
   "doc_id": "R-12",
   "revision": 3,
   "amendment": null,
   "clause": "4.3.3",
   "table_no": null,
   "row_key": null,
   "page": 3
  },
  {
   "source": "R-12-r3-6.2",
   "doc_id": "R-12",
   "revision": 3,
   "amendment": null,
   "clause": "6.2",
   "table_no": null,
   "row_key": null,
   "page": 5
  },
  {
   "source": "R-20-r2-table1",
   "doc_id": "R-20",
   "revision": 2,
   "amendment": null,
   "clause": null,
   "table_no": "1",
   "row_key": null,
   "page": 2
  }
 ],
 "sources": []
}
```

Checks: verified = True, exists = True, in force = True, supports = True. Citations: R-12-r3-table3 (exists True, in force True, supports False); R-12-r3-table1 (exists True, in force True, supports True); R-12-r3-4.3.3 (exists True, in force True, supports True); R-12-r3-6.2 (exists True, in force True, supports False); R-20-r2-table1 (exists True, in force True, supports False)

The audit record is in `audit_p101.json`.

## The contract on a conflict (demo: both sides of the P-205 conflict put into the context by hand)

Status: `answered`; answer: For pump P-205 (Class II), the vibration velocity limit for inspection is 4.5 mm/s. The alarm limit is 7.1 mm/s, requiring the cause to be found within 7 days. The trip limit is 11.2 mm/s, requiring the pump to be stopped.

## Regression gate S6 against S5 (practice 2 rule)

123 -> 129 right of 143, fixed 7, broke 1 (w02), p = 0.0703: **PASS**

## Five S0 failures for task 1 of the lecture

See `s0_failures.md`: m01 (search), g01 (search), v03 (search), m06 (assembly), i01 (model).
