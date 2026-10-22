"""Practice 2: the evaluation harness of the Plant Engineer Assistant.

- cases:     test cases with expected values and tags (the golden set), built from the data and from call logs
- harness:   score a run case by case, slice the results
- errors:    count error categories per field (the counting step of error analysis)
- checks:    cheap code checks for the free-text field
- judge:     LLM-as-judge with structured output, replayable like every other model call
- pool:      the free-text summaries of practice 1 and their human labels
- stats:     intervals, paired comparison, sample size, agreement and kappa, prediction-powered inference
- gate:      "not worse than the baseline" decision for CI (topic 6)
"""
