# Reference results of practice 2

Measured on the synthetic maintenance log.

## Judges against the human labels (213 summaries from practice 1)

| judge                        |   n |   unusable answers |   caught |   missed |   false alarm |   correct pass |   agreement, % |   kappa |   catches failures, % |   passes good answers, % |
|:-----------------------------|----:|-------------------:|---------:|---------:|--------------:|---------------:|---------------:|--------:|----------------------:|-------------------------:|
| judge-scale-course           | 213 |                  0 |       31 |        1 |           162 |             19 |           23.5 |    0.02 |                  96.9 |                     10.5 |
| judge-binary-course          | 213 |                  0 |       22 |       10 |             4 |            177 |           93.4 |    0.72 |                  68.8 |                     97.8 |
| judge-scale-course-thinking  | 213 |                  0 |       32 |        0 |           180 |              1 |           15.5 |    0    |                 100   |                      0.6 |
| judge-binary-course-thinking | 212 |                  1 |       28 |        4 |             6 |            174 |           95.3 |    0.82 |                  87.5 |                     96.7 |

The experiment below uses the binary judge `course-thinking`, which has the highest kappa.

## Prompt v1 vs v2, course model, 3 runs each, eval_50 + holdout_50 (100 entries)

| prompt   | setup        |   n |   valid, % |   field accuracy, % |   all fields right, % |   summary faithful, % | faithful 95% interval   |
|:---------|:-------------|----:|-----------:|--------------------:|----------------------:|----------------------:|:------------------------|
| v1       | course-v1-r1 | 100 |         98 |                96.8 |                    90 |                    91 | 84–95                   |
| v1       | course-v1-r2 | 100 |         98 |                96.8 |                    90 |                    93 | 86–97                   |
| v1       | course-v1-r3 | 100 |         98 |                96.6 |                    89 |                    91 | 84–95                   |
| v2       | course-v2-r1 | 100 |         98 |                96.4 |                    89 |                    97 | 92–99                   |
| v2       | course-v2-r2 | 100 |         98 |                96.6 |                    90 |                    98 | 93–99                   |
| v2       | course-v2-r3 | 100 |         98 |                96.6 |                    90 |                    98 | 93–99                   |

## Paired comparison of summary faithfulness, v1 vs v2, run by run

| A | B | n | A right | B right | B fixed | B broke | p-value | decision |
|---|---|---|---|---|---|---|---|---|
| course-v1-r1 | course-v2-r1 | 100 | 91 | 97 | 6 | 0 | 0.0312 | B is better |
| course-v1-r2 | course-v2-r2 | 100 | 93 | 98 | 5 | 0 | 0.0625 | no detectable difference |
| course-v1-r3 | course-v2-r3 | 100 | 91 | 98 | 7 | 0 | 0.0156 | B is better |
| v1, majority of 3 runs | v2, majority of 3 runs | 100 | 92 | 98 | 6 | 0 | 0.0312 | B is better |

## Prediction-powered inference on the pool (expert labels on 40 random summaries)

| estimate                           |   faithful, % |   low |   high |
|:-----------------------------------|--------------:|------:|-------:|
| expert on 40 only                  |          82.5 |  70.6 |   94.4 |
| judge on all 212                   |          84   | nan   |  nan   |
| PPI, judge on 172 and expert on 40 |          81.2 |  73.8 |   88.6 |
| truth, expert on all 212           |          84.9 | nan   |  nan   |
