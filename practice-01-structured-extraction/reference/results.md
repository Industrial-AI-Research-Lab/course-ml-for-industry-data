# Reference results: practice 1 (eval_50)

Measured by the lecturer on the synthetic maintenance log, split `eval_50` (50 records).
Compare your numbers with these. Small differences are normal: models are not fully deterministic.

| setup                            |   valid 1st try, % |   valid final, % |   field accuracy, % |   all fields right, % |   invented, % |   median latency, s |   USD per 1000 records |
|:---------------------------------|-------------------:|-----------------:|--------------------:|----------------------:|--------------:|--------------------:|-----------------------:|
| rules                            |                100 |              100 |                94.5 |                    56 |           0   |                0    |                  0     |
| local-baseline                   |                 20 |               20 |                18   |                    12 |           4.2 |                1.13 |                  0     |
| local-structured-schema-only     |                 24 |               96 |                76.2 |                    14 |          12.5 |                2.76 |                  0     |
| local-structured                 |                 50 |               98 |                89.8 |                    58 |          16.7 |                1.64 |                  0     |
| local-small-structured           |                 22 |               60 |                50.8 |                    18 |          27.1 |                1.69 |                  0     |
| course-baseline                  |                 80 |               80 |                77.8 |                    62 |           0   |                1.68 |                  0     |
| course-structured-schema-only    |                 58 |               98 |                85.2 |                    24 |           8.3 |                3.48 |                  0     |
| course-structured                |                 62 |              100 |                99.8 |                    98 |           0   |                3.12 |                  0     |
| course-thinking-structured       |                 72 |              100 |                98.8 |                    90 |           2.1 |               11.89 |                  0     |
| gpt-oss-120b-structured          |                 46 |              100 |                98.2 |                    88 |           2.1 |               11.93 |                  0     |
| deepseek-v3.2-structured         |                 58 |              100 |                97.5 |                    86 |           2.1 |               34.68 |                  0.287 |
| gpt-6-luna-structured            |                 72 |              100 |                98.8 |                    90 |           0   |                4.12 |                  0.238 |
| gpt-6.1-sol-structured           |                 42 |              100 |                99   |                    92 |           0   |                6.35 |                  5.02  |
| claude-sonnet-5-structured       |                 94 |              100 |                99.2 |                    94 |           4.2 |                4.8  |                  7.043 |
| local-structured+course-fallback |                 50 |              100 |                91.8 |                    60 |          16.7 |                1.76 |                  0     |

## Field accuracy by field, %

|                                  |   equipment_id |   equipment_type |   work_type |   failure_mode |   replaced_parts |   duration_hours |   work_date |   downtime |
|:---------------------------------|---------------:|-----------------:|------------:|---------------:|-----------------:|-----------------:|------------:|-----------:|
| rules                            |            100 |              100 |          74 |             92 |               92 |              100 |         100 |         98 |
| local-baseline                   |             20 |               12 |          18 |             20 |               20 |               18 |          20 |         16 |
| local-structured-schema-only     |             74 |               88 |          68 |             40 |               80 |               90 |          96 |         74 |
| local-structured                 |             76 |               88 |          90 |             96 |               90 |               96 |          98 |         84 |
| local-small-structured           |             50 |               50 |          54 |             52 |               52 |               56 |          52 |         40 |
| course-baseline                  |             80 |               64 |          78 |             80 |               80 |               80 |          80 |         80 |
| course-structured-schema-only    |             98 |               98 |          82 |             46 |               88 |               98 |          98 |         74 |
| course-structured                |            100 |              100 |          98 |            100 |              100 |              100 |         100 |        100 |
| course-thinking-structured       |            100 |              100 |         100 |             94 |               98 |              100 |         100 |         98 |
| gpt-oss-120b-structured          |            100 |              100 |          98 |             96 |               98 |               98 |          98 |         98 |
| deepseek-v3.2-structured         |             92 |               98 |         100 |            100 |               98 |               94 |         100 |         98 |
| gpt-6-luna-structured            |            100 |              100 |          98 |             96 |               98 |               98 |         100 |        100 |
| gpt-6.1-sol-structured           |            100 |              100 |          98 |             96 |               98 |              100 |         100 |        100 |
| claude-sonnet-5-structured       |            100 |              100 |         100 |            100 |               98 |              100 |         100 |         96 |
| local-structured+course-fallback |             78 |               90 |          92 |             98 |               92 |               98 |         100 |         86 |
