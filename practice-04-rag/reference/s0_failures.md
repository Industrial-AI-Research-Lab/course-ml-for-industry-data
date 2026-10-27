# Five failed answers of S0 (task 1 of lecture 4)

Configuration S0: BM25 over raw chunks, top-5, no filters, the course model answers from the top-5.
Task: for each answer write where it was lost: R = reading, S = search, C = context assembly, M = model.

## 1. m01 (multi_evidence) — lost in: **search**

- Question (asked in Russian): После замены подшипника насос P-101 вибрирует 4,8 мм/с. Это норма? Что делать и в какой срок?
- Translation: After a bearing change pump P-101 vibrates at 4.8 mm/s. Is this normal? What must be done, and by when?
- Expected: ['4,5', '24'] h
- System answer: The sources do not specify the vibration limit to determine if 4.8 mm/s is normal, but they mandate that measurements be taken within 8 hours after startup following a bearing replacement. You must perform the measurement within this 8-hour window. (value: None)
- Gold evidence: R-12 r3 cl 4.3.3; R-12 r3 t 3; R-12 r3 cl 6.1; R-12 r3 cl 6.2
- Top-5 of search: R-12 rev. 2 cl. 4.4.2; R-12 rev. 3 cl. 4.4.2; R-14 rev. 1 cl. 4.4.2; R-15 rev. 1 cl. 4.3; R-15 rev. 1 cl. 5.1
- Why this link: no chunk with the answer reached the context

## 2. g01 (tag_variant) — lost in: **search**

- Question (asked in Russian): Какая мощность двигателя у насоса Р-101?
- Translation: What is the motor power of pump Р-101 (Cyrillic Р)?
- Expected: 55 kW
- System answer: The provided sources do not contain the specific power rating of the motor for pump R-101. (value: None)
- Gold evidence: PS-101
- Top-5 of search: R-12 rev. 2 cl. 4.5.1; R-12 rev. 3 cl. 4.5.1; R-14 rev. 1 cl. 4.5.1; R-15 rev. 1 cl. 5.1; R-12 rev. 2 cl. 7.2
- Why this link: no chunk with the answer reached the context

## 3. v03 (point_in_time) — lost in: **search**

- Question (asked in Russian): С какой периодичностью сейчас нужно контролировать вибрацию насоса P-101 переносным виброметром?
- Translation: How often must the vibration of pump P-101 be checked with a portable meter now?
- Expected: 500 h
- System answer: Vibration of pump P-101 must be controlled with a portable vibrometer every 1000 operating hours, but not less than once a month. (value: 1000)
- Gold evidence: R-12 r3a1 cl 4.4.1
- Top-5 of search: R-12 rev. 2 cl. 3.7; R-12 rev. 3 cl. 3.7; R-14 rev. 1 cl. 3.7; R-12 rev. 2 cl. 4.2.1; R-12 rev. 2 cl. 4.4.1
- Why this link: no chunk with the answer reached the context

## 4. m06 (multi_evidence) — lost in: **assembly**

- Question (asked in Russian): После замены подшипника насос P-101 запустили. В течение скольких часов нужно измерить вибрацию и с какой границей сравнить результат?
- Translation: Pump P-101 was started after a bearing change. Within how many hours must the vibration be measured, and against which limit?
- Expected: ['8', '4,5'] h
- System answer: Vibration measurements must be taken within 8 hours after startup. The sources do not specify the comparison limit. (value: 8)
- Gold evidence: R-12 r3 cl 4.4.2; R-12 r3 t 3
- Top-5 of search: R-12 rev. 2 cl. 4.4.2; R-12 rev. 3 cl. 4.4.2; R-14 rev. 1 cl. 4.4.2; FS-07 rev. 2 cl. 3.3; R-12 rev. 2 cl. 6.1
- Why this link: the chunk with the answer is in the context, a required support or action chunk is not

## 5. i01 (incident) — lost in: **model**

- Question (asked in Russian): 10.02.2025 насос P-101 вибрировал 4,8 мм/с, осмотр в течение суток не провели. Было ли это нарушением регламента?
- Translation: On 10 February 2025 pump P-101 vibrated at 4.8 mm/s and nobody inspected it within 24 hours. Was this a violation of the regulation?
- Expected: ['5,0'] mm/s
- System answer: Yes, it was a violation of the regulation. The vibration of 4.8 mm/s exceeded the inspection threshold of 4.5 mm/s for a Class II pump, requiring an inspection within 24 hours, which was not performed. (value: 4,5)
- Gold evidence: R-12 r2 t 3; R-12 r2 cl 4.3.3
- Top-5 of search: R-12 rev. 2 cl. 4.3.3; R-12 rev. 3 cl. 4.3.3; R-14 rev. 1 cl. 4.3.3; R-12 rev. 2 Table 3; R-12 rev. 3 Table 3
- Why this link: every needed chunk was in the context, the answer is still wrong
