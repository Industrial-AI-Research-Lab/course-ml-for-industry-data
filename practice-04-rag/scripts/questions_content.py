"""The retrieval set of practice 4: questions built from the structure of the documents.

Rules of lecture 2: every question comes from a known place (a clause, a table, a row, a datasheet), the
answer is a value with a unit or a status, and the wording does not copy the clause. Every question has a
Russian wording (it goes to search and to the prompt: the engineer of a Russian plant asks in Russian) and an
English one (for the notebook). The `language` slice sends the English wording to search as is.

Evidence is written as locators, never as chunk ids: "R-12 r3 cl 4.3.3", "R-12 r3a1 cl 4.4.1" (amendment 1),
"R-12 r3 t 3" (Table 3), "VL-2 r4 t А.2 row V-12", "PS-101" (the datasheet). An evidence item is
(role, [locator, ...]): any of the locators satisfies it. Roles: answer (holds the value), support (needed to
choose it: the class, the table), action (what to do). `build_questions.py` turns locators into chunk ids.

Expected: (value, unit) as written in the document; ("values", [..], unit): every number must be in the answer
text (questions "what changed", and decisions like "is it normal, what to do": the limit and the deadline);
("status", "not_in_documents" | "conflict" | "needs_human").
"""

from corpus_content import TODAY

# ---------------------------------------------------------------------------------------------
# The 50 questions of practice 3, translated once.
# id -> (Russian wording, evidence). The English wording is the practice 3 text; questions that did not name
# the document get ", according to R-12": the archive now has R-14 with the same clauses.
# ---------------------------------------------------------------------------------------------
T3, A1 = "R-12 r3 t 3", "R-12 r3 t А.1"
BASE = {
    "t01": ("Какое значение виброскорости по R-12 требует осмотра машины класса I?", [("answer", [T3])]),
    "t02": ("При какой виброскорости по R-12 машину класса I нужно останавливать?", [("answer", [T3])]),
    "t03": ("Какое значение виброскорости по R-12 требует осмотра машины класса II?", [("answer", [T3])]),
    "t04": ("При какой виброскорости по R-12 машину класса II нужно останавливать?", [("answer", [T3])]),
    "t05": ("Какое значение виброскорости по R-12 требует осмотра машины класса III?", [("answer", [T3])]),
    "t06": ("При какой виброскорости по R-12 машину класса III нужно останавливать?", [("answer", [T3])]),
    "t07": ("Какое значение виброскорости по R-12 требует осмотра машины класса IV?", [("answer", [T3])]),
    "t08": ("При какой виброскорости по R-12 машину класса IV нужно останавливать?", [("answer", [T3])]),
    "t09": ("Какое значение виброскорости по R-12 требует осмотра вентилятора (класс V)?", [("answer", [T3])]),
    "t10": ("При какой виброскорости по R-12 вентилятор (класс V) нужно останавливать?", [("answer", [T3])]),
    "a01": ("Какая мощность привода у компрессора K-301 в перечне оборудования R-12?", [("answer", [A1])]),
    "a02": ("Какая мощность привода у вентилятора B-401 в перечне оборудования R-12?", [("answer", [A1])]),
    "a03": ("Какая мощность привода у насоса P-102 в перечне оборудования R-12?", [("answer", [A1, "PS-102"])]),
    "a04": ("Какая мощность привода у компрессора K-302 в перечне оборудования R-12?", [("answer", [A1])]),
    "c01": ("До какого значения виброскорости по R-12 должен измерять прибор (верхний предел диапазона)?",
            [("answer", ["R-12 r3 cl 4.2.1"])]),
    "c02": ("Какая наибольшая допустимая основная погрешность измерения виброскорости по R-12?",
            [("answer", ["R-12 r3 cl 4.2.2"])]),
    "c03": ("Как часто по R-12 нужно поверять или калибровать средства измерений, не реже?",
            [("answer", ["R-12 r3 cl 4.2.3"])]),
    "c04": ("Через сколько минут после пуска по R-12 можно начинать измерять вибрацию?",
            [("answer", ["R-12 r3 cl 4.1.2"])]),
    "c05": ("В течение какого времени по R-12 машину нужно осмотреть, если превышена граница осмотра?",
            [("answer", ["R-12 r3 cl 4.3.3"])]),
    # Amendment No. 1 changed this clause from 2026-09-01: the answer today is 500 h, not 720 h as in practice 3
    "c06": ("С какой периодичностью в часах наработки по R-12 выполняют контроль переносным виброметром?",
            [("answer", ["R-12 r3a1 cl 4.4.1"])]),
    "c07": ("В течение скольких часов после пуска по R-12 нужно измерить вибрацию после замены подшипников?",
            [("answer", ["R-12 r3 cl 4.4.2"])]),
    "c08": ("До какой мощности привода центробежный насос по R-12 относится к классу I?",
            [("answer", ["R-12 r3 cl 4.5.2", "R-12 r3 t 2"])]),
    "c09": ("Какая наибольшая температура наружного кольца подшипника допускается по R-12?",
            [("answer", ["R-12 r3 cl 5.1"])]),
    "c10": ("На сколько градусов за 10 минут должна вырасти температура подшипника, чтобы по R-12 остановить "
            "машину?", [("answer", ["R-12 r3 cl 5.2"])]),
    "c11": ("В течение какого времени по R-12 обходчик сообщает начальнику смены о превышении границы?",
            [("answer", ["R-12 r3 cl 6.1"])]),
    "c12": ("С какой периодичностью в часах наработки по R-12 смазывают подшипники качения?",
            [("answer", ["R-12 r3 cl 7.1"])]),
    "c13": ("Какое наибольшее радиальное смещение валов допускается по R-12 после ремонта?",
            [("answer", ["R-12 r3 cl 7.2"])]),
    "c14": ("Через сколько часов наработки по R-12 подшипники заменяют в любом случае?",
            [("answer", ["R-12 r3 cl 7.4"])]),
    "c15": ("Сколько лет по R-12 нужно хранить журналы измерений, не меньше?", [("answer", ["R-12 r3 cl 8.1"])]),
    "c16": ("Для оборудования с частотой вращения ниже какого значения R-12 не применяется?",
            [("answer", ["R-12 r3 cl 1.4"])]),
    "c17": ("Какая верхняя частота диапазона, в котором R-12 определяет виброскорость?",
            [("answer", ["R-12 r3 cl 3.1"])]),
    "d01": ("Какая номинальная подача насоса P-101?", [("answer", ["PS-101"])]),
    "d02": ("Какой номинальный напор насоса P-101?", [("answer", ["PS-101"])]),
    "d03": ("С какой частотой вращения работает насос P-101?", [("answer", ["PS-101"])]),
    "d04": ("Какая мощность двигателя насоса P-101?", [("answer", ["PS-101", A1])]),
    "d05": ("Какая масса насосного агрегата P-101?", [("answer", ["PS-101"])]),
    "d06": ("В каком году изготовлен насос P-101?", [("answer", ["PS-101"])]),
    "z01": ("Какая номинальная подача насоса P-205?", [("answer", ["DS-P-205", "OM-P-205 r2 t 1"])]),
    "z02": ("Какой номинальный напор насоса P-205?", [("answer", ["DS-P-205", "OM-P-205 r2 t 1"])]),
    "z03": ("С какой частотой вращения работает насос P-205?", [("answer", ["DS-P-205", "OM-P-205 r2 t 1"])]),
    "z04": ("Какая мощность двигателя насоса P-205?", [("answer", ["DS-P-205", "OM-P-205 r2 t 1", A1])]),
    "z05": ("Какая масса насоса P-205?", [("answer", ["DS-P-205", "OM-P-205 r2 t 1"])]),
    "n01": ("Какой наибольший уровень шума по R-12 допускается для насосов класса II?", []),
    "n02": ("При какой виброскорости по R-12 останавливают машины класса VI?", []),
    "n03": ("Как часто по R-12 меняют батареи в переносных виброметрах?", []),
    "n04": ("Какое биение вала по R-12 допускается после ремонта?", []),
    "n05": ("Какой допускаемый кавитационный запас (NPSH) у насоса P-101?", []),
    "n06": ("Сколько масла вмещает корпус подшипников насоса P-101?", []),
    "n07": ("Какое наибольшее давление на входе насоса P-205?", []),
    "n08": ("Какой гарантийный срок у насоса P-205?", []),
}

# English wordings of these base questions go to search as is: the `language` slice
LANGUAGE_IDS = ["t03", "t04", "t09", "a01", "a03", "c01", "c02", "c05", "c06", "c08", "c09", "c11", "c12", "c14",
                "d01", "d04", "z01", "z04", "n01", "n05"]

# ---------------------------------------------------------------------------------------------
# New questions: (id, slice, Russian, English, as_of or None (= asked today), expected, evidence, note)
# ---------------------------------------------------------------------------------------------
R14T3, R14A1 = "R-14 r1 t 3", "R-14 r1 t А.1"
CLASS_P101 = ("support", ["PS-101", A1])
NEW = [
    # --- applicability: near-duplicate regulations, the class and the station come from the registry ------
    ("p01", "applicability", "Насос P-310 вибрирует 4,0 мм/с. Нужно ли его осматривать?",
     "Pump P-310 vibrates at 4.0 mm/s. Does it need an inspection?", None, ("values", ["3,5"], "mm/s"),
     [("answer", [R14T3]), ("support", ["PS-310", R14A1])], "yes: R-14, class II, limit 3.5 (R-12 would say no)"),
    ("p02", "applicability", "Насос P-101 вибрирует 4,0 мм/с. Нужно ли его осматривать?",
     "Pump P-101 vibrates at 4.0 mm/s. Does it need an inspection?", None, ("values", ["4,5"], "mm/s"),
     [("answer", [T3]), CLASS_P101], "no: R-12, class II, limit 4.5"),
    ("p03", "applicability", "При какой виброскорости нужно остановить насос P-310?",
     "At what vibration velocity must pump P-310 be stopped?", None, ("5,6", "mm/s"),
     [("answer", [R14T3]), ("support", ["PS-310", R14A1])], ""),
    ("p04", "applicability", "При какой виброскорости нужно остановить насос P-101?",
     "At what vibration velocity must pump P-101 be stopped?", None, ("7,1", "mm/s"),
     [("answer", [T3]), CLASS_P101], ""),
    ("p05", "applicability", "С какого значения виброскорости насос P-311 нужно осматривать?",
     "From what vibration velocity must pump P-311 be inspected?", None, ("2,2", "mm/s"),
     [("answer", [R14T3]), ("support", ["PS-311", R14A1])], "class I by R-14"),
    ("p06", "applicability", "С какого значения виброскорости насос P-102 нужно осматривать?",
     "From what vibration velocity must pump P-102 be inspected?", None, ("2,8", "mm/s"),
     [("answer", [T3]), ("support", ["PS-102", A1])], "class I by R-12"),
    ("p07", "applicability", "Насос P-311 вибрирует 3,0 мм/с. Его нужно останавливать?",
     "Pump P-311 vibrates at 3.0 mm/s. Must it be stopped?", None, ("values", ["3,5"], "mm/s"),
     [("answer", [R14T3]), ("support", ["PS-311", R14A1])], "no: stop limit of class I by R-14 is 3.5"),
    ("p08", "applicability", "С какой виброскорости компрессор K-301 нужно осматривать?",
     "From what vibration velocity must compressor K-301 be inspected?", None, ("7,1", "mm/s"),
     [("answer", [T3]), ("support", [A1])], "class III by R-12"),
    ("p09", "applicability", "При какой виброскорости нужно остановить насос P-312?",
     "At what vibration velocity must pump P-312 be stopped?", None, ("5,6", "mm/s"),
     [("answer", [R14T3]), ("support", [R14A1])], "no datasheet: the class is in R-14 Table A.1"),
    ("p10", "applicability", "С какой виброскорости вентилятор B-401 нужно осматривать?",
     "From what vibration velocity must fan B-401 be inspected?", None, ("5,6", "mm/s"),
     [("answer", [T3]), ("support", [A1])], "class V by R-12"),
    ("p11", "applicability", "К какому классу машин относится насос P-310?",
     "Which machine class is pump P-310?", None, ("II", None),
     [("answer", ["PS-310", R14A1])], ""),
    ("p12", "applicability", "Насос насосной станции № 3 с двигателем 75 кВт вибрирует 4,0 мм/с. Нужен ли осмотр?",
     "A pump of pump station No. 3 with a 75 kW motor vibrates at 4.0 mm/s. Does it need an inspection?", None,
     ("values", ["3,5"], "mm/s"), [("answer", [R14T3]), ("support", ["R-14 r1 cl 4.5.2", "R-14 r1 t 2"])],
     "no tag: station 3 -> R-14, 75 kW -> class II"),
    # --- point in time: revisions and the amendment -------------------------------------------------------
    ("v01", "point_in_time", "Какая периодичность контроля вибрации насоса P-101 переносным виброметром "
                             "действовала на 01.05.2024?",
     "What was the interval of vibration checks of pump P-101 with a portable meter on 1 May 2024?",
     "2024-05-01", ("1000", "h"), [("answer", ["R-12 r2 cl 4.4.1"])], "rev. 2"),
    ("v02", "point_in_time", "Какая периодичность контроля вибрации насоса P-101 переносным виброметром "
                             "действовала на 01.06.2025?",
     "What was the interval of vibration checks of pump P-101 with a portable meter on 1 June 2025?",
     "2025-06-01", ("720", "h"), [("answer", ["R-12 r3 cl 4.4.1"])], "rev. 3"),
    ("v03", "point_in_time", "С какой периодичностью сейчас нужно контролировать вибрацию насоса P-101 переносным "
                             "виброметром?",
     "How often must the vibration of pump P-101 be checked with a portable meter now?", None, ("500", "h"),
     [("answer", ["R-12 r3a1 cl 4.4.1"])], "rev. 3 + Amendment No. 1"),
    ("v04", "point_in_time", "Какая периодичность контроля вибрации насоса P-101 переносным виброметром "
                             "действовала на 15 августа 2026 года?",
     "What was the interval of vibration checks of pump P-101 with a portable meter on 15 August 2026?",
     "2026-08-15", ("720", "h"), [("answer", ["R-12 r3 cl 4.4.1"])], "the amendment was approved but not yet in force"),
    ("v05", "point_in_time", "Какая граница осмотра по виброскорости действовала для насоса P-101 на 01.12.2024?",
     "What was the inspection limit of vibration velocity for pump P-101 on 1 December 2024?",
     "2024-12-01", ("5,0", "mm/s"), [("answer", ["R-12 r2 t 3"]), ("support", ["PS-101", "R-12 r2 t А.1"])], ""),
    ("v06", "point_in_time", "Какая граница осмотра по виброскорости действовала для насоса P-101 на 01.04.2025?",
     "What was the inspection limit of vibration velocity for pump P-101 on 1 April 2025?",
     "2025-04-01", ("4,5", "mm/s"), [("answer", [T3]), CLASS_P101], ""),
    ("v07", "point_in_time", "Какая температура подшипника насоса P-102 считалась допустимой на 10.01.2023?",
     "What bearing temperature of pump P-102 was permitted on 10 January 2023?",
     "2023-01-10", ("85", "°C"), [("answer", ["R-12 r2 cl 5.1"])], ""),
    ("v08", "point_in_time", "Какая температура подшипника насоса P-102 допустима сейчас?",
     "What bearing temperature of pump P-102 is permitted now?", None, ("80", "°C"),
     [("answer", ["R-12 r3 cl 5.1"])], ""),
    ("v09", "point_in_time", "С какой периодичностью сейчас контролируют вибрацию насоса P-310 переносным "
                             "виброметром?",
     "How often is the vibration of pump P-310 checked with a portable meter now?", None, ("720", "h"),
     [("answer", ["R-14 r1 cl 4.4.1"])], "the amendment of R-12 does not apply to R-14"),
    ("v10", "point_in_time", "При какой виброскорости нужно было останавливать насос P-101 на 01.03.2024?",
     "At what vibration velocity did pump P-101 have to be stopped on 1 March 2024?",
     "2024-03-01", ("7,1", "mm/s"), [("answer", ["R-12 r2 t 3"]), ("support", ["PS-101", "R-12 r2 t А.1"])],
     "the same value in both revisions"),
    # --- incident: task 2 of the lecture -----------------------------------------------------------------
    ("i01", "incident", "10.02.2025 насос P-101 вибрировал 4,8 мм/с, осмотр в течение суток не провели. Было ли это "
                        "нарушением регламента?",
     "On 10 February 2025 pump P-101 vibrated at 4.8 mm/s and nobody inspected it within 24 hours. Was this a "
     "violation of the regulation?", "2025-02-10", ("values", ["5,0"], "mm/s"),
     [("answer", ["R-12 r2 t 3"]), ("support", ["R-12 r2 cl 4.3.3"])], "no: rev. 2 was in force, limit 5.0"),
    ("i02", "incident", "20.04.2025 насос P-101 вибрировал 4,8 мм/с, осмотр в течение суток не провели. Было ли это "
                        "нарушением регламента?",
     "On 20 April 2025 pump P-101 vibrated at 4.8 mm/s and nobody inspected it within 24 hours. Was this a "
     "violation of the regulation?", "2025-04-20", ("values", ["4,5"], "mm/s"),
     [("answer", [T3]), ("support", ["R-12 r3 cl 4.3.3"])], "yes: rev. 3, limit 4.5"),
    ("i03", "incident", "05.10.2026 выяснилось, что вибрацию насоса P-101 последний раз контролировали 600 часов "
                        "наработки назад. Нарушена ли периодичность контроля?",
     "On 5 October 2026 it turned out that the vibration of pump P-101 was last checked 600 operating hours ago. "
     "Was the check interval violated?", "2026-10-05", ("values", ["500"], "h"), [("answer", ["R-12 r3a1 cl 4.4.1"])],
     "yes: Amendment No. 1, 500 h"),
    # --- what changed: the date filter must be off ------------------------------------------------------
    ("w01", "what_changed", "Как в ред. 3 R-12 изменилась граница осмотра для машин класса II по сравнению с ред. 2?",
     "How did the inspection limit for class II machines change in R-12 rev. 3 compared with rev. 2?", None,
     ("values", ["5,0", "4,5"], "mm/s"), [("answer", ["R-12 r2 t 3"]), ("answer", [T3])], ""),
    ("w02", "what_changed", "Как изменилась периодичность контроля переносным виброметром в ред. 3 R-12 по "
                            "сравнению с ред. 2?",
     "How did the interval of checks with a portable meter change in R-12 rev. 3 compared with rev. 2?", None,
     ("values", ["1000", "720"], "h"), [("answer", ["R-12 r2 cl 4.4.1"]), ("answer", ["R-12 r3 cl 4.4.1"])], ""),
    ("w03", "what_changed", "Как изменилась допустимая температура подшипника в ред. 3 R-12 по сравнению с ред. 2?",
     "How did the permitted bearing temperature change in R-12 rev. 3 compared with rev. 2?", None,
     ("values", ["85", "80"], "°C"), [("answer", ["R-12 r2 cl 5.1"]), ("answer", ["R-12 r3 cl 5.1"])], ""),
    ("w04", "what_changed", "Что поменяло Изменение № 1 к R-12 ред. 3 в периодичности контроля вибрации?",
     "What did Amendment No. 1 to R-12 rev. 3 change in the interval of vibration checks?", None,
     ("values", ["720", "500"], "h"), [("answer", ["R-12 r3 cl 4.4.1"]), ("answer", ["R-12 r3a1 cl 4.4.1"])], ""),
    # --- multi evidence: the clause, its table and the actions -------------------------------------------
    ("m01", "multi_evidence", "После замены подшипника насос P-101 вибрирует 4,8 мм/с. Это норма? Что делать и в "
                              "какой срок?",
     "After a bearing change pump P-101 vibrates at 4.8 mm/s. Is this normal? What must be done, and by when?",
     None, ("values", ["4,5", "24"], "h"),
     [("answer", ["R-12 r3 cl 4.3.3"]), ("support", [T3]), ("action", ["R-12 r3 cl 6.1"]),
      ("action", ["R-12 r3 cl 6.2"])], "the question of lectures 1-4: not normal, inspect within 24 hours"),
    ("m02", "multi_evidence", "Насос P-101 вибрирует 7,5 мм/с. Что нужно сделать?",
     "Pump P-101 vibrates at 7.5 mm/s. What must be done?", None, ("values", ["7,1"], "mm/s"),
     [("answer", [T3]), ("action", ["R-12 r3 cl 4.3.4"]), ("action", ["R-12 r3 cl 6.1"])],
     "stop at once: above the stop limit 7.1"),
    ("m03", "multi_evidence", "Насос P-102 вибрирует 3,0 мм/с. В какой срок его нужно осмотреть?",
     "Pump P-102 vibrates at 3.0 mm/s. Within what time must it be inspected?", None, ("values", ["2,8", "24"], "h"),
     [("answer", ["R-12 r3 cl 4.3.3"]), ("support", [T3]), ("support", ["PS-102", A1])], "class I, limit 2.8"),
    ("m04", "multi_evidence", "Компрессор K-302 вибрирует 12 мм/с. Что делать и в какой срок?",
     "Compressor K-302 vibrates at 12 mm/s. What must be done, and by when?", None, ("values", ["11,2", "24"], "h"),
     [("answer", ["R-12 r3 cl 4.3.3"]), ("support", [T3])], "class IV: above 11.2, below 18.0"),
    ("m05", "multi_evidence", "Насос P-101 вибрирует 4,8 мм/с. В какой срок обходчик должен сообщить об этом "
                              "начальнику смены?",
     "Pump P-101 vibrates at 4.8 mm/s. Within what time must the operator tell the shift supervisor?", None,
     ("values", ["1"], "h"), [("answer", ["R-12 r3 cl 6.1"]), ("support", [T3])], ""),
    ("m06", "multi_evidence", "После замены подшипника насос P-101 запустили. В течение скольких часов нужно "
                              "измерить вибрацию и с какой границей сравнить результат?",
     "Pump P-101 was started after a bearing change. Within how many hours must the vibration be measured, and "
     "against which limit?", None, ("values", ["8", "4,5"], "h"), [("answer", ["R-12 r3 cl 4.4.2"]), ("support", [T3])], ""),
    ("m07", "multi_evidence", "Насос P-310 вибрирует 4,0 мм/с. Что делать и в какой срок?",
     "Pump P-310 vibrates at 4.0 mm/s. What must be done, and by when?", None, ("values", ["3,5", "24"], "h"),
     [("answer", ["R-14 r1 cl 4.3.3"]), ("support", [R14T3]), ("action", ["R-14 r1 cl 6.2"])], "R-14, limit 3.5"),
    ("m08", "multi_evidence", "Насос P-102 вибрирует 5,0 мм/с. Какую границу он превысил и что делать?",
     "Pump P-102 vibrates at 5.0 mm/s. Which limit did it exceed, and what must be done?", None, ("values", ["4,5"], "mm/s"),
     [("answer", [T3]), ("action", ["R-12 r3 cl 4.3.4"]), ("support", ["PS-102", A1])], "class I: stop limit 4.5"),
    # --- cross document: R-12 clause 6.2 says "priority high", R-20 says what it means --------------------
    ("k01", "cross_doc", "Начальник смены создал заявку на осмотр по п. 6.2 R-12. В какой срок её должны принять "
                         "в работу?",
     "The shift supervisor created a work request under clause 6.2 of R-12. Within what time must it be taken "
     "into work?", None, ("4", "h"),
     [("answer", ["R-20 r2 t 1"]), ("support", ["R-20 r2 cl 3.3"]), ("support", ["R-12 r3 cl 6.2"])], ""),
    ("k02", "cross_doc", "Насос P-101 вибрирует 4,8 мм/с, заявку в CMMS создали. За сколько суток по ней должны "
                         "выполнить работы?",
     "Pump P-101 vibrates at 4.8 mm/s and a request was created in the CMMS. Within how many days must the work "
     "be done?", None, ("3", "days"), [("answer", ["R-20 r2 t 1"]), ("support", ["R-12 r3 cl 6.2"])], ""),
    ("k03", "cross_doc", "Заявку на осмотр насоса P-310 создали по п. 6.2 R-14. В какой срок её принимают в работу?",
     "A request to inspect pump P-310 was created under clause 6.2 of R-14. Within what time is it taken into "
     "work?", None, ("4", "h"), [("answer", ["R-20 r2 t 1"]), ("support", ["R-14 r1 cl 6.2"])], ""),
    # --- conflict: R-12 against the manufacturer's manual of P-205 (another language) ---------------------
    ("f01", "conflict", "Какая граница вибрации для осмотра или сигнализации установлена для насоса P-205?",
     "What vibration limit for an inspection or an alarm applies to pump P-205?", None, ("status", "conflict"),
     [("answer", [T3]), ("answer", ["OM-P-205 r2 t 2"])], "R-12 class II: 4.5; manual: alarm 7.1"),
    ("f02", "conflict", "При какой виброскорости нужно останавливать насос P-205?",
     "At what vibration velocity must pump P-205 be stopped?", None, ("status", "conflict"),
     [("answer", [T3]), ("answer", ["OM-P-205 r2 t 2"])], "R-12: 7.1; manual: trip 11.2"),
    ("f03", "conflict", "Насос P-205 вибрирует 6,0 мм/с. Это допустимо?",
     "Pump P-205 vibrates at 6.0 mm/s. Is this acceptable?", None, ("status", "conflict"),
     [("answer", [T3]), ("answer", ["OM-P-205 r2 t 2"])], "R-12: above 4.5, inspect; manual: normal up to 7.1"),
    ("f04", "conflict", "Насос P-205 вибрирует 9 мм/с. Его нужно останавливать?",
     "Pump P-205 vibrates at 9 mm/s. Must it be stopped?", None, ("status", "conflict"),
     [("answer", [T3]), ("answer", ["OM-P-205 r2 t 2"])], "R-12: above 7.1, stop; manual: alarm, trip at 11.2"),
    # same language: the maintenance regulation R-15 against the vibration regulations R-12 and R-14 (all in force)
    ("f05", "conflict", "Как часто нужно смазывать подшипники качения насоса P-101?",
     "How often must the rolling bearings of pump P-101 be lubricated?", None, ("status", "conflict"),
     [("answer", ["R-12 r3 cl 7.1"]), ("answer", ["R-15 r1 cl 4.1"])], "R-12: 2000 h; R-15: 3000 h"),
    ("f06", "conflict", "Через сколько часов наработки смазывают подшипники насоса P-310?",
     "After how many operating hours are the bearings of pump P-310 lubricated?", None, ("status", "conflict"),
     [("answer", ["R-14 r1 cl 7.1"]), ("answer", ["R-15 r1 cl 4.1"])], "R-14: 2000 h; R-15: 3000 h"),
    ("f07", "conflict", "Какое радиальное смещение валов допустимо после центровки насоса P-102?",
     "What radial offset of the shafts is permitted after aligning pump P-102?", None, ("status", "conflict"),
     [("answer", ["R-12 r3 cl 7.2"]), ("answer", ["R-15 r1 cl 5.2"])], "R-12: 0.05 mm; R-15: 0.08 mm"),
    ("f08", "conflict", "Какое радиальное смещение валов допустимо для насоса P-205 после монтажа или ремонта?",
     "What radial offset of the shafts is permitted for pump P-205 after installation or repair?", None,
     ("status", "conflict"), [("answer", ["R-12 r3 cl 7.2"]), ("answer", ["OM-P-205 r2 cl 3.2", "R-15 r1 cl 5.2"])],
     "R-12: 0.05 mm; the manual and R-15: 0.08 mm"),
    ("f09", "conflict", "Через сколько часов наработки нужно заменить подшипники насоса P-205?",
     "After how many operating hours must the bearings of pump P-205 be replaced?", None, ("status", "conflict"),
     [("answer", ["R-12 r3 cl 7.4"]), ("answer", ["OM-P-205 r2 cl 6.4"])], "R-12: 25 000 h; the manual: 20 000 h"),
    # --- table rows: the valve list of station 2 -------------------------------------------------------------
    ("r01", "table_row", "Какое допустимое давление для задвижки V-12?", "What is the allowed pressure for valve V-12?",
     None, ("1,6", "MPa"), [("answer", ["VL-2 r4 t А.2 row V-12"]), ("support", ["VL-2 r4 cl 1.3"])],
     "the question of lecture 1"),
    ("r02", "table_row", "Какой условный проход (DN) у крана V-37?", "What is the nominal size (DN) of valve V-37?",
     None, ("100", "mm"), [("answer", ["VL-2 r4 t А.2 row V-37"])], ""),
    ("r03", "table_row", "На какое номинальное давление рассчитана задвижка V-58?",
     "What nominal pressure is valve V-58 rated for?", None, ("4,0", "MPa"),
     [("answer", ["VL-2 r4 t А.2 row V-58"])], ""),
    ("r04", "table_row", "Какой диаметр DN у затвора V-73?", "What is the DN of butterfly valve V-73?", None,
     ("250", "mm"), [("answer", ["VL-2 r4 t А.2 row V-73"])], ""),
    ("r05", "table_row", "Какое PN у обратного клапана V-91?", "What is the PN of check valve V-91?", None,
     ("2,5", "MPa"), [("answer", ["VL-2 r4 t А.2 row V-91"])], ""),
    ("r06", "table_row", "Какой условный проход у крана V-104 на линии азота?",
     "What is the nominal size of valve V-104 on the nitrogen line?", None, ("50", "mm"),
     [("answer", ["VL-2 r4 t А.2 row V-104"])], ""),
    ("r07", "table_row", "Какой DN у задвижки V-110 на всасе сырьевой линии?",
     "What is the DN of valve V-110 on the suction of the feed line?", None, ("300", "mm"),
     [("answer", ["VL-2 r4 t А.2 row V-110"])], ""),
    ("r08", "table_row", "Какой диаметр у задвижки V-25 на линии охлаждения P-102?",
     "What is the size of valve V-25 on the cooling line of P-102?", None, ("80", "mm"),
     [("answer", ["VL-2 r4 t А.2 row V-25"])], "two tags: a valve and a pump"),
    # --- tag variants: Cyrillic letters, no hyphen, a space -----------------------------------------------
    ("g01", "tag_variant", "Какая мощность двигателя у насоса Р-101?", "What is the motor power of pump Р-101 "
     "(Cyrillic Р)?", None, ("55", "kW"), [("answer", ["PS-101", A1])], "Cyrillic Р"),
    ("g02", "tag_variant", "Какая подача у насоса P101?", "What is the flow of pump P101?", None, ("65", "m³/h"),
     [("answer", ["PS-101"])], "no hyphen"),
    ("g03", "tag_variant", "Какая мощность двигателя у насоса Р-310?", "What is the motor power of pump Р-310 "
     "(Cyrillic Р)?", None, ("45", "kW"), [("answer", ["PS-310", R14A1])], "Cyrillic Р"),
    ("g04", "tag_variant", "На какое давление рассчитан кран V37?", "What pressure is valve V37 rated for?", None,
     ("2,5", "MPa"), [("answer", ["VL-2 r4 t А.2 row V-37"])], "no hyphen"),
    ("g05", "tag_variant", "В каком году выпущен насос Р101?", "In which year was pump Р101 (Cyrillic Р) made?", None,
     ("2019", None), [("answer", ["PS-101"])], "Cyrillic Р, no hyphen"),
    ("g06", "tag_variant", "Какой напор у насоса P 102?", "What is the head of pump P 102?", None, ("32", "m"),
     [("answer", ["PS-102"])], "a space"),
    # --- not in documents: the right answer is to abstain -----------------------------------------------
    ("x01", "not_in_documents", "Какое допустимое давление для задвижки V-912?",
     "What is the allowed pressure for valve V-912?", None, ("status", "not_in_documents"), [], "no such valve"),
    ("x02", "not_in_documents", "Какой уровень шума у насоса P-101?", "What is the noise level of pump P-101?", None,
     ("status", "not_in_documents"), [], ""),
    ("x03", "not_in_documents", "Какое допустимое давление для задвижки V-112?",
     "What is the allowed pressure for valve V-112?", None, ("status", "not_in_documents"), [],
     "in the registry (station 1), not in the documents; looks like V-12"),
    ("x04", "not_in_documents", "Какая мощность двигателя у насоса P-999?", "What is the motor power of pump P-999?",
     None, ("status", "not_in_documents"), [], ""),
    ("x05", "not_in_documents", "При какой виброскорости по R-14 нужно осматривать машины класса VI?",
     "At what vibration velocity must class VI machines be inspected according to R-14?", None,
     ("status", "not_in_documents"), [], ""),
    ("x06", "not_in_documents", "Какой момент затяжки фундаментных болтов у насоса P-310?",
     "What is the tightening torque of the foundation bolts of pump P-310?", None, ("status", "not_in_documents"),
     [], ""),
    ("x07", "not_in_documents", "Какой гарантийный срок у насоса P-310?", "What is the warranty period of pump P-310?",
     None, ("status", "not_in_documents"), [], ""),
    ("x08", "not_in_documents", "Какое давление нагнетания у насоса P-102 по паспорту?",
     "What is the discharge pressure of pump P-102 in its datasheet?", None, ("status", "not_in_documents"), [], ""),
    ("x09", "not_in_documents", "Какая наибольшая температура перекачиваемой жидкости для насоса P-310?",
     "What is the highest temperature of the pumped liquid for pump P-310?", None, ("status", "not_in_documents"),
     [], ""),
    ("x10", "not_in_documents", "Какой номинальный ток двигателя насоса P-101?",
     "What is the rated current of the motor of pump P-101?", None, ("status", "not_in_documents"), [], ""),
]

__all__ = ["BASE", "LANGUAGE_IDS", "NEW", "TODAY"]

# ---------------------------------------------------------------------------------------------
# "Needs a person": the norm is in the documents, but the answer depends on data about the equipment (or the
# measurement) that no document and no registry card has. The right answer is the status needs_human: name what
# is missing and who has it. Not part of the retrieval set: data/needs_human.jsonl, reported as its own table.
# Registry rows: scripts/corpus_content.py (P-215, P-216, P-220A, P-220B). R-12 rev. 3: class I 2,8 / 4,5,
# class II 4,5 / 7,1 mm/s; R-14 (station 3): class II 3,5 / 5,6.
# ---------------------------------------------------------------------------------------------
NEEDS_HUMAN = [
    ("nh01", "Насос P-215 вибрирует 4,0 мм/с. Это норма?",
     "Pump P-215 vibrates at 4.0 mm/s. Is that normal?",
     [("answer", [T3]), ("support", ["R-12 r3 cl 4.5.2"])],
     "registry: station 2, no power, no class, no datasheet; not in Table A.1. Class I: above 2,8, inspect within "
     "24 h; class II: normal"),
    ("nh02", "Насос P-220 вибрирует 5,0 мм/с. Его нужно останавливать?",
     "Pump P-220 vibrates at 5.0 mm/s. Must it be stopped?",
     [("answer", [T3])],
     "no P-220 in the registry: P-220A (11 kW, class I) and P-220B (37 kW, class II). A: above 4,5, stop; "
     "B: inspect within 24 h"),
    ("nh03", "На насосной станции № 2 насос вибрирует 4,0 мм/с. Нужно ли его осматривать?",
     "A pump at pump station No. 2 vibrates at 4.0 mm/s. Must it be inspected?",
     [("answer", [T3])],
     "no tag: station 2 has class I and class II pumps. Class I: inspect; class II: normal"),
    ("nh04", "Насос P-216 после установки вибрирует 4,0 мм/с. Нужно ли его осматривать?",
     "Pump P-216 vibrates at 4.0 mm/s after installation. Must it be inspected?",
     [("answer", [T3]), ("answer", ["R-14 r1 t 3"])],
     "spare pump, class II, station not assigned: R-12 (station 2) 4,5, normal; R-14 (station 3) 3,5, inspect. "
     "Not a conflict: only one of the two applies, and the registry does not say which"),
    ("nh05", "Какой смазкой смазывать подшипники насоса P-215?",
     "Which grease must be used for the bearings of pump P-215?",
     [("answer", ["R-12 r3 cl 7.1"])],
     "R-12 7.1: the grease named in the datasheet; P-215 has no datasheet in the archive"),
    ("nh06", "Насос P-101: в вертикальном направлении 4,2 мм/с, горизонтальное и осевое направления не измеряли. "
             "Вибрация в норме?",
     "Pump P-101: 4.2 mm/s in the vertical direction; the horizontal and axial directions were not measured. "
     "Is the vibration normal?",
     [("answer", ["R-12 r3 cl 4.3.1"]), ("support", [T3])],
     "R-12 4.3.1: the result is the largest of three directions; two are missing, so 4,2 < 4,5 proves nothing. "
     "Needs a person: measure again. The status is arguable (a careful 'answered' that asks to re-measure is also "
     "safe); graded by status"),
]
