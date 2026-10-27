"""Content of the plant archive, version 2: the new documents of practice 4.

Everything here is teaching material: the plant, the regulations, the pumps, the valves and the numbers are
invented. Table 3 of R-12 and R-14 is in the style of ISO 20816 but its values are NOT taken from the standard.

The structure below is the ground truth. `generate_corpus.py` renders it to born-digital PDFs, and the
questions of the retrieval set are written against it. R-12 rev. 2 and rev. 3, PS-101 and DS-P-205 are not
here: they come from practice 3 (its chunk store is in inputs/practice-03/store/, its content in
practice03_content.py). The core numbers of practice 3 never change: class II, 4,5 (rev. 3) / 5,0 (rev. 2),
stop 7,1, clause 4.3.3, 24 hours, section 6.

A regulation is a dict with metadata, `sections` and `tables` in the format of practice 3:
a section is (number, title, items); an item is ("clause", number, text), ("subsection", number, title),
("table", table_id) or ("para", text). A table is a caption, header rows and cells (row, col, rowspan,
colspan, text).
"""

from __future__ import annotations

import copy
import random

import practice03_content as P3

ORG = P3.ORG
OWNER = P3.OWNER
DISCLAIMER = P3.DISCLAIMER

# The "today" of the practice: questions without a date are asked on this day.
TODAY = "2026-10-27"


def _cells_from_rows(header: list[str], rows: list[list[str]]) -> list[tuple]:
    cells = [(0, c, 1, 1, h) for c, h in enumerate(header)]
    for r, row in enumerate(rows, start=1):
        cells += [(r, c, 1, 1, v) for c, v in enumerate(row)]
    return cells


def _simple_table(caption: str, header: list[str], rows: list[list[str]], widths: list[float],
                  note: str | None = None) -> dict:
    return {"caption": caption, "header_rows": 1, "n_cols": len(header), "cells": _cells_from_rows(header, rows),
            "widths": widths, "note": note}


# ---------------------------------------------------------------------------------------------
# R-12 rev. 3, Amendment No. 1: changes two clauses, not the whole document
# ---------------------------------------------------------------------------------------------
AMENDMENT_R12 = {
    "key": "R-12_rev3_amd1", "file": "R-12_rev3_amd1.pdf", "doc_id": "R-12", "revision": 3, "amendment": 1,
    "kind": "amendment", "language": "ru",
    "title": P3.DOC_TITLE,
    "approved": "18.08.2026", "effective_from": "2026-09-01",
    "scope_text": P3.SCOPE, "scope_stations": ["2"], "scope_equipment_types": ["pump", "compressor", "fan"],
    "scope_tags": [],
    # (item number, amended clause or None, text). An item that sets a new wording of a clause replaces
    # the chunk of that clause in the base revision from the effective date.
    "items": [
        ("1", "4.4.1", "Пункт 4.4.1 изложить в новой редакции: «4.4.1 Периодический контроль переносным "
                       "виброметром выполняют каждые 500 часов наработки, но не реже одного раза в месяц.»"),
        ("2", "8.3", "Пункт 8.3 изложить в новой редакции: «8.3 Изменения в регламент вносят новой редакцией "
                     "или изменением к действующей редакции. Заменённые редакции и пункты хранятся в архиве "
                     "с отметкой «заменён»."),
        ("3", None, "Основание: анализ отказов подшипников насосов насосной станции № 2 за 2025 год "
                    "(учебный пример)."),
        ("4", None, "Остальные пункты редакции 3 действуют без изменений."),
    ],
}

# ---------------------------------------------------------------------------------------------
# R-14: a near-duplicate of R-12 for pump station No. 3. Same title, same clause numbers, stricter Table 3
# ---------------------------------------------------------------------------------------------
R14_TABLE3 = {"I": ("2,2", "3,5"), "II": ("3,5", "5,6"), "III": ("5,6", "9,0"), "IV": ("9,0", "14,0"),
              "V": ("4,5", "7,1")}


def _r14_sections() -> list:
    values = {"interval_h": "720", "temp_limit": "80"}
    out = []
    for num, title, items in copy.deepcopy(P3.SECTIONS):
        new_items = []
        for it in items:
            if it[0] in ("clause", "para"):
                text = it[-1].format(**values).replace("насосной станции № 2", "насосной станции № 3")
                it = (*it[:-1], text)
            new_items.append(it)
        out.append((num, title, new_items))
    return out


def _r14_table3_cells() -> list[tuple]:
    t = R14_TABLE3
    return [
        (0, 0, 2, 1, "Группа оборудования"), (0, 1, 2, 1, "Класс машины"),
        (0, 2, 1, 2, "Граница виброскорости, мм/с"),
        (1, 2, 1, 1, "осмотр"), (1, 3, 1, 1, "останов"),
        (2, 0, 2, 1, "Насосы"), (2, 1, 1, 1, "I"), (2, 2, 1, 1, t["I"][0]), (2, 3, 1, 1, t["I"][1]),
        (3, 1, 1, 1, "II"), (3, 2, 1, 1, t["II"][0]), (3, 3, 1, 1, t["II"][1]),
        (4, 0, 2, 1, "Компрессоры"), (4, 1, 1, 1, "III"), (4, 2, 1, 1, t["III"][0]), (4, 3, 1, 1, t["III"][1]),
        (5, 1, 1, 1, "IV"), (5, 2, 1, 1, t["IV"][0]), (5, 3, 1, 1, t["IV"][1]),
        (6, 0, 1, 1, "Вентиляторы"), (6, 1, 1, 1, "V"), (6, 2, 1, 1, t["V"][0]), (6, 3, 1, 1, t["V"][1]),
    ]


R14_EQUIPMENT = [
    ("P-310", "Насос циркуляционный", "45", "II", "ПС-310"),
    ("P-311", "Насос дренажный", "7,5", "I", "ПС-311"),
    ("P-312", "Насос сырьевой", "75", "II", "ПС-312"),
    ("K-330", "Компрессор воздушный", "160", "III", "ПС-330"),
    ("B-340", "Вентилятор вытяжной", "22", "V", "ПС-340"),
]

R14 = {
    "key": "R-14_rev1", "file": "R-14_rev1.pdf", "doc_id": "R-14", "revision": 1, "amendment": None,
    "kind": "regulation", "kind_label": "РЕГЛАМЕНТ", "language": "ru",
    "title": P3.DOC_TITLE,  # the same title as R-12: on purpose
    "approved": "12.02.2024", "effective_from": "2024-03-01",  # in force later than approved
    "replaces": "Введён впервые", "scope_text": "Насосная станция № 3",
    "scope_stations": ["3"], "scope_equipment_types": ["pump", "compressor", "fan"], "scope_tags": [],
    "sections": _r14_sections(),
    "two_column": {"3"},
    "tables": {
        "table3": {"caption": "Таблица 3 — Границы виброскорости по классам машин (СКЗ, мм/с)", "header_rows": 2,
                   "n_cols": 4, "cells": _r14_table3_cells(), "widths": [55, 32, 38, 38],
                   "note": "Примечание — Значения установлены предприятием для учебного примера и не являются "
                           "нормами ГОСТ ISO 20816. Нормы строже, чем в R-12: насосы станции № 3 работают на "
                           "лёгких фундаментах."},
        "table2": {**{k: v for k, v in P3.TABLES["table2"].items() if k not in ("cells", "rows")},
                   "cells": P3.TABLES["table2"]["cells"](3), "widths": [60, 50, 30]},
        "tableA1": _simple_table("Таблица А.1 — Оборудование насосной станции № 3",
                                 ["Тег", "Наименование", "Мощность, кВт", "Класс", "Паспорт"],
                                 [list(r) for r in R14_EQUIPMENT], [20, 62, 30, 20, 28]),
    },
    "appendix": ("А", "Перечень оборудования насосной станции № 3 (справочное)", [
        ("para", "Класс машины по таблице 2 указан для каждой единицы оборудования. При замене оборудования "
                 "перечень обновляют в новой редакции регламента."),
        ("table", "tableA1"),
    ]),
}

# ---------------------------------------------------------------------------------------------
# R-15: maintenance of centrifugal pumps. A distractor on the same topic, refers to R-12, R-14 and R-20
# ---------------------------------------------------------------------------------------------
R15 = {
    "key": "R-15_rev1", "file": "R-15_rev1.pdf", "doc_id": "R-15", "revision": 1, "amendment": None,
    "kind": "regulation", "kind_label": "РЕГЛАМЕНТ", "language": "ru",
    "title": "Техническое обслуживание центробежных насосов",
    "approved": "01.09.2023", "effective_from": "2023-09-01",
    "replaces": "Введён впервые", "scope_text": "Насосные станции № 2 и № 3",
    "scope_stations": ["2", "3"], "scope_equipment_types": ["pump"], "scope_tags": [],
    "sections": [
        ("1", "Область применения", [
            ("clause", "1.1", "Настоящий регламент устанавливает виды, периодичность и объём технического "
                              "обслуживания центробежных насосов насосных станций № 2 и № 3."),
            ("clause", "1.2", "Вибрационный контроль насосов выполняют по регламенту R-12 для станции № 2 и "
                              "по регламенту R-14 для станции № 3. Настоящий регламент норм вибрации не "
                              "устанавливает."),
            ("clause", "1.3", "Работы по настоящему регламенту выполняет персонал службы главного механика."),
        ]),
        ("2", "Нормативные ссылки", [
            ("para", "— R-12 Вибрационный контроль насосного оборудования (насосная станция № 2);"),
            ("para", "— R-14 Вибрационный контроль насосного оборудования (насосная станция № 3);"),
            ("para", "— R-20 Порядок работы с заявками в CMMS."),
        ]),
        ("3", "Виды и периодичность обслуживания", [
            ("clause", "3.1", "Ежесменное обслуживание выполняет оперативный персонал: внешний осмотр насоса, "
                              "проверка утечек через уплотнения и контроль шума подшипников на слух."),
            ("clause", "3.2", "Техническое обслуживание ТО-1 выполняют каждые 1500 часов наработки."),
            ("clause", "3.3", "Техническое обслуживание ТО-2 выполняют каждые 4500 часов наработки."),
            ("clause", "3.4", "Капитальный ремонт выполняют по техническому состоянию, но не реже одного раза "
                              "в 6 лет."),
            ("clause", "3.5", "Объём работ каждого вида обслуживания приведён в таблице 1."),
            ("table", "table1"),
        ]),
        ("4", "Смазка", [
            # a same-language conflict: R-12 and R-14, clause 7.1, say every 2000 hours
            ("clause", "4.1", "Подшипники качения насосов смазывают каждые 3000 часов наработки смазкой, "
                              "указанной в паспорте насоса."),
            ("clause", "4.2", "Масло в масляной ванне подшипникового узла заменяют при каждом ТО-2."),
            ("clause", "4.3", "Марку смазочного материала берут из паспорта насоса. Замена марки без "
                              "согласования с главным механиком не допускается."),
        ]),
        ("5", "Центровка", [
            ("clause", "5.1", "Центровку валов насоса и двигателя проверяют при каждом ТО-1 и после каждого "
                              "ремонта."),
            # a same-language conflict: R-12 and R-14, clause 7.2, say 0,05 mm
            ("clause", "5.2", "Радиальное смещение валов после центровки допускается не более 0,08 мм."),
            ("clause", "5.3", "Результаты центровки заносят в CMMS в порядке, установленном R-20."),
        ]),
        ("6", "Торцевые уплотнения", [
            ("clause", "6.1", "Допустимая утечка через торцевое уплотнение — не более 10 капель в минуту."),
            ("clause", "6.2", "При превышении допустимой утечки начальник смены создаёт заявку с приоритетом "
                              "«средний» по R-20."),
        ]),
        ("7", "Записи", [
            ("clause", "7.1", "Результаты технического обслуживания заносят в CMMS в течение смены."),
            ("clause", "7.2", "График ТО-1 и ТО-2 на год утверждает главный механик до 15 декабря."),
        ]),
    ],
    "tables": {
        "table1": _simple_table("Таблица 1 — Объём технического обслуживания",
                                ["Вид обслуживания", "Периодичность", "Основные работы"],
                                [["Ежесменное", "каждую смену", "внешний осмотр, утечки, шум подшипников"],
                                 ["ТО-1", "1500 ч", "подтяжка крепежа, проверка центровки, очистка фильтров"],
                                 ["ТО-2", "4500 ч", "замена масла, ревизия торцевого уплотнения, проверка муфты"],
                                 ["Капитальный ремонт", "по состоянию, не реже 1 раза в 6 лет",
                                  "разборка, дефектация, замена изношенных деталей"]],
                                [40, 45, 80]),
    },
    "appendix": None,
}

# ---------------------------------------------------------------------------------------------
# R-20: work requests in the CMMS. Defines what "priority high" in clause 6.2 of R-12 means
# ---------------------------------------------------------------------------------------------
R20 = {
    "key": "R-20_rev2", "file": "R-20_rev2.pdf", "doc_id": "R-20", "revision": 2, "amendment": None,
    "kind": "procedure", "kind_label": "ПОРЯДОК", "language": "ru",
    "title": "Порядок работы с заявками в CMMS",
    "approved": "10.01.2023", "effective_from": "2023-01-10",
    "replaces": "Взамен редакции 1 от 05.04.2019", "scope_text": "Все подразделения предприятия",
    "scope_stations": ["all"], "scope_equipment_types": ["all"], "scope_tags": [],
    "sections": [
        ("1", "Область применения", [
            ("clause", "1.1", "Настоящий порядок устанавливает правила создания, приоритизации и выполнения "
                              "заявок на техническое обслуживание и ремонт оборудования в CMMS."),
            ("clause", "1.2", "Требования порядка распространяются на все подразделения предприятия."),
        ]),
        ("2", "Термины", [
            ("clause", "2.1", "Заявка: запись в CMMS о необходимости работ на единице оборудования."),
            ("clause", "2.2", "Приоритет: признак заявки, который определяет сроки её принятия в работу и "
                              "выполнения."),
            ("clause", "2.3", "Срок принятия в работу: время от создания заявки до назначения исполнителя."),
            ("clause", "2.4", "Срок выполнения: время от создания заявки до её закрытия исполнителем."),
        ]),
        ("3", "Приоритеты заявок", [
            ("clause", "3.1", "Приоритет заявки устанавливает её автор по таблице 1."),
            ("clause", "3.2", "Приоритет «аварийный» присваивают при отказе оборудования, угрозе жизни и "
                              "здоровью людей, пожаре или утечке опасного вещества."),
            ("clause", "3.3", "Приоритет «высокий» присваивают при превышении норм вибрации и температуры, "
                              "установленных регламентами вибрационного контроля (п. 6.2 R-12, п. 6.2 R-14)."),
            ("clause", "3.4", "Приоритет «средний» присваивают при неисправности, которая не грозит остановом "
                              "оборудования."),
            ("clause", "3.5", "Приоритет «плановый» присваивают работам по графику планово-предупредительного "
                              "ремонта (ППР)."),
            ("table", "table1"),
        ]),
        ("4", "Создание заявки", [
            ("clause", "4.1", "В заявке указывают тег оборудования, описание неисправности, приоритет и номер "
                              "пункта документа, по которому создана заявка."),
            ("clause", "4.2", "Заявку создаёт начальник смены или механик участка."),
            ("clause", "4.3", "Результаты измерений и фотографии прикладывают к заявке."),
        ]),
        ("5", "Контроль сроков", [
            ("clause", "5.1", "Если заявку не приняли в работу в срок по таблице 1, CMMS направляет "
                              "уведомление начальнику цеха."),
            ("clause", "5.2", "Если заявку не выполнили в срок, её рассматривают на ежедневном совещании "
                              "службы главного механика."),
            ("clause", "5.3", "Закрытые заявки хранят в CMMS не менее 5 лет."),
        ]),
    ],
    "tables": {
        "table1": _simple_table("Таблица 1 — Приоритеты заявок и сроки",
                                ["Приоритет", "Срок принятия в работу", "Срок выполнения"],
                                [["аварийный", "30 минут", "24 часа"],
                                 ["высокий", "4 часа", "3 суток"],
                                 ["средний", "1 рабочий день", "10 рабочих дней"],
                                 ["плановый", "по графику ППР", "по графику ППР"]],
                                [45, 60, 60]),
    },
    "appendix": None,
}

# ---------------------------------------------------------------------------------------------
# Manufacturer's manual of P-205, in English: conflicts with R-12 on the vibration limits
# ---------------------------------------------------------------------------------------------
MANUAL_P205 = {
    "key": "OM-P-205", "file": "OM-P-205_manual_en.pdf", "doc_id": "OM-P-205", "revision": 2, "amendment": None,
    "kind": "manual", "kind_label": "OPERATION AND MAINTENANCE MANUAL", "language": "en",
    "title": "Centrifugal pump XB80-200 (plant tag P-205)",
    "approved": "15.05.2023", "effective_from": "2023-05-15",
    "replaces": "Edition B, replaces edition A of 2021", "scope_text": "Pump XB80-200, plant tag P-205",
    "scope_stations": ["2"], "scope_equipment_types": ["pump"], "scope_tags": ["P-205"],
    "manufacturer": "Huadong Pump Co., Ltd. (teaching example)",
    "sections": [
        ("1", "Safety", [
            ("clause", "1.1", "Read this manual before installation, start-up and maintenance of the pump."),
            ("clause", "1.2", "Only trained personnel may operate and service the pump."),
            ("clause", "1.3", "Stop the pump and lock out the motor before any maintenance work."),
        ]),
        ("2", "Description", [
            ("clause", "2.1", "The XB80-200 is a single-stage end-suction centrifugal pump with a "
                              "mechanical seal and oil-lubricated rolling bearings."),
            ("clause", "2.2", "The rated data of the pump are given in Table 1."),
            ("table", "table1"),
        ]),
        ("3", "Installation", [
            ("clause", "3.1", "Install the pump on a rigid concrete foundation and grout the base plate."),
            ("clause", "3.2", "After installation align the pump and the motor shafts. The radial offset "
                              "must not exceed 0.08 mm."),
            ("clause", "3.3", "Support the suction and discharge pipes so that they do not load the pump "
                              "casing."),
        ]),
        ("4", "Start-up", [
            ("clause", "4.1", "Fill the pump and the suction pipe with liquid before start-up."),
            ("clause", "4.2", "Never run the pump dry, even for a short time."),
            ("clause", "4.3", "Check the direction of rotation by a short start of the motor."),
        ]),
        ("5", "Condition monitoring", [
            ("clause", "5.1", "Measure the vibration velocity (RMS, 10 to 1000 Hz) on both bearing housings "
                              "once a week."),
            ("clause", "5.2", "The vibration limits of this pump are given in Table 2. They apply to the pump "
                              "on a rigid foundation."),
            ("clause", "5.3", "If the alarm limit is exceeded, find the cause within 7 days (see section 7)."),
            ("clause", "5.4", "If the trip limit is exceeded, stop the pump at once."),
            ("clause", "5.5", "The temperature of the bearing housing must not exceed 95 °C."),
            ("table", "table2"),
        ]),
        ("6", "Maintenance", [
            ("clause", "6.1", "The bearings are oil-lubricated. Change the oil every 3000 operating hours."),
            ("clause", "6.2", "The oil volume of the bearing housing is 0.6 L."),
            ("clause", "6.3", "Replace the mechanical seal when the leakage exceeds 5 drops per minute."),
            ("clause", "6.4", "Replace the rolling bearings after 20 000 operating hours."),
        ]),
        ("7", "Troubleshooting", [
            ("clause", "7.1", "Typical causes of high vibration and their remedies are given in Table 3."),
            ("table", "table3"),
        ]),
    ],
    "tables": {
        "table1": _simple_table("Table 1 — Rated data", ["Parameter", "Unit", "Value"],
                                [["Flow", "m³/h", "50"], ["Head", "m", "80"], ["Speed", "r/min", "2900"],
                                 ["Motor power", "kW", "37"], ["Mass", "kg", "520"]],
                                [70, 40, 50]),
        "table2": _simple_table("Table 2 — Vibration limits (velocity, mm/s RMS)", ["Zone", "Velocity, mm/s",
                                                                                     "Action"],
                                [["Normal operation", "up to 7.1", "none"],
                                 ["Alarm", "above 7.1", "find the cause within 7 days"],
                                 ["Trip", "above 11.2", "stop the pump"]],
                                [50, 45, 70]),
        "table3": _simple_table("Table 3 — Troubleshooting", ["Symptom", "Possible cause", "Remedy"],
                                [["High vibration", "misalignment", "align the shafts again"],
                                 ["High vibration", "worn bearings", "replace the bearings"],
                                 ["High vibration", "cavitation", "check the suction pressure and the filter"],
                                 ["Hot bearing", "too much or too little oil", "set the correct oil level"]],
                                [45, 55, 65]),
    },
    "appendix": None,
}

# ---------------------------------------------------------------------------------------------
# Valve list of pump station No. 2: a long multi-page table, a row is the answer
# ---------------------------------------------------------------------------------------------
VALVE_TYPES = [("ЗД", "задвижка"), ("КШ", "кран шаровой"), ("КО", "клапан обратный"), ("ЗТ", "затвор дисковый")]
VALVE_FIXED = {  # rows that the questions ask about: tag -> (type, DN mm, PN MPa, medium, pipeline, drive)
    "V-12": ("ЗД", "150", "1,6", "нефть", "нагнетание P-101", "ручной"),
    "V-25": ("ЗД", "80", "1,6", "вода оборотная", "охлаждение P-102", "ручной"),
    "V-37": ("КШ", "100", "2,5", "нефть", "байпас P-205", "электропривод"),
    "V-58": ("ЗД", "200", "4,0", "нефть", "коллектор нагнетания", "электропривод"),
    "V-73": ("ЗТ", "250", "1,0", "вода оборотная", "коллектор оборотной воды", "пневмопривод"),
    "V-91": ("КО", "65", "2,5", "конденсат", "линия конденсата", "—"),
    "V-104": ("КШ", "50", "6,3", "азот", "азот на уплотнения", "ручной"),
    "V-110": ("ЗД", "300", "1,6", "нефть", "всас сырьевой линии", "электропривод"),
}
N_VALVES = 110


def valve_rows() -> list[list[str]]:
    rng = random.Random(2026)
    pipes = ["всас P-101", "нагнетание P-101", "всас P-102", "нагнетание P-102", "всас P-205",
             "нагнетание P-205", "дренаж T-201", "коллектор нагнетания", "линия конденсата", "азот на уплотнения",
             "коллектор оборотной воды", "факельная линия"]
    media = {"всас": "нефть", "нагнетание": "нефть", "дренаж": "дренаж", "коллектор нагнетания": "нефть",
             "линия конденсата": "конденсат", "азот": "азот", "коллектор оборотной": "вода оборотная",
             "факельная": "газ"}
    rows = []
    for i in range(1, N_VALVES + 1):
        tag = f"V-{i}"
        if tag in VALVE_FIXED:
            t, dn, pn, med, pipe, drive = VALVE_FIXED[tag]
        else:
            t = rng.choice([v[0] for v in VALVE_TYPES])
            dn = rng.choice(["25", "32", "40", "50", "65", "80", "100", "150", "200", "250"])
            pn = rng.choice(["1,0", "1,6", "1,6", "2,5", "4,0", "6,3"])
            pipe = rng.choice(pipes)
            med = next(v for k, v in media.items() if pipe.startswith(k))
            drive = "—" if t == "КО" else rng.choice(["ручной", "ручной", "электропривод", "пневмопривод"])
        rows.append([tag, t, dn, pn, med, pipe, drive])
    return rows


VALVE_LIST = {
    "key": "VL-2_rev4", "file": "VL-2_valve_list_rev4.pdf", "doc_id": "VL-2", "revision": 4, "amendment": None,
    "kind": "list", "kind_label": "ПЕРЕЧЕНЬ", "language": "ru",
    "title": "Запорная арматура насосной станции № 2",
    "approved": "20.11.2025", "effective_from": "2025-11-20",
    "replaces": "Взамен редакции 3 от 02.02.2023", "scope_text": "Насосная станция № 2",
    "scope_stations": ["2"], "scope_equipment_types": ["valve"], "scope_tags": [],
    "sections": [
        ("1", "Общие положения", [
            ("clause", "1.1", "Перечень содержит запорную и обратную арматуру технологических трубопроводов "
                              "насосной станции № 2 (таблица А.2)."),
            ("clause", "1.2", "Условные обозначения типов арматуры приведены в таблице А.1."),
            ("clause", "1.3", "Арматуру эксплуатируют при рабочем давлении, не превышающем номинальное "
                              "давление PN, указанное в таблице А.2."),
            ("clause", "1.4", "Перечень пересматривают при замене арматуры, но не реже одного раза в 3 года."),
            ("clause", "1.5", "Изменения в перечень вносит служба главного механика."),
        ]),
    ],
    "tables": {
        "tableA1": _simple_table("Таблица А.1 — Условные обозначения типов арматуры", ["Обозначение", "Тип"],
                                 [list(v) for v in VALVE_TYPES], [40, 80]),
        "tableA2": _simple_table("Таблица А.2 — Перечень запорной арматуры насосной станции № 2",
                                 ["Тег", "Тип", "DN, мм", "PN, МПа", "Среда", "Трубопровод", "Привод"],
                                 valve_rows(), [14, 12.5, 15, 17, 27.5, 44, 30]),
    },
    "appendix": ("А", "Перечень арматуры (обязательное)", [("table", "tableA1"), ("table", "tableA2")]),
    "long_tables": {"tableA2"},
}

# ---------------------------------------------------------------------------------------------
# Noise: realistic background on other topics
# ---------------------------------------------------------------------------------------------
FIRE_SAFETY = {
    "key": "FS-07_rev2", "file": "FS-07_fire_safety_rev2.pdf", "doc_id": "FS-07", "revision": 2,
    "amendment": None, "kind": "instruction", "kind_label": "ИНСТРУКЦИЯ", "language": "ru",
    "title": "О мерах пожарной безопасности в помещении насосной станции № 2",
    "approved": "15.01.2024", "effective_from": "2024-01-15",
    "replaces": "Взамен редакции 1 от 10.02.2020", "scope_text": "Насосная станция № 2",
    "scope_stations": ["2"], "scope_equipment_types": ["all"], "scope_tags": [],
    "sections": [
        ("1", "Общие положения", [
            ("clause", "1.1", "Инструкция устанавливает требования пожарной безопасности в помещении насосной "
                              "станции № 2 и обязательна для всех работников, которые в нём находятся."),
            ("clause", "1.2", "Ответственный за пожарную безопасность помещения назначается приказом "
                              "начальника цеха."),
            ("clause", "1.3", "Курение и применение открытого огня в помещении запрещены, кроме огневых работ "
                              "по наряду-допуску."),
        ]),
        ("2", "Содержание помещения", [
            ("clause", "2.1", "Ежесменный осмотр помещения выполняет оперативный персонал перед сдачей смены."),
            ("clause", "2.2", "Проходы к насосам и электрическим щитам шириной не менее 1 м содержат "
                              "свободными."),
            ("clause", "2.3", "Промасленную ветошь собирают в металлические ящики с крышкой и удаляют из "
                              "помещения в конце смены."),
            ("clause", "2.4", "Разлитые нефтепродукты немедленно убирают, место разлива засыпают песком."),
        ]),
        ("3", "Огневые работы", [
            ("clause", "3.1", "Огневые работы выполняют по наряду-допуску. Наряд-допуск действует не более "
                              "одной смены."),
            ("clause", "3.2", "Перед началом огневых работ выполняют анализ воздушной среды. Повторный анализ "
                              "выполняют после каждого перерыва в работе."),
            ("clause", "3.3", "После окончания огневых работ место их проведения осматривают в течение "
                              "4 часов."),
        ]),
        ("4", "Первичные средства пожаротушения", [
            ("clause", "4.1", "Огнетушители осматривают не реже одного раза в месяц, результаты заносят в "
                              "журнал."),
            ("clause", "4.2", "Огнетушители перезаряжают не реже одного раза в 5 лет и после каждого "
                              "применения."),
            ("clause", "4.3", "Нормы оснащения помещения первичными средствами пожаротушения приведены в "
                              "таблице 1."),
            ("table", "table1"),
        ]),
        ("5", "Действия при пожаре", [
            ("clause", "5.1", "Работник, обнаруживший пожар, сообщает о нём по телефону 01 и начальнику смены."),
            ("clause", "5.2", "Начальник смены останавливает насосы кнопкой аварийного останова и закрывает "
                              "задвижки на входе и выходе станции."),
            ("clause", "5.3", "До прибытия пожарной охраны работники тушат пожар первичными средствами, если "
                              "это не угрожает их жизни."),
        ]),
    ],
    "tables": {
        "table1": _simple_table("Таблица 1 — Нормы оснащения помещения насосной станции № 2",
                                ["Средство", "Количество", "Место"],
                                [["Огнетушитель порошковый ОП-10", "4", "у входов и у щита управления"],
                                 ["Огнетушитель углекислотный ОУ-5", "2", "у электрических щитов"],
                                 ["Ящик с песком 0,5 м³", "1", "у насосов P-101 и P-102"],
                                 ["Покрывало для изоляции очага возгорания", "2", "у входов"]],
                                [70, 30, 65]),
    },
    "appendix": None,
}

LAB_ACCESS = {
    "key": "LAB-03_rev1", "file": "LAB-03_lab_access_rev1.pdf", "doc_id": "LAB-03", "revision": 1,
    "amendment": None, "kind": "procedure", "kind_label": "ПОРЯДОК", "language": "ru",
    "title": "Допуск персонала к работе в химико-аналитической лаборатории",
    "approved": "01.03.2022", "effective_from": "2022-03-01",
    "replaces": "Введён впервые", "scope_text": "Химико-аналитическая лаборатория",
    "scope_stations": ["lab"], "scope_equipment_types": ["all"], "scope_tags": [],
    "sections": [
        ("1", "Общие положения", [
            ("clause", "1.1", "Порядок устанавливает требования к допуску работников к самостоятельной работе "
                              "в химико-аналитической лаборатории."),
            ("clause", "1.2", "Работы с вредными веществами выполняют только работники, допущенные по "
                              "настоящему порядку."),
        ]),
        ("2", "Допуск к работе", [
            ("clause", "2.1", "Перед допуском работник проходит вводный инструктаж и инструктаж на рабочем "
                              "месте."),
            ("clause", "2.2", "Стажировка на рабочем месте длится от 2 до 14 смен в зависимости от "
                              "квалификации работника."),
            ("clause", "2.3", "Проверку знаний выполняют не реже одного раза в 12 месяцев."),
            ("clause", "2.4", "Допуск к самостоятельной работе оформляют распоряжением начальника лаборатории."),
        ]),
        ("3", "Работа с пробами", [
            ("clause", "3.1", "Пробы нефтепродуктов хранят в лаборатории 30 суток после анализа."),
            ("clause", "3.2", "Температура воздуха в помещении лаборатории поддерживается от 18 до 25 °C."),
            ("clause", "3.3", "Скорость воздуха в проёме вытяжного шкафа — не менее 0,5 м/с."),
            ("clause", "3.4", "Пробы маркируют: номер пробы, место отбора, дата и время отбора."),
        ]),
        ("4", "Средства защиты", [
            ("clause", "4.1", "В лаборатории работают в халате, защитных очках и перчатках."),
            ("clause", "4.2", "Аптечку и средства для нейтрализации кислот проверяют еженедельно."),
        ]),
    ],
    "tables": {},
    "appendix": None,
}

WAREHOUSE = {
    "key": "WH-04_rev3", "file": "WH-04_lubricants_rev3.pdf", "doc_id": "WH-04", "revision": 3,
    "amendment": None, "kind": "instruction", "kind_label": "ИНСТРУКЦИЯ", "language": "ru",
    "title": "Приём, хранение и выдача смазочных материалов на складе ГСМ",
    "approved": "01.10.2021", "effective_from": "2021-10-01",
    "replaces": "Взамен редакции 2 от 14.05.2018", "scope_text": "Склад горюче-смазочных материалов",
    "scope_stations": ["warehouse"], "scope_equipment_types": ["all"], "scope_tags": [],
    "sections": [
        ("1", "Общие положения", [
            ("clause", "1.1", "Инструкция устанавливает порядок приёма, хранения и выдачи смазочных "
                              "материалов на складе ГСМ."),
            ("clause", "1.2", "Ответственный за склад — кладовщик склада ГСМ."),
        ]),
        ("2", "Приём", [
            ("clause", "2.1", "Каждую партию смазочного материала принимают с паспортом качества "
                              "изготовителя."),
            ("clause", "2.2", "Входной контроль партии выполняет лаборатория в течение 3 рабочих дней."),
        ]),
        ("3", "Хранение", [
            ("clause", "3.1", "Пластичные смазки (Литол-24, ЦИАТИМ-201) хранят в закрытой таре при "
                              "температуре не выше 30 °C."),
            ("clause", "3.2", "Сроки хранения смазочных материалов приведены в таблице 1."),
            ("clause", "3.3", "Масла хранят в бочках на поддонах, пробками вверх."),
            ("table", "table1"),
        ]),
        ("4", "Выдача", [
            ("clause", "4.1", "Смазочные материалы выдают по требованию механика участка."),
            ("clause", "4.2", "Материал с истёкшим сроком хранения не выдают до повторного контроля "
                              "качества."),
        ]),
    ],
    "tables": {
        "table1": _simple_table("Таблица 1 — Сроки хранения смазочных материалов",
                                ["Материал", "Срок хранения, лет"],
                                [["Литол-24", "5"], ["ЦИАТИМ-201", "5"], ["Масло И-40А", "5"],
                                 ["Масло ТП-22С", "3"]], [80, 50]),
    },
    "appendix": None,
}

REGULATIONS = [R14, R15, R20, MANUAL_P205, VALVE_LIST, FIRE_SAFETY, LAB_ACCESS, WAREHOUSE]

# ---------------------------------------------------------------------------------------------
# Datasheets: format of PS-101 in practice 3
# ---------------------------------------------------------------------------------------------


def _datasheet(doc_id, tag, model, serial, year, flow, head, speed, power, cls, regulation, mass, bearings):
    return {
        "doc_id": doc_id, "tag": tag, "year": year,
        "title_ru": "ПАСПОРТ НАСОСА", "title_en": "PUMP DATASHEET", "font": "dejavu",
        "head": [("Тег / Tag", tag), ("Изготовитель / Manufacturer", "АО «Учебный насосный завод»"),
                 ("Модель / Model", model), ("Заводской номер / Serial No.", serial), ("Год выпуска / Year", year)],
        "params_caption": "Технические характеристики / Technical data",
        "params": [("Подача / Flow", "м³/ч", flow), ("Напор / Head", "м", head),
                   ("Частота вращения / Speed", "об/мин", speed), ("Мощность двигателя / Motor power", "кВт", power),
                   (f"Класс машины по {regulation} / Machine class", "—", cls), ("Масса агрегата / Unit mass", "кг", mass),
                   ("Подшипники / Bearings", "—", bearings), ("Смазка / Lubricant", "—", "Литол-24")],
        "materials_caption": "Материалы / Materials",
        "materials": [("Корпус / Casing", "Сталь 25Л"), ("Рабочее колесо / Impeller", "Сталь 20Х13Л"),
                      ("Вал / Shaft", "Сталь 40Х")],
        "footer": f"Вибрационный контроль — по регламенту {regulation}. " + DISCLAIMER,
    }


DATASHEETS = [
    _datasheet("PS-102", "P-102", "НК 25/32-80", "102-2018-0211", "2018", "25", "32", "2900", "11", "I", "R-12",
               "210", "6306"),
    _datasheet("PS-310", "P-310", "НК 80/50-160", "310-2021-0388", "2021", "80", "50", "2950", "45", "II", "R-14",
               "520", "6310 C3"),
    _datasheet("PS-311", "P-311", "НК 20/25-65", "311-2021-0390", "2021", "20", "25", "2900", "7,5", "I", "R-14",
               "150", "6305"),
]

# ---------------------------------------------------------------------------------------------
# Metadata of the documents that come from practice 3 (their chunks are in inputs/practice-03/store/)
# ---------------------------------------------------------------------------------------------
PRACTICE3_DOCS = {
    ("R-12", 2): {"effective_from": "2022-06-01", "effective_to": "2025-03-14", "language": "ru",
                  "kind": "regulation", "scope_stations": ["2"], "scope_equipment_types": ["pump", "compressor", "fan"],
                  "scope_tags": []},
    ("R-12", 3): {"effective_from": "2025-03-14", "effective_to": None, "language": "ru", "kind": "regulation",
                  "scope_stations": ["2"], "scope_equipment_types": ["pump", "compressor", "fan"], "scope_tags": []},
    ("PS-101", None): {"effective_from": "2019-01-01", "effective_to": None, "language": "ru", "kind": "datasheet",
                       "scope_stations": ["2"], "scope_equipment_types": ["pump"], "scope_tags": ["P-101"]},
    ("DS-P-205", None): {"effective_from": "2023-01-01", "effective_to": None, "language": "zh", "kind": "datasheet",
                         "scope_stations": ["2"], "scope_equipment_types": ["pump"], "scope_tags": ["P-205"]},
}

# ---------------------------------------------------------------------------------------------
# Equipment registry of practice 4: practice 1 registry, P-101 moved to pump station 2 (as R-12 Table A.1
# says), plus station, power, machine class and datasheet for the equipment of the regulations
# ---------------------------------------------------------------------------------------------
REGISTRY_FIX = {"P-101": "Pump station 2"}
# The documents of a unit of equipment (datasheet, manufacturer's manual). The manual does not name the plant tag:
# a manufacturer does not know it. Only the registry links the two.
REGISTRY_DOCUMENTS = {"P-101": ["PS-101"], "P-102": ["PS-102"], "P-205": ["DS-P-205", "OM-P-205"],
                      "P-310": ["PS-310"], "P-311": ["PS-311"]}
REGISTRY_EXTRA = {
    # tag: (type, area, name, power_kw, class, datasheet)
    "P-101": ("pump", "Pump station 2", "pump P-101", "55", "II", "PS-101"),
    "P-102": ("pump", "Pump station 2", "pump P-102", "11", "I", "PS-102"),
    "P-205": ("pump", "Pump station 2", "pump P-205 (imported)", "37", "II", "DS-P-205"),
    "K-301": ("compressor", "Pump station 2", "air compressor K-301", "250", "III", ""),
    "K-302": ("compressor", "Pump station 2", "nitrogen compressor K-302", "400", "IV", ""),
    "B-401": ("fan", "Pump station 2", "exhaust fan B-401", "30", "V", ""),
    "P-310": ("pump", "Pump station 3", "pump P-310", "45", "II", "PS-310"),
    "P-311": ("pump", "Pump station 3", "pump P-311", "7.5", "I", "PS-311"),
    "P-312": ("pump", "Pump station 3", "pump P-312", "75", "II", ""),
    "K-330": ("compressor", "Pump station 3", "air compressor K-330", "160", "III", ""),
    "B-340": ("fan", "Pump station 3", "exhaust fan B-340", "22", "V", ""),
    # the needs_human set (data/needs_human.jsonl): equipment whose card lacks what the norm depends on.
    # None of them is in R-12 Table A.1 (it is updated only with a new revision) or has a datasheet.
    "P-215": ("pump", "Pump station 2", "pump P-215 (installed 2026, card not filled)", "", "", ""),
    "P-216": ("pump", "Spare equipment", "spare pump P-216 (station not assigned)", "45", "II", ""),
    "P-220A": ("pump", "Pump station 2", "pump P-220A", "11", "I", ""),
    "P-220B": ("pump", "Pump station 2", "pump P-220B", "37", "II", ""),
}
