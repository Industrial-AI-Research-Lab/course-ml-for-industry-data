"""Questions of the mini-eval: 50 questions about the scanned documents, answers checked by code.

Built by the rules of lecture 2: every question comes from a known place in a document (a clause, a table
row, a datasheet line), the answer is a number with a unit, and 8 of 50 are negative: the answer is not in
the document ("not stated"). The wording does not copy the clause, and the questions are in English while
the documents are in Russian and Chinese, so there is no lexical leak.

Fields: id, doc, question, value (as written in the document, or None), unit (canonical, or None),
where (clause number, table or datasheet line), slice, anchor (a string on the page with the answer;
the generator uses it to find the page).
"""

R12 = "R-12"  # revision 3, the scanned copy
P101 = "PS-101"
P205 = "DS-P-205"
T3 = "Таблица 3 —"

QUESTIONS = [
    # --- Table 3: the answer is a cell of a table with merged cells -------------------------
    ("t01", R12, "What is the inspection limit of vibration velocity for class I machines in R-12?", "2,8", "mm/s", "Table 3", "table", T3),
    ("t02", R12, "What is the stop limit of vibration velocity for class I machines in R-12?", "4,5", "mm/s", "Table 3", "table", T3),
    ("t03", R12, "What is the inspection limit of vibration velocity for class II machines in R-12?", "4,5", "mm/s", "Table 3", "table", T3),
    ("t04", R12, "What is the stop limit of vibration velocity for class II machines in R-12?", "7,1", "mm/s", "Table 3", "table", T3),
    ("t05", R12, "What is the inspection limit of vibration velocity for class III machines in R-12?", "7,1", "mm/s", "Table 3", "table", T3),
    ("t06", R12, "What is the stop limit of vibration velocity for class III machines in R-12?", "11,2", "mm/s", "Table 3", "table", T3),
    ("t07", R12, "What is the inspection limit of vibration velocity for class IV machines in R-12?", "11,2", "mm/s", "Table 3", "table", T3),
    ("t08", R12, "What is the stop limit of vibration velocity for class IV machines in R-12?", "18,0", "mm/s", "Table 3", "table", T3),
    ("t09", R12, "What is the inspection limit of vibration velocity for fans (class V) in R-12?", "5,6", "mm/s", "Table 3", "table", T3),
    ("t10", R12, "What is the stop limit of vibration velocity for fans (class V) in R-12?", "9,0", "mm/s", "Table 3", "table", T3),
    # --- Appendix A: a plain table -------------------------------------------------------------
    ("a01", R12, "What is the drive power of compressor K-301 in the equipment list of R-12?", "250", "kW", "Table A.1", "table", "Таблица А.1"),
    ("a02", R12, "What is the drive power of fan B-401 in the equipment list of R-12?", "30", "kW", "Table A.1", "table", "Таблица А.1"),
    ("a03", R12, "What is the drive power of pump P-102 in the equipment list of R-12?", "11", "kW", "Table A.1", "table", "Таблица А.1"),
    ("a04", R12, "What is the drive power of compressor K-302 in the equipment list of R-12?", "400", "kW", "Table A.1", "table", "Таблица А.1"),
    # --- Clauses: the answer is a number in running text ----------------------------------------
    ("c01", R12, "What is the upper end of the measuring range of vibration velocity for the instruments?", "100", "mm/s", "4.2.1", "text", "от 0,1 до 100"),
    ("c02", R12, "What is the largest permitted basic error of a vibration velocity measurement?", "5", "%", "4.2.2", "text", "погрешность измерения"),
    ("c03", R12, "How often must the measuring instruments be verified or calibrated, at least?", "12", "months", "4.2.3", "text", "калибруют не реже"),
    ("c04", R12, "How many minutes after start-up may vibration measurements begin, at the earliest?", "30", "min", "4.1.2", "text", "после пуска оборудования"),
    ("c05", R12, "Within what time must a machine be inspected after its vibration exceeds the inspection limit?", "24", "h", "4.3.3", "text", "4.3.3"),
    ("c06", R12, "How often must periodic measurements with a portable vibration meter be done, in operating hours?", "720", "h", "4.4.1", "text", "4.4.1"),
    ("c07", R12, "Within what time after start-up must vibration be measured after a bearing replacement?", "8", "h", "4.4.2", "text", "4.4.2"),
    ("c08", R12, "Up to what drive power is a centrifugal pump a class I machine?", "15", "kW", "4.5.2", "text", "4.5.2"),
    ("c09", R12, "What is the highest permitted temperature of the bearing outer ring?", "80", "°C", "5.1", "text", "наружного кольца"),
    ("c10", R12, "What rise of bearing temperature within 10 minutes requires a stop?", "5", "°C", "5.2", "text", "за 10 минут"),
    ("c11", R12, "Within what time must the operator report an exceeded limit to the shift supervisor?", "1", "h", "6.1", "text", "6.1"),
    ("c12", R12, "How often must rolling bearings be lubricated, in operating hours?", "2000", "h", "7.1", "text", "7.1"),
    ("c13", R12, "What is the largest permitted radial misalignment of the shafts after a repair?", "0,05", "mm", "7.2", "text", "радиального смещения"),
    ("c14", R12, "After how many operating hours must bearings be replaced at the latest?", "25 000", "h", "7.4", "text", "7.4"),
    ("c15", R12, "For how many years must the measurement logs be kept, at least?", "5", "years", "8.1", "text", "8.1"),
    ("c16", R12, "Below what rotation speed does R-12 not apply?", "600", "rpm", "1.4", "text", "1.4"),
    ("c17", R12, "What is the upper frequency of the band in which vibration velocity is defined?", "1000", "Hz", "3.1", "text", "3.1"),
    # --- Datasheet P-101, Russian and English ---------------------------------------------------
    ("d01", P101, "What is the rated flow of pump P-101?", "65", "m³/h", "datasheet", "datasheet", "P-101"),
    ("d02", P101, "What is the rated head of pump P-101?", "125", "m", "datasheet", "datasheet", "P-101"),
    ("d03", P101, "What is the rotation speed of pump P-101?", "2950", "rpm", "datasheet", "datasheet", "P-101"),
    ("d04", P101, "What is the motor power of pump P-101?", "55", "kW", "datasheet", "datasheet", "P-101"),
    ("d05", P101, "What is the mass of the P-101 pump unit?", "640", "kg", "datasheet", "datasheet", "P-101"),
    ("d06", P101, "In which year was pump P-101 manufactured?", "2019", None, "datasheet", "datasheet", "P-101"),
    # --- Datasheet P-205, Chinese and English: a language that is rare for a Russian OCR setup ---
    ("z01", P205, "What is the rated flow of pump P-205?", "50", "m³/h", "datasheet", "rare_language", "P-205"),
    ("z02", P205, "What is the rated head of pump P-205?", "80", "m", "datasheet", "rare_language", "P-205"),
    ("z03", P205, "What is the rotation speed of pump P-205?", "2900", "rpm", "datasheet", "rare_language", "P-205"),
    ("z04", P205, "What is the motor power of pump P-205?", "37", "kW", "datasheet", "rare_language", "P-205"),
    ("z05", P205, "What is the mass of pump P-205?", "520", "kg", "datasheet", "rare_language", "P-205"),
    # --- Not stated: the right answer is "not stated" ------------------------------------------
    ("n01", R12, "What is the maximum noise level for class II pumps in R-12?", None, None, "Table 3", "not_stated", T3),
    ("n02", R12, "What is the vibration velocity stop limit for class VI machines in R-12?", None, None, "Table 3", "not_stated", T3),
    ("n03", R12, "How often must the batteries of portable vibration meters be replaced according to R-12?", None, None, "4.2", "not_stated", "калибруют не реже"),
    ("n04", R12, "What is the permitted shaft run-out after a repair according to R-12?", None, None, "7.2", "not_stated", "радиального смещения"),
    ("n05", P101, "What is the required NPSH of pump P-101?", None, None, "datasheet", "not_stated", "P-101"),
    ("n06", P101, "What oil volume does the bearing housing of P-101 hold?", None, None, "datasheet", "not_stated", "P-101"),
    ("n07", P205, "What is the maximum suction pressure of pump P-205?", None, None, "datasheet", "not_stated", "P-205"),
    ("n08", P205, "What is the warranty period of pump P-205?", None, None, "datasheet", "not_stated", "P-205"),
]
