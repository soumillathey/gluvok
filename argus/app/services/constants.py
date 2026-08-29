import re

# Indian State & Union Territory Codes Mapping
STATE_CODES = {
    "AN": "Andaman and Nicobar Islands",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Daman and Diu",
    "DL": "Delhi",
    "DN": "Dadra and Nagar Haveli",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HP": "Himachal Pradesh",
    "HR": "Haryana",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "OR": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TS": "Telangana",
    "UK": "Uttarakhand",
    "UA": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
    "BP": "Police / Government Series",
    "BH": "Bharat Series (National)",
}

STATE_PREFIX_PATTERN = "|".join(sorted([k for k in STATE_CODES if k != "BH"], key=len, reverse=True))

# Regex matching Indian License Plates.
# Standard format: State(2L) + District(1-2 digits: 01-99 or 1-9) + Series(1-3L) + Serial(3-4 digits)
# BH series: Year(2 digits) + BH + Serial(4 digits) + Series(1-2L)
# Examples: MH04BG649, DL1CX2744, RJ09GA0165, RJ14GJ4976, 22BH1234AA, BP2A4904, GJ7UU1804, KA1A1234
INDIAN_PLATE_REGEX = re.compile(
    r"(?:"
    rf"({STATE_PREFIX_PATTERN})[\s.-]?(?:0[1-9]|[1-9]\d|[1-9])[\s.-]?([A-Za-z]{{1,3}})[\s.-]?(\d{{3,4}})"
    r"|"
    r"(\d{2})[\s.-]?(BH)[\s.-]?(\d{4})[\s.-]?([A-Za-z]{1,2})"
    r")",
    re.IGNORECASE,
)

# Positional character confusions for Indian license plates
CHAR_TO_DIGIT = {
    "O": "0",
    "D": "0",
    "Q": "0",
    "I": "1",
    "L": "1",
    "H": "1",
    "Z": "2",
    "A": "4",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
}

DIGIT_TO_CHAR = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "J",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}

SERIES_CORRECTIONS = {"G3": "GJ", "GT": "GJ", "GI": "GJ", "GB": "GB", "D3": "DJ", "DT": "DJ", "DI": "DJ"}

STATE_PREFIX_CORRECTIONS = {
    "W8": "WB",
    "RT": "RJ",
    "R3": "RJ",
    "D1": "DL",
    "D7": "DL",
    "H8": "HR",
    "0D": "OD",
    "0R": "OR",
    "00": "OD",
    "0L": "DL",
    "K1": "KL",
    "T1": "TN",
    "A1": "AP",
    "VB": "WB",
    "NB": "WB",
    "2B": "WB",
    "MB": "WB",
    "38": "JH",
    "28": "JH",
}

# Common non-plate commercial vehicle words/decals
NON_PLATE_WORDS = {
    "GOOD",
    "GOODS",
    "LUCK",
    "CARRIER",
    "SPEED",
    "TATA",
    "ASHOK",
    "LEYLAND",
    "EICHER",
    "INDIAN",
    "NATIONAL",
    "PERMIT",
    "DIESEL",
    "STOP",
    "HORN",
    "PLEASE",
    "FAST",
    "SUPER",
    "INDIA",
    "ROAD",
    "LINES",
    "TRANSPORT",
    "MOTORS",
    "SUPREME",
    "CEMENT",
    "COACH",
    "AIR",
    "BRAKE",
    "ALL",
    "STATE",
    "40KM",
    "PUBLIC",
    "AUTO",
    "SAFETY",
    "FIRST",
}


def normalize_candidate_strings(raw_str: str) -> list[str]:
    """
    Generate normalized plate candidate variants using positional character rules for Indian plates.
    """
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_str).upper()
    if not cleaned or len(cleaned) < 6:
        return []

    candidates = [cleaned]

    # State prefix corrections
    for prefix, repl in STATE_PREFIX_CORRECTIONS.items():
        if cleaned.startswith(prefix):
            candidates.append(repl + cleaned[len(prefix) :])

    results = list(candidates)
    for cand in candidates:
        # Standard 10-character plate: [State 2L][District 2D][Series 2L][Serial 4D]
        if len(cand) == 10:
            st = cand[:2]
            dist = cand[2:4]
            series = cand[4:6]
            serial = cand[6:10]

            st_corr = STATE_PREFIX_CORRECTIONS.get(st, st)
            dist_corr = "".join(CHAR_TO_DIGIT.get(c, c) for c in dist)
            series_corr = SERIES_CORRECTIONS.get(series, "".join(DIGIT_TO_CHAR.get(c, c) for c in series))
            serial_corr = "".join(CHAR_TO_DIGIT.get(c, c) for c in serial)

            corrected = st_corr + dist_corr + series_corr + serial_corr
            if corrected not in results:
                results.append(corrected)

            if dist_corr.startswith("4"):
                alt_corr = st_corr + "0" + dist_corr[1:] + series_corr + serial_corr
                if alt_corr not in results:
                    results.append(alt_corr)

        # 9-character plate:
        elif len(cand) == 9:
            st = cand[:2]
            st_corr = STATE_PREFIX_CORRECTIONS.get(st, st)

            # Case A: State 2L + District 2D + Series 1L + Serial 4D (e.g. RJ09G4017)
            dist_a = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[2:4])
            ser_a = "".join(DIGIT_TO_CHAR.get(c, c) for c in cand[4:5])
            num_a = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[5:9])
            cand_a = st_corr + dist_a + ser_a + num_a
            if cand_a not in results:
                results.append(cand_a)

            # Case B: State 2L + District 1D + Series 2L + Serial 4D (e.g. GJ7UU1804, DL1CX2744)
            dist_b = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[2:3])
            ser_b = "".join(DIGIT_TO_CHAR.get(c, c) for c in cand[3:5])
            num_b = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[5:9])
            cand_b = st_corr + dist_b + ser_b + num_b
            if cand_b not in results:
                results.append(cand_b)

        # 8-character plate:
        elif len(cand) == 8:
            st = cand[:2]
            st_corr = STATE_PREFIX_CORRECTIONS.get(st, st)

            # Case A: State 2L + District 1D + Series 1L + Serial 4D (e.g. BP2A4904, BP1A2453)
            dist_a = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[2:3])
            ser_a = "".join(DIGIT_TO_CHAR.get(c, c) for c in cand[3:4])
            num_a = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[4:8])
            cand_a = st_corr + dist_a + ser_a + num_a
            if cand_a not in results:
                results.append(cand_a)

            # Case B: State 2L + District 2D + Series 1L + Serial 3D (e.g. KA25B315)
            dist_b = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[2:4])
            ser_b = "".join(DIGIT_TO_CHAR.get(c, c) for c in cand[4:5])
            num_b = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[5:8])
            cand_b = st_corr + dist_b + ser_b + num_b
            if cand_b not in results:
                results.append(cand_b)

        # Bharat (BH) series
        if "BH" in cand:
            idx = cand.find("BH")
            if idx >= 2 and len(cand) >= idx + 6:
                yr = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[idx - 2 : idx])
                serial = "".join(CHAR_TO_DIGIT.get(c, c) for c in cand[idx + 2 : idx + 6])
                ser = "".join(DIGIT_TO_CHAR.get(c, c) for c in cand[idx + 6 :])
                bh_cand = yr + "BH" + serial + ser
                if bh_cand not in results:
                    results.append(bh_cand)

    return results
