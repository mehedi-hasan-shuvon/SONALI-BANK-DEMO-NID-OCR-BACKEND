import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

BENGALI_TO_ENGLISH_DIGITS = {
    "০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
    "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9"
}

DIGIT_FIXES = {
    "O": "0", "o": "0",
    "I": "1", "l": "1",
    "B": "8",
    "S": "5", "s": "5",
    "Z": "2", "z": "2"
}

ENGLISH_MONTHS = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04",
    "may": "05", "jun": "06", "jul": "07", "aug": "08",
    "sep": "09", "oct": "10", "nov": "11", "dec": "12"
}

BENGALI_MONTHS = {
    "জানু": "01", "জানুয়ারি": "01", "জানুয়ারি": "01",
    "ফেব": "02", "ফেব্রুয়ারি": "02", "ফেব্রুয়ারি": "02",
    "মার্চ": "03",
    "এপ্রি": "04", "এপ্রিল": "04",
    "মে": "05",
    "জুন": "06",
    "জুল": "07", "জুলাই": "07",
    "আগ": "08", "আগস্ট": "08",
    "সেপ": "09", "সেপ্টেম্বর": "09",
    "অক্টো": "10", "অক্টোবর": "10",
    "নভে": "11", "নভেম্বর": "11",
    "ডিসে": "12", "ডিসেম্বর": "12"
}

HEADER_KEYWORDS = [
    "GOVERNMENT", "PEOPLE", "REPUBLIC", "BANGLADESH",
    "NATIONAL", "CARD", "IDENTITY", "ELECTION", "COMMISSION",
    "SIGNATURE", "HOLDER", "গণপ্রজাতন্ত্রী", "বাংলাদেশ", "জাতীয়", "পরিচয়পত্র", "পরিচয়",
    "প্রদানের", "তারিখ", "রক্তের", "গ্রুপ", "ঠিকানা", "ডাকঘর", "বাসা", "হোল্ডিং", "গ্রাম", "রাস্তা",
    "BLOOD", "GROUP", "ISSUE", "DATE", "BIRTH", "PLACE"
]

FEMALE_MARKERS = [
    "বেগম", "পারভীন", "পারভিন", "খাতুন", "আক্তার", "বানু", "বিবি",
    "সুলতানা", "জাহান", "মাকসুদা", "মোসাঃ", "মোছাম্মৎ", "ফাতেমা", "রোকেয়া", "নাসরিন"
]


def convert_bengali_digits(text: str) -> str:
    """Convert Bengali numerals (০-৯) to English digits (0-9)."""
    if not text:
        return ""
    for bn, en in BENGALI_TO_ENGLISH_DIGITS.items():
        text = text.replace(bn, en)
    return text


def clean_field_value(value: str) -> str:
    """Trim leading/trailing delimiters, punctuation, and extraneous whitespace."""
    if not value:
        return ""
    value = re.sub(r"^[\s:ঃ\-\.,|;/=~]+", "", value)
    value = re.sub(r"[\s:ঃ\-\.,|;/=~]+$", "", value)
    return " ".join(value.split())


def is_pure_back_side(text: str) -> bool:
    """Check if the text represents purely the back side of an NID card."""
    has_back_markers = any(k in text for k in ["ঠিকানা", "বাসা/হোল্ডিং", "গ্রাম/রাস্তা", "ডাকঘর", "Blood Group", "রক্তের গ্রুপ", "Place of Birth", "জন্মস্থান"])
    has_front_markers = any(k in text for k in ["Date of Birth", "DOB", "জন্ম তারিখ", "পিতা", "মাতা", "NID No", "Name:"])
    return has_back_markers and not has_front_markers


def correct_ocr_century(year_str: str) -> str:
    """
    Correct OCR century misreads.
    If OCR misreads '1' as '2' (e.g. 2080 for 1980, 2095 for 1995), auto-correct century prefix.
    """
    try:
        y = int(year_str)
        if 2030 <= y <= 2099:
            return str(1900 + (y - 2000))
    except Exception:
        pass
    return year_str


def classify_nid_type(nid_number: str) -> str:
    """Classify NID type based on digit count."""
    length = len(nid_number)
    if length == 10:
        return "Smart NID (10 digits)"
    elif length == 17:
        return "Old NID (17 digits)"
    elif length == 13:
        return "Old NID (13 digits)"
    return f"NID ({length} digits)"


def extract_nid_number(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract National ID number and determine card type:
    - Smart NID: 10 digits (e.g. 196 390 8916 -> 1963908916)
    - Old NID with birth year prefix: 17 digits
    - Old NID without birth year prefix: 13 digits
    """
    clean_text = convert_bengali_digits(text)

    # 1. Match label followed immediately (same line or next line) by digits
    label_pat = r"(?:NID\s*(?:No|NO|Number)?|ID\s*(?:NO|No)?|আইডি\s*নং|এনআইডি\s*নম্বর)\s*[\n:ঃ.\s-]*([0-9a-zA-Z\s-]{9,25})"
    m = re.search(label_pat, clean_text, re.IGNORECASE)
    if m:
        raw_val = m.group(1)
        for k, v in DIGIT_FIXES.items():
            raw_val = raw_val.replace(k, v)
        digits = re.sub(r"\D", "", raw_val)
        if len(digits) in (10, 13, 17):
            return digits, classify_nid_type(digits)

    # 2. Extract standard 10-digit Smart Card numbers (e.g. 196 390 8916 or 196-390-8916)
    pat_10 = re.findall(r"\b(\d{3}[\s-]?\d{3}[\s-]?\d{4})\b", clean_text)
    norm_10 = [
        re.sub(r"\D", "", x)
        for x in pat_10
        if len(re.sub(r"\D", "", x)) == 10 and not re.sub(r"\D", "", x).startswith("01")
    ]
    if norm_10:
        most_common_10 = Counter(norm_10).most_common(1)[0][0]
        return most_common_10, "Smart NID (10 digits)"

    # 3. If not pure back side, find standalone 10, 17, or 13 digits with frequency voting
    if not is_pure_back_side(text):
        spaced = re.sub(r"(?<=\d)\s+(?=\d)", "", clean_text)
        # Prioritize 10-digit Smart NID then 17/13 Old NID
        for length in (10, 17, 13):
            matches = re.findall(rf"\b(\d{{{length}}})\b", spaced)
            candidates = [c for c in matches if not (length == 10 and c.startswith("01"))]
            if candidates:
                best_match = Counter(candidates).most_common(1)[0][0]
                return best_match, classify_nid_type(best_match)

    return None, None


def extract_date_of_birth(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract Date of Birth from OCR text using frequency voting across passes
    and standardizing to ISO format (YYYY-MM-DD).
    """
    if is_pure_back_side(text):
        return None, None, None

    clean_text = convert_bengali_digits(text)
    # Fix letter O used as zero in days (e.g. O1 Jul 2000 -> 01 Jul 2000)
    clean_text = re.sub(r"\b[Oo](\d)\b", r"0\1", clean_text)

    # Pattern A: DD Mon YYYY (e.g. 01 Jul 2000, 14 Aug 1995)
    found_dates = re.findall(r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b", clean_text)
    if found_dates:
        # Use frequency voting to pick the most recognized date
        most_common_raw = Counter(found_dates).most_common(1)[0][0]
        match = re.match(r"(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})", most_common_raw)
        if match:
            day, mon_str, year = match.groups()
            year = correct_ocr_century(year)
            mon_key = mon_str[:3].lower()
            if mon_key in ENGLISH_MONTHS:
                mm = ENGLISH_MONTHS[mon_key]
                iso = f"{year}-{mm}-{int(day):02d}"
                display = f"{int(day):02d} {mon_str.capitalize()} {year}"
                return iso, display, most_common_raw

    # Pattern B: DD/MM/YYYY or DD-MM-YYYY
    found_numeric = re.findall(r"\b(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{4})\b", clean_text)
    if found_numeric:
        day, mm, year = found_numeric[0]
        year = correct_ocr_century(year)
        if 1 <= int(mm) <= 12 and 1 <= int(day) <= 31:
            iso = f"{year}-{int(mm):02d}-{int(day):02d}"
            display = f"{int(day):02d}/{int(mm):02d}/{year}"
            return iso, display, f"{day}/{mm}/{year}"

    # Pattern C: Bengali month text (e.g. ০১ জানু ১৯৮৫ or ০১ জুলাই ২০০০)
    for b_m, mm in BENGALI_MONTHS.items():
        p_bn = re.search(rf"\b(\d{{1,2}})\s+{b_m}[^\s]*\s+(\d{{4}})\b", clean_text)
        if p_bn:
            day, year = p_bn.groups()
            year = correct_ocr_century(year)
            iso = f"{year}-{mm}-{int(day):02d}"
            display = f"{int(day):02d}/{mm}/{year}"
            return iso, display, p_bn.group(0)

    return None, None, None


def extract_english_name(text: str) -> Optional[str]:
    """
    Extract the cardholder's English Name.
    Uses candidate scoring: multi-word all-caps Latin names with valid vowel syllables,
    avoiding header words and random OCR noise (such as 'ARM SPT CTSA').
    """
    if is_pure_back_side(text):
        return None

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    # 1. Direct match with 'Name:' prefix
    for line in lines:
        m = re.match(r"^(?:Name|NAME|Nam|Names|Name\.)\s*[:ঃ.\s-]+(.*)$", line, re.IGNORECASE)
        if m:
            val = clean_field_value(m.group(1))
            cleaned = re.sub(r"[^A-Za-z\s\.-]", "", val).strip()
            if len(cleaned) >= 4 and not any(h in cleaned.upper() for h in HEADER_KEYWORDS):
                return cleaned.upper()

    # 2. Candidate scoring across lines
    best_candidate = None
    best_score = -1

    for line in lines:
        cleaned = re.sub(r"^[:ঃ\-.,\s=~]+|[:ঃ\-.,\s=~]+$", "", line).strip()
        latin = re.sub(r"[^A-Za-z\s\.-]", "", cleaned).strip()
        words = [w for w in latin.split() if len(w) >= 2]

        if len(words) >= 2 and latin.isupper():
            if any(h in latin for h in HEADER_KEYWORDS):
                continue
            if re.search(r"(?:BIRTH|DATE|NID|NO|CARD|REPUBLIC|PEOPLE|GOVERNMENT)", latin):
                continue

            # Every word in a real human name must have at least one vowel
            if all(re.search(r"[AEIOUYaeiouy]", w) for w in words):
                # Score based on number of words and word lengths
                score = len(words) * 3 + sum(len(w) for w in words)
                if score > best_score:
                    best_score = score
                    best_candidate = " ".join(words)

    return best_candidate


def extract_bengali_names(text: str, name_en: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract Bengali Name, Father's Name, and Mother's Name.
    Distinguishes:
    - Cardholder's Bengali Name (e.g. মেহেদী হাসান শোভন)
    - Father's Name (e.g. এস, এম, এমদাদুল হক)
    - Mother's Name (identified by female markers e.g. মাকসুদা পারভীন)
    """
    if is_pure_back_side(text):
        return None, None, None

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    name_bn = None
    father_name = None
    mother_name = None

    # Step 1: Check for explicit labels
    for line in lines:
        if not name_bn:
            m = re.match(r"^(?:নাম|Name/নাম)\s*[:ঃ.\s-]+(.*)$", line, re.IGNORECASE)
            if m:
                val = clean_field_value(m.group(1))
                if re.search(r"[\u0980-\u09FF]", val) and len(val) >= 3:
                    name_bn = val

        if not father_name:
            m = re.match(r"^(?:পিতা|পিতার\s*নাম|Father(?:\'s\s*Name)?)\s*[:ঃ.\s-]+(.*)$", line, re.IGNORECASE)
            if m:
                val = clean_field_value(m.group(1))
                if len(val) >= 2 and not any(h in val for h in HEADER_KEYWORDS):
                    father_name = val

        if not mother_name:
            m = re.match(r"^(?:মাতা|মাতার\s*নাম|Mother(?:\'s\s*Name)?)\s*[:ঃ.\s-]+(.*)$", line, re.IGNORECASE)
            if m:
                val = clean_field_value(m.group(1))
                if len(val) >= 2 and not any(h in val for h in HEADER_KEYWORDS):
                    mother_name = val

    # Step 2: Extract clean valid Bengali name candidate lines
    valid_candidates = []
    for line in lines:
        cleaned = re.sub(r"^(?:See\s*[0-9২:]+|[:ঃ\-.,\s=~0-9\(\)\/])+", "", line).strip()
        cleaned = re.sub(r"[:ঃ\-.,\s=~0-9\(\)\/]+$", "", cleaned).strip()

        if any(h in cleaned for h in HEADER_KEYWORDS):
            continue
        # Discard lines with more than 1 digit or barcode numbers
        if len(re.findall(r"[\d০-৯]", cleaned)) > 1:
            continue

        bn_alpha = len(re.findall(r"[\u0985-\u09B9\u09CE\u09DC-\u09DF\u09BE-\u09CC]", cleaned))
        words = [w for w in re.split(r"[,.\s]+", cleaned) if len(w) >= 2]

        if bn_alpha >= 5 and len(words) >= 2:
            # Clean duplicate trailing word (e.g. 'হক হক' -> 'হক')
            cleaned_line = re.sub(r"(\b\w+\b)\s+\1$", r"\1", cleaned)
            if cleaned_line not in valid_candidates:
                valid_candidates.append(cleaned_line)

    # Step 3: Assign candidate names
    for cand in valid_candidates:
        # Check if Mother's name (contains female name marker like পারভীন, বেগম, etc.)
        if any(m in cand for m in FEMALE_MARKERS):
            if not mother_name:
                mother_name = cand
        elif not name_bn:
            name_bn = cand
        elif not father_name:
            father_name = cand

    return name_bn, father_name, mother_name


def extract_blood_group(text: str) -> Optional[str]:
    """Extract and standardize Blood Group (A+, A-, B+, B-, AB+, AB-, O+, O-)."""
    pat = r"(?:Blood\s*Group|রক্তের\s*গ্রুপ)\s*[:ঃ.\s/-]*([ABOabo]{1,2}\s*[\(]?(?:\+|\-|\+ve|\-ve|positive|negative)[\)]?)"
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        raw = m.group(1).upper()
        if "AB" in raw:
            grp = "AB"
        elif "A" in raw:
            grp = "A"
        elif "B" in raw:
            grp = "B"
        elif "O" in raw:
            grp = "O"
        else:
            grp = ""
        sign = "-" if ("-" in raw or "NEG" in raw) else "+"
        if grp:
            return f"{grp}{sign}"

    stand_alone = re.search(r"\b(A|B|AB|O)\s*(\+|\-|\(\+\)|\(\-\))\b", text)
    if stand_alone:
        grp = stand_alone.group(1).upper()
        sign = "-" if "-" in stand_alone.group(2) else "+"
        return f"{grp}{sign}"

    return None


def extract_place_of_birth(text: str) -> Optional[str]:
    """Extract Place of Birth (District name)."""
    pat = r"(?:(?:Place\s*of\s*Birth\s*/\s*জন্মস্থান|জন্মস্থান\s*/\s*Place\s*of\s*Birth|Place\s*of\s*Birth|জন্মস্থান))\s*[:ঃ.\s/-]*([A-Za-z\u0980-\u09FF\s]+)"
    m = re.search(pat, text, re.IGNORECASE)
    if m:
        val = clean_field_value(m.group(1).split("\n")[0])
        val = re.split(r"[:ঃ/]", val)[0].strip()
        if len(val) >= 2:
            return val
    return None


def extract_issue_date(text: str) -> Optional[str]:
    """Extract card issue date."""
    clean = convert_bengali_digits(text)
    pat = r"(?:প্রদানের\s*তারিখ|Date\s*of\s*Issue|Issue\s*Date)\s*[:ঃ.\s/-]*([0-9\u0980-\u09FF/-]{8,12})"
    m = re.search(pat, clean, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def extract_address(text: str) -> Optional[Dict[str, Any]]:
    """Extract raw and structured address components from back of card."""
    pat = r"(?:ঠিকানা|Address)\s*[:ঃ.\s-]*([\s\S]+?)(?=(?:রক্তের\s*গ্রুপ|Blood\s*Group|জন্মস্থান|Place\s*of\s*Birth|প্রদানের\s*তারিখ|Date\s*of\s*Issue|$))"
    m = re.search(pat, text, re.IGNORECASE)
    if not m:
        return None

    full_addr = clean_field_value(m.group(1))
    full_addr = re.sub(r"[\s,.-]+$", "", full_addr)

    holding = re.search(r"(?:বাসা/হোল্ডিং|হোল্ডিং|বাসা)\s*[:ঃ.\s-]*([^,;\n]+)", full_addr)
    village = re.search(r"(?:গ্রাম/রাস্তা|গ্রাম|রাস্তা)\s*[:ঃ.\s-]*([^,;\n]+)", full_addr)
    post_office = re.search(r"ডাকঘর\s*[:ঃ.\s-]*([^,;\n]+)", full_addr)
    upazila = re.search(r"(?:উপজেলা|থানা)\s*[:ঃ.\s-]*([^,;\n]+)", full_addr)
    district = re.search(r"জেলা\s*[:ঃ.\s-]*([^,;\n]+)", full_addr)
    postal_code = re.search(r"\b(\d{4})\b", full_addr)

    return {
        "full_address": full_addr,
        "holding": clean_field_value(holding.group(1)) if holding else None,
        "village_or_road": clean_field_value(village.group(1)) if village else None,
        "post_office": clean_field_value(post_office.group(1)) if post_office else None,
        "postal_code": postal_code.group(1) if postal_code else None,
        "upazila_or_thana": clean_field_value(upazila.group(1)) if upazila else None,
        "district": clean_field_value(district.group(1)) if district else None,
    }


def detect_card_side(data: Dict[str, Any], text: str) -> str:
    """Detect whether uploaded image is Front, Back, Both, or Unknown."""
    has_front = bool(
        data.get("nid_number")
        or data.get("name_en")
        or data.get("name_bn")
        or data.get("date_of_birth")
        or data.get("father_name")
        or data.get("mother_name")
    )
    has_back = bool(
        data.get("address")
        or data.get("blood_group")
        or data.get("place_of_birth")
        or data.get("issue_date")
        or "ঠিকানা" in text
    )

    if has_front and has_back:
        return "both"
    elif has_front:
        return "front"
    elif has_back:
        return "back"
    return "unknown"


def evaluate_confidence(data: Dict[str, Any], card_side: str) -> Tuple[float, bool, List[str], List[str]]:
    """Calculate confidence score (0.0 to 1.0) and KYC validation status."""
    score = 0.0
    missing_fields = []
    warnings = []

    if card_side in ("front", "both", "unknown"):
        if data.get("nid_number"):
            score += 0.35
        else:
            missing_fields.append("nid_number")

        if data.get("name_en"):
            score += 0.20
        else:
            missing_fields.append("name_en")

        if data.get("name_bn"):
            score += 0.20
        else:
            warnings.append("Bengali name could not be extracted.")

        if data.get("date_of_birth"):
            score += 0.15
        else:
            missing_fields.append("date_of_birth")

        if data.get("father_name"):
            score += 0.05
        if data.get("mother_name"):
            score += 0.05

        is_valid = bool(data.get("nid_number") and (data.get("name_en") or data.get("name_bn")))

    elif card_side == "back":
        if data.get("address"):
            score += 0.40
        else:
            missing_fields.append("address")

        if data.get("blood_group"):
            score += 0.30
        if data.get("place_of_birth"):
            score += 0.15
        if data.get("issue_date"):
            score += 0.15

        is_valid = bool(data.get("address") or data.get("blood_group"))
    else:
        is_valid = False

    score = min(1.0, round(score, 2))
    return score, is_valid, missing_fields, warnings


def extract_nid_data(text: str) -> Dict[str, Any]:
    """Parse OCR raw text and extract structured Bangladeshi NID data for Bank KYC."""
    nid_number, nid_type = extract_nid_number(text)
    dob_iso, dob_display, dob_raw = extract_date_of_birth(text)
    name_en = extract_english_name(text)
    name_bn, father_name, mother_name = extract_bengali_names(text, name_en)

    blood_group = extract_blood_group(text)
    place_of_birth = extract_place_of_birth(text)
    issue_date = extract_issue_date(text)
    address = extract_address(text)

    result = {
        "nid_number": nid_number,
        "nid_type": nid_type,
        "name_en": name_en,
        "name_bn": name_bn,
        "father_name": father_name,
        "mother_name": mother_name,
        "date_of_birth": dob_iso,
        "date_of_birth_display": dob_display,
        "date_of_birth_raw": dob_raw,
        "blood_group": blood_group,
        "place_of_birth": place_of_birth,
        "issue_date": issue_date,
        "address": address,
    }

    card_side = detect_card_side(result, text)
    result["card_side"] = card_side

    confidence, is_valid, missing, warnings = evaluate_confidence(result, card_side)
    result["validation"] = {
        "is_valid_nid": is_valid,
        "confidence_score": confidence,
        "missing_critical_fields": missing,
        "warnings": warnings,
    }

    return result
