"""
Bangladeshi NID field extraction.

Design rules
------------
1. Every OCR pass is parsed on its OWN, in its original line order. Passes are never
   merged into one text blob (that destroys the row sequence).
2. FRONT rows are always in this order:
       নাম  / <bn name> / Name / <EN NAME> / পিতা / <father> / মাতা / <mother> /
       Date of Birth / <date> / NID No / <number>
   so the English-name line is used as an anchor: Bengali value lines before it are
   the cardholder's name, Bengali value lines after it are father then mother.
3. BACK rows are always:
       ঠিকানা: <address, 1-3 lines>
       Blood Group / Place of Birth / Issue Date   (one single row)
       MRZ (3 lines)
4. The MRZ has ICAO check digits, so NID number, date of birth, sex and expiry read
   from it can be VERIFIED instead of guessed.
5. Final values are chosen by consensus across passes.
"""
import difflib
import re
import unicodedata
from datetime import date
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s) if s else ""


def _rx(pattern: str, flags: int = 0):
    """Compile a pattern after NFC-normalising it (Tesseract returns NFC-ish Bengali)."""
    return re.compile(_nfc(pattern), flags)


BN_DIGIT_TABLE = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# OCR letter -> digit confusions. Only ever applied to tokens already known to be numeric.
NUM_FIX_TABLE = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "i": "1", "|": "1", "!": "1", "]": "1", "[": "1",
    "Z": "2", "z": "2", "S": "5", "s": "5", "G": "6", "B": "8",
})

ENGLISH_MONTHS = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}
MONTH_WORDS = {
    "january": "01", "february": "02", "march": "03", "april": "04", "june": "06",
    "july": "07", "august": "08", "september": "09", "october": "10",
    "november": "11", "december": "12",
}
MON_ABBR = {v: k.capitalize() for k, v in ENGLISH_MONTHS.items()}

HEADER_EN = ("GOVERNMENT", "REPUBLIC", "BANGLADESH", "NATIONAL", "CARD", "PEOPLE",
             "IDENTITY", "COMMISSION", "ELECTION")
HEADER_BN_STRONG = tuple(_nfc(w) for w in ("গণপ্রজাতন্ত্রী", "পরিচয়পত্র"))
HEADER_BN_WEAK = tuple(_nfc(w) for w in ("বাংলাদেশ", "সরকার", "জাতীয়"))  # 'সরকার' is also a surname

BN_LETTER_RE = re.compile(r"[\u0985-\u09B9\u09CE\u09DC-\u09DF\u09F0\u09F1]")


def bn_letter_count(s: str) -> int:
    return len(BN_LETTER_RE.findall(s))


def latin_letter_count(s: str) -> int:
    return len(re.findall(r"[A-Za-z]", s))


def convert_bengali_digits(text: str) -> str:
    return text.translate(BN_DIGIT_TABLE) if text else ""


def fix_digits(token: str) -> str:
    """Turn an OCR'd numeric token into pure digits (O->0, l->1 ...)."""
    return re.sub(r"\D", "", (token or "").translate(BN_DIGIT_TABLE).translate(NUM_FIX_TABLE))


def tidy(value: str) -> str:
    """Trim delimiters / punctuation around a field value."""
    if not value:
        return ""
    value = re.sub(r"^[\s:;ঃ\-–—.,|/=~_]+", "", value)
    value = re.sub(r"[\s:;ঃ\-–—.,|/=~_]+$", "", value)
    return " ".join(value.split())


def consensus(cands: List[Optional[str]], total: Optional[int] = None) -> Tuple[Optional[str], float]:
    """
    Pick the candidate that agrees most with all the others (medoid by string similarity).
    Returns (value, support) where support = share of passes that agree (>=0.9 similar).
    Ties go to the earliest pass, so order passes from most to least trustworthy.
    """
    vals = [c for c in cands if c]
    total = total or len(cands) or 1
    if not vals:
        return None, 0.0
    best, best_score = None, -1.0
    for v in dict.fromkeys(vals):
        score = sum(difflib.SequenceMatcher(None, v, o).ratio() for o in vals)
        if score > best_score + 1e-9:
            best, best_score = v, score
    support = sum(1 for o in vals if difflib.SequenceMatcher(None, best, o).ratio() >= 0.9) / total
    return best, round(min(support, 1.0), 2)


# --------------------------------------------------------------------------- #
# dates
# --------------------------------------------------------------------------- #
def month_number(token: str) -> Optional[str]:
    t = re.sub(r"[^a-z]", "", (token or "").lower())
    if len(t) < 3:
        return None
    if t[:3] in ENGLISH_MONTHS:
        return ENGLISH_MONTHS[t[:3]]
    hit = difflib.get_close_matches(t, list(MONTH_WORDS) + list(ENGLISH_MONTHS), n=1, cutoff=0.6)
    if hit:
        return MONTH_WORDS.get(hit[0]) or ENGLISH_MONTHS.get(hit[0])
    return None


def fix_year(year_token: str) -> Optional[int]:
    digits = fix_digits(year_token)
    if len(digits) != 4:
        return None
    y = int(digits)
    this_year = date.today().year
    if y > this_year:                           # OCR read '1' as '2' (2085 -> 1985)
        y -= 100
    if y > this_year or y < 1890:
        return None
    return y


def build_date(day_tok: str, mon_tok: str, year_tok: str) -> Optional[Tuple[str, str]]:
    day = fix_digits(day_tok)
    mm = month_number(mon_tok)
    yy = fix_year(year_tok)
    if not day or len(day) > 2 or not mm or not yy:
        return None
    try:
        date(yy, int(mm), int(day))
    except ValueError:
        return None
    return f"{yy:04d}-{mm}-{int(day):02d}", f"{int(day):02d} {MON_ABBR[mm]} {yy}"


DATE_RE = re.compile(r"([0-9OoIl|\[\]]{1,2})\s*([A-Za-z]{3,9})\.?,?\s*([0-9OoIlSBZ]{4})")


def find_dates(line: str) -> List[Tuple[str, str, str]]:
    out = []
    for m in DATE_RE.finditer(line):
        built = build_date(*m.groups())
        if built:
            out.append((built[0], built[1], m.group(0).strip()))
    return out


# --------------------------------------------------------------------------- #
# NID number
# --------------------------------------------------------------------------- #
VALID_NID_LENGTHS = (10, 13, 17)
NID_LABEL_RE = re.compile(r"(?:N[I1l]D|\b[I1l]D)\s*(?:N[o0O]|Number)?\s*[.:ঃ\-]*", re.IGNORECASE)


def classify_nid_type(nid_number: str) -> str:
    n = len(nid_number)
    if n == 10:
        return "Smart NID (10 digits)"
    if n == 17:
        return "Old NID (17 digits)"
    if n == 13:
        return "Old NID (13 digits)"
    return f"NID ({n} digits)"


def digits_from(fragment: str) -> str:
    """Digits from a fragment, but only if the fragment really looks numeric."""
    frag = (fragment or "").translate(BN_DIGIT_TABLE)
    compact = re.sub(r"[\s\-.:ঃ]", "", frag)
    if not compact:
        return ""
    digit_like = sum(1 for c in compact if c.isdigit() or c in "OoQDIl|!SsZzBG")
    if digit_like / len(compact) < 0.8:
        return ""
    return re.sub(r"\D", "", compact.translate(NUM_FIX_TABLE))


def find_nid(lines: List[str]) -> Optional[str]:
    # 1) label ("NID No") followed by digits on the same / next line
    for i, line in enumerate(lines):
        for m in NID_LABEL_RE.finditer(line):
            cand = digits_from(line[m.end():])
            if not cand and i + 1 < len(lines):
                cand = digits_from(lines[i + 1])
            if len(cand) in VALID_NID_LENGTHS:
                return cand
    # 2) number-shaped text anywhere, bottom-up (NID is the last row on the front)
    for line in reversed(lines):
        t = convert_bengali_digits(line)
        m = re.search(r"(?<!\d)(\d{3})[\s-]?(\d{3})[\s-]?(\d{4})(?!\d)", t)
        if m and not m.group(0).startswith("01"):
            return re.sub(r"\D", "", m.group(0))
        for n in (17, 13):
            m = re.search(rf"(?<!\d)\d{{{n}}}(?!\d)", t)
            if m:
                return m.group(0)
    return None


# --------------------------------------------------------------------------- #
# FRONT side
# --------------------------------------------------------------------------- #
LABEL_RE = {
    "name_bn": _rx(r"^নাম(?=$|[\s:ঃ.\-])\s*[:ঃ.\-]*\s*(.*)$"),
    "father": _rx(r"^পিতা(?:র\s*নাম)?(?=$|[\s:ঃ.\-])\s*[:ঃ.\-]*\s*(.*)$"),
    "mother": _rx(r"^মাতা(?:র\s*নাম)?(?=$|[\s:ঃ.\-])\s*[:ঃ.\-]*\s*(.*)$"),
}
EN_NAME_LABEL_RE = re.compile(r"^(?:name|narne|nane|mame|nama|nam)\b[\s:ঃ.\-]*(.*)$", re.IGNORECASE)


def is_header(line: str) -> bool:
    up = line.upper()
    if sum(1 for w in HEADER_EN if w in up) >= 2:
        return True
    if any(w in line for w in HEADER_BN_STRONG):
        return True
    return sum(1 for w in HEADER_BN_WEAK if w in line) >= 2


def clean_bn_value(s: str) -> str:
    s = re.sub(r"[0-9০-৯A-Za-z|_=~()\[\]{}<>\"'`“”‘’*#@$%^&+\\!?;:]", " ", s or "")
    tokens = []
    for t in s.split():
        t = re.sub(r"^[.,\-–—।]+|[,\-–—।]+$", "", t)
        if not t or t == "ঃ":
            continue
        if bn_letter_count(t) == 0 and len(t) <= 2:
            continue
        tokens.append(t)
    # 'হক হক' -> 'হক' (duplicated trailing token from overlapping lines)
    if len(tokens) >= 2 and tokens[-1] == tokens[-2]:
        tokens.pop()
    return " ".join(tokens)


def is_bn_value(val: str) -> bool:
    return bn_letter_count(val) >= 3


def clean_en_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z\s.\-']", " ", s or "")
    return " ".join(s.split()).strip(" .-'").upper()


def is_en_name(s: str) -> bool:
    if len(re.sub(r"[^A-Z]", "", s)) < 3 or not re.search(r"[AEIOUY]", s):
        return False
    return not any(h in s for h in HEADER_EN) and not re.search(r"BIRTH|DATE|\bNID\b|\bNAME\b", s)


def find_english_name(lines: List[str], dob_idx: Optional[int]) -> Tuple[Optional[int], Optional[str]]:
    stop = dob_idx if dob_idx is not None else len(lines)
    # A) the row right after the "Name" label (sequence rule)
    for i, line in enumerate(lines[:stop]):
        m = EN_NAME_LABEL_RE.match(line.strip())
        if not m:
            continue
        inline = clean_en_name(m.group(1))
        if is_en_name(inline):
            return i, inline
        for j in range(i + 1, min(i + 3, stop)):
            nxt = lines[j]
            if bn_letter_count(nxt) > latin_letter_count(nxt):
                continue
            cand = clean_en_name(nxt)
            if is_en_name(cand):
                return j, cand
    # B) fallback: best all-caps, multi-word Latin line before the DOB row
    best, best_idx, best_score = None, None, -1
    for i, line in enumerate(lines[:stop]):
        if bn_letter_count(line) > 0 or is_header(line):
            continue
        cand = clean_en_name(line)
        words = cand.split()
        if len(words) >= 2 and is_en_name(cand) and all(re.search(r"[AEIOUY]", w) or len(w) <= 2 for w in words):
            score = len(words) * 3 + sum(len(w) for w in words)
            if score > best_score:
                best, best_idx, best_score = cand, i, score
    return best_idx, best


def find_dob(lines: List[str]) -> Tuple[Optional[int], Optional[Tuple[str, str, str]]]:
    """Prefer the row labelled 'Date of Birth'; otherwise the LAST date on the card
    (the hologram near the top repeats the date as a ghost image)."""
    last = None
    for i, line in enumerate(lines):
        found = find_dates(line)
        if not found:
            continue
        if re.search(r"b[i1l]rth|\bdob\b", line, re.IGNORECASE):
            return i, found[0]
        last = (i, found[0])
    if last:
        return last
    return None, None


def parse_front(text: str) -> Dict[str, Any]:
    """Parse ONE OCR pass of the front side, using the fixed row sequence."""
    lines = [l.strip() for l in _nfc(text).split("\n") if l.strip()]
    dob_idx, dob = find_dob(lines)
    en_idx, name_en = find_english_name(lines, dob_idx)
    nid = find_nid(lines)

    stop = dob_idx if dob_idx is not None else len(lines)
    entries: List[Tuple[int, Optional[str], str]] = []   # (line idx, explicit label, value)
    pending: Optional[str] = None
    for i, line in enumerate(lines[:stop]):
        if is_header(line):
            pending = None
            continue
        kind, rest = None, line
        for k, rx in LABEL_RE.items():
            m = rx.match(line)
            if m:
                kind, rest = k, m.group(1)
                break
        val = clean_bn_value(rest)
        ok = is_bn_value(val)
        if kind and not ok:
            pending = kind            # label-only row; value is on the next row
            continue
        if kind and ok:
            entries.append((i, kind, val))
            pending = None
        elif ok and i != en_idx:
            entries.append((i, pending, val))
            pending = None
        else:
            pending = None

    slots: Dict[str, Optional[Tuple[int, str]]] = {"name_bn": None, "father": None, "mother": None}
    leftovers: List[Tuple[int, str]] = []
    for idx, kind, val in entries:
        if kind and slots[kind] is None:
            slots[kind] = (idx, val)
        else:
            leftovers.append((idx, val))

    if en_idx is not None:
        before = [e for e in leftovers if e[0] < en_idx]
        after = [e for e in leftovers if e[0] > en_idx]
        if slots["name_bn"] is None and before:
            slots["name_bn"] = before[-1]            # row just above the "Name" label
        # Rows below it, in order. A SINGLE unlabeled value with both slots empty is
        # ambiguous (its own label was lost by OCR), so it is NOT guessed: guessing is what
        # copied the mother's name into the father field.
        empty = [s for s in ("father", "mother") if slots[s] is None]
        if len(empty) == 1 and after:
            slots[empty[0]] = after.pop(0)
        elif len(empty) == 2 and len(after) >= 2:
            slots["father"], slots["mother"] = after[0], after[1]
    else:
        for slot in ("name_bn", "father", "mother"):
            if slots[slot] is None and leftovers:
                slots[slot] = leftovers.pop(0)

    return {
        "nid_number": nid,
        "name_en": name_en,
        "name_bn": slots["name_bn"][1] if slots["name_bn"] else None,
        "father_name": slots["father"][1] if slots["father"] else None,
        "mother_name": slots["mother"][1] if slots["mother"] else None,
        "dob_iso": dob[0] if dob else None,
        "dob_display": dob[1] if dob else None,
        "dob_raw": dob[2] if dob else None,
    }


# --------------------------------------------------------------------------- #
# BACK side: address + (blood group | place of birth | issue date) row
# --------------------------------------------------------------------------- #
DISTRICTS = [
    "BAGERHAT", "BANDARBAN", "BARGUNA", "BARISHAL", "BARISAL", "BHOLA", "BOGURA", "BOGRA",
    "BRAHMANBARIA", "CHANDPUR", "CHATTOGRAM", "CHITTAGONG", "CHUADANGA", "COMILLA", "CUMILLA",
    "COXS BAZAR", "COX BAZAR", "DHAKA", "DINAJPUR", "FARIDPUR", "FENI", "GAIBANDHA", "GAZIPUR",
    "GOPALGANJ", "HABIGANJ", "JAMALPUR", "JASHORE", "JESSORE", "JHALOKATI", "JHALAKATHI",
    "JHENAIDAH", "JOYPURHAT", "KHAGRACHHARI", "KHAGRACHARI", "KHULNA", "KISHOREGANJ", "KURIGRAM",
    "KUSHTIA", "LAKSHMIPUR", "LAXMIPUR", "LALMONIRHAT", "MADARIPUR", "MAGURA", "MANIKGANJ",
    "MEHERPUR", "MOULVIBAZAR", "MUNSHIGANJ", "MYMENSINGH", "NAOGAON", "NARAIL", "NARAYANGANJ",
    "NARSINGDI", "NATORE", "NETROKONA", "NETRAKONA", "NILPHAMARI", "NOAKHALI", "PABNA",
    "PANCHAGARH", "PATUAKHALI", "PIROJPUR", "RAJBARI", "RAJSHAHI", "RANGAMATI", "RANGPUR",
    "SATKHIRA", "SHARIATPUR", "SHERPUR", "SIRAJGANJ", "SUNAMGANJ", "SYLHET", "TANGAIL",
    "THAKURGAON", "CHAPAINAWABGANJ", "NAWABGANJ", "NARAYANGONJ",
]

ADDR_LABEL = _rx(r"ঠিকানা\s*[:;ঃ.\-]*|Address\s*[:;.\-]*", re.IGNORECASE)
HOLDING_LABEL = _rx(r"বাসা\s*[/|।\\]?\s*হোল্ডিং\s*[:;ঃ.\-]*")
VILLAGE_LABEL = _rx(r"গ্রাম\s*[/|।\\]?\s*রাস্তা\s*[:;ঃ.\-]*")
PO_LABEL = _rx(r"ডাকঘর\s*[:;ঃ.\-]*")

BLOOD_LABEL = re.compile(r"[BG][l1I|]{1,2}[o0O]{1,2}d\s*(?:G\w{1,4})?|\bG[rn][o0O]u?p\b", re.IGNORECASE)
PLACE_LABEL = re.compile(r"(?:P.{1,4}?e\s*)?o[fF]\s*B[i1l]rth|P[l1I]a[cç]e\s*o[fF]", re.IGNORECASE)
ISSUE_LABEL = re.compile(r"[I1l|]ssue\s*(?:D[a-z]{2,4})?", re.IGNORECASE)


def is_details_line(line: str) -> bool:
    return bool(BLOOD_LABEL.search(line) or PLACE_LABEL.search(line) or ISSUE_LABEL.search(line))


def looks_like_mrz(line: str) -> bool:
    return "<<" in line or bool(re.search(r"I<?BGD", line))


def snap_district(value: str) -> str:
    up = value.upper()
    if up in DISTRICTS:
        return up
    hit = difflib.get_close_matches(up, DISTRICTS, n=1, cutoff=0.78)
    return hit[0] if hit else up


def parse_blood(rest: str) -> Optional[str]:
    rest = re.sub(r"^[\s:;._|]+", "", rest or "")
    m = re.match(r"(AB|A|B|O|0)\s*([\s\S]*)", rest, re.IGNORECASE)
    if not m:
        return None
    group = m.group(1).upper().replace("0", "O")
    tail = m.group(2).strip().lstrip("(").strip()
    if not tail:
        return None
    low, first = tail.lower(), tail[0]
    if low.startswith("pos") or first in "+*tT4F\u2020":          # thin '+' is often read as F / t / 4
        return f"{group}+"
    if low.startswith("neg") or first in "-\u2013\u2014_~":
        return f"{group}-"
    return None     # sign unreadable: better "unknown" than a silent wrong KYC value


def parse_place(rest: str) -> Optional[str]:
    val = re.sub(r"^[\s:;.\-_|]+", "", rest or "")
    val = re.sub(r"^birth\b[\s:;.\-]*", "", val, flags=re.IGNORECASE)
    val = re.sub(r"[^A-Za-z\u0980-\u09FF\s]", " ", val)
    val = " ".join(val.split())
    return snap_district(val) if len(val) >= 3 else None


def parse_details_lines(lines: List[str]) -> Dict[str, Optional[str]]:
    """Blood Group / Place of Birth / Issue Date are always ONE row, in that order."""
    rows = [l for l in lines if is_details_line(l)]
    out: Dict[str, Optional[str]] = {
        "blood_group": None, "place_of_birth": None, "issue_date": None, "issue_date_iso": None}
    if not rows:
        return out
    row = " ".join(rows)
    spans = []
    for name, rx in (("blood", BLOOD_LABEL), ("place", PLACE_LABEL), ("issue", ISSUE_LABEL)):
        m = rx.search(row)
        if m:
            spans.append((m.start(), m.end(), name))
    spans.sort()
    values = {}
    for n, (_, end, name) in enumerate(spans):
        nxt = spans[n + 1][0] if n + 1 < len(spans) else len(row)
        values[name] = row[end:nxt]
    if "blood" in values:
        out["blood_group"] = parse_blood(values["blood"])
    if "place" in values:
        out["place_of_birth"] = parse_place(values["place"])
    if "issue" in values:
        found = find_dates(values["issue"])
        if found:
            out["issue_date_iso"], out["issue_date"] = found[0][0], found[0][1]
    return out


def parse_address_text(text: str) -> Optional[Dict[str, Any]]:
    text = " ".join(re.sub(r"[|_=~]", " ", text or "").split())
    if len(text) < 6:
        return None
    found = []
    for key, rx in (("holding", HOLDING_LABEL), ("village", VILLAGE_LABEL), ("po", PO_LABEL)):
        m = rx.search(text)
        if m:
            found.append((m.start(), m.end(), key))
    found.sort()
    parts: Dict[str, str] = {}
    for n, (_, end, key) in enumerate(found):
        nxt = found[n + 1][0] if n + 1 < len(found) else len(text)
        parts[key] = text[end:nxt]

    po_name = postal = None
    tail = ""
    po_val = parts.get("po", "")
    if po_val:
        m = re.match(r"\s*(.+?)\s*[-–—]\s*([০-৯0-9]{4})\b[\s,]*(.*)$", po_val)
        if m:
            po_name, postal, tail = tidy(m.group(1)), convert_bengali_digits(m.group(2)), m.group(3)
        else:
            first, _, tail = po_val.partition(",")
            po_name = tidy(first)

    upazila = district = corp = None
    lab_up = re.search(_nfc(r"(?:উপজেলা|থানা)\s*[:;ঃ.\-]*\s*([^,]+)"), tail)
    lab_di = re.search(_nfc(r"(?<!উপ)জেলা\s*[:;ঃ.\-]*\s*([^,]+)"), tail)
    if lab_up:
        upazila = tidy(lab_up.group(1))
    if lab_di:
        district = tidy(lab_di.group(1))
    if not (upazila or district):
        segs = [tidy(s) for s in re.split(r"\s*,\s*", tail)]
        segs = [s for s in segs if s]
        if segs:
            district = segs[-1]
            mid = segs[:-1]
            corp = next((s for s in mid if re.search(_nfc(r"সিটি|কর্পোরেশন|পৌরসভা|ইউনিয়ন"), s)), None)
            upazila = next((s for s in mid if s != corp), None)

    return {
        "full_address": tidy(text),
        "holding": tidy(parts.get("holding", "")) or None,
        "village_or_road": tidy(parts.get("village", "")) or None,
        "post_office": po_name or None,
        "postal_code": postal,
        "upazila_or_thana": upazila,
        "city_corporation_or_municipality": corp,
        "district": district,
    }


def parse_address_lines(lines: List[str]) -> Optional[Dict[str, Any]]:
    start = None
    for i, l in enumerate(lines):
        if ADDR_LABEL.search(l):
            start = i
            break
    if start is None:
        for i, l in enumerate(lines):
            if HOLDING_LABEL.search(l) or VILLAGE_LABEL.search(l) or PO_LABEL.search(l):
                start = i
                break
    if start is None:
        return None
    block = []
    for l in lines[start:start + 6]:
        if is_details_line(l) or looks_like_mrz(l):
            break                                   # address ends where the details row starts
        if bn_letter_count(l) >= 2:                 # drop barcode / hologram junk rows
            block.append(l)
    if not block:
        return None
    text = ADDR_LABEL.sub("", " ".join(block), count=1)
    return parse_address_text(text)


def parse_back(text: str) -> Dict[str, Any]:
    lines = [l.strip() for l in _nfc(text).split("\n") if l.strip()]
    out = parse_details_lines(lines)
    out["address"] = parse_address_lines(lines)
    return out


# --------------------------------------------------------------------------- #
# MRZ (ICAO TD1, 3 x 30) - check digits make this the most trustworthy source
# --------------------------------------------------------------------------- #
_MRZ_DIGITLIKE = "0-9OIQDSBZG"
_MRZ_FIX = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "S": "5", "B": "8", "Z": "2", "G": "6"})


def mrz_check_digit(s: str) -> int:
    total = 0
    for i, c in enumerate(s):
        if c.isdigit():
            v = int(c)
        elif "A" <= c <= "Z":
            v = ord(c) - 55
        else:
            v = 0
        total += v * (7, 3, 1)[i % 3]
    return total % 10


def clean_mrz_line(line: str) -> str:
    s = (line or "").upper().replace(" ", "")
    s = s.translate(str.maketrans({"«": "<", "‹": "<", "(": "<", "[": "<", "{": "<"}))
    return re.sub(r"[^A-Z0-9<]", "<", s)


def _mrz_date(yymmdd: str, kind: str) -> Optional[str]:
    d = yymmdd.translate(_MRZ_FIX)
    if not d.isdigit() or len(d) != 6:
        return None
    yy, mm, dd = int(d[:2]), int(d[2:4]), int(d[4:6])
    year = 2000 + yy
    if kind == "dob" and year > date.today().year:
        year -= 100
    try:
        date(year, mm, dd)
    except ValueError:
        return None
    return f"{year:04d}-{mm:02d}-{dd:02d}"


def parse_mrz(text: str) -> Optional[Dict[str, Any]]:
    lines = [clean_mrz_line(l) for l in (text or "").split("\n")]
    lines = [l for l in lines if len(l) >= 15]
    l1 = l2 = l3 = None
    r1 = re.compile(rf"BGD([{_MRZ_DIGITLIKE}]{{9}})([<{_MRZ_DIGITLIKE}])([{_MRZ_DIGITLIKE}<]{{0,15}})")
    r2 = re.compile(rf"([{_MRZ_DIGITLIKE}]{{6}})([{_MRZ_DIGITLIKE}<])([MF<])([{_MRZ_DIGITLIKE}]{{6}})([{_MRZ_DIGITLIKE}<])BGD")
    m1 = m2 = None
    for s in lines:
        if m1 is None and (m := r1.search(s)):
            l1, m1 = s, m
        elif m2 is None and (m := r2.search(s)):
            l2, m2 = s, m
        elif l3 is None and "<" in s and len(re.findall(r"[A-Z]", s)) >= 6 and not re.search(r"\d{4}", s):
            l3 = s
    if not (m1 or m2):
        return None

    out: Dict[str, Any] = {
        "nid_number": None, "nid_check_ok": False,
        "date_of_birth": None, "dob_check_ok": False,
        "gender": None, "expiry_date": None, "expiry_check_ok": False,
        "name": None, "raw_lines": [l for l in (l1, l2, l3) if l],
    }
    if m1:
        doc9 = m1.group(1).translate(_MRZ_FIX)
        sep = m1.group(2)
        lead = m1.group(3).split("<")[0].translate(_MRZ_FIX)
        if sep == "<" and len(lead) >= 2 and lead.isdigit():
            # >9-digit document numbers overflow into the optional-data field:
            # '196390891' + '<' + '64'  ->  NID 1963908916, check digit 4
            nid, cd = doc9 + lead[:-1], int(lead[-1])
        else:
            nid, cd = doc9, (int(sep.translate(_MRZ_FIX)) if sep.translate(_MRZ_FIX).isdigit() else -1)
        if nid.isdigit():
            out["nid_number"] = nid
            out["nid_check_ok"] = mrz_check_digit(nid) == cd
    if m2:
        dob, dcd, sex, exp, ecd = m2.groups()
        out["date_of_birth"] = _mrz_date(dob, "dob")
        d6 = dob.translate(_MRZ_FIX)
        out["dob_check_ok"] = d6.isdigit() and dcd.translate(_MRZ_FIX).isdigit() and mrz_check_digit(d6) == int(dcd.translate(_MRZ_FIX))
        out["gender"] = sex if sex in "MF" else None
        out["expiry_date"] = _mrz_date(exp, "exp")
        e6 = exp.translate(_MRZ_FIX)
        out["expiry_check_ok"] = e6.isdigit() and ecd.translate(_MRZ_FIX).isdigit() and mrz_check_digit(e6) == int(ecd.translate(_MRZ_FIX))
    if l3 and "<<" in l3:
        parts = l3.strip("<").split("<<")
        toks = [t for p in parts for t in p.replace("<", " ").split() if len(t) > 1]   # 1-letter tokens = misread '<'
        out["name"] = " ".join(toks) or None
    return out


def _mrz_score(m: Dict[str, Any]) -> int:
    return (3 * m["nid_check_ok"] + 2 * m["dob_check_ok"] + m["expiry_check_ok"]
            + bool(m["nid_number"]) + bool(m["date_of_birth"]))


def best_mrz(texts: List[str]) -> Optional[Dict[str, Any]]:
    parsed = [p for p in (parse_mrz(t) for t in texts) if p]
    if not parsed:
        return None
    parsed.sort(key=_mrz_score, reverse=True)
    best = dict(parsed[0])
    for other in parsed[1:]:       # fill gaps only with check-digit-verified values
        if not best["nid_check_ok"] and other["nid_check_ok"]:
            best["nid_number"], best["nid_check_ok"] = other["nid_number"], True
        if not best["dob_check_ok"] and other["dob_check_ok"]:
            best["date_of_birth"], best["dob_check_ok"] = other["date_of_birth"], True
        if not best["expiry_check_ok"] and other["expiry_check_ok"]:
            best["expiry_date"], best["expiry_check_ok"] = other["expiry_date"], True
        for k in ("gender", "name"):
            if not best[k] and other[k]:
                best[k] = other[k]
    return best


# --------------------------------------------------------------------------- #
# result builders (vote across passes)
# --------------------------------------------------------------------------- #
def _empty_result() -> Dict[str, Any]:
    return {
        "nid_number": None, "nid_type": None, "nid_source": None,
        "name_en": None, "name_bn": None, "father_name": None, "mother_name": None,
        "date_of_birth": None, "date_of_birth_display": None, "date_of_birth_raw": None,
        "gender": None,
        "blood_group": None, "place_of_birth": None, "issue_date": None, "issue_date_iso": None,
        "address": None, "mrz": None,
    }


LABEL_WORDS = {"name_bn": _nfc("নাম"), "father": _nfc("পিতা"), "mother": _nfc("মাতা")}


def parse_row_value(text: str, kind: str) -> Optional[str]:
    """Value of ONE single-row crop (father / mother / bn-name). The row identity is known
    from its position, so a lost or garbled label cannot move the value to another field."""
    best = ""
    for line in _nfc(text or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        m = LABEL_RE[kind].match(line)
        val = clean_bn_value(m.group(1) if m else line)
        toks = val.split()
        if len(toks) >= 2 and difflib.SequenceMatcher(None, toks[0], LABEL_WORDS[kind]).ratio() >= 0.6:
            val = " ".join(toks[1:])                      # garbled label such as 'লাম' / 'পিত'
        if is_bn_value(val) and len(val) > len(best):
            best = val
    return best or None


def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def build_front_result(texts: List[str], rows: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """`texts` = OCR passes of the front, most trustworthy first.
    `rows`  = optional single-row crops {'name_bn': [...], 'father': [...], 'mother': [...]}."""
    texts = [t for t in texts if t and t.strip()]
    parsed = [parse_front(t) for t in texts]
    n = max(len(parsed), 1)
    result = _empty_result()
    agreement: Dict[str, float] = {}
    warnings: List[str] = []
    row_c = {k: [parse_row_value(t, k) for t in (rows or {}).get(k, [])] for k in ("name_bn", "father", "mother")}

    for field in ("nid_number", "name_en"):
        val, sup = consensus([p[field] for p in parsed], n)
        result[field], agreement[field] = val, sup

    for field, key, rk in (("name_bn", "name_bn", "name_bn"),
                           ("father_name", "father_name", "father"),
                           ("mother_name", "mother_name", "mother")):
        cands = row_c[rk] + [p[key] for p in parsed]          # row crops first = most trusted
        val, sup = consensus(cands, len(row_c[rk]) + n)
        result[field], agreement[field] = val, sup

    f, m = result["father_name"], result["mother_name"]
    if f and m and _similar(f, m) >= 0.85:                    # same text in both fields = wrong
        alt = [c for c in row_c["father"] + [p["father_name"] for p in parsed] if c and _similar(c, m) < 0.85]
        if alt:
            result["father_name"], agreement["father_name"] = consensus(alt, len(alt))
        else:
            result["father_name"], agreement["father_name"] = None, 0.0
            warnings.append("Father's name could not be read separately from the mother's; verify manually.")

    dob, sup = consensus([p["dob_iso"] for p in parsed], n)
    agreement["date_of_birth"] = sup
    result["date_of_birth"] = dob
    if dob:
        src = next(p for p in parsed if p["dob_iso"] == dob)
        result["date_of_birth_display"], result["date_of_birth_raw"] = src["dob_display"], src["dob_raw"]

    if result["nid_number"]:
        result["nid_type"] = classify_nid_type(result["nid_number"])
        result["nid_source"] = "front"
    result["card_side"] = "front"
    result["_agreement"] = agreement
    result["_warnings"] = warnings
    return result


def build_back_result(docs: Dict[str, List[str]]) -> Dict[str, Any]:
    """`docs` = {'address': [...], 'details': [...], 'full': [...], 'mrz': [...]} OCR passes."""
    result = _empty_result()
    agreement: Dict[str, float] = {}

    addr_docs = [t for t in docs.get("address", []) + docs.get("full", []) if t]
    detail_docs = [t for t in docs.get("details", []) + docs.get("full", []) + docs.get("address", []) if t]

    addrs = [a for a in (parse_address_lines([l.strip() for l in _nfc(t).split("\n") if l.strip()]) for t in addr_docs) if a]
    val, sup = consensus([a["full_address"] for a in addrs], max(len(addr_docs), 1))
    if val:
        result["address"] = next(a for a in addrs if a["full_address"] == val)
    agreement["address"] = sup

    details = [parse_details_lines([l.strip() for l in _nfc(t).split("\n") if l.strip()]) for t in detail_docs]
    n = max(len(details), 1)
    for key in ("blood_group", "place_of_birth", "issue_date_iso"):
        val, sup = consensus([d[key] for d in details], n)
        result[key] = val
        agreement[key] = sup
    if result["issue_date_iso"]:
        src = next(d for d in details if d["issue_date_iso"] == result["issue_date_iso"])
        result["issue_date"] = src["issue_date"]

    mrz = best_mrz(docs.get("mrz", []))
    if mrz:
        result["mrz"] = mrz
        result["gender"] = mrz["gender"]
        if mrz["nid_number"] and mrz["nid_check_ok"]:
            result["nid_number"], result["nid_source"] = mrz["nid_number"], "mrz"
            result["nid_type"] = classify_nid_type(mrz["nid_number"])
        if mrz["date_of_birth"] and mrz["dob_check_ok"]:
            result["date_of_birth"] = mrz["date_of_birth"]
    result["card_side"] = "back"
    result["_agreement"] = agreement
    return result


# --------------------------------------------------------------------------- #
# confidence + dual-side merge
# --------------------------------------------------------------------------- #
def _front_score(d, missing, warnings) -> float:
    s = 0.0
    if d.get("nid_number"):
        s += 0.35
    else:
        missing.append("nid_number")
    if d.get("name_en"):
        s += 0.20
    else:
        missing.append("name_en")
    if d.get("name_bn"):
        s += 0.20
    else:
        warnings.append("Bengali name could not be extracted.")
    if d.get("date_of_birth"):
        s += 0.15
    else:
        missing.append("date_of_birth")
    s += 0.05 * bool(d.get("father_name")) + 0.05 * bool(d.get("mother_name"))
    return s


def _back_score(d, missing, warnings) -> float:
    s = 0.0
    mrz = d.get("mrz") or {}
    if d.get("address"):
        s += 0.35
    else:
        missing.append("address")
    s += 0.15 * bool(d.get("blood_group")) + 0.15 * bool(d.get("place_of_birth")) + 0.10 * bool(d.get("issue_date"))
    if mrz.get("nid_check_ok"):
        s += 0.25
    elif d.get("mrz"):
        warnings.append("MRZ was read but its check digits did not verify.")
    if not d.get("blood_group"):
        warnings.append("Blood group missing or sign unreadable.")
    return s


def evaluate_confidence(data: Dict[str, Any], side: str, mismatch: bool = False):
    missing: List[str] = []
    warnings: List[str] = []
    if side == "front":
        score = _front_score(data, missing, warnings)
        is_valid = bool(data.get("nid_number") and (data.get("name_en") or data.get("name_bn")))
    elif side == "back":
        score = _back_score(data, missing, warnings)
        is_valid = bool(data.get("address") or (data.get("mrz") or {}).get("nid_check_ok"))
    elif side == "both":
        f, b = _front_score(data, missing, warnings), _back_score(data, missing, warnings)
        score = 0.6 * f + 0.4 * b
        is_valid = bool(data.get("nid_number") and (data.get("name_en") or data.get("name_bn"))) and not mismatch
    else:
        score, is_valid = 0.0, False
    if mismatch:
        score -= 0.25
    return min(1.0, max(0.0, round(score, 2))), is_valid, missing, warnings


def finalize(data: Dict[str, Any], side: str, mismatch: bool = False,
             extra_warnings: Optional[List[str]] = None) -> Dict[str, Any]:
    agreement = data.pop("_agreement", {}) or {}
    own_warnings = data.pop("_warnings", []) or []
    score, is_valid, missing, warnings = evaluate_confidence(data, side, mismatch)
    warnings = (extra_warnings or []) + own_warnings + warnings
    for field in ("nid_number", "date_of_birth"):
        if field in agreement and data.get(field) and agreement[field] < 0.5 and data.get("nid_source") != "mrz":
            warnings.append(f"Low agreement between OCR passes for {field}; verify manually.")
    data["card_side"] = side
    data["validation"] = {
        "is_valid_nid": is_valid,
        "confidence_score": score,
        "missing_critical_fields": missing,
        "warnings": warnings,
        "field_agreement": agreement,
        "needs_manual_review": bool(missing or mismatch or score < 0.7),
    }
    return data


def merge_dual(front: Dict[str, Any], back: Dict[str, Any]) -> Dict[str, Any]:
    """Combine front + back and cross-check them against each other (via the MRZ)."""
    f_val, b_val = front.get("validation", {}), back.get("validation", {})
    merged = dict(front)
    for key in ("address", "blood_group", "place_of_birth", "issue_date", "issue_date_iso", "mrz", "gender"):
        if back.get(key) and not merged.get(key):
            merged[key] = back[key]

    warnings = list(f_val.get("warnings", []))
    warnings += [w for w in b_val.get("warnings", []) if w not in warnings]
    mismatch = False

    if back.get("nid_source") == "mrz":
        if not merged.get("nid_number"):
            merged["nid_number"], merged["nid_source"] = back["nid_number"], "mrz"
            merged["nid_type"] = back["nid_type"]
        elif len(merged["nid_number"]) == 10 and merged["nid_number"] != back["nid_number"]:
            mismatch = True
            warnings.append("NID number on the front does not match the verified MRZ on the back. "
                            "Are these two different cards, or is the front OCR wrong?")
    mrz = back.get("mrz") or {}
    if mrz.get("dob_check_ok") and mrz.get("date_of_birth"):
        if not merged.get("date_of_birth"):
            merged["date_of_birth"] = mrz["date_of_birth"]
        elif merged["date_of_birth"] != mrz["date_of_birth"]:
            mismatch = True
            warnings.append("Date of birth on the front does not match the MRZ on the back.")
    nid = merged.get("nid_number") or ""
    if len(nid) == 17 and merged.get("date_of_birth") and nid[:4] != merged["date_of_birth"][:4]:
        warnings.append("17-digit NID does not start with the birth year; one of them may be misread.")

    merged["_agreement"] = front.get("validation", {}).get("field_agreement", {})
    return finalize(merged, "both", mismatch=mismatch, extra_warnings=warnings)