import io
import re
 
import pdfplumber
 
# ---------------------------------------------------------------------------
# 1. Test-name normalisation (canonical id -> aliases seen on Indian lab reports)
#    Canonical ids match the document ids in rag.py's knowledge base.
# ---------------------------------------------------------------------------
ALIASES = {
    "hemoglobin": ["hemoglobin", "haemoglobin", "hb", "hgb"],
    "rbc": ["rbc", "rbc count", "total rbc count", "red blood cell count", "red blood cells",
            "erythrocyte count", "red cell count"],
    "wbc": ["wbc", "wbc count", "total wbc count", "white blood cell count", "total leucocyte count",
            "total leukocyte count", "tlc", "leucocyte count", "leukocyte count", "total count"],
    "neutrophils": ["neutrophils", "neutrophil", "polymorphs", "neutrophil count"],
    "lymphocytes": ["lymphocytes", "lymphocyte"],
    "platelets": ["platelet count", "platelets", "platelet", "plt"],
    "hematocrit": ["hematocrit", "haematocrit", "pcv", "packed cell volume", "hct"],
    "mcv": ["mcv", "mean corpuscular volume", "mean cell volume"],
    "esr": ["esr", "erythrocyte sedimentation rate"],
    "crp": ["crp", "c reactive protein", "hs crp"],
    "glucose_fasting": ["fasting blood sugar", "fbs", "fasting blood glucose", "glucose fasting",
                        "fasting glucose", "fasting plasma glucose", "fpg", "blood sugar fasting",
                        "plasma glucose fasting"],
    "glucose_pp": ["post prandial blood sugar", "postprandial blood sugar", "ppbs", "pp blood sugar",
                   "blood sugar pp", "glucose pp", "post prandial glucose", "plasma glucose pp"],
    "glucose_random": ["random blood sugar", "rbs", "random glucose", "blood sugar random",
                       "glucose random"],
    "hba1c": ["hba1c", "glycated hemoglobin", "glycosylated hemoglobin", "glycated haemoglobin",
              "glycosylated haemoglobin", "a1c"],
    "total_cholesterol": ["total cholesterol", "cholesterol total", "serum cholesterol", "cholesterol"],
    "ldl": ["ldl", "ldl cholesterol", "ldl c", "low density lipoprotein"],
    "hdl": ["hdl", "hdl cholesterol", "hdl c", "high density lipoprotein"],
    "vldl": ["vldl", "vldl cholesterol"],
    "triglycerides": ["triglycerides", "triglyceride", "serum triglycerides", "tg"],
    "creatinine": ["creatinine", "serum creatinine", "s creatinine"],
    "urea": ["urea", "blood urea", "serum urea", "bun", "blood urea nitrogen"],
    "uric_acid": ["uric acid", "serum uric acid"],
    "alt": ["alt", "sgpt", "alanine aminotransferase"],
    "ast": ["ast", "sgot", "aspartate aminotransferase"],
    "alp": ["alp", "alkaline phosphatase"],
    "bilirubin": ["bilirubin", "total bilirubin", "bilirubin total", "serum bilirubin",
                  "direct bilirubin", "bilirubin direct", "indirect bilirubin"],
    "albumin": ["albumin", "serum albumin"],
    "tsh": ["tsh", "thyroid stimulating hormone", "ultrasensitive tsh", "us tsh"],
    "t3": ["t3", "total t3", "free t3", "ft3", "triiodothyronine"],
    "t4": ["t4", "total t4", "free t4", "ft4", "thyroxine"],
    "vitamin_d": ["vitamin d", "vit d", "vitamin d3", "25 oh vitamin d", "25 hydroxy vitamin d",
                  "vitamin d 25 oh", "vitamin d total"],
    "vitamin_b12": ["vitamin b12", "vit b12", "b12", "cobalamin", "cyanocobalamin"],
    "ferritin": ["ferritin", "serum ferritin"],
    "iron": ["iron", "serum iron"],
    "sodium": ["sodium", "serum sodium"],
    "potassium": ["potassium", "serum potassium"],
    "calcium": ["calcium", "serum calcium", "total calcium"],
}
def _clean(text: str) -> str:
    """lowercase, punctuation -> space, collapse spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()
 
 
# longest alias first, so "hdl cholesterol" wins over "cholesterol"
_ALIAS_LIST = sorted(
    ((_clean(a), canon) for canon, aliases in ALIASES.items() for a in aliases),
    key=lambda pair: len(pair[0]),
    reverse=True,
)
 
 
def normalize_test_name(name: str):
    """Return the canonical id for a test name, or None if unknown."""
    cleaned = _clean(name)
    for alias, canon in _ALIAS_LIST:
        if re.search(rf"\b{re.escape(alias)}\b", cleaned):
            return canon
    return None
 
 
# ---------------------------------------------------------------------------
# 2. PDF -> text
# ---------------------------------------------------------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a text-based PDF (scanned/image PDFs return '')."""
    pages = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# 3. Text -> structured results
# ---------------------------------------------------------------------------
NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
 
HEAD_RE = re.compile(
    rf"^(?P<name>[A-Za-z][A-Za-z0-9 ()/,.'\-]*?)\s*:?\s+"
    rf"(?P<value>{NUM})(?P<flag>[HhLl])?(?=\s|$)(?P<rest>.*)$"
)
RANGE_RE = re.compile(rf"(?P<low>{NUM})\s*(?:-|–|—|to)\s*(?P<high>{NUM})")
LT_RE = re.compile(rf"(?:<=?|≤|up\s*to|upto|less\s+than|below)\s*(?P<v>{NUM})", re.I)
GT_RE = re.compile(rf"(?:>=?|≥|more\s+than|greater\s+than|above)\s*(?P<v>{NUM})", re.I)
 
FLAG_TOKENS = {"h", "l", "high", "low", "*", "(h)", "(l)", "↑", "↓", "normal", "abnormal"}
SKIP_WORDS = {"age", "sex", "gender", "patient", "name", "date", "page", "ref", "reference",
              "sample", "report", "id", "uhid", "mobile", "phone", "collected", "received",
              "lab", "dr", "doctor", "time", "method", "specimen", "test", "unit", "result",
              "investigation", "biological", "normal", "registered", "printed", "bill"}
 
 
def _num(x):
    if x is None or x == "":
        return None
    try:
        return float(str(x).replace(",", ""))
    except ValueError:
        return None
 
 
def format_range(low, high) -> str:
    if low is not None and high is not None:
        return f"{low:g} - {high:g}"
    if high is not None:
        return f"< {high:g}"
    if low is not None:
        return f"> {low:g}"
    return "not given"
 
 
def make_result(name, value, unit="", low=None, high=None, raw_line=""):
    """Build one result dict (used by the parser and by the agent's Gemini fallback)."""
    return {
        "name": str(name).strip(),
        "canonical": normalize_test_name(str(name)),
        "value": float(value),
        "unit": (unit or "").strip(),
        "low": low,
        "high": high,
        "range_text": format_range(low, high),
        "status": "UNCHECKED",
        "deviation_pct": None,
        "raw_line": raw_line,
    }
 
 
def _parse_line(line: str):
    m = HEAD_RE.match(line.strip())
    if not m:
        return None
    name = m.group("name").strip(" -:.,")
    first_word = _clean(name).split(" ")[0] if _clean(name) else ""
    if not name or first_word in SKIP_WORDS:
        return None
 
    value = _num(m.group("value"))
    rest = m.group("rest")
 
    low = high = None
    unit_part = rest
    rm = RANGE_RE.search(rest)
    if rm:
        low, high = _num(rm.group("low")), _num(rm.group("high"))
        unit_part = rest[: rm.start()]
    else:
        lt, gt = LT_RE.search(rest), GT_RE.search(rest)
        if lt:
            high, unit_part = _num(lt.group("v")), rest[: lt.start()]
        elif gt:
            low, unit_part = _num(gt.group("v")), rest[: gt.start()]
 
    unit_tokens = [t for t in unit_part.split() if t.lower() not in FLAG_TOKENS]
    unit = unit_tokens[0] if unit_tokens else ""
 
    result = make_result(name, value, unit, low, high, raw_line=line.strip())
    # keep a line only if it has a range OR is a test we recognise
    if low is None and high is None and result["canonical"] is None:
        return None
    return result
 
 
def parse_report(text: str):
    """Parse every 'Test  Value  Unit  Range' line in the report text."""
    results, seen = [], set()
    for line in text.splitlines():
        r = _parse_line(line)
        if r and _clean(r["name"]) not in seen:
            seen.add(_clean(r["name"]))
            results.append(r)
    return results
 
 
# ---------------------------------------------------------------------------
# 4. Range-checking tool (the agent calls this - no LLM guessing on numbers)
# ---------------------------------------------------------------------------
def check_ranges(results):
    """Mark each result LOW / HIGH / NORMAL / NO_RANGE using the report's own range."""
    checked = []
    for r in results:
        r = dict(r)
        v, low, high = r["value"], r["low"], r["high"]
        if low is None and high is None:
            r["status"] = "NO_RANGE"
        elif low is not None and v < low:
            r["status"] = "LOW"
            r["deviation_pct"] = round((v - low) / low * 100, 1) if low else None
        elif high is not None and v > high:
            r["status"] = "HIGH"
            r["deviation_pct"] = round((v - high) / high * 100, 1) if high else None
        else:
            r["status"] = "NORMAL"
        checked.append(r)
    return checked
 
 
def summary_text(results) -> str:
    """Plain, deterministic one-paragraph summary of the checked results."""
    if not results:
        return "No test values could be read from this report."
    out = [r for r in results if r["status"] in ("LOW", "HIGH")]
    normal = [r for r in results if r["status"] == "NORMAL"]
    no_range = [r for r in results if r["status"] == "NO_RANGE"]
    s = f"{len(results)} tests found: {len(normal)} within the report's reference range, {len(out)} outside it"
    s += f", {len(no_range)} without a printed range." if no_range else "."
    if out:
        s += " Outside range: " + ", ".join(f"{r['name']} ({r['status'].lower()})" for r in out) + "."
    return s
 
 
def results_to_text(results) -> str:
    """Compact table for LLM prompts."""
    return "\n".join(
        f"- {r['name']}: {r['value']:g} {r['unit']} (report range {r['range_text']}) -> {r['status']}"
        for r in results
    )
 
