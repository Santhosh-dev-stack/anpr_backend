import re

# Indian plate format: 2-letter state code, 1-2 digit RTO code, 0-3 letter
# series, 3-4 digit number (e.g. TN01BA7320, TN01AB1234, KA05MH1234). PaddleOCR
# on a small, often blurry crop regularly returns near-miss garbage (missing
# leading letters, digit/letter confusion, truncated reads) — those are
# rejected here rather than shown as if they were a confident, correct read.
_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{3,4}$")
_MIN_LENGTH = 8
_MAX_LENGTH = 11

# The textbook 4-part Indian plate shape: 2-letter state code, exactly 2
# RTO digits, 1-2 series letters, exactly 4 number digits (e.g. TN97A6500,
# KA05MH1234). Stricter than _PLATE_PATTERN above on purpose — this is an
# additional, informational check (see is_standard_format), not a
# replacement: _PLATE_PATTERN's wider ranges (1-2 RTO digits, 0-3 series
# letters, 3-4 number digits) stay the actual accept/reject gate, since
# real plates legitimately fall outside this exact shape (older 3-digit
# numbers, series-less older plates, single-digit RTO codes on some
# states) and normalize_plate's OCR-confusion correction already targets
# that wider range deliberately.
_STANDARD_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}$")

# All current Indian state/UT RTO codes, plus BH (the unified "Bharat" series).
# Used to correct a single misread letter in the state-code prefix (e.g. OCR
# confusing visually similar letters like N/H) — a real plate's first two
# letters must be one of these, so a near-miss with exactly one of these as
# its unique closest match is overwhelmingly likely the intended code rather
# than a coincidence.
_VALID_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UP", "UA", "WB", "BH",
}

# Letter pairs PaddleOCR commonly confuses at small plate-crop resolution,
# because they're visually similar in the font Indian plates use (e.g. N and
# H both being tall verticals joined by a stroke). Used to break ties when
# more than one real state code is a single-letter edit away from a misread
# prefix — plain edit distance alone can't tell 'TH' -> 'TN' (a real, common
# confusion) apart from 'TH' -> 'TR' or 'TH' -> 'TS' (not realistic misreads),
# so without this every one of those looks equally "close" and the correction
# would have to give up as ambiguous.
_COMMON_CONFUSIONS = {
    frozenset({"N", "H"}), frozenset({"O", "Q"}), frozenset({"I", "L"}),
    frozenset({"B", "R"}), frozenset({"D", "O"}), frozenset({"E", "F"}),
    frozenset({"G", "C"}), frozenset({"M", "N"}), frozenset({"V", "Y"}),
    # 'I' misread for 'T' (e.g. 'IN' -> 'TN') -- a bold vertical stroke with
    # a top serif reads as either letter at low plate-crop resolution.
    frozenset({"I", "T"}),
    frozenset({"U", "V"}), frozenset({"P", "R"}), frozenset({"W", "V"}),
}


# Digit <-> letter look-alikes PaddleOCR commonly swaps in the *series*
# segment (the 0-3 letters between the RTO digits and the final number,
# e.g. the "B" in TN09B1234) — this segment is letters, but a digit shape
# that looks like one often gets read instead (e.g. '8' for 'B').
# Inverted, the same map corrects the opposite mistake (a look-alike letter
# read where a digit segment was expected).
_DIGIT_TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "6": "G"}
_LETTER_TO_DIGIT = {letter: digit for digit, letter in _DIGIT_TO_LETTER.items()}


def is_valid_plate(text: str) -> bool:
    if not (_MIN_LENGTH <= len(text) <= _MAX_LENGTH):
        return False
    if not _PLATE_PATTERN.fullmatch(text):
        return False
    return text[:2] in _VALID_STATE_CODES


def is_standard_format(text: str) -> bool:
    """Checks a plate string (expected to already be a normalize_plate()
    output) against the textbook 4-part shape — state code, 2 RTO digits,
    1-2 series letters, 4 number digits — rather than the wider range
    is_valid_plate() actually accepts. Purely informational: use this to
    flag/log whether a final accepted reading happens to match the classic
    format, not to reject readings that don't (see module comment above).
    """
    return bool(_STANDARD_PLATE_PATTERN.fullmatch(text)) and text[:2] in _VALID_STATE_CODES


def _differing_letter_pair(a: str, b: str) -> frozenset | None:
    diffs = [frozenset({x, y}) for x, y in zip(a, b) if x != y]
    return diffs[0] if len(diffs) == 1 else None


def _correct_state_code(text: str) -> str:
    """If text's first two letters aren't a real state code, but exactly one
    known code is both a single letter away *and* that letter swap is a
    known OCR confusion, use it. Falls back to plain single-letter distance
    only when that alone is unique; otherwise leaves text unchanged (rather
    than guess between multiple equally-plausible corrections).
    """
    prefix, rest = text[:2], text[2:]
    if prefix in _VALID_STATE_CODES:
        return text
    # A digit in the prefix (e.g. 'N0...') is a digit/letter misread, not a
    # state-code letter typo -- that's _fix_digit_letter_confusions' job via
    # its own explicit digit->letter map. Leaving it alone here avoids the
    # unique-closest-code fallback below "correcting" it against a code that
    # just happens to be the only one differing by one position, dropping a
    # real digit in the process (e.g. 'N0981234' -> 'NL981234').
    if not prefix.isalpha():
        return text

    close_matches = [
        code for code in _VALID_STATE_CODES if _differing_letter_pair(code, prefix) is not None
    ]
    plausible_matches = [
        code for code in close_matches if _differing_letter_pair(code, prefix) in _COMMON_CONFUSIONS
    ]
    if len(plausible_matches) == 1:
        return plausible_matches[0] + rest
    if len(close_matches) == 1:
        return close_matches[0] + rest
    return text


def _fix_digit_letter_confusions(text: str) -> str | None:
    """Indian plates alternate letter/digit segments (2 state letters, 1-2
    RTO digits, 0-3 series letters, 3-4 number digits) whose exact split
    varies per plate, so this tries every valid segment-length combination
    that adds up to len(text); for each, substitutes a digit for its letter
    look-alike in segments expecting letters (and vice versa for digit
    segments) via _DIGIT_TO_LETTER/_LETTER_TO_DIGIT, and returns the first
    combination that produces a fully consistent, valid plate. Returns None
    if no segmentation works.

    rto_len=2 is tried before rto_len=1: a 2-digit RTO code is the norm on
    Indian plates, a 1-digit one is comparatively rare. Without this order,
    a misread like 'TN0SOB4398' (should be RTO '05', 'S' misread for '5')
    parses just fine taken literally as RTO '0' + series 'SOB' -- a real,
    syntactically valid plate shape, so nothing downstream would ever
    suspect it's wrong. Preferring the 2-digit interpretation whenever a
    look-alike letter (O/I/S/B/Z/G) sits right where a second RTO digit
    would go catches this without needing anything beyond the format itself.
    """
    n = len(text)
    for rto_len in (2, 1):
        for series_len in (0, 1, 2, 3):
            for number_len in (3, 4):
                if 2 + rto_len + series_len + number_len != n:
                    continue
                chars = list(text)
                ok = True

                def _fix_segment(start: int, length: int, expect_letter: bool) -> None:
                    nonlocal ok
                    for i in range(start, start + length):
                        ch = chars[i]
                        if expect_letter:
                            if ch.isdigit():
                                if ch in _DIGIT_TO_LETTER:
                                    chars[i] = _DIGIT_TO_LETTER[ch]
                                else:
                                    ok = False
                            elif not ch.isalpha():
                                ok = False
                        else:
                            if ch.isalpha():
                                if ch in _LETTER_TO_DIGIT:
                                    chars[i] = _LETTER_TO_DIGIT[ch]
                                else:
                                    ok = False
                            elif not ch.isdigit():
                                ok = False

                _fix_segment(0, 2, expect_letter=True)
                _fix_segment(2, rto_len, expect_letter=False)
                _fix_segment(2 + rto_len, series_len, expect_letter=True)
                _fix_segment(2 + rto_len + series_len, number_len, expect_letter=False)

                if ok:
                    candidate = "".join(chars)
                    corrected = _correct_state_code(candidate)
                    if is_valid_plate(corrected):
                        return corrected
                    if is_valid_plate(candidate):
                        return candidate
    return None


def normalize_plate(text: str) -> str | None:
    """Returns a valid plate reading for `text`, correcting common OCR
    artifacts, or None if it can't be made valid.

    Two corrections are tried, independently and combined:
    - A plate crop's top edge sometimes catches a sliver of a sticker/screw/
      state-emblem text above the actual plate, which PaddleOCR concatenates
      onto the real reading as one spurious leading character (observed e.g.
      'STN09CS1812' for an actual 'TN09CS1812').
    - The state-code prefix is corrected against the real list of Indian
      state/UT codes when a single-letter OCR confusion (e.g. 'TH' misread
      for 'TN') is the only plausible explanation (see _correct_state_code).

    Each candidate still has to pass the strict format check, so a
    correction is only applied when it actually produces something
    plate-shaped — this doesn't loosen what counts as a valid plate.

    _fix_digit_letter_confusions is tried first, not as a last resort: it
    already re-runs the state-code correction internally, and — critically —
    prefers a standard 2-digit RTO code over accepting a reading literally
    even when that literal reading already happens to be syntactically
    valid (e.g. 'TN0SOB4398', a plate-shaped string on its own, but almost
    certainly RTO '05' with '5' misread as 'S'). Checking plain validity
    first would accept that literal reading and never reconsider it.
    """
    candidates = [text]
    if len(text) > _MIN_LENGTH:
        candidates.append(text[1:])

    for candidate in candidates:
        fixed = _fix_digit_letter_confusions(candidate)
        if fixed is not None:
            return fixed

    for candidate in candidates:
        corrected = _correct_state_code(candidate)
        if is_valid_plate(corrected):
            return corrected
        if is_valid_plate(candidate):
            return candidate
    return None
