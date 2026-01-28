REPLACEMENTS = {
    "Ĝ": "İ",
    "ĝ": "i",
    "Ĝl": "İl",
    "bĜlgĜ": "bilgi",
    "ver Ĝ": "veri",
    "Ĝş": "İş",
    "Ĝç": "İç",
    "Ĝn": "İn",
}

def fix_turkish_encoding(text: str) -> str:
    for wrong, correct in REPLACEMENTS.items():
        text = text.replace(wrong, correct)
    return text
