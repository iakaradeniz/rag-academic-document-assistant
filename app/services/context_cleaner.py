import re
import unicodedata


class ContextCleaner:
    """
    Cleans retrieved text chunks before sending them to LLM.
    Fixes common PDF extraction artifacts.
    """

    CID_PATTERN = re.compile(r"\(cid:\d+\)")

    @staticmethod
    def clean(text: str) -> str:
        if not text:
            return ""

        # 1. Remove (cid:123) artifacts
        text = ContextCleaner.CID_PATTERN.sub("", text)

        # 2. Normalize unicode (important for Turkish chars)
        text = unicodedata.normalize("NFKC", text)

        # 3. Fix excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # 4. Trim
        return text.strip()
