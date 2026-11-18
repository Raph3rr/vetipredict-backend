# scraper.py
import re
import requests
from bs4 import BeautifulSoup
from email_validator import validate_email, EmailNotValidError
from PyPDF2 import PdfReader
import io
import magic   # optional, for mime-type detection
from typing import List, Tuple

# robust email regex (simple, used for initial extraction)
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", re.I)

def extract_emails_from_text(text: str) -> List[str]:
    """Extract candidate emails and validate them."""
    if not text:
        return []
    candidates = set(re.findall(EMAIL_REGEX, text))
    valid = set()
    for c in candidates:
        try:
            v = validate_email(c)
            valid.add(v.email)
        except EmailNotValidError:
            continue
    return sorted(valid)

def scrape_url_text(url: str, timeout=10) -> Tuple[str, str]:
    """Fetch a URL and return (final_url, text content)"""
    headers = {"User-Agent": "EmailScraper/1.0 (+https://example.com)"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # remove scripts/styles
    for s in soup(["script", "style", "noscript"]):
        s.decompose()

    text = soup.get_text(separator="\n", strip=True)
    final_url = r.url
    return final_url, text

def extract_from_pdf_bytes(b: bytes) -> str:
    """Extract text from PDF bytes using PyPDF2 (simple approach)."""
    try:
        reader = PdfReader(io.BytesIO(b))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                continue
        return "\n".join(pages)
    except Exception:
        return ""

def parse_uploaded_file(file_storage) -> Tuple[str, str]:
    """
    Determine mime type and extract text. Returns (filename, text)
    Supports text, pdf, docx(optional).
    """
    content = file_storage.read()
    # reset file pointer for any later uses
    try:
        file_storage.seek(0)
    except Exception:
        pass

    mime = None
    try:
        mime = magic.from_buffer(content, mime=True)
    except Exception:
        pass

    filename = getattr(file_storage, 'filename', 'uploaded')

    if mime and 'pdf' in mime:
        text = extract_from_pdf_bytes(content)
    else:
        # fallback: decode as utf-8 - may fail for binary docs
        try:
            text = content.decode('utf-8', errors='ignore')
        except Exception:
            text = ""
    return filename, text
