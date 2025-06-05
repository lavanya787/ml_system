import pandas as pd
import fitz  # PyMuPDF
from docx import Document

def parse_file(filepath):
    ext = filepath.split('.')[-1].lower()

    if ext in ['csv', 'xlsx']:
        df = pd.read_csv(filepath) if ext == 'csv' else pd.read_excel(filepath)
        return df, 'structured'

    elif ext == 'txt':
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read(), 'text'

    elif ext == 'pdf':
        text = ""
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text()
        return text, 'text'

    elif ext == 'docx':
        doc = Document(filepath)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text, 'text'

    else:
        raise ValueError("Unsupported file type")
