import sys
import os

pdf_path = sys.argv[1]
txt_path = sys.argv[2]
if not os.path.exists(pdf_path):
    print(f"File not found: {pdf_path}")
    sys.exit(1)

text = ""
try:
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    for i in range(min(5, len(doc))): 
        text += doc[i].get_text()
except Exception as e:
    print(f"PyMuPDF failed: {e}")
    try:
        import PyPDF2
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for i in range(min(5, len(reader.pages))):
                text += reader.pages[i].extract_text()
    except Exception as e2:
        print(f"PyPDF2 failed: {e2}")

with open(txt_path, 'w', encoding='utf-8') as f:
    f.write(text)
print(f"Extraction complete to {txt_path}")
