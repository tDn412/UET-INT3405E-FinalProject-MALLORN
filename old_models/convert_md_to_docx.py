import markdown
from docx import Document
from htmldocx import HtmlToDocx
import os

def convert_md_to_docx(md_file, docx_file):
    if not os.path.exists(md_file):
        print(f"Error: {md_file} not found.")
        return

    # Read Markdown
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert to HTML
    html = markdown.markdown(md_content, extensions=['extra'])

    # Create Docx
    doc = Document()
    new_parser = HtmlToDocx()
    new_parser.add_html_to_document(html, doc)
    
    doc.save(docx_file)
    print(f"Successfully converted {md_file} to {docx_file}")

if __name__ == "__main__":
    convert_md_to_docx('bao-cao-mallorn-final.md', 'bao-cao-mallorn-final.docx')
