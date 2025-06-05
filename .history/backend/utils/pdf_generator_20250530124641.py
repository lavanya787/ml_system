from fpdf import FPDF
import os

def generate_pdf_report(file_id, extracted, summary=None, keywords=None, entities=None, full_text='', output_folder='analysis'):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Document Summary Report", ln=True, align='C')
    pdf.ln(10)

    if summary:
        pdf.set_font("Arial", size=10, style='B')
        pdf.cell(0, 10, "Summary:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 10, summary)
        pdf.ln(5)

    if keywords:
        pdf.set_font("Arial", size=10, style='B')
        pdf.cell(0, 10, "Keywords:", ln=True)
        pdf.set_font("Arial", size=10)
        pdf.multi_cell(0, 10, ', '.join(keywords))
        pdf.ln(5)

    if entities:
        pdf.set_font("Arial", size=10, style='B')
        pdf.cell(0, 10, "Named Entities:", ln=True)
        pdf.set_font("Arial", size=10)
        for ent in entities:
            pdf.cell(0, 10, f"{ent['text']} ({ent['label']})", ln=True)
        pdf.ln(5)

    pdf.set_font("Arial", size=10, style='B')
    pdf.cell(0, 10, "Extracted Content:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 10, full_text)

    for style, content in extracted:
        lines = content.split('\n')
        for line in lines:
            pdf.multi_cell(0, 10, txt=line.strip())

    pdf_path = os.path.join(output_folder, f"{file_id}_summary.pdf")
    pdf.output(pdf_path)
    return pdf_path

def generate_pdf_summary(data, output_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Project Analysis Summary", ln=True, align="C")
    pdf.ln(10)

    for heading, content in data.items():
        pdf.set_font("Arial", "B", 14)
        pdf.multi_cell(0, 10, heading)
        pdf.set_font("Arial", "", 12)
        pdf.multi_cell(0, 10, content)
        pdf.ln(5)

    pdf.output(output_path)
