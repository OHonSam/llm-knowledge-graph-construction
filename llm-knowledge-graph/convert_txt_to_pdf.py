from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from argparse import ArgumentParser
import os

def convert_txt_to_pdf(txt_file_path: str, pdf_file_path: str = None, 
                       encoding: str = 'utf-8', page_size=A4):
    """
    Convert a text file to PDF with proper formatting
    
    Args:
        txt_file_path: Path to input .txt file
        pdf_file_path: Path for output .pdf file (optional)
        encoding: Text file encoding (default: utf-8)
        page_size: PDF page size (default: A4)
    """
    
    # Default output path if not provided
    if pdf_file_path is None:
        pdf_file_path = txt_file_path.replace('.txt', '.pdf')
    
    try:
        # Read the text file
        with open(txt_file_path, 'r', encoding=encoding) as file:
            content = file.read()
        
        # Create PDF document
        doc = SimpleDocTemplate(
            pdf_file_path,
            pagesize=page_size,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Get styles
        styles = getSampleStyleSheet()

        font_path = r"C:\Windows\Fonts\arial.ttf"
        pdfmetrics.registerFont(TTFont('UnicodeFont', font_path))
        font_name = "UnicodeFont"
        
        # Create custom style for body text
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            fontName=font_name,
            leading=14
        )
        
        # Split content into paragraphs
        paragraphs = content.split('\n\n')
        story = []
        
        for para in paragraphs:
            if para.strip():  # Skip empty paragraphs
                # Replace line breaks within paragraphs
                para = para.replace('\n', '<br/>')
                story.append(Paragraph(para, body_style))
                story.append(Spacer(1, 6))
        
        # Build PDF
        doc.build(story)
        print(f"✅ Successfully converted {txt_file_path} to {pdf_file_path}")
        return pdf_file_path
        
    except Exception as e:
        print(f"❌ Error converting {txt_file_path}: {e}")
        return None

def main():
    parser = ArgumentParser()
    parser.add_argument("--txt_path", type=str, required=True)
    parser.add_argument("--pdf_path", type=str)
    args = parser.parse_args()

    convert_txt_to_pdf(txt_file_path=args.txt_path, pdf_file_path=args.pdf_path)

if __name__ == "__main__":
    main()

# python .\llm-knowledge-graph\convert_txt_to_pdf.py --txt_path .\llm-knowledge-graph\data\custom_pdfs\Thuong_Han_Luan.txt