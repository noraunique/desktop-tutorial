#!/usr/bin/env python3
"""
Extract specific pages from PDF using pdfplumber for better text quality.
"""

import pdfplumber
import sys
import os

def extract_pages_pdfplumber(pdf_path, start_page, end_page, output_file):
    """Extract specific pages from PDF using pdfplumber."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_content = []
            
            for page_num in range(start_page - 1, min(end_page, len(pdf.pages))):
                page = pdf.pages[page_num]
                page_text = page.extract_text()
                
                if page_text:
                    # Add page header
                    text_content.append(f"\n--- Page {page_num + 1} ---\n")
                    text_content.append(page_text)
                    text_content.append("\n")
            
            # Write to output file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(''.join(text_content))
            
            print(f"Extracted pages {start_page}-{end_page} from {pdf_path}")
            print(f"Output saved to: {output_file}")
            print(f"Total characters: {len(''.join(text_content))}")
            
    except Exception as e:
        print(f"Error extracting pages: {e}")
        return False
    
    return True

def extract_full_pdf(pdf_path, output_file):
    """Extract all text from PDF using pdfplumber."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text_content = []
            
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                
                if page_text:
                    # Add page header
                    text_content.append(f"\n--- Page {page_num + 1} ---\n")
                    text_content.append(page_text)
                    text_content.append("\n")
            
            # Write to output file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(''.join(text_content))
            
            print(f"Extracted all pages from {pdf_path}")
            print(f"Output saved to: {output_file}")
            print(f"Total pages: {len(pdf.pages)}")
            print(f"Total characters: {len(''.join(text_content))}")
            
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  Extract specific pages: python extract_pages.py <pdf_file> <start_page> <end_page> [output_file]")
        print("  Extract all pages: python extract_pages.py <pdf_file> all [output_file]")
        print("\nExamples:")
        print('  python extract_pages.py "Chapter 4.pdf" 13 14 system_maintenance.txt')
        print('  python extract_pages.py "Chapter 4.pdf" all full_chapter.txt')
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    
    if not os.path.exists(pdf_file):
        print(f"Error: PDF file '{pdf_file}' not found")
        sys.exit(1)
    
    if sys.argv[2].lower() == "all":
        # Extract all pages
        output_file = sys.argv[3] if len(sys.argv) > 3 else f"{os.path.splitext(pdf_file)[0]}_full.txt"
        extract_full_pdf(pdf_file, output_file)
    else:
        # Extract specific pages
        try:
            start_page = int(sys.argv[2])
            end_page = int(sys.argv[3])
            output_file = sys.argv[4] if len(sys.argv) > 4 else f"{os.path.splitext(pdf_file)[0]}_pages_{start_page}-{end_page}.txt"
            
            extract_pages_pdfplumber(pdf_file, start_page, end_page, output_file)
            
        except ValueError:
            print("Error: Start and end page must be integers")
            sys.exit(1)
