
import os
import pandas as pd
from langchain.agents import initialize_agent,AgentType
from langchain.tools import Tool
from langchain_google_genai.llms import ChatGoogleGenerativeAI
from agents import initialize_llm
from typing import List
import re
from langchain.tools import StructuredTool
from typing import Dict
#For PDF Generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, 
    Image, Table, TableStyle, Frame, KeepInFrame
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import requests
from typing import List, Dict, Optional
import re
import json
from PIL import Image as PILImage

#-----------------------------------------------------------------------------------------------------------------
# For DataScout with Excel Generatio
# Data Extraction Tools
def extract_num_rows_from_prompt(user_prompt: str) -> int:
    match = re.search(r'(\d+)\s+(rows|records)', user_prompt, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def extract_columns_from_prompt(user_prompt: str) -> List[str]:
    match = re.search(r'(field names|column names|fields|columns|field_names|column_names)[\s:]*([a-zA-Z0-9_,\s\.]*)',
                      user_prompt, re.IGNORECASE)
    if match:
        raw_columns = match.group(2).split(',')
    else:
        return []

    formatted_columns = [
        re.sub(r'[^a-zA-Z0-9]', '_', col.strip()).lower()
        for col in raw_columns
    ]

    formatted_columns = list(filter(bool, formatted_columns))
    return list(dict.fromkeys(formatted_columns))


# Synthetic Excel Data Generator
def generate_data_from_text(text_sample: str, column_names: List[str], num_rows: int = 10, chunk_size: int = 50) -> str:
    llm = initialize_llm()
    
    sysp = "You are a data generator that produces only specified formatted data with no extra text or code fences."
    
    generated_rows = []
    rows_generated = 0
    column_names_str = ", ".join(column_names)

    while rows_generated < num_rows:
        rows_to_generate = min(chunk_size, num_rows - rows_generated)

        if rows_generated == 0:
            prompt = (
                f"{sysp}\n\n"
                f"Based on the following description:\n'{text_sample}'\n"
                f"Generate {rows_to_generate} rows of synthetic data with the following columns:\n"
                f"Columns: {column_names_str}\n"
                "Ensure that all columns are present and the data is realistic, varied, and maintains logical relationships. "
                "Format the data as tilde-separated values ('~') without including column names or any extra text."
                "Return ONLY the raw data with no additional commentary or formatting."
            )
        else:
            reference_data = "\n".join(["~".join(row) for row in generated_rows[-5:]])
            prompt = (
                f"{sysp}\n\n"
                f"Based on the following description:\n'{text_sample}'\n"
                f"Generate {rows_to_generate} rows of synthetic data with the following columns:\n"
                f"Columns: {column_names_str}\n"
                f"Follow the format of these recently generated rows:\n{reference_data}\n"
                "Ensure that all columns are present and the data is realistic, varied, and maintains logical relationships. "
                "Format the data as tilde-separated values ('~') without including column names or any extra text."
                "Return ONLY the raw data with no additional commentary or formatting."
            )

        response = llm.invoke(prompt)
        
        # Extract content from AIMessage object
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Clean and split the response
        rows = []
        for line in content.split('\n'):
            line = line.strip()
            if line and '~' in line:
                rows.append(line.split('~'))

        generated_rows.extend(rows[:num_rows - rows_generated])
        rows_generated = len(generated_rows)

    df = pd.DataFrame(generated_rows, columns=column_names)
    print(df.head())
    print(f"Generated {len(generated_rows)} rows of data.")
    file_path = "data_output.xlsx"
    df.to_excel(file_path, index=False)
    return file_path

#----------------------------------------------------------------------------------------------------------------
#For DataScout with pdf Generation
# Data Extraction Tools for PDF
# Function to extract number of pages from the user promp
def extract_num_pages_for_pdf_prompt(user_prompt: str) -> int:
    match = re.search(r'(\d+)\s+(pages|page)', user_prompt, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

# Function to extract sections from the user prompt
def extract_sections_from_prompt(user_prompt: str) -> List[str]:
    match = re.search(
        r'(section names|sections|topics|headings|subsections|chapters|parts|content areas|section name|section|topic|heading|subsection|chapter|part|content area)[\s:]*([a-zA-Z0-9_,\s\.]*)',
        user_prompt, 
        re.IGNORECASE
    )
    if match:
        raw_sections = match.group(2).split(',')
    else:
        return []

    formatted_sections = [
        section.strip().title() for section in raw_sections
    ]

    formatted_sections = list(filter(bool, formatted_sections))
    return list(dict.fromkeys(formatted_sections))

# Function to generate PDF data from text
# This function generates a PDF document with specified sections and content
def generate_pdf_from_text(text_sample: str, sections: List[str], num_pages: int = 1) -> str:
    llm = initialize_llm()
    file_path = "smart_document.pdf"
    
    # Step 1: Get document structure from LLM analysis
    try:
        analysis_prompt = f"""Analyze this document request and return JSON:
        {{
            "title": "Document title",
            "style": "professional/academic",
            "sections": [
                {{
                    "name": "Section name",
                    "content_type": "text/mixed",
                    "needs_visuals": true/false
                }}
            ]
        }}
        Request: Create a {num_pages}-page document about {text_sample} with sections {sections}"""
        
        analysis = llm.invoke(analysis_prompt)
        structure = json.loads(analysis.content)
    except Exception as e:
        print(f"Analysis failed, using defaults: {e}")
        structure = {
            "title": "Generated Document",
            "style": "professional",
            "sections": [{"name": s, "content_type": "text"} for s in sections]
        }

    # Step 2: Configure document with proper styles
    styles = getSampleStyleSheet()
    
    # Modify existing styles instead of redefining
    styles['Title'].fontSize = 18
    styles['Title'].leading = 22
    styles['Title'].alignment = 1  # Center
    
    # Add new styles with unique names
    styles.add(ParagraphStyle(
        name='SectionHeader',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12
    ))
    
    styles.add(ParagraphStyle(
        name='BodyTextEnhanced',
        parent=styles['BodyText'],
        spaceAfter=8,
        leading=14
    ))

    # Create document with margins
    doc = SimpleDocTemplate(
        file_path,
        pagesize=letter,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=1*inch,
        bottomMargin=1*inch
    )
    
    elements = []
    
    # Cover Page
    elements.append(Spacer(1, 3*inch))
    elements.append(Paragraph(structure["title"], styles['Title']))
    elements.append(Spacer(1, 2*inch))

    elements.append(PageBreak())

    # Content Sections
    for section in structure["sections"]:
        # Section Header
        elements.append(Paragraph(section["name"], styles['SectionHeader']))
        elements.append(Spacer(1, 0.25*inch))
        
        # Generate Content
        content_prompt = f"""Generate professional content for section: {section["name"]}
        About: {text_sample}
        Format: Several well-structured paragraphs
        Length: About {int(500/len(sections))} words
        Tone: Professional
        """
        
        response = llm.invoke(content_prompt)
        content = response.content
        
        # Add Content Paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        for para in paragraphs[:3]:  # Limit to 3 paragraphs
            elements.append(Paragraph(para, styles['BodyTextEnhanced']))
            elements.append(Spacer(1, 0.15*inch))
        
        # Add Page Break between sections
        elements.append(PageBreak())
    
    # Build PDF
    doc.build(elements)
    print(f"Successfully generated PDF at: {file_path}")
    return file_path

#----------------------------------------------------------------------------------------------------------------
# Tool Wrapping for LangChain For PDF Generation
def pdf_generator_tool(prompt: str, sections: List[str], number_of_pages: int) -> str:
    if not sections:
        sections = ["Introduction", "Content", "Conclusion"]
    if not number_of_pages or number_of_pages <= 0:
        number_of_pages = 1
    return generate_pdf_from_text(prompt, sections, number_of_pages)

def extract_sections_tool(prompt: str) -> List[str]:
    return extract_sections_from_prompt(prompt)

def extract_num_pages_tool(prompt: str) -> int:
    return extract_num_pages_for_pdf_prompt(prompt)


#-----------------------------------------------------------------------------------------------------------------
# Tool Wrapping for LangChain For Excel Generation
def excel_generator_tool(prompt: str, columns: List[str], number_of_rows: int) -> str:
    if not columns:
        raise ValueError("Column names cannot be empty.")
    if not number_of_rows or number_of_rows <= 0:
        raise ValueError("Number of rows must be a positive integer.")
    return generate_data_from_text(prompt, columns, number_of_rows)

def extract_columns_tool(prompt: str) -> List[str]:
    return extract_columns_from_prompt(prompt)

def extract_num_rows_tool(prompt: str) -> int:
    return extract_num_rows_from_prompt(prompt)

#-----------------------------------------------------------------------------------------------------------------
# Agent Setup
def DataScout_agent():
    llm = initialize_llm()
    tools = [
        Tool(
            name="ExtractRowCount",
            func=extract_num_rows_tool,
            description="Extracts number of rows/records from the user's prompt."
        ),
        Tool(
            name="ExtractColumnNames",
            func=extract_columns_tool,
            description="Extracts field/column names from the user's prompt."
        ),
        StructuredTool.from_function(
            func=excel_generator_tool,
            name="GenerateExcelFromPrompt",
            description="Generates Excel data. Args: prompt (str), columns (List[str]), number_of_rows (int)",
            return_direct=True
        )
    ]
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent

#-----------------------------------------------------------------------------------------------------------------
# Agent Setup for PDF Generation
def DataScout_agent_with_pdf():
    llm = initialize_llm()
    tools = [
        Tool(
            name="ExtractPageCount",
            func=extract_num_pages_tool,
            description="Extracts number of pages from the user's prompt."
        ),
        Tool(
            name="ExtractSectionNames",
            func=extract_sections_tool,
            description="Extracts section names from the user's prompt."
        ),
        StructuredTool.from_function(
            func=pdf_generator_tool,
            name="GeneratePDFFromPrompt",
            description="Generates PDF document. Args: prompt (str), sections (List[str]), number_of_pages (int)",
            return_direct=True
        )
    ]
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    return agent


# #Testing the pipeline
# if __name__ == "__main__":
#     test_prompt = "Generate 25 records with field names: Name, Age, Email, Purchase Date, Amount"
#     try:
#         agent = DataScout_agent()
#         response = agent.invoke(test_prompt)
#         print(f"Response: {response}")
#     except Exception as e:
#         print(f"❌ Error: {e}")
if __name__ == "__main__":
    test_prompt = "Create a pdf  with 3 pages about Artificial Intelligence with sections: Introduction, Methodology, Conclusion in 500 words per each page"
    try:
        agent = DataScout_agent_with_pdf()
        response = agent.invoke(test_prompt)
        print(f"Response: {response}")
    except Exception as e:
        print(f"❌ Error: {e}")
