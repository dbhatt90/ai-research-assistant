# backend/tools/summarize_tool.py
from langchain_core.tools import tool
import PyPDF2
import requests
from io import BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import get_settings

settings = get_settings()

@tool
def summarize_paper(pdf_url: str, focus_area: str = "main findings") -> str:
    """
    Download and summarize a research paper from a PDF URL.
    
    This tool:
    1. Downloads PDF from arXiv or other sources
    2. Extracts text content
    3. Generates a focused summary using LLM
    
    Args:
        pdf_url: Direct URL to PDF file (e.g., arXiv PDF link)
        focus_area: What to focus on - options: "main findings", "methodology", 
                   "results", "introduction", "full summary" (default: "main findings")
    
    Returns:
        Structured summary with:
        - Title and authors (if found)
        - Key points based on focus_area
        - Main contributions
        - Limitations (if mentioned)
    
    Example:
        summarize_paper("https://arxiv.org/pdf/2301.12345.pdf", "methodology")
    """
    try:
        print(f"📥 Downloading PDF from {pdf_url[:50]}...")
        
        # Download PDF
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        
        # Read PDF
        pdf_file = BytesIO(response.content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        # Extract text from first 10 pages (to stay within token limits)
        num_pages = min(len(pdf_reader.pages), 10)
        text_content = []
        
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text_content.append(page.extract_text())
        
        full_text = "\n".join(text_content)
        
        # Limit text length (approximately 8000 tokens worth)
        if len(full_text) > 30000:
            full_text = full_text[:30000] + "\n\n[Content truncated due to length...]"
        
        print(f"✅ Extracted {len(full_text)} characters from {num_pages} pages")
        
        # Create summarization prompt based on focus area
        focus_prompts = {
            "main findings": "Focus on the main findings, results, and conclusions.",
            "methodology": "Focus on the research methodology, experimental setup, and techniques used.",
            "results": "Focus on the quantitative and qualitative results, performance metrics, and outcomes.",
            "introduction": "Focus on the problem statement, motivation, and background.",
            "full summary": "Provide a comprehensive summary covering all aspects."
        }
        
        focus_instruction = focus_prompts.get(focus_area.lower(), focus_prompts["main findings"])
        
        prompt = f"""You are analyzing a research paper. {focus_instruction}

Please provide a structured summary with:

1. **Title & Authors** (if identifiable from the text)
2. **Main Topic** (1-2 sentences)
3. **Key Points** (3-5 bullet points based on focus area)
4. **Main Contributions** (what's new/novel)
5. **Limitations** (if mentioned)

Keep the summary concise but informative (300-500 words).

---

PAPER CONTENT:
{full_text}

---

STRUCTURED SUMMARY:"""

        # Generate summary using LLM
        llm = ChatGoogleGenerativeAI(
    model=settings.llm_model,
    temperature=settings.llm_temperature,
    google_api_key=settings.google_api_key,
    convert_system_message_to_human=True  # Gemini doesn't support system messages
)

        
        summary = llm.invoke(prompt).content
        
        return f"📄 **Paper Summary** ({focus_area}):\n\n{summary}"
        
    except requests.exceptions.RequestException as e:
        return f"❌ Error downloading PDF: {str(e)}. Please check if the URL is accessible."
    except PyPDF2.errors.PdfReadError as e:
        return f"❌ Error reading PDF: {str(e)}. The PDF may be corrupted or password-protected."
    except Exception as e:
        return f"❌ Error summarizing paper: {str(e)}"


if __name__ == "__main__":
    # Test with a real arXiv paper
    test_url = "https://arxiv.org/pdf/2103.00020.pdf"  # CLIP paper
    result = summarize_paper.invoke({
        "pdf_url": test_url,
        "focus_area": "main findings"
    })
    print(result)