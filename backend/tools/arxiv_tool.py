# backend/tools/arxiv_tool.py
from langchain_core.tools import tool
import arxiv

@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Search arXiv for academic research papers.
    
    This tool searches the arXiv repository for papers matching the query.
    It returns paper titles, authors, summaries, and PDF links.
    
    Args:
        query: Search query (e.g., "machine learning", "AI agents", "transformers")
        max_results: Maximum number of papers to return (default: 5, max: 10)
    
    Returns:
        Formatted string with paper information including:
        - Title
        - Authors
        - Publication date
        - Summary (first 300 chars)
        - PDF URL
        - arXiv URL
    
    Example:
        search_arxiv("reinforcement learning", max_results=3)
    """
    try:
        # Limit max_results to prevent abuse
        max_results = min(max_results, 10)
        
        # Create arXiv client and search
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        results = []
        for paper in client.results(search):
            # Extract author names (limit to first 3)
            authors = [author.name for author in paper.authors[:3]]
            if len(paper.authors) > 3:
                authors.append(f"... +{len(paper.authors) - 3} more")
            
            results.append({
                "title": paper.title.strip(),
                "authors": ", ".join(authors),
                "published": paper.published.strftime("%Y-%m-%d"),
                "summary": paper.summary.replace("\n", " ").strip()[:300] + "...",
                "pdf_url": paper.pdf_url,
                "arxiv_url": paper.entry_id,
                "categories": ", ".join(paper.categories[:3])
            })
        
        if not results:
            return f"No papers found for query: '{query}'. Try different keywords or broader terms."
        
        # Format output
        output = [f"Found {len(results)} research papers for '{query}':\n"]
        
        for i, paper in enumerate(results, 1):
            output.append(f"\n{i}. **{paper['title']}**")
            output.append(f"   📅 Published: {paper['published']}")
            output.append(f"   👥 Authors: {paper['authors']}")
            output.append(f"   🏷️  Categories: {paper['categories']}")
            output.append(f"   📄 Summary: {paper['summary']}")
            output.append(f"   🔗 PDF: {paper['pdf_url']}")
            output.append(f"   🔗 arXiv: {paper['arxiv_url']}")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error searching arXiv: {str(e)}. Please try again with a different query."


# Test function (optional - for development)
if __name__ == "__main__":
    # Test the tool
    result = search_arxiv.invoke({"query": "AI agents", "max_results": 2})
    print(result)