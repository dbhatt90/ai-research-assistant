# backend/tools/web_search_tool.py
from langchain_core.tools import tool
from ddgs import DDGS  # ← Changed import
from typing import Optional

@tool
def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo for current information and context.
    
    This tool performs a web search to find relevant articles, tutorials,
    courses, and general information. Useful for finding recent content
    that may not be in academic papers.
    
    Args:
        query: Search query (e.g., "best data science courses", "Python tutorials")
        max_results: Maximum number of results (default: 5, max: 10)
    
    Returns:
        Formatted string with search results including:
        - Title
        - URL
        - Description/snippet
    
    Example:
        search_web("machine learning online courses", max_results=5)
    """
    try:
        # Limit max_results
        max_results = min(max_results, 10)
        
        # Perform search
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=max_results,
                region='wt-wt',  # Worldwide
                safesearch='moderate'
            ))
        
        if not results:
            return f"No web results found for query: '{query}'. Try rephrasing or using different keywords."
        
        # Format output
        output = [f"Found {len(results)} web results for '{query}':\n"]
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('href', 'No URL')
            body = result.get('body', 'No description')
            
            output.append(f"\n{i}. **{title}**")
            output.append(f"   🔗 {url}")
            output.append(f"   📝 {body[:200]}{'...' if len(body) > 200 else ''}")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Error searching web: {str(e)}. Please try again."


# Test function
if __name__ == "__main__":
    result = search_web.invoke({"query": "data science courses", "max_results": 3})
    print(result)