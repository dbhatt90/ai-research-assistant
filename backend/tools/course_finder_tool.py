# backend/tools/course_finder_tool.py
from langchain_core.tools import tool
from ddgs import DDGS 


@tool
def find_learning_resources(topic: str, resource_type: str = "courses") -> str:
    """
    Find curated learning resources on a topic.
    
    Args:
        topic: Subject to learn about (e.g., "machine learning", "LangGraph")
        resource_type: Type of resource - "courses", "tutorials", "books", "videos"
    
    Returns:
        Curated list of high-quality learning resources with links
    """
    try:
        # Craft specific search queries for quality resources
        search_queries = {
            "courses": f"{topic} online course Coursera edX Udacity",
            "tutorials": f"{topic} tutorial documentation getting started",
            "books": f"best {topic} books textbook O'Reilly Manning",
            "videos": f"{topic} video lecture tutorial YouTube"
        }
        
        query = search_queries.get(resource_type, search_queries["courses"])
        
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=8))
        
        if not results:
            return f"No {resource_type} found for '{topic}'"
        
        # Filter and rank by quality indicators
        quality_domains = [
            'coursera.org', 'edx.org', 'udacity.com', 'mit.edu', 'stanford.edu',
            'youtube.com', 'github.com', 'medium.com', 'towardsdatascience.com',
            'kaggle.com', 'fast.ai', 'deeplearning.ai'
        ]
        
        # Prioritize results from quality domains
        quality_results = [r for r in results 
                          if any(domain in r.get('href', '') for domain in quality_domains)]
        other_results = [r for r in results 
                        if r not in quality_results]
        
        ranked_results = quality_results[:5] + other_results[:3]
        
        output = [f"🎓 **Learning Resources: {topic}** ({resource_type})\n"]
        
        for i, result in enumerate(ranked_results[:6], 1):
            title = result.get('title', 'No title')
            url = result.get('href', 'No URL')
            snippet = result.get('body', 'No description')[:150]
            
            # Add quality badge
            badge = "⭐ " if any(d in url for d in quality_domains) else ""
            
            output.append(f"{i}. {badge}**{title}**")
            output.append(f"   🔗 {url}")
            output.append(f"   📝 {snippet}...\n")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"❌ Error finding resources: {str(e)}"


if __name__ == "__main__":
    result = find_learning_resources.invoke({
        "topic": "LangGraph agent development",
        "resource_type": "courses"
    })
    print(result)