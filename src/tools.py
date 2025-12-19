
import os
from datetime import datetime
import time
import asyncio
import aiohttp
from urllib.parse import urlparse, urlunparse
import markdownify
import readabilipy.simple_json
from protego import Protego
import inspect
from typing import get_type_hints
from functools import wraps

_tools = {}

def tool(func):
    """Register function as agentic tool, auto-generate OpenAI schema from signature/docstring."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    doc = inspect.getdoc(func) or ""
    
    props, req = {}, []
    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD): continue
        ptype = hints.get(name, str)
        schema = {"type": {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}.get(ptype.__name__, "string")}
        for line in doc.split('\n'):
            if line.strip().startswith(f"{name}:"):
                schema["description"] = line.split(':', 1)[1].strip()
                break
        props[name] = schema
        if param.default is param.empty: req.append(name)
    
    _tools[func.__name__] = {"type": "function", "function": {
        "name": func.__name__, 
        "description": doc.split('\n')[0] if doc else f"Execute {func.__name__}",
        "parameters": {"type": "object", "properties": props, "required": req}
    }}
    return wrapper


def bool_(x):
    return True if x is True or (isinstance(x, str) and 'true' in x.lower()) else False
def int_(x, default):
    try:
        return int(x)
    except:
        return default

@tool
def get_current_time(*args, **kwargs):
    """Getting the current system time with timezone information."""
    try:
        current_time = datetime.now()
        timezone = datetime.now().astimezone().tzinfo
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        return {"status": "success", "time": formatted_time, "timezone": str(timezone), "timestamp": current_time.timestamp()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# = = = Brave Search = = = 

DEFAULT_USER_AGENT = "LMStudio/1.0 (AI Research Tool)"
last_brave_api_request = 0 # Last time an API request was made, used for enforcing rate limits

async def enforce_api_rate_limit(): # 1/sec, 15000/m
    """ Enforce API rate limiting, ensures 1 second delay between requests """
    global last_brave_api_request
    now = time.time()
    time_since_last_request = now - last_brave_api_request
    if time_since_last_request < 1.1 and last_brave_api_request > 0:
        delay = 1.1 - time_since_last_request
        await asyncio.sleep(delay)
    last_brave_api_request = time.time()

async def search_async(query, count=10, offset=0): # max 20, offset for pagination
    """ Perform a web search using Brave Search API """
    await enforce_api_rate_limit()
    api_key = os.environ.get('BRAVE_API_KEY')
    assert api_key is not None, '!! Brave API key not set !!'
    if not query:
        raise ValueError("Search query cannot be empty")
    if count < 1: count = 10
    elif count > 20: count = 20 # API limit
    
    try:
        headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
        params = {"q": query, "count": count, "offset": offset}
        
        async with aiohttp.ClientSession() as session:
            async with session.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status >= 400:
                    error_text = await response.text()
                    raise Exception(f"Brave API error: {response.status} {response.reason}\n{error_text}")
                
                data = await response.json()

                results = []
                if "web" in data and "results" in data["web"]:
                    for item in data["web"]["results"]:
                        results.append({x: item.get(x, "") for x in ['title', 'description', 'url']}) # x: item.get(x, "") or ""

                # Format results as text (similar to Node.js implementation)
                formatted_results = "\n\n".join([f"Title: {r['title']}\nDescription: {r['description']}\nURL: {r['url']}" for r in results])
                return formatted_results
    
    except asyncio.TimeoutError:
        raise Exception("Search request timed out")
    except Exception as e:
        raise Exception(f"Search failed: {str(e)}")

# @tool
def brave_search(query: str, count: int = 10, offset: int = 0, *args, **kwargs): # max 20, offset for pagination
    """Performing a web search using the Brave Search API.
    
    query: Search query (max 400 chars, 50 words)
    count: Number of results (min 1, max 20, default 10)
    offset: Pagination offset (max 9, default 0)
    """
    try:
        formatted_results = asyncio.run(search_async(query=query, count=int_(count, 10), offset=int_(offset, 0)))
        return {"status": "success", "content": formatted_results}
    except Exception as e:
        # return {"status": "error", "message": str(e)}
        error_message = str(e)
        print(f"Brave search error: {error_message}")
        return {"status": "error", "message": f"Search failed: {error_message}"}

# = = = Fetch URL = = = 

def extract_content_from_html(html):
    """Extract and convert HTML content to Markdown format """
    try:
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=False)
        if not ret["content"]:
            return "<e>Page failed to be simplified from HTML</e>"
        content = markdownify.markdownify(ret["content"], heading_style="atx")
        return content
    except Exception as e:
        return f"<e>Failed to process HTML: {str(e)}</e>"

async def check_robots_txt(url, user_agent):
    """Check if the URL can be fetched according to robots.txt """
    parsed = urlparse(url)
    robots_url = urlunparse((parsed.scheme, parsed.netloc, "/robots.txt", "", "", ""))
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(robots_url, headers={"User-Agent": user_agent}) as response:
                if response.status in (401, 403):
                    return {"allowed": False, "message": f"When fetching robots.txt ({robots_url}), received {response.status}, assuming fetching is not allowed"}
                elif 400 <= response.status < 500:
                    return {"allowed": True, "message": "No robots.txt or client error, assuming allowed"}
                robot_txt = await response.text()
        processed_robot_txt = "\n".join(line for line in robot_txt.splitlines() if not line.strip().startswith("#"))
        robot_parser = Protego.parse(processed_robot_txt)
        if not robot_parser.can_fetch(str(url), user_agent):
            return {"allowed": False, "message": f"The site's robots.txt specifies that fetching this page is not allowed for user agent: {user_agent}"}
        return {"allowed": True, "message": "Robots.txt allows fetching"}
    except Exception as e:
        return {"allowed": True, "message": f"Failed to check robots.txt: {str(e)}, assuming allowed"}

async def fetch_content(url, user_agent, force_raw=False):
    """Fetch content from URL """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"User-Agent": user_agent}, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status >= 400:
                    return "", f"Failed to fetch {url} - status code {response.status}", response.status
                
                page_raw = await response.text()
                content_type = response.headers.get("content-type", "")
                is_page_html = "<html" in page_raw[:100] or "text/html" in content_type or not content_type
                if is_page_html and not force_raw:
                    return extract_content_from_html(page_raw), "", response.status
                
                return (page_raw, f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n", response.status)
    except Exception as e:
        return "", f"Failed to fetch {url}: {str(e)}", 500

async def fetch_url_content_async(url, max_length=32000, start_index=0, raw=False, check_robots=True):
    """Fetch URL content and return as markdown or raw text """
    try:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        if not bool(urlparse(url).netloc): # not a valid network location
            return {"status": "error", "message": f"Invalid URL format: {url}"}
        if check_robots:
            robots_check = await check_robots_txt(url, DEFAULT_USER_AGENT)
            if not robots_check["allowed"]:
                return {"status": "error", "message": robots_check["message"]}
        
        content, prefix, status_code = await fetch_content(url, DEFAULT_USER_AGENT, raw)
        
        if status_code >= 400:
            return {"status": "error", "message": prefix}
        
        # pagination
        original_length = len(content)
        if start_index >= original_length:
            return {"status": "error", "message": "No more content available"}
        # truncated_content = content[start_index:start_index + max_length]
        truncated_content = str(content)
        if not truncated_content:
            return {"status": "error", "message": "No content available"}
        # Add pagination info if needed
        result = truncated_content
        actual_content_length = len(truncated_content)
        remaining_content = original_length - (start_index + actual_content_length)
        pagination_info = ""
        if actual_content_length == max_length and remaining_content > 0:
            next_start = start_index + actual_content_length
            pagination_info = f"\n\nContent truncated. {remaining_content} characters remaining. Call with start_index={next_start} to get more."
        
        return {
            "status": "success", 
            "url": url,
            "content": prefix + result + pagination_info,
            "total_length": original_length,
            "current_position": start_index,
            "remaining": remaining_content,
            "content_type": "markdown" if not raw and prefix == "" else "raw"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@tool
def fetch_url_content(url: str, max_length: int = 100000, start_index: int = 0, raw: bool = False, check_robots: bool = False, *args, **kwargs):
    """Fetching content from a URL and converting to markdown when possible.
    
    url: The URL to fetch content from
    max_length: Maximum number of characters to return (default: 100000)
    start_index: Starting index for pagination (default: 0)
    raw: Return raw content instead of processing HTML (default: false)
    check_robots: Check robots.txt before fetching (default: false)
    """
    try:
        return asyncio.run(fetch_url_content_async(url=url, max_length=int_(max_length, 100000), start_index=int_(start_index, 0), raw=bool_(raw), check_robots=bool_(check_robots)))
    except Exception as e:
        return {"status": "error", "message": str(e)}

toolset = list(_tools.values())

if __name__ == "__main__":
    for x in toolset: print(x)
