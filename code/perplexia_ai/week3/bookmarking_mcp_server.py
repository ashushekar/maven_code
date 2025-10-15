import json
import os
from typing import List, Dict, Any
from fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("Bookmarking")

# Path to store bookmarks - created in the same directory as the server
BOOKMARKS_FILE = os.path.join(os.path.dirname(__file__), "bookmarks.json")

def _load_bookmarks() -> List[Dict[str, Any]]:
    """Load bookmarks from JSON file"""
    if not os.path.exists(BOOKMARKS_FILE):
        return []
    
    try:
        with open(BOOKMARKS_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

def _save_bookmarks(bookmarks: List[Dict[str, Any]]) -> None:
    """Save bookmarks to JSON file"""
    try:
        with open(BOOKMARKS_FILE, 'w') as f:
            json.dump(bookmarks, f, indent=2)
    except IOError as e:
        raise Exception(f"Failed to save bookmarks: {e}")

@mcp.tool()
def add_bookmark(urls: List[str]) -> str:
    """
    Add a list of URLs to bookmarks
    
    Args:
        urls: List of URLs to bookmark
    
    Returns:
        Success message with number of bookmarks added
    """
    if not urls:
        return "No URLs provided to bookmark"
    
    # Load existing bookmarks
    bookmarks = _load_bookmarks()
    
    # Get current timestamp
    from datetime import datetime
    timestamp = datetime.now().isoformat()
    
    # Add new bookmarks
    existing_urls = {bookmark['url'] for bookmark in bookmarks}
    new_bookmarks_count = 0
    
    for url in urls:
        if url not in existing_urls:
            bookmark = {
                'url': url,
                'created_at': timestamp
            }
            bookmarks.append(bookmark)
            existing_urls.add(url)
            new_bookmarks_count += 1
    
    # Save updated bookmarks
    _save_bookmarks(bookmarks)
    
    return f"Successfully added {new_bookmarks_count} new bookmark(s). {len(urls) - new_bookmarks_count} duplicate(s) skipped."

@mcp.tool()
def get_bookmarks() -> List[Dict[str, Any]]:
    """
    Retrieve all bookmarks
    
    Returns:
        List of all bookmark dictionaries
    """
    bookmarks = _load_bookmarks()
    return bookmarks

@mcp.tool()
def remove_bookmark(url: str) -> str:
    """
    Remove a bookmark by URL
    
    Args:
        url: URL of the bookmark to remove
    
    Returns:
        Success or failure message
    """
    bookmarks = _load_bookmarks()
    
    # Find and remove the bookmark
    original_count = len(bookmarks)
    bookmarks = [bookmark for bookmark in bookmarks if bookmark['url'] != url]
    
    if len(bookmarks) < original_count:
        _save_bookmarks(bookmarks)
        return f"Successfully removed bookmark for: {url}"
    else:
        return f"Bookmark not found for URL: {url}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
