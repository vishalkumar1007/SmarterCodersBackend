import requests
from bs4 import BeautifulSoup

def fetch_html_content(url: str) -> str:
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch URL: {url}")
    return response.text

def parse_html_content(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove unnecessary tags
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    return soup.get_text(separator=" ", strip=True)
