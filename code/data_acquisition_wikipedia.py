import requests
import json
from datetime import datetime
from typing import List

def fetch_wikipedia_articles(topics: List[str], lang: str = "en", out_path: str = "ethiopian_history.json"):
    """
    Fetches Wikipedia articles using the MediaWiki API and saves them with metadata.
    Args:
        topics (List[str]): List of Wikipedia article titles to fetch.
        lang (str): Language code (e.g., 'en', 'am').
        out_path (str): Output JSON file path.
    """
    session = requests.Session()
    url = f"https://{lang}.wikipedia.org/w/api.php"
    results = []
    for topic in topics:
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts|pageprops|info",
            "explaintext": True,
            "titles": topic,
            "redirects": 1,
            "inprop": "url",
        }
        resp = session.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if "extract" in page and page.get("extract"):
                results.append({
                    "id": str(page.get("pageid", topic.replace(" ", "_").lower())),
                    "title": page.get("title", topic),
                    "text": page["extract"],
                    "source": f"https://{lang}.wikipedia.org/?curid={page.get('pageid', '')}",
                    "language": lang,
                    "retrieved_at": datetime.now().isoformat(),
                    "wikidata_id": page.get("pageprops", {}).get("wikibase_item"),
                    "fullurl": page.get("fullurl"),
                })
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} articles to {out_path}")

if __name__ == "__main__":
    topics = [
        "History of Ethiopia",
        "Menelik II",
        "Battle of Adwa",
        "Haile Selassie",
        "Lucy (Australopithecus)",
        "Ethiopian Empire",
        "Derg",
        "Ethiopian Orthodox Tewahedo Church",
        "Coffee",
        "Kebra Nagast",
        "Tewodros II",
        "Amharic language",
        "Ethiopian calendar",
        "Ethiopian cuisine",
        "Ethiopian Jews"
        "Solomonic dynasty"
    ]
    fetch_wikipedia_articles(topics, lang="en", out_path="../data/ethiopian_history_en.json")

    titles_am = [
        "የኢትዮጵያ ታሪክ",  # History of Ethiopia
        "መንለክ ሁለተኛ",    # Menelik II
        "የአድዋ ጦርነት",    # Battle of Adwa
        "ኃይለ ሥላሴ",      # Haile Selassie
    ]
    fetch_wikipedia_articles(titles_am, lang="am", out_path="../data/ethiopian_history_am.json")
