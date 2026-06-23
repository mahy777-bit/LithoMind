import json
import os
import requests
from semanticscholar import SemanticScholar

PAPERS_FILE = "scripts/papers.json"

def search_crossref(title: str) -> str | None:
    """
    Fallback: search CrossRef by title, return a usable link if found.
    """
    try:
        response = requests.get(
            "https://api.crossref.org/works",
            params={"query.bibliographic": title, "rows": 1},
            timeout=15
        )
        response.raise_for_status()
        items = response.json().get("message", {}).get("items", [])
        if not items:
            return None

        item = items[0]
        # CrossRef sometimes provides a direct link list
        links = item.get("link", [])
        for link in links:
            if link.get("content-type") == "application/pdf":
                return link.get("URL")

        # Fallback to the DOI resolver URL if no direct PDF link exists
        doi = item.get("DOI")
        if doi:
            return f"https://doi.org/{doi}"

        return None
    except Exception as e:
        print(f"  CrossRef search failed: {e}")
        return None

def fix_missing_urls():
    with open(PAPERS_FILE, "r") as f:
        papers = json.load(f)

    sch = SemanticScholar()
    fixed = 0
    removed = 0
    updated_papers = []

    for paper in papers:
        if paper.get("url"):
            updated_papers.append(paper)
            continue

        print(f"Missing URL: {paper['title'][:60]}...")

        try:
            results = sch.search_paper(
                paper["title"],
                limit=1,
                fields=["openAccessPdf", "externalIds"]
            )
            arxiv_id = None
            open_access_url = None

            if results:
                result = results[0]
                arxiv_id = result.externalIds.get("ArXiv") if result.externalIds else None
                if result.openAccessPdf and result.openAccessPdf.get("url"):
                    open_access_url = result.openAccessPdf.get("url")

            if arxiv_id:
                paper["url"] = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                paper["id"] = arxiv_id
                paper["source"] = "arxiv"
                fixed += 1
                print(f"  ✓ Fixed via arXiv: {paper['url']}")

            elif open_access_url:
                paper["url"] = open_access_url
                fixed += 1
                print(f"  ✓ Fixed via Semantic Scholar open access: {paper['url']}")

            else:
                crossref_url = search_crossref(paper["title"])
                if crossref_url:
                    paper["url"] = crossref_url
                    fixed += 1
                    print(f"  ✓ Fixed via CrossRef: {paper['url']}")
                else:
                    print(f"  ✗ No accessible URL found via any source — removing")
                    removed += 1
                    continue

        except Exception as e:
            print(f"  ✗ Error: {e} — removing")
            removed += 1
            continue

        updated_papers.append(paper)

    with open(PAPERS_FILE, "w") as f:
        json.dump(updated_papers, f, indent=2)

    print(f"\nDone — {fixed} fixed, {removed} removed")
    print(f"Total papers remaining: {len(updated_papers)}")

if __name__ == "__main__":
    fix_missing_urls()