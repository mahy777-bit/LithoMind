import json
import os
import sys
from semanticscholar import SemanticScholar

PAPERS_FILE = "scripts/papers.json"

def fix_missing_urls():
    with open(PAPERS_FILE, "r") as f:
        papers = json.load(f)

    sch = SemanticScholar()
    fixed = 0
    removed = 0
    updated_papers = []

    for paper in papers:
        # Already has URL — keep as is
        if paper.get("url"):
            updated_papers.append(paper)
            continue

        print(f"Missing URL: {paper['title'][:60]}...")

        # Try to find URL via Semantic Scholar using title
        try:
            results = sch.search_paper(
                paper["title"],
                limit=1,
                fields=["openAccessPdf", "externalIds"]
            )
            if results:
                result = results[0]
                arxiv_id = result.externalIds.get("ArXiv") if result.externalIds else None

                if arxiv_id:
                    paper["url"] = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                    paper["id"] = arxiv_id
                    paper["source"] = "arxiv"
                    fixed += 1
                    print(f"  ✓ Fixed via arXiv: {paper['url']}")
                elif result.openAccessPdf:
                    paper["url"] = result.openAccessPdf.get("url")
                    fixed += 1
                    print(f"  ✓ Fixed via open access: {paper['url']}")
                else:
                    print(f"  ✗ No accessible PDF found — removing")
                    removed += 1
                    continue
            else:
                print(f"  ✗ Not found — removing")
                removed += 1
                continue
        except Exception as e:
            print(f"  ✗ Error: {e} — removing")
            removed += 1
            continue

        updated_papers.append(paper)

    # Save updated papers.json
    with open(PAPERS_FILE, "w") as f:
        json.dump(updated_papers, f, indent=2)

    print(f"\nDone — {fixed} fixed, {removed} removed")
    print(f"Total papers remaining: {len(updated_papers)}")

if __name__ == "__main__":
    fix_missing_urls()