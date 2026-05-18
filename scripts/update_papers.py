import arxiv
import json
import os
from semanticscholar import SemanticScholar
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))
from app.config import PAPER_SOURCES

# ── Configuration ──────────────────────────────────────────

PAPERS_FILE = "scripts/papers.json"

SEARCH_QUERIES = [
    "optical proximity correction OPC",
    "inverse lithography technology ILT",
    "EUV lithography stochastics",
    "computational lithography simulation",
    "machine learning lithography hotspot",
    "semiconductor defect inspection SEM",
    "source mask optimization SMO",
    "design for manufacturability DFM",
    "multi-patterning lithography",
    "mask error enhancement factor MEEF",
    "process variation control CD",
    "deep learning optical proximity correction",
    "lithography process window",
    "extreme ultraviolet mask defect",
]

CATEGORY_KEYWORDS = {
    "OPC": ["optical proximity correction", "opc", "model-based opc"],
    "ILT": ["inverse lithography", "ilt", "source mask optimization", "smo"],
    "EUV": ["euv", "extreme ultraviolet", "euvl", "stochastic"],
    "Computational Lithography": ["aerial image", "lithography simulation", "process model", "resist model"],
    "ML for Lithography": ["deep learning", "neural network", "machine learning", "hotspot", "cnn"],
    "Defect Inspection": ["defect", "sem image", "inspection", "yield", "classification"],
    "Process Control": ["cd control", "overlay", "focus exposure", "process window"],
    "Mask Technology": ["mask fabrication", "phase shift", "attenuated psk", "euvl mask"],
    "DFM": ["design for manufacturability", "dfm", "lithography friendly"],
    "Patterning": ["multi-patterning", "sadp", "directed self assembly", "dsa"],
}

MAX_RESULTS_PER_QUERY = 10

# ── Load/save papers.json ──────────────────────────────────
def load_papers() -> list:
    if not os.path.exists(PAPERS_FILE):
        return []
    with open(PAPERS_FILE, "r") as f:
        content = f.read().strip()
        if not content:
            return []
        return json.loads(content)

def save_papers(papers: list):
    with open(PAPERS_FILE, "w") as f:
        json.dump(papers, f, indent=2)
    print(f"Saved {len(papers)} papers to {PAPERS_FILE}")

def get_existing_ids(papers: list) -> set:
    return {p["id"] for p in papers}

# ── Category detection ─────────────────────────────────────

def detect_category(title: str, abstract: str) -> str:
    text = (title + " " + abstract).lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return category
    return "General Lithography"

# ── Private source implementations ────────────────────────

def _search_arxiv(query: str, existing_ids: set) -> list:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=MAX_RESULTS_PER_QUERY,
        sort_by=arxiv.SortCriterion.Relevance
    )
    candidates = []
    try:
        for result in client.results(search):
            arxiv_id = result.entry_id.split("/")[-1]
            if arxiv_id in existing_ids:
                continue
            candidates.append({
                "id": arxiv_id,
                "source": "arxiv",
                "title": result.title,
                "abstract": result.summary[:300] + "...",
                "category": detect_category(result.title, result.summary),
                "year": result.published.year,
                "url": result.pdf_url,
            })
    except Exception as e:
        print(f"arXiv search failed for '{query}': {e}")
    return candidates

def _search_semantic_scholar(query: str, existing_ids: set) -> list:
    sch = SemanticScholar()
    candidates = []
    try:
        results = sch.search_paper(
            query,
            limit=MAX_RESULTS_PER_QUERY,
            fields=["title", "abstract", "year", "openAccessPdf", "externalIds"]
        )
        for paper in results:
            if not paper.openAccessPdf:
                continue
            paper_id = paper.externalIds.get("ArXiv") or str(paper.paperId)
            if paper_id in existing_ids:
                continue
            abstract = paper.abstract or ""
            candidates.append({
                "id": paper_id,
                "source": "semantic_scholar",
                "title": paper.title,
                "abstract": abstract[:300] + "...",
                "category": detect_category(paper.title, abstract),
                "year": paper.year,
                "url": paper.openAccessPdf.get("url") if paper.openAccessPdf else None,
            })
    except Exception as e:
        print(f"Semantic Scholar search failed for '{query}': {e}")
    return candidates

# ── Single search router ───────────────────────────────────

def search_papers(query: str, source: str, existing_ids: set) -> list:
    """
    Single entry point for all paper sources.
    Add new sources here as elif blocks — no changes needed elsewhere.
    """
    if source == "arxiv":
        return _search_arxiv(query, existing_ids)
    elif source == "semantic_scholar":
        return _search_semantic_scholar(query, existing_ids)
    else:
        print(f"Unknown source: {source} — skipping")
        return []

# ── User approval ──────────────────────────────────────────

def approve_candidates(candidates: list) -> list:
    if not candidates:
        print("No new candidates found.")
        return []

    approved = []
    print(f"\n{'='*60}")
    print(f"Found {len(candidates)} new candidate papers.")
    print(f"{'='*60}\n")

    for i, paper in enumerate(candidates):
        print(f"[{i+1}/{len(candidates)}]")
        print(f"Title    : {paper['title']}")
        print(f"Category : {paper['category']}")
        print(f"Year     : {paper['year']}")
        print(f"Source   : {paper['source']}")
        print(f"Abstract : {paper['abstract']}")
        print()
        choice = input("Add to knowledge base? (y/n/q to quit): ").strip().lower()
        if choice == "q":
            print("Stopping review.")
            break
        elif choice == "y":
            paper.pop("abstract", None)
            approved.append(paper)
            print("  ✓ Added\n")
        else:
            print("  ✗ Skipped\n")

    return approved

# ── Main ───────────────────────────────────────────────────

def main():
    print("LithoMind — Paper Discovery Tool")
    print(f"Sources: {PAPER_SOURCES}")
    print(f"Searching {len(SEARCH_QUERIES)} queries...\n")

    papers = load_papers()
    existing_ids = get_existing_ids(papers)
    print(f"Existing papers in knowledge base: {len(papers)}")

    # Search all queries across all sources
    all_candidates = []
    seen_in_session = set()

    for query in SEARCH_QUERIES:
        for source in PAPER_SOURCES:
            print(f"Searching [{source}]: {query}")
            results = search_papers(query, source, existing_ids)
            for candidate in results:
                if candidate["id"] not in seen_in_session:
                    seen_in_session.add(candidate["id"])
                    all_candidates.append(candidate)

    print(f"\nTotal new candidates found: {len(all_candidates)}")

    # User approval
    approved = approve_candidates(all_candidates)

    if not approved:
        print("No papers approved. Exiting.")
        return

    # Save to papers.json
    papers.extend(approved)
    save_papers(papers)
    print(f"\n{len(approved)} papers added to knowledge base list.")

    # Ask to ingest now
    ingest_now = input("\nIngest approved papers into ChromaDB now? (y/n): ").strip().lower()
    if ingest_now == "y":
        print("Starting ingestion...")
        # Will call build_knowledge_base here once built
    else:
        print("Papers saved to papers.json. Run build_knowledge_base.py when ready.")

if __name__ == "__main__":
    main()