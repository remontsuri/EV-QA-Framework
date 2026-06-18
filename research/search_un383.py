from ddgs import DDGS

with DDGS() as ddgs:
    results = list(ddgs.text("UN 38.3 lithium battery transport standard test requirements", max_results=5))
    for r in results:
        print(r.get("title", ""))
        print(r.get("href", ""))
        print(r.get("body", "")[:300])
        print("---")