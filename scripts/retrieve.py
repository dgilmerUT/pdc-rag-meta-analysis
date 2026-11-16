import requests
import csv
import os
import time

# ── EDIT THESE TWO THINGS ──────────────────────────────
EMAIL = os.environ["UNPAYWALL_EMAIL"]   # Unpaywall asks for a contact email

SEARCH_TERMS = [
    "polysilazane ceramic yield pyrolysis",
    "polycarbosilane SiC precursor synthesis",
    "preceramic polymer descriptor properties",
    "Polymer derived ceramic precursor",
    "polymer derived ceramic",
    "Preceramic polymer", 
]
# ───────────────────────────────────────────────────────

PDF_DIR = "data/raw/pdfs"
os.makedirs(PDF_DIR, exist_ok=True)

print("Searching OpenAlex...")
all_papers = []

for term in SEARCH_TERMS:
    url = "https://api.openalex.org/works"
    params = {
        "search": term,
        "filter": "from_publication_date:2000-01-01",
        "per_page": 50,
        "select": "doi,title,publication_year,open_access"
    }
    r = requests.get(url, params=params)
    works = r.json().get("results", [])
    print(f"  '{term}': found {len(works)} papers")
    all_papers.extend(works)

# Remove duplicates by DOI
seen = set()
unique = []
for p in all_papers:
    doi = p.get("doi", "")
    if doi and doi not in seen:
        seen.add(doi)
        unique.append(p)

print(f"\nTotal unique papers found: {len(unique)}")

# Save metadata
with open("data/metadata/corpus.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["doi", "title", "year", "open_access", "pdf_url"])
    for p in unique:
        oa = p.get("open_access", {})
        writer.writerow([
            p.get("doi", ""),
            p.get("title", ""),
            p.get("publication_year", ""),
            oa.get("is_oa", False),
            oa.get("oa_url", "")
        ])

print("Metadata saved to data/metadata/corpus.csv")
print("\nDownloading open-access PDFs...")

downloaded = 0
skipped = 0

for p in unique:
    oa = p.get("open_access", {})
    if not oa.get("is_oa"):
        skipped += 1
        continue

    doi = p.get("doi", "").replace("https://doi.org/", "")
    if not doi:
        continue

    # Check Unpaywall for a direct PDF link
    unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email={EMAIL}"
    try:
        r = requests.get(unpaywall_url, timeout=10)
        if r.status_code != 200:
            continue
        data = r.json()
        best = data.get("best_oa_location", {})
        pdf_url = best.get("url_for_pdf") if best else None
    except Exception as e:
        print(f"  Unpaywall lookup failed for {doi}: {e}")
        continue

    if not pdf_url:
        continue

    # Make a safe filename
    filename = doi.replace("/", "_").replace(":", "_") + ".pdf"
    filepath = os.path.join(PDF_DIR, filename)

    if os.path.exists(filepath):
        continue  # already downloaded

    try:
        r = requests.get(pdf_url, timeout=30)
        if r.status_code == 200 and len(r.content) > 10000:
            with open(filepath, "wb") as f:
                f.write(r.content)
            print(f"  Downloaded: {filename[:60]}")
            downloaded += 1
        time.sleep(1)  # be polite to servers
    except Exception as e:
        print(f"  Download failed for {doi}: {e}")

print(f"\nDone. Downloaded {downloaded} PDFs.")
print(f"Papers behind paywalls (need manual download): {skipped}")
print(f"PDFs are in: {PDF_DIR}")