# PDC RAG meta-analysis: code and data

Code and extracted data for:

> Gilmer D, Walker R. *Mapping the Polymer-to-Ceramic Design Space: A Retrieval-Augmented Meta-Analysis of Preceramic Polymer Architecture and Microstructural Evolution.* (journal and DOI to be added)

Preprint v1: ChemRxiv, doi:[10.26434/chemrxiv.15006390/v1](https://doi.org/10.26434/chemrxiv.15006390/v1) (949-paper corpus).
This repository covers the expanded, deduplicated 1,763-paper corpus used in the journal version.

## What is here

| Path | Contents |
|---|---|
| `outputs/descriptors.json` | The descriptor dataset: one record per paper (1,763), extracted by Claude from the paper text. This is the reference dataset for every statistic in the paper. |
| `outputs/descriptors.csv` | The same table flattened to one column per field. |
| `outputs/sources.csv` | Source list: the record key (`source`), tier, and DOI or bibliographic data for each paper, so the corpus can be reassembled. See "Source list" below. |
| `prompt/extraction_prompt.txt` | The extraction schema and prompt, verbatim. |
| `outputs/dedup_2026-10-28/` | First deduplication round (shared chunk text): the records and files removed, the duplicate groups, and the pre-deduplication descriptor table (its first 949 records are the v1 preprint corpus). |
| `outputs/dedup_2026-11-16/` | Second round: 11 duplicate records found through shared DOIs, with the copy kept for each. |
| `outputs/manuscript_v2/` | Manuscript figures, `numbers.json` (every number quoted in the text), and `mech_by_method.json` (strength and hardness split by test method). |
| `data/chunk_counts.csv` | Number of text chunks per paper (the chunk text itself is not distributed). |
| `data/residual_duplicate_groups.json` | Record pairs that still share most of their chunk text after deduplication (reported in the paper). |
| `scripts/` | The pipeline: retrieval, PDF parsing, chunking, embedding, extraction, deduplication, search. |
| `scripts/analysis/` | Statistics (`stats_lib.py`) and figure generation (`make_figures.py`). |
| `docs/validation_pilot/` | Annotation protocol and results of the 30-paper validation pilot. |

The full-text corpus (PDFs, parsed text, chunks and the vector index) is not included because the papers are under publisher copyright.

## Reproducing the statistics and figures

Needs Python 3.12 and only the light dependencies:

```bash
pip install -r requirements-analysis.txt
PYTHONPATH=scripts/analysis python3 scripts/analysis/make_figures.py
```

This rewrites `outputs/manuscript_v2/` from `outputs/descriptors.json`. The resulting `numbers.json` matches the deposited one exactly.

## Running the full pipeline

The full pipeline needs the papers themselves, assembled from `outputs/sources.csv`, plus `pip install -r requirements.txt` and an Anthropic API key in `ANTHROPIC_API_KEY`. Run the scripts from the repository root in this order:

1. `scripts/retrieve.py`: OpenAlex search and open-access download (set `UNPAYWALL_EMAIL`). Most of the corpus was collected by hand from Web of Science, Scopus and citation chaining, as described in the paper.
2. `scripts/parse_pdfs.py`: PDF to structured JSON with Docling, into `data/parsed/`. The tier comes from the folder a PDF sits in (`pdfs`, `conventional_ceramics`, `applications`, `alternative_routes`, ...).
3. `scripts/chunk.py`: 250-word chunks with a 50-word overlap, into `data/chunks.json`.
4. `scripts/ingest.py`: SciBERT (`allenai/scibert_scivocab_uncased`) embeddings into ChromaDB (`vector_store/chroma_db`).
5. `scripts/extract.py`: for each paper, the 40 highest-priority chunks (up to 60,000 characters) are sent to `claude-sonnet-4-6` with the prompt in `prompt/`; the JSON answers go to `outputs/descriptors.json`.
6. `scripts/dedup.py` (shared chunk text) and `scripts/remove_records.py` (a list of records, used for the DOI duplicates): remove records and log them so `scripts/undo_dedup.py` can restore them.
7. `scripts/analysis/make_figures.py`: statistics and figures.

`scripts/query.py` searches the vector index from the command line. `scripts/build_sources.py` builds `outputs/sources.csv`.

LLM extraction is not deterministic: re-running step 5 gives slightly different values. The deposited `descriptors.json` is the reference dataset.

## Descriptor fields

Each record has these categories, with the sub-fields given in `prompt/extraction_prompt.txt`: `polymer_type`, `polymer_name`, `elemental_composition`, `bonding_structure`, `functional_groups`, `thermal_pyrolysis`, `crosslink_density`, `molecular_weight`, `backbone_architecture`, `microstructural`, `mechanical_properties`, `conformation_crystallinity`, `notes`, plus `source` (the record key) and `source_type` (tier; missing means `pdc`).

Tiers: `pdc` 1,327 papers, `conventional_ceramic` 343, `application` 86, `alternative_route` 7.

Two unpublished documents (research proposals) were in earlier versions of the corpus, including the v1 preprint, and were removed in the second round. They are withheld from every file here; in the round-one pre-deduplication table their records are replaced by empty placeholders, so v1-corpus comparison values recomputed from this repository differ slightly from the preprint.

A field counts as reported unless it is null, false or empty. The `notes` and free-text `description` fields are model-written summaries of the paper, not quotations.

## Source list

`outputs/sources.csv` columns: `source`, `tier`, `title_guess` (from the filename or title page), `doi`, `title`, `authors`, `year`, `container`, `type`, `match`, `title_similarity`.

`match` says how the bibliographic data was found:
- `doi_in_text`: a DOI printed on the paper, confirmed on Crossref with a matching title.
- `title_search`: a Crossref title search with a close title match.
- `check`: a probable match not yet confirmed.
- `manual`: entered by hand (books, theses, reports, patents).
- `none`: not yet found.

## Licenses

Code: MIT (`LICENSE`). Data and figures: CC BY 4.0 (`LICENSE-DATA.md`).

## Citation

See `CITATION.cff`, or cite the paper above and the Zenodo archive (doi to be added).
