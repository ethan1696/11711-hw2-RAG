# Data Pipeline Layout

This folder contains crawler and data-prep code for the RAG assignment.

## Install

```bash
pip install -r requirements.txt
```

or

```bash
pip install -r data/requirements.txt
```

## Crawler package

- `data/crawler/types.py`: shared dataclasses and typed records.
- `data/crawler/constants.py`: defaults and config constants.
- `data/crawler/config.py`: typed config loading/saving and domain policies.
- `data/crawler/url.py`: URL normalization, scope filtering, and link extraction.
- `data/crawler/frontier.py`: thread-safe frontier queue with budget/scope enforcement.
- `data/crawler/fetcher.py`: requests/selenium fetch backend with retries and rate-limits.
- `data/crawler/stats.py`: thread-safe runtime stats collector and JSON summary.
- `data/crawler/pipeline.py`: orchestrates crawl/fetch/parse-only execution modes.
- `data/crawler/parsers/`: parser modules (HTML/PDF) live here.

## Run

```bash
python data/crawl.py --config data/crawler/config.example.yaml --output_dir data/crawled_output
```

```bash
python data/crawl.py --config data/crawler/config.example.yaml --mode parse_only --output_dir data/crawled_output
```

## Notes

- Per-domain fetch backends are supported via `DomainConfig.backend`.
- Use `backend: selenium` for dynamic pages.
