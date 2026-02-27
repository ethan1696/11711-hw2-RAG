# Minimal Pipeline (Current Configs)

This is the minimal path to run the pipeline and produce leaderboard answers with the current config files:
- `data/crawler/config.yaml`
- `data/chunker/config.full.yaml`
- `rag/config.yaml`

## 1) Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 2) Crawl

```bash
python data/crawl.py \
  --config data/crawler/config.yaml \
  --output_dir data/crawled_output
```

Main output: `data/crawled_output/parsed/docs.jsonl`

## 3) Chunk

```bash
python data/chunker/chunk.py --config data/chunker/config.full.yaml
```

Main output: `data/chunked_output2/processed/chunks.jsonl`

## 4) Build Dense Embeddings

```bash
python retrieval/embed/embed.py \
  --input_chunks data/chunked_output2/processed/chunks.jsonl \
  --output_dir retrieval/output_embed \
  --device cuda:6
```

## 5) Build BM25 Index

```bash
python retrieval/build_bm25_index.py \
  --input_chunks data/chunked_output2/processed/chunks.jsonl \
  --output_dir retrieval/output_embed
```

## 6) Generate Leaderboard Answers

```bash
python rag/run_leaderboard.py \
  --config rag/config.yaml \
  --queries_json rag/leaderboard_queries.json \
  --output_json rag/leaderboard_answers.json \
  --andrewid ethanwan
```

Output file: `rag/leaderboard_answers.json`

## Notes

- `rag/config.yaml` currently controls retrieval mode (`closed_book`, `dense`, `sparse`, or `hybrid`).
- If you use `dense` or `hybrid`, run steps 4 and 5 first so `retrieval/output_embed` has all retrieval artifacts.
