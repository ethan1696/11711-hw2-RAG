"""CLI entrypoint for crawler pipeline execution."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from data.crawler import CrawlConfig, Pipeline, PipelineMode, host_from_url, load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CMU ANLP RAG crawler pipeline.",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON/YAML crawl config.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/crawled_output"),
        help="Root output directory for raw/parsed/manifests/logs.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=[mode.value for mode in PipelineMode],
        default=PipelineMode.CRAWL.value,
        help="Pipeline mode: crawl, fetch_only, or parse_only.",
    )

    parser.add_argument(
        "--seed",
        action="append",
        default=[],
        help="Seed URL (repeatable). Overrides config seeds if provided.",
    )
    parser.add_argument(
        "--domain",
        action="append",
        default=[],
        help=(
            "Allowed domain spec (repeatable). Format: DOMAIN or DOMAIN=BACKEND, "
            "where BACKEND is requests or selenium. Overrides config domains if provided."
        ),
    )

    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--max_pages", type=int, default=None)
    parser.add_argument(
        "--per_domain_cap",
        type=int,
        default=None,
        help="Use 0 or negative to disable per-domain cap.",
    )
    parser.add_argument(
        "--per_seed_cap",
        type=int,
        default=None,
        help="Use 0 or negative to disable per-seed cap.",
    )
    parser.add_argument("--concurrency", type=int, default=None)

    parser.add_argument("--timeout_seconds", type=float, default=None)
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument("--retry_backoff_seconds", type=float, default=None)
    parser.add_argument("--rate_limit_seconds", type=float, default=None)

    parser.add_argument("--user_agent", type=str, default=None)
    parser.add_argument(
        "--respect_robots",
        dest="respect_robots",
        action="store_true",
        default=None,
        help="Respect robots.txt (default comes from config).",
    )
    parser.add_argument(
        "--no_respect_robots",
        dest="respect_robots",
        action="store_false",
        help="Ignore robots.txt.",
    )

    parser.add_argument(
        "--force_download",
        action="store_true",
        help="Always re-download URLs even if raw cache exists.",
    )
    parser.add_argument(
        "--force_parse",
        action="store_true",
        help="Always re-parse documents even if parsed cache exists.",
    )

    parser.add_argument(
        "--print_stats_json",
        action="store_true",
        help="Print full stats JSON in stdout after run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    return parser.parse_args(argv)


def _parse_domain_specs(specs: list[str]) -> list[dict[str, Any]]:
    domains: list[dict[str, Any]] = []

    for spec in specs:
        raw = spec.strip()
        if not raw:
            continue

        if "=" in raw:
            domain, backend = raw.split("=", maxsplit=1)
            domain = domain.strip()
            backend = backend.strip().lower()
            if backend not in {"requests", "selenium"}:
                raise ValueError(
                    f"Invalid backend in --domain '{spec}'. Use requests or selenium."
                )
            domains.append({"domain": domain, "backend": backend})
        else:
            domains.append({"domain": raw})

    return domains


def build_config(args: argparse.Namespace) -> CrawlConfig:
    if args.config is not None:
        config = load_config(args.config)
        payload = config.to_dict()
    else:
        payload = {
            "seeds": list(args.seed),
            "domains": _parse_domain_specs(args.domain),
        }

    if args.seed:
        payload["seeds"] = list(args.seed)

    if args.domain:
        payload["domains"] = _parse_domain_specs(args.domain)

    if not payload.get("seeds"):
        raise ValueError("No seeds provided. Use --config or at least one --seed.")

    if not payload.get("domains"):
        payload["domains"] = [{"domain": host_from_url(seed)} for seed in payload["seeds"]]

    if args.max_depth is not None:
        payload["max_depth"] = args.max_depth
    if args.max_pages is not None:
        payload["max_pages"] = args.max_pages
    if args.per_domain_cap is not None:
        payload["per_domain_cap"] = None if args.per_domain_cap <= 0 else args.per_domain_cap
    if args.per_seed_cap is not None:
        payload["per_seed_cap"] = None if args.per_seed_cap <= 0 else args.per_seed_cap
    if args.concurrency is not None:
        payload["concurrency"] = args.concurrency

    if args.timeout_seconds is not None:
        payload["timeout_seconds"] = args.timeout_seconds
    if args.retries is not None:
        payload["retries"] = args.retries
    if args.retry_backoff_seconds is not None:
        payload["retry_backoff_seconds"] = args.retry_backoff_seconds
    if args.rate_limit_seconds is not None:
        payload["rate_limit_seconds"] = args.rate_limit_seconds

    if args.user_agent is not None:
        payload["user_agent"] = args.user_agent
    if args.respect_robots is not None:
        payload["respect_robots"] = args.respect_robots

    if args.force_download:
        payload["force_download"] = True
    if args.force_parse:
        payload["force_parse"] = True

    return CrawlConfig.from_dict(payload)


def setup_logging(output_dir: Path, verbose: bool) -> None:
    log_level = logging.DEBUG if verbose else logging.INFO

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "crawl.log"

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(log_level)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Trafilatura can emit frequent non-actionable warnings like
    # "discarding data: None" on noisy pages. Keep crawler logs readable.
    logging.getLogger("trafilatura").setLevel(logging.ERROR)
    logging.getLogger("trafilatura.core").setLevel(logging.ERROR)


def print_summary(result: dict[str, Any], *, print_stats_json: bool) -> None:
    paths = result.get("paths", {})
    stats = result.get("stats", {})

    print("\n=== Crawl Complete ===")
    print(f"mode: {result.get('mode')}")
    print(f"output_dir: {paths.get('output_dir')}")
    print(f"docs: {paths.get('parsed_docs')}")
    print(f"errors: {paths.get('parsed_errors')}")
    print(f"url_meta: {paths.get('url_meta')}")
    print(f"stats: {paths.get('crawl_stats')}")

    print("\n--- Core Stats ---")
    for key in [
        "frontier_enqueued",
        "frontier_skipped_visited",
        "frontier_skipped_depth",
        "frontier_skipped_budget",
        "fetched_ok",
        "fetched_error",
        "parsed_ok",
        "parsed_error",
        "stored_docs",
        "duration_seconds",
    ]:
        if key in stats:
            print(f"{key}: {stats[key]}")

    if print_stats_json:
        print("\n--- Full Stats JSON ---")
        print(json.dumps(stats, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.output_dir, verbose=args.verbose)

    try:
        config = build_config(args)
    except Exception as exc:
        logging.error("Failed to build config: %s", exc)
        return 2

    logging.info(
        "Starting pipeline: mode=%s, output_dir=%s, seeds=%d, domains=%d",
        args.mode,
        args.output_dir,
        len(config.seeds),
        len(config.domains),
    )

    try:
        pipeline = Pipeline(config, output_dir=args.output_dir)
        result = pipeline.run(PipelineMode(args.mode))
    except KeyboardInterrupt:
        logging.error("Interrupted by user")
        return 130
    except Exception:
        logging.exception("Pipeline execution failed")
        return 1

    print_summary(result, print_stats_json=args.print_stats_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
