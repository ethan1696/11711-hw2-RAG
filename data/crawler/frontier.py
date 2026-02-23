"""Thread-safe frontier queue with scope and budget enforcement."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .config import CrawlConfig
from .types import FrontierItem
from .url import host_from_url, normalize_url


class EnqueueStatus(str, Enum):
    """Result status for frontier enqueue attempts."""

    ENQUEUED = "enqueued"
    SKIPPED_INVALID_URL = "skipped_invalid_url"
    SKIPPED_OUT_OF_SCOPE = "skipped_out_of_scope"
    SKIPPED_DEPTH = "skipped_depth"
    SKIPPED_SEEN = "skipped_seen"
    SKIPPED_GLOBAL_BUDGET = "skipped_global_budget"
    SKIPPED_DOMAIN_BUDGET = "skipped_domain_budget"
    SKIPPED_SEED_BUDGET = "skipped_seed_budget"
    SKIPPED_CLOSED = "skipped_closed"


@dataclass(frozen=True, slots=True)
class EnqueueResult:
    """Outcome of one enqueue attempt."""

    status: EnqueueStatus
    normalized_url: str | None = None
    item: FrontierItem | None = None

    @property
    def accepted(self) -> bool:
        return self.status == EnqueueStatus.ENQUEUED


class Frontier:
    """Frontier queue used by producer/consumer crawl workers.

    - Thread-safe `push` and `pop` for multi-worker crawling.
    - Enforces max depth, deduplication, and crawl budgets.
    - Treats URLs as seen at enqueue-time to avoid duplicate work.
    """

    def __init__(
        self,
        config: CrawlConfig,
        *,
        initial_seen_urls: Iterable[str] | None = None,
    ) -> None:
        self.config = config

        self._queue: queue.Queue[FrontierItem] = queue.Queue()
        self._lock = threading.Lock()

        self._seen_urls: set[str] = set()
        self._accepted_this_run = 0
        self._accepted_by_domain: dict[str, int] = {}
        self._accepted_by_seed: dict[str, int] = {}

        self._enqueued_count = 0
        self._dequeued_count = 0
        self._skipped_seen_count = 0
        self._skipped_depth_count = 0
        self._skipped_budget_count = 0
        self._skipped_invalid_count = 0
        self._skipped_out_of_scope_count = 0

        self._closed = False

        if initial_seen_urls:
            for url in initial_seen_urls:
                normalized = normalize_url(url)
                if normalized:
                    self._seen_urls.add(normalized)

    def seed(self, seeds: Iterable[str]) -> list[EnqueueResult]:
        """Seed frontier with depth=0 URLs."""

        return [self.push(seed, depth=0, referrer=None, seed_url=seed) for seed in seeds]

    def push(
        self,
        url: str,
        *,
        depth: int,
        referrer: str | None = None,
        seed_url: str | None = None,
    ) -> EnqueueResult:
        """Attempt to enqueue one URL with constraints enforced."""

        normalized = normalize_url(url)
        if not normalized:
            with self._lock:
                self._skipped_invalid_count += 1
            return EnqueueResult(EnqueueStatus.SKIPPED_INVALID_URL)

        normalized_seed = normalize_url(seed_url) if seed_url else None
        seed_key = normalized_seed or normalized

        if depth > self.config.max_depth_for_seed(seed_key):
            with self._lock:
                self._skipped_depth_count += 1
            return EnqueueResult(EnqueueStatus.SKIPPED_DEPTH, normalized_url=normalized)

        if not self.config.is_url_allowed(normalized):
            with self._lock:
                self._skipped_out_of_scope_count += 1
            return EnqueueResult(EnqueueStatus.SKIPPED_OUT_OF_SCOPE, normalized_url=normalized)

        with self._lock:
            if self._closed:
                return EnqueueResult(EnqueueStatus.SKIPPED_CLOSED, normalized_url=normalized)

            if normalized in self._seen_urls:
                self._skipped_seen_count += 1
                return EnqueueResult(EnqueueStatus.SKIPPED_SEEN, normalized_url=normalized)

            if self._accepted_this_run >= self.config.max_pages:
                self._skipped_budget_count += 1
                return EnqueueResult(EnqueueStatus.SKIPPED_GLOBAL_BUDGET, normalized_url=normalized)

            domain_cfg = self.config.get_domain_config(normalized)
            domain_key = domain_cfg.domain if domain_cfg else host_from_url(normalized)
            per_domain_cap = self.config.per_domain_cap_for(normalized)
            if (
                per_domain_cap is not None
                and self._accepted_by_domain.get(domain_key, 0) >= per_domain_cap
            ):
                self._skipped_budget_count += 1
                return EnqueueResult(EnqueueStatus.SKIPPED_DOMAIN_BUDGET, normalized_url=normalized)

            per_seed_cap = self.config.per_seed_cap_for(seed_key)
            if (
                per_seed_cap is not None
                and self._accepted_by_seed.get(seed_key, 0) >= per_seed_cap
            ):
                self._skipped_budget_count += 1
                return EnqueueResult(EnqueueStatus.SKIPPED_SEED_BUDGET, normalized_url=normalized)

            self._seen_urls.add(normalized)
            self._accepted_this_run += 1
            self._accepted_by_domain[domain_key] = self._accepted_by_domain.get(domain_key, 0) + 1
            self._accepted_by_seed[seed_key] = self._accepted_by_seed.get(seed_key, 0) + 1

            item = FrontierItem(
                url=normalized,
                depth=depth,
                seed_url=seed_key,
                referrer=referrer,
            )
            self._queue.put(item)
            self._enqueued_count += 1

        return EnqueueResult(
            EnqueueStatus.ENQUEUED,
            normalized_url=normalized,
            item=item,
        )

    def push_many(
        self,
        urls: Iterable[str],
        *,
        depth: int,
        referrer: str | None = None,
        seed_url: str | None = None,
    ) -> list[EnqueueResult]:
        """Attempt to enqueue multiple URLs, preserving input order."""

        return [
            self.push(url, depth=depth, referrer=referrer, seed_url=seed_url)
            for url in urls
        ]

    def pop(self, *, block: bool = True, timeout: float | None = None) -> FrontierItem | None:
        """Pop one frontier item for a worker thread.

        Returns `None` when no item is available under the requested blocking mode.
        """

        try:
            if block:
                item = self._queue.get(block=True, timeout=timeout)
            else:
                item = self._queue.get(block=False)
        except queue.Empty:
            return None

        with self._lock:
            self._dequeued_count += 1
        return item

    def pop_many(
        self,
        max_items: int,
        *,
        block: bool = True,
        timeout: float | None = None,
    ) -> list[FrontierItem]:
        """Pop up to `max_items` items.

        This helps batch-oriented workers while staying thread-safe.
        """

        if max_items <= 0:
            raise ValueError("max_items must be > 0")

        first = self.pop(block=block, timeout=timeout)
        if first is None:
            return []

        items = [first]
        while len(items) < max_items:
            next_item = self.pop(block=False)
            if next_item is None:
                break
            items.append(next_item)
        return items

    def task_done(self) -> None:
        """Mark one popped task as finished (delegates to Queue.task_done)."""

        self._queue.task_done()

    def join(self) -> None:
        """Block until all queued tasks are marked done."""

        self._queue.join()

    def close(self) -> None:
        """Close frontier to future enqueue attempts."""

        with self._lock:
            self._closed = True

    @property
    def closed(self) -> bool:
        """Whether frontier has been closed for new enqueue attempts."""

        with self._lock:
            return self._closed

    def qsize(self) -> int:
        """Approximate queue size."""

        return self._queue.qsize()

    def empty(self) -> bool:
        """Return True if queue is currently empty."""

        return self._queue.empty()

    def seen_urls(self) -> set[str]:
        """Return snapshot of seen URLs (seeded + accepted this run)."""

        with self._lock:
            return set(self._seen_urls)

    def accepted_by_domain(self) -> dict[str, int]:
        """Return snapshot of accepted-per-domain counts."""

        with self._lock:
            return dict(self._accepted_by_domain)

    def accepted_by_seed(self) -> dict[str, int]:
        """Return snapshot of accepted-per-seed counts."""

        with self._lock:
            return dict(self._accepted_by_seed)

    def snapshot(self) -> dict[str, int | bool]:
        """Return frontier counters for logs/stats reporting."""

        with self._lock:
            return {
                "closed": self._closed,
                "queue_size": self._queue.qsize(),
                "seen_urls": len(self._seen_urls),
                "accepted_this_run": self._accepted_this_run,
                "accepted_seed_count": len(self._accepted_by_seed),
                "enqueued": self._enqueued_count,
                "dequeued": self._dequeued_count,
                "skipped_seen": self._skipped_seen_count,
                "skipped_depth": self._skipped_depth_count,
                "skipped_budget": self._skipped_budget_count,
                "skipped_invalid": self._skipped_invalid_count,
                "skipped_out_of_scope": self._skipped_out_of_scope_count,
            }


__all__ = [
    "EnqueueResult",
    "EnqueueStatus",
    "Frontier",
]
