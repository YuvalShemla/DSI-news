"""Constrained decoding for chrono-semantic DocIDs."""

from __future__ import annotations

import json
import logging
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


class Prefixer:
    """Prefix trie for constrained beam search over valid DocID sequences.

    Builds a trie from docid_to_tokenids mapping. At each decoding step,
    returns the set of allowed next tokens given the prefix generated so far.
    """

    def __init__(
        self,
        docid_to_tokenids_path: str | Path | None = None,
        tokenizer=None,
        prefix_path: str | Path | None = None,
        save_prefix: bool = True,
    ):
        self.prefix_dict: dict[tuple, set] = defaultdict(set)

        if prefix_path is not None and os.path.exists(prefix_path):
            logger.info("Loading prefix trie from %s", prefix_path)
            with open(prefix_path, "rb") as fin:
                self.prefix_dict = pickle.load(fin)
        else:
            if docid_to_tokenids_path is None or tokenizer is None:
                raise ValueError(
                    "Must provide docid_to_tokenids_path and tokenizer "
                    "when prefix_path is not available"
                )
            logger.info("Building prefix trie from %s", docid_to_tokenids_path)
            with open(docid_to_tokenids_path) as fin:
                docid_to_tokenids = json.load(fin)

            for docid, tokenids in docid_to_tokenids.items():
                extended_tokenids = [tokenizer.pad_token_id] + tokenids
                for i in range(1, len(extended_tokenids)):
                    self.prefix_dict[tuple(extended_tokenids[:i])].add(
                        extended_tokenids[i]
                    )

            if save_prefix:
                prefix_dir = os.path.dirname(docid_to_tokenids_path)
                out_path = os.path.join(prefix_dir, "prefix.pickle")
                with open(out_path, "wb") as fout:
                    pickle.dump(dict(self.prefix_dict), fout)
                logger.info("Saved prefix trie to %s", out_path)

        logger.info("Prefix trie has %d entries", len(self.prefix_dict))

    def __call__(self, batch_id: int, sent) -> list[int]:
        """Return allowed next token IDs for the given prefix."""
        return list(self.prefix_dict[tuple(sent.cpu().tolist())])


class FilteredPrefixer:
    """Extends Prefixer with time-based filtering.

    When a time filter is active, prunes the trie to only allow DocIDs with
    matching year/month prefix tokens at positions 0 and 1.
    """

    def __init__(
        self,
        docid_to_tokenids_path: str | Path,
        tokenizer,
        token_id_ranges: dict,
        prefix_path: str | Path | None = None,
        save_prefix: bool = True,
    ):
        """
        Args:
            token_id_ranges: Output of docid_tokenizer.get_token_id_ranges().
                             Dict with keys "year" (list), "month" (list), "rq" (dict).
        """
        # Build the base trie
        self._base = Prefixer(
            docid_to_tokenids_path=docid_to_tokenids_path,
            tokenizer=tokenizer,
            prefix_path=prefix_path,
            save_prefix=save_prefix,
        )
        self.prefix_dict = self._base.prefix_dict

        # Store token ID sets for filtering
        self._year_ids = set(token_id_ranges["year"])
        self._month_ids = set(token_id_ranges["month"])

        # Build year_token -> token_id and month_token -> token_id lookups
        # token_id_ranges["year"] is ordered by year_range
        self._year_token_map = {}  # year_int -> token_id
        self._month_token_map = {}  # month_int -> token_id
        for i, tid in enumerate(token_id_ranges["year"]):
            # Years start from the first year in the range
            # We infer from the token string via tokenizer
            self._year_token_map[tid] = tid
        for i, tid in enumerate(token_id_ranges["month"]):
            self._month_token_map[tid] = tid

        # Reverse: we need year_int -> token_id. Parse from token_id_ranges order.
        # The caller passes year_range info — we store the raw lists for lookup.
        self._year_id_list = token_id_ranges["year"]  # index 0 = first year
        self._month_id_list = token_id_ranges["month"]  # index 0 = month 01

        self._filter_year_id: int | None = None
        self._filter_month_id: int | None = None

    def set_filter(
        self, filter_year: int | None = None, filter_month: int | None = None,
        year_range_start: int = 2017,
    ) -> None:
        """Set the current time filter. Call before generation for filtered queries.

        Args:
            filter_year: Year to filter to (e.g. 2023), or None.
            filter_month: Month to filter to (e.g. 6), or None.
            year_range_start: First year in the year_range (default 2017).
        """
        if filter_year is not None:
            year_idx = filter_year - year_range_start
            if 0 <= year_idx < len(self._year_id_list):
                self._filter_year_id = self._year_id_list[year_idx]
            else:
                logger.warning("Year %d out of range, ignoring filter", filter_year)
                self._filter_year_id = None
        else:
            self._filter_year_id = None

        if filter_month is not None:
            month_idx = filter_month - 1
            if 0 <= month_idx < len(self._month_id_list):
                self._filter_month_id = self._month_id_list[month_idx]
            else:
                logger.warning("Month %d out of range, ignoring filter", filter_month)
                self._filter_month_id = None
        else:
            self._filter_month_id = None

    def clear_filter(self) -> None:
        """Remove any active filter."""
        self._filter_year_id = None
        self._filter_month_id = None

    def __call__(self, batch_id: int, sent) -> list[int]:
        """Return allowed next token IDs, respecting any active time filter.

        At position 0 (first token): if filter_year is set, only allow that year's token.
        At position 1 (second token): if filter_month is set, only allow that month's token.
        At positions 2-7: standard trie lookup (RQ tokens).
        """
        prefix = tuple(sent.cpu().tolist())
        allowed = self.prefix_dict.get(prefix, set())

        if not allowed:
            return []

        # Position in the DocID sequence = len(prefix) - 1 (prefix includes pad_token_id)
        # prefix[0] is pad, so position 0 of DocID is at len(prefix)==1
        docid_pos = len(prefix) - 1

        if docid_pos == 0 and self._filter_year_id is not None:
            # Only allow the filtered year token
            if self._filter_year_id in allowed:
                return [self._filter_year_id]
            return []

        if docid_pos == 1 and self._filter_month_id is not None:
            # Only allow the filtered month token
            if self._filter_month_id in allowed:
                return [self._filter_month_id]
            return []

        return list(allowed)


_FILTER_PATTERN = re.compile(r"\[FILTER:(\d{4})(?:-(\d{2}))?\]\s*")


def parse_filter_from_query(query: str) -> tuple[str, int | None, int | None]:
    """Extract filter prefix from query string.

    Examples:
        "[FILTER:2023] climate change" -> ("climate change", 2023, None)
        "[FILTER:2023-06] climate change" -> ("climate change", 2023, 6)
        "climate change" -> ("climate change", None, None)

    Returns:
        (clean_query, filter_year, filter_month)
    """
    match = _FILTER_PATTERN.search(query)
    if match is None:
        return query, None, None

    year = int(match.group(1))
    month = int(match.group(2)) if match.group(2) else None
    clean_query = query[: match.start()] + query[match.end() :]
    clean_query = clean_query.strip()
    return clean_query, year, month


def build_smtid_to_docids(
    docid_to_tokenids_path: str | Path,
    max_new_tokens: int = 8,
) -> dict[str, list[str]]:
    """Build reverse mapping from string-joined token IDs to doc IDs.

    This is needed to map generated sequences back to document IDs.
    E.g., "262145_262158_262400_..." -> ["doc_123", "doc_456"]

    Args:
        docid_to_tokenids_path: Path to docid_to_tokenids.json.
        max_new_tokens: Expected number of tokens per DocID (for validation).

    Returns:
        Dict mapping underscore-joined token ID string -> list of doc IDs.
    """
    with open(docid_to_tokenids_path) as fin:
        docid_to_tokenids = json.load(fin)

    smtid_to_docids: dict[str, list[str]] = defaultdict(list)
    for docid, tokenids in docid_to_tokenids.items():
        key = "_".join(str(t) for t in tokenids[:max_new_tokens])
        smtid_to_docids[key].append(docid)

    logger.info(
        "Built smtid_to_docids: %d unique sequences -> %d docs",
        len(smtid_to_docids),
        len(docid_to_tokenids),
    )
    return dict(smtid_to_docids)


if __name__ == "__main__":
    import tempfile

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    # Test parse_filter_from_query
    tests = [
        ("[FILTER:2023] climate change", ("climate change", 2023, None)),
        ("[FILTER:2023-06] climate change", ("climate change", 2023, 6)),
        ("climate change", ("climate change", None, None)),
        ("[FILTER:2019-12] stock market crash", ("stock market crash", 2019, 12)),
    ]
    for query, expected in tests:
        result = parse_filter_from_query(query)
        assert result == expected, f"Failed: {query} -> {result} != {expected}"
    logger.info("All parse_filter_from_query tests passed")

    # Test Prefixer with dummy data
    dummy_docids = {
        "doc_0": [100, 200, 300, 301, 302, 303, 304, 305],
        "doc_1": [100, 200, 300, 301, 302, 303, 304, 306],
        "doc_2": [100, 201, 310, 311, 312, 313, 314, 315],
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "docid_to_tokenids.json")
        with open(path, "w") as f:
            json.dump(dummy_docids, f)

        # Minimal mock tokenizer
        class MockTokenizer:
            pad_token_id = 0

        prefixer = Prefixer(
            docid_to_tokenids_path=path,
            tokenizer=MockTokenizer(),
            save_prefix=False,
        )

        import torch

        # From pad, first allowed should be {100}
        allowed = prefixer(0, torch.tensor([0]))
        assert set(allowed) == {100}, f"Expected {{100}}, got {set(allowed)}"

        # From [pad, 100], should be {200, 201}
        allowed = prefixer(0, torch.tensor([0, 100]))
        assert set(allowed) == {200, 201}, f"Expected {{200, 201}}, got {set(allowed)}"

        logger.info("All Prefixer tests passed")

        # Test build_smtid_to_docids
        smtid_map = build_smtid_to_docids(path)
        assert len(smtid_map) == 3  # all three docs have unique sequences
        logger.info("All build_smtid_to_docids tests passed")
