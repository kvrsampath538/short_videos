#!/usr/bin/env python3
"""
Full ingestion of all local JSON files into the vector DB.
Safe to re-run — ChromaDB upserts by id so existing records are updated, not duplicated.

Run from project root:
    python3 scripts/ingest_vectordb.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vectordb import (
    VectorDB, _idea_text, _idea_meta, _channel_short_text, _channel_meta, _safe_id
)


def _load(path: str | Path):
    p = ROOT / path
    if not p.exists():
        print(f"  ⚠ {path} not found — skipping")
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def ingest_ideas(db: VectorDB) -> None:
    docs = _load("ideas_history.json")
    if not isinstance(docs, list) or not docs:
        return
    print(f"\n[ideas] {len(docs)} records")
    n = db.bulk_ingest(
        "ideas", docs,
        text_fn=_idea_text,
        id_fn=lambda doc, i: _safe_id(_idea_text(doc), f"idea_{i}"),
        meta_fn=_idea_meta,
    )
    print(f"[ideas] ✓ {n} upserted  (total: {db.count('ideas')})")


def ingest_saturated(db: VectorDB) -> None:
    docs = _load("saturated_history.json")
    if not isinstance(docs, list) or not docs:
        return
    print(f"\n[saturated] {len(docs)} records")

    def _text(doc):
        return " | ".join(filter(None, [doc.get("title",""), doc.get("area",""), doc.get("concept","")]))

    n = db.bulk_ingest(
        "saturated", docs,
        text_fn=_text,
        id_fn=lambda doc, i: _safe_id(_text(doc), f"sat_{i}"),
        meta_fn=_idea_meta,
    )
    print(f"[saturated] ✓ {n} upserted  (total: {db.count('saturated')})")


def ingest_blacklist(db: VectorDB) -> None:
    docs = _load("blacklist.json")
    if not isinstance(docs, list) or not docs:
        return
    print(f"\n[blacklist] {len(docs)} records")
    n = db.bulk_ingest(
        "blacklist", docs,
        text_fn=_idea_text,
        id_fn=lambda doc, i: _safe_id(_idea_text(doc), f"bl_{i}"),
        meta_fn=_idea_meta,
    )
    print(f"[blacklist] ✓ {n} upserted  (total: {db.count('blacklist')})")


def ingest_bookmarks(db: VectorDB) -> None:
    docs = _load("bookmarks.json")
    if not isinstance(docs, list) or not docs:
        return
    print(f"\n[bookmarks] {len(docs)} records")
    n = db.bulk_ingest(
        "bookmarks", docs,
        text_fn=_idea_text,
        id_fn=lambda doc, i: _safe_id(_idea_text(doc), f"bm_{i}"),
        meta_fn=_idea_meta,
    )
    print(f"[bookmarks] ✓ {n} upserted  (total: {db.count('bookmarks')})")


def ingest_favorites(db: VectorDB) -> None:
    docs = _load("favorites.json")
    if not isinstance(docs, list) or not docs:
        return
    print(f"\n[favorites] {len(docs)} records")
    n = db.bulk_ingest(
        "favorites", docs,
        text_fn=_idea_text,
        id_fn=lambda doc, i: _safe_id(_idea_text(doc), f"fav_{i}"),
        meta_fn=_idea_meta,
    )
    print(f"[favorites] ✓ {n} upserted  (total: {db.count('favorites')})")


def ingest_channel(db: VectorDB) -> None:
    raw = _load("channel_analysis.json")
    if not isinstance(raw, dict):
        return
    shorts = raw.get("all_shorts", [])
    print(f"\n[channel] {len(shorts)} shorts")
    if shorts:
        n = db.bulk_ingest(
            "channel", shorts,
            text_fn=_channel_short_text,
            id_fn=lambda doc, i: _safe_id(doc.get("id", doc.get("title", "")), f"ch_{i}"),
            meta_fn=_channel_meta,
        )
        print(f"[channel] ✓ {n} upserted  (total: {db.count('channel')})")
    # Store the full channel analysis (without shorts) as a config blob
    meta = {k: v for k, v in raw.items() if k != "all_shorts"}
    db.save_config("channel_analysis", meta)
    print("[channel] ✓ channel_analysis config saved")


def ingest_audience_insights(db: VectorDB) -> None:
    raw = _load("audience_insights.json")
    if not isinstance(raw, dict) or not raw:
        return
    db.save_config("audience_insights", raw)
    print("\n[audience_insights] ✓ config saved")


def ingest_channel_performance(db: VectorDB) -> None:
    raw = _load("channel_performance.json")
    if not isinstance(raw, dict) or not raw:
        return
    db.save_config("channel_performance", raw)
    print("\n[channel_performance] ✓ config saved")


def main() -> None:
    print("=" * 60)
    print("  VectorDB ingestion — ChromaDB + all-MiniLM-L6-v2")
    print(f"  DB: {ROOT / 'vector_db'}")
    print("=" * 60)

    db = VectorDB()

    ingest_ideas(db)
    ingest_saturated(db)
    ingest_blacklist(db)
    ingest_bookmarks(db)
    ingest_favorites(db)
    ingest_channel(db)
    ingest_audience_insights(db)
    ingest_channel_performance(db)

    print("\n" + "=" * 60)
    print("  Final collection sizes:")
    for name, count in db.info().items():
        print(f"  {'✓' if count else '○'}  {name:<14} {count:>5} docs")
    print("=" * 60)
    print("  Done — all data in vector_db/, JSON files no longer needed.")


if __name__ == "__main__":
    main()
