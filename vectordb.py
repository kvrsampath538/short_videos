"""
Local vector database using ChromaDB + sentence-transformers.

Collections stored on disk at  <project_root>/vector_db/
- ideas        — all generated ideas  (replaces ideas_history.json)
- saturated    — rejected/saturated   (replaces saturated_history.json)
- blacklist    — blacklisted ideas    (replaces blacklist.json)
- bookmarks    — bookmarked ideas     (replaces bookmarks.json)
- favorites    — favourited ideas     (replaces favorites.json)
- channel      — channel shorts       (replaces channel_analysis all_shorts)
- app_config   — singleton JSON blobs (audience_insights, channel_analysis meta,
                                       channel_performance)

Every idea document stores its full JSON in a `_json` metadata field so
get_all() can reconstruct the original dict without loss.

Typical usage
-------------
from vectordb import get_db

db = get_db()                                       # singleton — safe to call anywhere
db.upsert_idea(idea_dict)                           # add/update in ideas collection
db.upsert_idea(idea_dict, collection="bookmarks")   # add to bookmarks
results   = db.search("nuclear reactor water", n=5)
similar   = db.find_similar_ideas(new_idea, threshold=0.25)
is_dup, m = db.is_semantically_duplicate(new_idea)
all_ideas = db.get_all("ideas")
db.delete_by_title("bookmarks", "Some Title")
db.save_config("audience_insights", {...})
data      = db.load_config("audience_insights")
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

_ROOT     = Path(__file__).parent
_DB_DIR   = _ROOT / "vector_db"
_MODEL_ID = "all-MiniLM-L6-v2"   # 384-dim, fast, good quality for short texts

COLLECTIONS = ["ideas", "saturated", "blacklist", "bookmarks", "favorites", "channel"]

# ── Singleton model ───────────────────────────────────────────────────────────

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[VectorDB] Loading embedding model '{_MODEL_ID}'…")
        _model = SentenceTransformer(_MODEL_ID)
        print("[VectorDB] Model ready.")
    return _model


def _embed(texts: list[str]) -> list[list[float]]:
    return _get_model().encode(texts, show_progress_bar=False).tolist()


# ── Singleton DB ──────────────────────────────────────────────────────────────

_vdb: "VectorDB | None" = None


def get_db() -> "VectorDB":
    """Return the process-level VectorDB singleton. Thread-safe for reads."""
    global _vdb
    if _vdb is None:
        _vdb = VectorDB()
    return _vdb


# ── Text / ID / metadata helpers ──────────────────────────────────────────────

def _idea_text(doc: dict) -> str:
    parts = [
        doc.get("title", ""),
        doc.get("area", ""),
        doc.get("concept", ""),
        doc.get("viral_hook", ""),
    ]
    return " | ".join(p for p in parts if p).strip()


def _channel_short_text(doc: dict) -> str:
    parts = [
        doc.get("title", ""),
        doc.get("area", ""),
        doc.get("description", ""),
    ]
    return " | ".join(p for p in parts if p).strip()


def _safe_id(text: str, fallback: str) -> str:
    """
    Stable, unique ChromaDB document id.
    Uses first 60 clean chars of text + 8-char SHA1 to guarantee uniqueness
    even when two titles share the same prefix.
    Re-hashing the same full text always produces the same id → idempotent upserts.
    """
    cleaned = re.sub(r"[^\w\-]", "_", text)[:60].strip("_")
    h = hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:8]
    base = cleaned or fallback
    return f"{base}_{h}"


def _idea_meta(idea: dict) -> dict:
    """Build ChromaDB metadata for an idea, including the full JSON for reconstruction."""
    return {
        "title":        idea.get("title", "")[:500],
        "area":         idea.get("area", ""),
        "viral_hook":   idea.get("viral_hook", "")[:500],
        "shock_score":  str(idea.get("shock_score", "")),
        "pattern_used": idea.get("pattern_used", ""),
        "generated_at": idea.get("generated_at", ""),
        "_json":        json.dumps(idea, ensure_ascii=False),
    }


def _channel_meta(doc: dict) -> dict:
    return {
        "title":        doc.get("title", "")[:500],
        "area":         doc.get("area", ""),
        "view_count":   str(doc.get("view_count", 0)),
        "like_count":   str(doc.get("like_count", 0)),
        "published_at": doc.get("published_at", ""),
        "url":          doc.get("url", ""),
        "_json":        json.dumps(doc, ensure_ascii=False),
    }


# ── VectorDB class ─────────────────────────────────────────────────────────────

class VectorDB:
    """Persistent ChromaDB-backed store for all app data (no JSON files needed)."""

    def __init__(self, db_dir: str | Path = _DB_DIR):
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(db_dir),
            settings=Settings(anonymized_telemetry=False),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _col(self, name: str) -> chromadb.Collection:
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def _upsert_batch(
        self,
        collection: str,
        ids: list[str],
        texts: list[str],
        metadatas: list[dict],
    ) -> None:
        if not ids:
            return
        embeddings = _embed(texts)
        self._col(collection).upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    # ── Count / info ──────────────────────────────────────────────────────────

    def count(self, collection: str) -> int:
        return self._col(collection).count()

    def info(self) -> dict[str, int]:
        return {c: self.count(c) for c in COLLECTIONS}

    # ── Upsert ────────────────────────────────────────────────────────────────

    def upsert_idea(self, idea: dict, collection: str = "ideas") -> None:
        """Upsert a single idea dict into the given collection."""
        title = idea.get("title", "").strip()
        if not title:
            return
        doc_id = _safe_id(_idea_text(idea), f"idea_{hash(title)}")
        self._upsert_batch(collection, [doc_id], [_idea_text(idea)], [_idea_meta(idea)])

    def upsert_channel_short(self, doc: dict) -> None:
        """Upsert a single channel short into the 'channel' collection."""
        vid_id = doc.get("id", doc.get("title", ""))
        doc_id = _safe_id(vid_id, f"ch_{hash(vid_id)}")
        self._upsert_batch("channel", [doc_id], [_channel_short_text(doc)], [_channel_meta(doc)])

    # ── Read all ──────────────────────────────────────────────────────────────

    def get_all(self, collection: str, include_ids: bool = False) -> list[dict]:
        """
        Return all documents in a collection as a list of dicts.
        Reconstructs the full original dict from the stored `_json` metadata field.
        """
        col = self._col(collection)
        if col.count() == 0:
            return []
        result = col.get(include=["metadatas"])  # ids are always returned automatically
        docs = []
        for i, meta in enumerate(result.get("metadatas", [])):
            raw_json = meta.get("_json", "")
            if raw_json:
                try:
                    doc = json.loads(raw_json)
                    if include_ids:
                        doc["_chroma_id"] = result["ids"][i]
                    docs.append(doc)
                    continue
                except Exception:
                    pass
            # Fallback: reconstruct from flat metadata fields
            doc = {k: v for k, v in meta.items() if not k.startswith("_")}
            if include_ids:
                doc["_chroma_id"] = result["ids"][i]
            docs.append(doc)
        return docs

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete_doc(self, collection: str, doc_id: str) -> None:
        """Delete a document by its ChromaDB id."""
        try:
            self._col(collection).delete(ids=[doc_id])
        except Exception:
            pass

    def delete_by_title(self, collection: str, title: str) -> None:
        """Find and delete all documents in a collection whose title matches exactly."""
        col = self._col(collection)
        if col.count() == 0:
            return
        try:
            # ids are always returned; don't put them in include
            result = col.get(where={"title": {"$eq": title[:500]}})
            ids = result.get("ids", [])
            if ids:
                col.delete(ids=ids)
        except Exception:
            pass

    def delete_all(self, collection: str) -> None:
        """Delete all documents in a collection (by re-creating it)."""
        try:
            self._client.delete_collection(collection)
        except Exception:
            pass

    # ── Search & similarity ───────────────────────────────────────────────────

    def search(
        self,
        query: str,
        collection: str = "ideas",
        n: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Semantic search. Returns list of dicts:
          {id, document, distance, metadata, full_doc (if _json present)}
        """
        col = self._col(collection)
        if col.count() == 0:
            return []
        embedding = _embed([query])[0]
        kwargs: dict[str, Any] = dict(
            query_embeddings=[embedding],
            n_results=min(n, col.count()),
            include=["documents", "distances", "metadatas"],
        )
        if where:
            kwargs["where"] = where
        result = col.query(**kwargs)
        hits = []
        for i, doc_id in enumerate(result["ids"][0]):
            meta = result["metadatas"][0][i]
            raw_json = meta.get("_json", "")
            full_doc = None
            if raw_json:
                try:
                    full_doc = json.loads(raw_json)
                except Exception:
                    pass
            hits.append({
                "id":       doc_id,
                "document": result["documents"][0][i],
                "distance": round(result["distances"][0][i], 4),
                "metadata": meta,
                "full_doc": full_doc,
            })
        return hits

    def find_similar_ideas(
        self,
        idea: dict,
        collection: str = "ideas",
        n: int = 5,
        threshold: float = 0.25,
    ) -> list[dict]:
        """Find the n most semantically similar ideas within cosine-distance threshold."""
        hits = self.search(_idea_text(idea), collection=collection, n=n)
        return [h for h in hits if h["distance"] <= threshold]

    def is_semantically_duplicate(
        self,
        idea: dict,
        collection: str = "ideas",
        threshold: float = 0.12,
    ) -> tuple[bool, str]:
        """
        Returns (is_duplicate, matched_title).
        Tight threshold — only very close matches flag as duplicates.
        0.0 = identical, 1.0 = completely unrelated.
        """
        hits = self.find_similar_ideas(idea, collection=collection, n=1, threshold=threshold)
        if hits:
            matched = hits[0]["metadata"].get("title", hits[0]["id"])
            return True, matched
        return False, ""

    # ── Config blobs (audience_insights, channel_analysis, channel_performance) ─

    def save_config(self, key: str, data: dict) -> None:
        """
        Store a single JSON document under a key in the app_config collection.
        Replaces audience_insights.json, channel_analysis.json, channel_performance.json.
        """
        col = self._col("app_config")
        payload = json.dumps(data, ensure_ascii=False)
        metadata = {"key": key, "_json": payload}
        embedding = _embed([key])[0]
        col.upsert(ids=[key], embeddings=[embedding], documents=[key], metadatas=[metadata])

    def load_config(self, key: str) -> dict | None:
        """Retrieve a stored config document by key. Returns None if not found."""
        col = self._col("app_config")
        try:
            result = col.get(ids=[key], include=["metadatas"])
            metas = result.get("metadatas") or []
            if metas:
                raw = metas[0].get("_json", "")
                if raw:
                    return json.loads(raw)
        except Exception:
            pass
        return None

    # ── Bulk ingestion (used by ingest script) ────────────────────────────────

    def bulk_ingest(
        self,
        collection: str,
        docs: list[dict],
        text_fn=None,
        id_fn=None,
        meta_fn=None,
        batch_size: int = 128,
        show_progress: bool = True,
    ) -> int:
        """
        Ingest a list of dicts in batches. Returns number of documents upserted.
        text_fn(doc)     -> str  — text to embed (defaults to _idea_text)
        id_fn(doc, i)    -> str  — unique document id
        meta_fn(doc)     -> dict — metadata (defaults to _idea_meta which includes _json)
        """
        if text_fn is None:
            text_fn = _idea_text
        if id_fn is None:
            id_fn = lambda doc, i: _safe_id(text_fn(doc), f"doc_{i}")
        if meta_fn is None:
            meta_fn = _idea_meta  # includes _json for full reconstruction

        total = 0
        for start in range(0, len(docs), batch_size):
            chunk = docs[start: start + batch_size]
            ids, texts, metas = [], [], []
            seen_ids: set[str] = set()
            for i, doc in enumerate(chunk):
                text = text_fn(doc)
                if not text.strip():
                    continue
                doc_id = id_fn(doc, start + i)
                if doc_id in seen_ids:
                    # Collision within batch: append position to break tie
                    doc_id = f"{doc_id}_{start + i}"
                seen_ids.add(doc_id)
                ids.append(doc_id)
                texts.append(text)
                metas.append(meta_fn(doc))
            self._upsert_batch(collection, ids, texts, metas)
            total += len(ids)
            if show_progress:
                pct = min(start + batch_size, len(docs))
                print(f"  [{collection}] {pct}/{len(docs)} docs embedded…", end="\r")
        if show_progress:
            print(f"  [{collection}] {total} docs embedded.      ")
        return total
