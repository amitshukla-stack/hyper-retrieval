"""
Stage 7 — Chunk euler-docs and euler-documentation markdown into a searchable
documentation collection.

Input:   euler-docs/*.md, euler-documentation/*.md
         pipeline/output/public_docs.json  (if exists — optional public API docs)
Output:  pipeline/output/doc_chunks.json   (raw chunks with metadata)
         pipeline/output/docs.lance        (embedded chunks, LanceDB collection)

Chunking strategy:
  - Split each markdown file by H2 / H3 headings
  - Each chunk: heading + content, ~500 tokens max
  - Metadata: source_file, section_title, tags (auto-extracted)
  - Embed with the same Qwen3-Embedding-8B model
"""
import re, json, pathlib, sys, os, time
import numpy as np

REPO_ROOT  = pathlib.Path(os.environ.get("REPO_ROOT", pathlib.Path(__file__).parent.parent / "repo"))
OUT_DIR    = pathlib.Path(__file__).parent / "output"
DOCS_DIRS  = [
    pathlib.Path(__file__).parent.parent / "euler-docs",
    pathlib.Path(__file__).parent.parent / "euler-documentation",
]
PUBLIC_DOCS_JSON = OUT_DIR / "public_docs.json"
OUT_CHUNKS = OUT_DIR / "doc_chunks.json"
OUT_LANCE  = OUT_DIR / "docs.lance"
BATCH_SIZE = 32
MAX_CHUNK_CHARS = 3000

# ── Tagging ──────────────────────────────────────────────────────────────────

TAG_PATTERNS = {
    "database":    re.compile(r'\b(?:beam|sql|query|table|schema|migration|db|postgres|mysql)\b', re.I),
    "caching":     re.compile(r'\b(?:cache|redis|kvstore|hedis|cacheable)\b', re.I),
    "api":         re.compile(r'\b(?:api|endpoint|route|handler|request|response|http)\b', re.I),
    "payment":     re.compile(r'\b(?:payment|order|txn|transaction|gateway|connector|upi|card|wallet)\b', re.I),
    "types":       re.compile(r'\b(?:type|newtype|data|record|sum type|algebraic)\b', re.I),
    "error":       re.compile(r'\b(?:error|exception|failure|either|maybe|result)\b', re.I),
    "testing":     re.compile(r'\b(?:test|spec|hspec|quickcheck|mock|stub)\b', re.I),
    "logging":     re.compile(r'\b(?:log|logger|trace|span|metric|observability)\b', re.I),
    "style":       re.compile(r'\b(?:style|convention|pattern|naming|format)\b', re.I),
    "integration": re.compile(r'\b(?:integrat|sdk|webhook|callback|notification)\b', re.I),
}

def auto_tag(text: str) -> list[str]:
    return [tag for tag, pat in TAG_PATTERNS.items() if pat.search(text)]


# ── Markdown chunker ─────────────────────────────────────────────────────────

def chunk_markdown(path: pathlib.Path, source_label: str) -> list[dict]:
    """
    Split a markdown file into chunks at each H2 or H3 boundary.
    Returns list of chunk dicts with metadata.
    """
    src = path.read_text(encoding="utf-8", errors="replace")
    chunks = []

    # Split on H2/H3 headings
    sections = re.split(r'\n(?=#{2,3}\s)', src)

    # First section (before any H2/H3) is the file preamble
    preamble = sections[0].strip()
    file_title = ""
    m = re.match(r'^#\s+(.+)', preamble)
    if m:
        file_title = m.group(1).strip()

    if len(preamble) > 100:
        chunks.append({
            "id":            f"{source_label}::intro",
            "source_file":   source_label,
            "section_title": file_title or path.stem,
            "heading_level": 1,
            "text":          preamble[:MAX_CHUNK_CHARS],
            "chars":         len(preamble),
            "tags":          auto_tag(preamble),
        })

    for section in sections[1:]:
        # Parse heading
        hm = re.match(r'^(#{2,3})\s+(.+)', section)
        if not hm:
            continue
        level = len(hm.group(1))
        title = hm.group(2).strip()
        content = section.strip()

        # Further split on H3 if section is too long
        if level == 2 and len(content) > MAX_CHUNK_CHARS:
            subsections = re.split(r'\n(?=###\s)', content)
            for sub in subsections:
                shm = re.match(r'^(#{2,3})\s+(.+)', sub)
                sub_title = shm.group(2).strip() if shm else title
                sub_level = len(shm.group(1)) if shm else level
                text = sub.strip()
                if len(text) < 80:
                    continue
                slug = re.sub(r'[^a-z0-9]+', '_', sub_title.lower())[:40]
                chunks.append({
                    "id":            f"{source_label}::{slug}",
                    "source_file":   source_label,
                    "section_title": sub_title,
                    "heading_level": sub_level,
                    "text":          text[:MAX_CHUNK_CHARS],
                    "chars":         len(text),
                    "tags":          auto_tag(text),
                })
        else:
            if len(content) < 80:
                continue
            slug = re.sub(r'[^a-z0-9]+', '_', title.lower())[:40]
            chunks.append({
                "id":            f"{source_label}::{slug}",
                "source_file":   source_label,
                "section_title": title,
                "heading_level": level,
                "text":          content[:MAX_CHUNK_CHARS],
                "chars":         len(content),
                "tags":          auto_tag(content),
            })

    return chunks


# ── Public docs chunker ──────────────────────────────────────────────────────

def load_public_docs() -> list[dict]:
    if not PUBLIC_DOCS_JSON.exists():
        print(f"  [public_docs] Not found: {PUBLIC_DOCS_JSON} — skipping")
        return []

    data = json.loads(PUBLIC_DOCS_JSON.read_text())
    chunks = []
    for page in data.get("pages", []):
        url     = page["url"]
        title   = page["title"]
        md      = page["markdown"]
        section = page["section"]

        # Treat each page as one chunk; split further if very long
        if len(md) <= MAX_CHUNK_CHARS:
            slug = re.sub(r'[^a-z0-9]+', '_', title.lower())[:40]
            chunks.append({
                "id":            f"public_docs::{slug}",
                "source_file":   "public_docs",
                "section_title": title,
                "heading_level": 1,
                "text":          md,
                "chars":         len(md),
                "tags":          auto_tag(md) + [section],
                "url":           url,
            })
        else:
            # Split by H2/H3 within the page markdown
            sub_chunks = re.split(r'\n(?=#{2,3}\s)', md)
            for i, sub in enumerate(sub_chunks):
                if len(sub) < 80:
                    continue
                hm = re.match(r'^(#{2,3})\s+(.+)', sub)
                sub_title = hm.group(2).strip() if hm else f"{title} ({i})"
                slug = re.sub(r'[^a-z0-9]+', '_', sub_title.lower())[:40]
                chunks.append({
                    "id":            f"public_docs::{slug}_{i}",
                    "source_file":   "public_docs",
                    "section_title": sub_title,
                    "heading_level": len(hm.group(1)) if hm else 2,
                    "text":          sub[:MAX_CHUNK_CHARS],
                    "chars":         len(sub),
                    "tags":          auto_tag(sub) + [section],
                    "url":           url,
                })

    print(f"  [public_docs] {len(data.get('pages',[]))} pages → {len(chunks)} chunks")
    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict], model) -> np.ndarray:
    texts = [
        f"passage: {c['section_title']}\n{c['text'][:1500]}"
        for c in chunks
    ]
    all_vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        vecs = model.encode(batch, normalize_embeddings=True, batch_size=BATCH_SIZE)
        all_vecs.append(vecs)
        print(f"  Embedded {min(i+BATCH_SIZE, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.vstack(all_vecs).astype("float32")


# ── Write LanceDB ─────────────────────────────────────────────────────────────

def write_lance(chunks: list[dict], vecs: np.ndarray):
    import lancedb
    import pyarrow as pa

    db = lancedb.connect(str(OUT_DIR))
    rows = []
    for chunk, vec in zip(chunks, vecs):
        rows.append({
            "id":            chunk["id"],
            "source_file":   chunk["source_file"],
            "section_title": chunk["section_title"],
            "heading_level": chunk.get("heading_level", 2),
            "text":          chunk["text"],
            "tags":          ",".join(chunk.get("tags", [])),
            "url":           chunk.get("url", ""),
            "vector":        vec.tolist(),
        })

    schema = pa.schema([
        pa.field("id",            pa.string()),
        pa.field("source_file",   pa.string()),
        pa.field("section_title", pa.string()),
        pa.field("heading_level", pa.int32()),
        pa.field("text",          pa.string()),
        pa.field("tags",          pa.string()),
        pa.field("url",           pa.string()),
        pa.field("vector",        pa.list_(pa.float32(), 4096)),
    ])

    tbl = db.create_table("docs", data=rows, schema=schema, mode="overwrite")
    print(f"✓ docs.lance: {len(rows)} chunks at 4096d")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    all_chunks = []

    # 1. Internal euler-docs
    for docs_dir in DOCS_DIRS:
        if not docs_dir.exists():
            print(f"[skip] {docs_dir} not found")
            continue
        md_files = list(docs_dir.glob("*.md"))
        print(f"\n{docs_dir.name}: {len(md_files)} markdown files")
        for f in sorted(md_files):
            label = f"{docs_dir.name}/{f.name}"
            chunks = chunk_markdown(f, label)
            print(f"  {f.name}: {len(chunks)} chunks")
            all_chunks.extend(chunks)

    # 2. Public docs (if available)
    print("\nPublic docs:")
    all_chunks.extend(load_public_docs())

    # Deduplicate by id
    seen = set()
    deduped = []
    for c in all_chunks:
        if c["id"] not in seen:
            seen.add(c["id"])
            deduped.append(c)
    all_chunks = deduped

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Save raw chunks
    OUT_CHUNKS.write_text(json.dumps(all_chunks, indent=2))
    print(f"✓ doc_chunks.json: {len(all_chunks)} chunks")

    # Embed
    print("\nLoading embedding model...")
    sys.path.insert(0, str(pathlib.Path(__file__).parent / "models"))
    from sentence_transformers import SentenceTransformer
    model_path = pathlib.Path(__file__).parent / "models" / "Qwen3-Embedding-8B"
    if not model_path.exists():
        # fallback to HF hub name
        model_path = "Qwen/Qwen3-Embedding-8B"
    model = SentenceTransformer(str(model_path), device="cuda", trust_remote_code=True)
    print(f"Model loaded")

    t0 = time.time()
    vecs = embed_chunks(all_chunks, model)
    print(f"Embedding done in {time.time()-t0:.1f}s")

    # Write LanceDB
    write_lance(all_chunks, vecs)


if __name__ == "__main__":
    main()
