"""RAG retrieval tests with a hashing embedder (no model download, so CI stays fast and offline)."""
import pytest

from shipsense.rag import HashingEmbedder, PolicyIndex, chunk_markdown, load_knowledge


@pytest.fixture(scope="module")
def index():
    return PolicyIndex.build(load_knowledge(), HashingEmbedder())


def test_chunker_splits_by_section_and_skips_banner(tmp_path):
    doc = tmp_path / "x.md"
    doc.write_text("# Title\n\n> **FICTIONAL DOCUMENT.** banner\n\n## Alpha\none two\n\n## Beta\nthree four\n")
    chunks = chunk_markdown(doc)
    assert [c.section for c in chunks] == ["Alpha", "Beta"]
    assert all("FICTIONAL" not in c.text for c in chunks)
    assert chunks[0].citation == "x.md § Alpha"


def test_long_section_becomes_overlapping_windows(tmp_path):
    doc = tmp_path / "long.md"
    doc.write_text("# T\n## S\n" + " ".join(f"w{i}" for i in range(250)))
    chunks = chunk_markdown(doc, max_words=100, overlap=20)
    assert [len(c.text.split()) for c in chunks] == [100, 100, 90]
    assert chunks[0].text.split()[-20:] == chunks[1].text.split()[:20]  # overlap carries context


def test_knowledge_base_loads_all_docs_but_not_readme():
    docs = {c.doc for c in load_knowledge()}
    assert len(docs) == 11
    assert "README.md" not in docs


@pytest.mark.parametrize("query, expected_doc", [
    ("no carrier scan for 10 business days lost shipment", "09_damaged_and_lost_shipments.md"),
    ("peak season November 15 January 10", "10_peak_season_and_weather.md"),
    ("certificate of origin commercial invoice HS code", "04_customs_and_international_delays.md"),
])
def test_search_finds_the_right_document(index, query, expected_doc):
    hits = index.search(query, k=3)
    assert hits[0].chunk.doc == expected_doc
    assert hits[0].score >= hits[-1].score  # sorted best-first


def test_index_save_and_load_roundtrip(index, tmp_path):
    index.save(tmp_path)
    loaded = PolicyIndex.load(HashingEmbedder(), tmp_path)
    q = "who approves an upgrade to Same Day"
    assert [h.chunk.citation for h in loaded.search(q)] == [h.chunk.citation for h in index.search(q)]
