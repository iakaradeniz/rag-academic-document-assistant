from app.services.chunker import PageChunker


def test_chunking_produces_chunks():
    docs = [
        {
            "text": "A" * 2000,
            "metadata": {"source": "test.pdf", "page": 1},
        }
    ]

    chunker = PageChunker(chunk_size=500, overlap=0.2)
    chunks = chunker.chunk(docs)

    assert len(chunks) > 1
    assert chunks[0]["metadata"]["page"] == 1
