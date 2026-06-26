from data.synthetic.embedder import FakeEmbedder


def test_fake_embedder_returns_mapped_vectors():
    e = FakeEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    assert e.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
