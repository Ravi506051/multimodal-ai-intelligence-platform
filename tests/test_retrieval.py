from src.retrieval import LocalRetriever

def test_retriever_returns_relevant_source():
    retriever = LocalRetriever()
    retriever.add_documents([
        ("climate.txt", "Climate change increases average global temperatures and affects weather."),
        ("cooking.txt", "A recipe for pasta uses tomatoes, garlic, and olive oil."),
    ])
    results = retriever.search("global temperatures climate", top_k=2)
    assert results
    assert results[0]["source"] == "climate.txt"

def test_empty_query_returns_no_results():
    retriever = LocalRetriever()
    retriever.add_documents([("notes.txt", "Some useful notes about machine learning.")])
    assert retriever.search("") == []
