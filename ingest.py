"""
ingest.py — Sprint 1
Loads product data, generates embeddings, and stores in FAISS vector store.
Run this ONCE before starting the chatbot: python ingest.py
"""

import json
import os
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/products.json"
VECTORSTORE_PATH = "vectorstore/"


def load_products(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def product_to_document(product: dict) -> Document:
    """Convert a product dict into a rich text chunk for embedding."""
    text = f"""
Product Name: {product['name']}
Category: {product['category']}
Brand: {product['brand']}
Price: {product['price']}
Description: {product['description']}
Features: {product['features']}
Warranty: {product['warranty']}
Return Policy: {product['return_policy']}
Stock Status: {product['stock']}
""".strip()

    return Document(
        page_content=text,
        metadata={
            "id": product["id"],
            "name": product["name"],
            "category": product["category"],
            "price": product["price"],
        },
    )


def build_vectorstore():
    print("📦 Loading product data...")
    products = load_products(DATA_PATH)
    print(f"   Found {len(products)} products.")

    print("\n📝 Converting to documents...")
    documents = [product_to_document(p) for p in products]

    print("\n🔢 Generating embeddings (this may take a minute on first run)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 4},
    )

    print("\n💾 Storing in FAISS vector store...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

    print(f"\n✅ Done! Vector store saved to '{VECTORSTORE_PATH}'")
    print(f"   {len(documents)} product chunks indexed and ready.")


if __name__ == "__main__":
    build_vectorstore()
