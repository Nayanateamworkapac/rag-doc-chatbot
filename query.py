from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

while True:
    query = input("\nAsk a question (type exit to quit): ")

    if query.lower() == "exit":
        break

    THRESHOLD = 1.2

    docs = vectorstore.similarity_search_with_score(query, k=1)

    if docs:
        doc, score = docs[0]
        print(f"\n🔍 Similarity score: {score}")

        if score <= THRESHOLD:
            print("\n📌 Answer:\n")
            print(doc.page_content)
        else:
            print("\n❌ Sorry, I couldn’t find a relevant answer.")
    else:
        print("❌ No relevant information found")

