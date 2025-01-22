from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
import time

folder_path = "db"
embedding = FastEmbedEmbeddings()

vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs=dict(k=10, score_threshold=0.4),
)


def rag_query(query):
    start_time = time.time()
    docs = retriever.get_relevant_documents(query)
    if not docs:
        print("No relevant documents found.")
    else:
        print("\nRetrieved Documents:")
        res = ""
        for i, doc in enumerate(docs):
            res += f"Document {i + 1}:\n"
            res += doc.page_content+"\n"
            res += "\n"
    print(f"Execution time: {time.time() - start_time:.2f} seconds")
    return res

