from split_into_chunks import load_and_chunk_folder
from embeddings import load_chunks_from_json, generate_embeddings_from_json
from faiss_index import create_faiss_index
import json

knowledge_base = "../task-2/knowledge_base"

documents = load_and_chunk_folder(knowledge_base)

chunks_data = []
for doc in documents:
    chunks_data.append({
        "content": doc.page_content,
        "metadata": doc.metadata
    })

with open('./data/chunks.json', 'w', encoding='utf-8') as f:
    json.dump(chunks_data, f, ensure_ascii=False, indent=2)

print(f"✅ Чанки сохранены в chunks.json")

result = generate_embeddings_from_json(
    json_file='./data/chunks.json',
    model_name='all-MiniLM-L6-v2',
    output_file='./data/embeddings_data.pkl'
)

index, metadata = create_faiss_index(
    embeddings_file='./data/embeddings_data.pkl',
    index_type='Flat',  # или 'IVF', 'HNSW'
    output_index='./data/faiss_index.bin',
    output_metadata='./data/faiss_metadata.pkl'
)