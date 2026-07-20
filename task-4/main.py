
import requests
import numpy as np
from search import VectorSearchEngine

search_engine = VectorSearchEngine(
    index_file='../task-3/data/faiss_index.bin',
    metadata_file='../task-3/data/faiss_metadata.pkl'
)
search_engine.load()

# ФИКС: переопределяем метод поиска для совместимости
def safe_search(query, k=10):
    """Безопасный поиск с правильным энкодингом"""
    try:
        return search_engine.search(query, k=k)
    except TypeError:
        print("⚠️ Использую альтернативный метод поиска...")

        query_embedding = search_engine.model.encode(
            query,  # строка, не список
            convert_to_numpy=True,
            normalize_embeddings=True
        ).reshape(1, -1).astype(np.float32)

        distances, indices = search_engine.index.search(query_embedding, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(search_engine.metadata['texts']):
                continue

            distance = distances[0][i]
            results.append({
                'text': search_engine.metadata['texts'][idx],
                'metadata': search_engine.metadata['metadata'][idx],
                'distance': float(distance),
                'score': 1.0 / (1.0 + distance),
                'index': int(idx)
            })

        return results

FEW_SHOT = [
    ("Как называется столица планеты Ти'лора?",
     "Столица планеты Ти'лора называется Сайрон."),
    ("Кто был учителем Ами Шани?",
     "Учителем Ами Шани был мастер Зен.")
]

def build_prompt(question, chunks):
    context = "\n\n".join([f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)])

    few_shot = ""
    for q, a in FEW_SHOT:
        few_shot += f"Q: {q}\nA: {a}\n\n"

    return f"""System: Ты помощник, который сначала размышляет, а потом отвечает. Всегда пиши свои шаги. Отвечай на русском.

    Твоя задача - ответить на вопрос, используя ТОЛЬКО информацию из документов.

    Правила:
    1. Сначала проанализируй документы
    2. Найди релевантную информацию
    3. Сформулируй ответ
    4. Если информации нет - скажи об этом

    Примеры правильных ответов:
    {few_shot}

    Документы:
    {context}

    Вопрос: {question}

    Пожалуйста, объясни свои шаги и дай ответ.
    Ответ:"""

def ask_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "temperature": 0.7
        }
    )
    return response.json()['response']

def rag_query(question):
    # Используем безопасный поиск
    chunks = safe_search(question, k=5)

    if not chunks:
        return "Не найдено документов."

    prompt = build_prompt(question, chunks)
    return ask_ollama(prompt)

# Запуск
print("\n🤖 RAG Чат-бот запущен!")
print("Введите 'exit' для выхода\n")

while True:
    q = input("❓ Вопрос: ")

    if q.lower() in ['exit', 'quit', 'q']:
        print("👋 До свидания!")
        break

    if not q.strip():
        continue

    print("⏳ Думаю...")
    answer = rag_query(q)
    print(f"💬 Ответ: {answer}\n")