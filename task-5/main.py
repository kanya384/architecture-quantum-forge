from safety import check_safety
import requests
import numpy as np
from search import VectorSearchEngine

search_engine = VectorSearchEngine(
    index_file='../task-3/data/faiss_index.bin',
    metadata_file='../task-3/data/faiss_metadata.pkl'
)
search_engine.load()

def safe_search(query, k=100):
    """Безопасный поиск с правильным энкодингом"""
    try:
        return
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

def ask_ollama(prompt, model="llama3.1"):
    """
    Отправляет запрос к Ollama с обработкой ошибок
    """
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "temperature": 0.7,
        "options": {
            "num_ctx": 4096
        }
    }

    try:
        print(f"📤 Отправка запроса к модели {model}...")
        response = requests.post(url, json=payload, timeout=60)

        if response.status_code != 200:
            print(f"❌ HTTP ошибка: {response.status_code}")
            print(f"Ответ сервера: {response.text[:500]}")
            return f"Ошибка API: {response.status_code}"

        try:
            data = response.json()

            # Проверяем наличие ключа 'response'
            if 'response' in data:
                return data['response']
            elif 'error' in data:
                return f"Ошибка модели: {data['error']}"
            else:
                print(f"⚠️ Неожиданный формат ответа: {json.dumps(data, indent=2)[:500]}")
                return f"Неизвестный формат ответа. Ключи: {list(data.keys())}"

        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга JSON: {e}")
            print(f"Сырой ответ: {response.text[:500]}")
            return f"Ошибка парсинга: {e}"

    except requests.exceptions.Timeout:
        return "⏱️ Превышено время ожидания"
    except requests.exceptions.ConnectionError:
        return "🔌 Нет соединения с Ollama. Проверьте, запущен ли контейнер."
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        return f"Ошибка: {e}"

def rag_query(question):
    if not check_safety(question, role="user"):
           return "❌ Ваш запрос не прошел проверку безопасности."
    chunks = search_engine.search(question, k=10)

    if not chunks:
        return "Не найдено документов."

    prompt = build_prompt(question, chunks)
    answer = ask_ollama(prompt)
    if not check_safety(answer, role="assistant"):
            return "⚠️ Сгенерированный ответ не прошел проверку безопасности."

    return answer

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