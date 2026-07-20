import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from datetime import datetime

def load_chunks_from_json(json_file='chunks.json'):
    """
    Загружает чанки из JSON файла
    """
    print(f"📂 Загрузка чанков из {json_file}...")

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Проверяем структуру JSON
    if isinstance(data, list):
        # Если это список чанков
        chunks = data
    elif isinstance(data, dict) and 'chunks' in data:
        # Если это объект с полем chunks
        chunks = data['chunks']
    else:
        # Пробуем другие варианты
        chunks = data

    print(f"✅ Загружено {len(chunks)} чанков")

    # Показываем пример
    if chunks:
        print(f"\n📝 Пример первого чанка:")
        print(f"  Содержимое: {chunks[0].get('content', chunks[0].get('text', ''))[:100]}...")
        print(f"  Метаданные: {chunks[0].get('metadata', {})}")

    return chunks

def generate_embeddings_from_json(json_file='chunks.json',
                                 model_name='all-MiniLM-L6-v2',
                                 output_file='embeddings_data.pkl'):
    """
    Генерирует эмбеддинги из JSON файла с чанками
    """

    print("🚀 ГЕНЕРАЦИЯ ЭМБЕДДИНГОВ ИЗ JSON")
    print("="*50)

    # 1. Загружаем чанки из JSON
    chunks_data = load_chunks_from_json(json_file)

    # 2. Извлекаем тексты и метаданные
    texts = []
    metadata_list = []

    for chunk in chunks_data:
        # Извлекаем текст (поддерживаем разные форматы)
        if 'content' in chunk:
            text = chunk['content']
        elif 'page_content' in chunk:
            text = chunk['page_content']
        elif 'text' in chunk:
            text = chunk['text']
        else:
            # Если ключа нет, берем весь объект как текст
            text = str(chunk)

        # Извлекаем метаданные
        if 'metadata' in chunk:
            metadata = chunk['metadata']
        elif 'meta' in chunk:
            metadata = chunk['meta']
        else:
            # Создаем метаданные из оставшихся полей
            metadata = {k: v for k, v in chunk.items() if k not in ['content', 'page_content', 'text']}

        texts.append(text)
        metadata_list.append(metadata)

    print(f"\n📊 Извлечено {len(texts)} текстов для эмбеддингов")

    # 3. Загружаем модель
    print(f"\n🤖 Загрузка модели: {model_name}...")
    model = SentenceTransformer(model_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"✅ Размер эмбеддинга: {embedding_dim}")

    # 4. Генерируем эмбеддинги
    print(f"\n⚡ Генерация эмбеддингов для {len(texts)} чанков...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    print(f"✅ Сгенерировано {len(embeddings)} эмбеддингов")
    print(f"📊 Форма: {embeddings.shape}")

    # 5. Сохраняем результат
    result = {
        'embeddings': embeddings,
        'texts': texts,
        'metadata': metadata_list,
        'model_info': {
            'name': model_name,
            'dimension': embedding_dim,
            'created_at': str(datetime.now())
        },
        'statistics': {
            'total_chunks': len(texts),
            'embedding_shape': embeddings.shape
        }
    }

    # Сохраняем в разных форматах
    with open(output_file, 'wb') as f:
        pickle.dump(result, f)
    print(f"\n💾 Сохранено в {output_file}")

    np.save(output_file.replace('.pkl', '.npy'), embeddings)
    print(f"💾 Эмбеддинги сохранены в {output_file.replace('.pkl', '.npy')}")

    # Сохраняем метаданные в JSON
    metadata_export = {
        'metadata': metadata_list,
        'texts_preview': [t[:100] + '...' for t in texts[:5]],
        'model_info': result['model_info'],
        'statistics': result['statistics']
    }

    json_out = output_file.replace('.pkl', '_metadata.json')
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(metadata_export, f, ensure_ascii=False, indent=2)
    print(f"💾 Метаданные сохранены в {json_out}")

    # 6. Показываем статистику
    print("\n" + "="*50)
    print("📊 СТАТИСТИКА:")
    print("="*50)
    print(f"Всего чанков: {len(texts)}")
    print(f"Размер эмбеддингов: {embedding_dim}")
    print(f"Средняя длина текста: {np.mean([len(t) for t in texts]):.0f} символов")

    # Уникальные источники
    sources = {}
    for meta in metadata_list:
        source = meta.get('source', meta.get('file', 'unknown'))
        if source not in sources:
            sources[source] = 0
        sources[source] += 1

    if len(sources) > 1:
        print(f"\n📁 Источники:")
        for source, count in sources.items():
            print(f"  - {source}: {count} чанков")

    return result