import numpy as np
import pickle
import faiss
import json
from datetime import datetime
import os


def create_faiss_index(embeddings_file='embeddings.pkl',
                       index_type='Flat',
                       output_index='faiss_index.bin',
                       output_metadata='faiss_metadata.pkl'):
    """
    Создает FAISS индекс из загруженных эмбеддингов

    Args:
        embeddings_file (str): Файл с эмбеддингами (.pkl)
        index_type (str): Тип индекса ('Flat', 'IVF', 'HNSW')
        output_index (str): Имя файла для индекса FAISS
        output_metadata (str): Имя файла для метаданных
    """

    print("🏗️ СОЗДАНИЕ FAISS ИНДЕКСА")
    print("="*60)

    # 1. Загружаем эмбеддинги
    print(f"📂 Загрузка эмбеддингов из {embeddings_file}...")
    with open(embeddings_file, 'rb') as f:
        data = pickle.load(f)

    embeddings = data['embeddings']
    texts = data['texts']
    metadata = data['metadata']
    model_info = data['model_info']

    print(f"✅ Загружено {len(embeddings)} эмбеддингов")
    print(f"📐 Размерность: {embeddings.shape[1]}")

    # 2. Создаем FAISS индекс
    dimension = embeddings.shape[1]
    print(f"\n🔧 Создание индекса типа '{index_type}'...")

    if index_type == 'Flat':
        # Простой плоский индекс (точный поиск)
        index = faiss.IndexFlatL2(dimension)
        print("  - Используется точный поиск (L2 расстояние)")

    elif index_type == 'IVF':
        # Индекс с кластеризацией для ускорения
        nlist = min(100, len(embeddings) // 10)  # Количество кластеров
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
        print(f"  - Используется IVF с {nlist} кластерами")
        print("  - Требуется обучение индекса...")

        # Обучаем индекс
        index.train(embeddings)
        print("  - Обучение завершено")

    elif index_type == 'HNSW':
        # Индекс с иерархической навигацией (быстрый поиск)
        index = faiss.IndexHNSWFlat(dimension, 32)
        print("  - Используется HNSW с 32 связями")

    else:
        raise ValueError(f"Неизвестный тип индекса: {index_type}")

    # 3. Добавляем эмбеддинги в индекс
    print(f"\n📥 Добавление {len(embeddings)} векторов в индекс...")
    index.add(embeddings)

    print(f"✅ Векторы добавлены")
    print(f"📊 Всего векторов в индексе: {index.ntotal}")

    # 4. Сохраняем индекс
    print(f"\n💾 Сохранение индекса в {output_index}...")
    faiss.write_index(index, output_index)
    print(f"✅ Индекс сохранен (размер: {os.path.getsize(output_index) / 1024:.1f} KB)")

    # 5. Сохраняем метаданные отдельно
    print(f"💾 Сохранение метаданных в {output_metadata}...")
    metadata_for_search = {
        'texts': texts,
        'metadata': metadata,
        'model_info': model_info,
        'index_info': {
            'type': index_type,
            'dimension': dimension,
            'total_vectors': index.ntotal,
            'created_at': str(datetime.now()),
            'embeddings_file': embeddings_file
        }
    }

    with open(output_metadata, 'wb') as f:
        pickle.dump(metadata_for_search, f)
    print(f"✅ Метаданные сохранены")

    # 6. Сохраняем метаданные в JSON для просмотра
    json_metadata = {
        'index_info': metadata_for_search['index_info'],
        'model_info': metadata_for_search['model_info'],
        'texts_preview': [t[:100] + '...' for t in texts[:100]],
        'metadata_preview': metadata[:100],
        'total_chunks': len(texts)
    }

    with open(output_metadata.replace('.pkl', '.json'), 'w', encoding='utf-8') as f:
        json.dump(json_metadata, f, ensure_ascii=False, indent=2)
    print(f"✅ Метаданные также сохранены в JSON формате")

    # 7. Статистика
    print("\n" + "="*60)
    print("✅ FAISS ИНДЕКС СОЗДАН!")
    print("="*60)
    print(f"📊 Всего векторов: {index.ntotal}")
    print(f"📐 Размерность: {dimension}")
    print(f"📁 Тип индекса: {index_type}")
    print(f"\n📂 Созданные файлы:")
    print(f"  - {output_index} (индекс FAISS)")
    print(f"  - {output_metadata} (метаданные)")
    print(f"  - {output_metadata.replace('.pkl', '.json')} (метаданные JSON)")

    return index, metadata_for_search