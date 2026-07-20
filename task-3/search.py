import os
import faiss
import pickle

class VectorSearchEngine:
    """
    Класс для поиска по векторному индексу FAISS
    """

    def __init__(self, index_file='faiss_index.bin', metadata_file='faiss_metadata.pkl'):
        """
        Инициализация поискового движка

        Args:
            index_file (str): Путь к файлу индекса FAISS
            metadata_file (str): Путь к файлу с метаданными
        """
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.index = None
        self.metadata = None
        self.model = None

    def load(self):
        """Загружает индекс и метаданные"""
        print("📂 ЗАГРУЗКА ПОИСКОВОГО ДВИЖКА")
        print("="*50)

        # Загружаем индекс FAISS
        if os.path.exists(self.index_file):
            print(f"✅ Загрузка индекса из {self.index_file}...")
            self.index = faiss.read_index(self.index_file)
            print(f"  - Всего векторов: {self.index.ntotal}")
        else:
            raise FileNotFoundError(f"Индекс {self.index_file} не найден")

        # Загружаем метаданные
        if os.path.exists(self.metadata_file):
            print(f"✅ Загрузка метаданных из {self.metadata_file}...")
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"  - Всего чанков: {len(self.metadata['texts'])}")
        else:
            raise FileNotFoundError(f"Метаданные {self.metadata_file} не найдены")

        # Загружаем модель для эмбеддингов
        from sentence_transformers import SentenceTransformer
        model_name = self.metadata['model_info']['name']
        print(f"✅ Загрузка модели {model_name}...")
        self.model = SentenceTransformer(model_name)

        print("\n✅ ПОИСКОВЫЙ ДВИЖОК ГОТОВ!")
        return self

    def search(self, query, k=5, threshold=1.0):
        """
        Поиск по запросу

        Args:
            query (str): Поисковый запрос
            k (int): Количество результатов
            threshold (float): Порог расстояния (чем меньше, тем точнее)

        Returns:
            list: Список результатов с метаданными
        """
        if self.index is None or self.metadata is None:
            raise ValueError("Движок не загружен. Вызовите .load()")

        # Генерируем эмбеддинг для запроса
        query_embedding = self.model.encode([query], convert_to_numpy=True)

        # Ищем в FAISS
        distances, indices = self.index.search(query_embedding, k)

        # Формируем результаты
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.metadata['texts']):
                continue

            distance = distances[0][i]

            # Если расстояние больше порога, пропускаем
            if distance > threshold:
                continue

            results.append({
                'text': self.metadata['texts'][idx],
                'metadata': self.metadata['metadata'][idx],
                'distance': float(distance),
                'score': 1.0 / (1.0 + distance),  # Преобразуем расстояние в score
                'index': int(idx)
            })

        return results

    def search_with_context(self, query, k=5, threshold=1.0):
        """
        Поиск с контекстом (возвращает больше информации)
        """
        results = self.search(query, k, threshold)

        # Добавляем контекстную информацию
        for result in results:
            meta = result['metadata']
            result['context'] = {
                'source': meta.get('source', meta.get('file', 'unknown')),
                'chunk_index': meta.get('chunk_index', 0),
                'file_path': meta.get('file_path', ''),
                'chunk_id': meta.get('chunk_id', '')
            }

        return results

    def get_stats(self):
        """Возвращает статистику индекса"""
        if self.index is None or self.metadata is None:
            return None

        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.metadata['index_info']['dimension'],
            'index_type': self.metadata['index_info']['type'],
            'total_texts': len(self.metadata['texts']),
            'model': self.metadata['model_info']['name']
        }

search_engine = VectorSearchEngine(
    index_file='./data/faiss_index.bin',
    metadata_file='./data/faiss_metadata.pkl'
)

search_engine.load()

results = search_engine.search("Кто был учителем Ами Шани?", k=5)

for i, result in enumerate(results, 1):
    print(f"\n📌 Результат {i}:")
    print(f"  Текст: {result['text'][:200]}...")
    print(f"  Источник: {result['metadata'].get('source', 'unknown')}")
    print(f"  Сходство: {result['score']:.3f}")
    print(f"  Расстояние: {result['distance']:.3f}")