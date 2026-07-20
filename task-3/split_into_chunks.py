import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# split_into_chunks_fixed.py - исправленная версия

import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def load_and_chunk_folder(folder_path, chunk_size=500, chunk_overlap=100):
    """
    Загрузка и разбивка на чанки всех файлов в папке
    С правильным сохранением source в метаданных
    """
    documents = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    # Поддерживаемые расширения
    extensions = ['.txt', '.md', '.pdf', '.docx']

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if os.path.isdir(file_path):
            continue

        _, ext = os.path.splitext(file_name)
        if ext.lower() not in extensions:
            continue

        print(f"📄 Обработка: {file_name}")

        # Читаем файл
        try:
            if ext.lower() in ['.txt', '.md']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            else:
                # Для других форматов используйте соответствующие библиотеки
                continue

            # Создаем базовый документ
            base_metadata = {
                'source': file_name,  # ВАЖНО: имя файла
                'file_path': file_path,
                'file_name': file_name,
                'file_size': os.path.getsize(file_path),
                'file_type': ext
            }

            # Разбиваем на чанки
            chunks = text_splitter.split_text(content)

            # Создаем Document для каждого чанка
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 50:  # Пропускаем слишком короткие
                    continue

                metadata = base_metadata.copy()
                metadata.update({
                    'chunk_index': i,
                    'chunk_total': len(chunks),
                    'chunk_length': len(chunk)
                })

                doc = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                documents.append(doc)

            print(f"  ✅ Создано {len(chunks)} чанков из {file_name}")

        except Exception as e:
            print(f"  ❌ Ошибка при обработке {file_name}: {e}")

    print(f"\n✅ Всего загружено {len(documents)} чанков")

    # Выводим статистику по источникам
    sources = {}
    for doc in documents:
        source = doc.metadata.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1

    print(f"📄 Уникальных источников: {len(sources)}")
    for source, count in sorted(sources.items())[:5]:
        print(f"  - {source}: {count} чанков")

    return documents