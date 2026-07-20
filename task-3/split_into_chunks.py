import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

def load_and_chunk_folder(folder_path, chunk_size=500, chunk_overlap=50):
    """
    Загружает все текстовые файлы из папки и разбивает на чанки

    Args:
        folder_path (str): Путь к папке с файлами
        chunk_size (int): Размер чанка в символах
        chunk_overlap (int): Перекрытие между чанками

    Returns:
        list: Список объектов Document с чанками и метаданными
    """

    # Настраиваем сплиттер
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True,
    )

    all_documents = []

    # Проходим по всем файлам в папке
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Пропускаем папки
        if os.path.isdir(file_path):
            continue

        # Проверяем расширение файла
        if not filename.endswith(('.txt', '.md', '.rst', '.json')):
            print(f"Пропускаем файл (неподдерживаемый формат): {filename}")
            continue

        try:
            # Читаем содержимое файла
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Создаем метаданные для файла
            metadata = {
                "source": filename,           # Имя файла
                "file_path": file_path,       # Полный путь
                "file_size": os.path.getsize(file_path),  # Размер файла
            }

            # Создаем документ
            doc = Document(
                page_content=content,
                metadata=metadata
            )

            # Разбиваем на чанки
            chunks = text_splitter.split_documents([doc])

            # Добавляем номер чанка в метаданные
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_index"] = i
                chunk.metadata["total_chunks"] = len(chunks)

            all_documents.extend(chunks)
            print(f"✅ Обработан файл: {filename} -> {len(chunks)} чанков")

        except Exception as e:
            print(f"❌ Ошибка при обработке {filename}: {e}")

    print(f"\n📊 Всего создано чанков: {len(all_documents)}")
    return all_documents