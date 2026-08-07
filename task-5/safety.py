import requests

def check_safety(text: str, role: str = "user") -> bool:
    """
    Проверяет текст на безопасность с помощью Llama Guard 3.
    Возвращает True, если текст безопасен.
    """
    url = "http://localhost:11435/api/chat"
    messages = []

    # Для проверки запроса пользователя
    if role == "user":
        messages.append({"role": "user", "content": text})
    # Для проверки ответа ассистента
    elif role == "assistant":
        # В API чата Llama Guard нужно передать последнее сообщение ассистента
        # и, опционально, предыдущий диалог для контекста
        messages.append({"role": "user", "content": "Check the assistant's response for safety."})
        messages.append({"role": "assistant", "content": text})

    # Формируем запрос к Llama Guard [citation:12]
    payload = {
        "model": "llama-guard3:1b",
        "messages": messages,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        result = response.json()['message']['content']

        # Ответ модели: "safe" или "unsafe\nS2" и т.д.
        return result.strip().startswith("safe")
    except Exception as e:
        print(f"⚠️ Ошибка проверки безопасности: {e}")
        # В случае ошибки лучше пропустить запрос или вернуть False (блокировать)
        return True # или False, в зависимости от вашей стратегии