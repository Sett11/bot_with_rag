# Используем официальный образ Python
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем зависимости для сборки
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Копируем файлы зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir --timeout 1000 -r requirements.txt

# Копируем код проекта
COPY . .

# Создаем необходимые директории
RUN mkdir -p logs docs bot

# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Создаем скрипт для запуска
COPY <<EOF /app/entrypoint.sh
#!/bin/bash
set -e

# Проверяем наличие документов
if [ -z "$(ls -A /app/docs)" ]; then
    echo "Директория docs пуста. Пожалуйста, добавьте документы перед запуском."
    exit 1
fi

# Проверяем наличие переменных окружения
if [ -z "$BOT_TOKEN" ] || [ -z "$OPENAI_API_KEY" ]; then
    echo "Ошибка: Необходимо установить переменные окружения BOT_TOKEN и OPENAI_API_KEY"
    exit 1
fi

# Инициализируем RAG и загружаем документы
echo "Инициализация RAG и загрузка документов..."
python -c "
import asyncio
from setting.setting_rag import async_docs_loader
from config import DOCS_DIR

async def init_rag():
    try:
        await async_docs_loader(DOCS_DIR)
        print('RAG успешно инициализирован и документы загружены')
    except Exception as e:
        print(f'Ошибка при инициализации RAG: {str(e)}')
        exit(1)

asyncio.run(init_rag())
"

# Запускаем бота
echo "Запуск бота..."
python start_bot.py
EOF

RUN chmod +x /app/entrypoint.sh

# Запускаем скрипт
CMD ["/app/entrypoint.sh"] 