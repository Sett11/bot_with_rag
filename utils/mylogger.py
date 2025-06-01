import logging
from logging.handlers import TimedRotatingFileHandler
import os
import sys

def ensure_log_directory(log_file):
    """Создает директорию для логов, если она не существует"""
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    print(f"Директория для логов: {log_dir}")  # Отладочный вывод
    
    # Проверяем права доступа
    if os.path.exists(log_file):
        try:
            with open(log_file, 'a') as f:
                f.write("")
            print(f"Права на запись в файл {log_file} подтверждены")
        except Exception as e:
            print(f"Ошибка при проверке прав доступа к файлу {log_file}: {str(e)}")
            raise

class Logger(logging.Logger):
    """Класс, обеспечивающий настройку логгирования с ротацией логов по времени."""
    def __init__(self, name, log_file, level=logging.INFO):
        """
        Инициализация класса Logger.

        Args:
            name: Имя логгера
            log_file: Путь к файлу для сохранения логов
            level: Уровень логгирования
        """
        super().__init__(name, level)
        print(f"Инициализация логгера {name} с файлом {log_file}")  # Отладочный вывод

        # Форматтер для логов
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Создаем директорию для логов, если она не существует
        ensure_log_directory(log_file)

        try:
            # Обработчик для файла с ротацией по времени
            file_handler = TimedRotatingFileHandler(
                log_file,
                when="midnight",
                interval=1,
                encoding='utf-8'
            )
            file_handler.suffix = "%Y-%m-%d"
            file_handler.setFormatter(formatter)
            self.addHandler(file_handler)
            print(f"Файловый обработчик добавлен: {log_file}")  # Отладочный вывод
            
            # Проверяем, что логгер работает
            self.info("Тестовое сообщение для проверки логгера")
            print("Тестовое сообщение записано в лог")
        except Exception as e:
            print(f"Ошибка при создании файлового обработчика: {str(e)}")  # Отладочный вывод
            raise

        # Обработчик для вывода в консоль
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)
        print("Консольный обработчик добавлен")  # Отладочный вывод

        # Устанавливаем уровень логирования
        self.setLevel(level)
        print(f"Уровень логирования установлен: {level}")  # Отладочный вывод

    def info(self, msg, *args, **kwargs):
        """Переопределяем метод info для отладки"""
        print(f"Попытка записи в лог: {msg}")  # Отладочный вывод
        super().info(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        """Переопределяем метод error для отладки"""
        print(f"Попытка записи ошибки в лог: {msg}")  # Отладочный вывод
        super().error(msg, *args, **kwargs)