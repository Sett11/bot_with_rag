import logging
from logging.handlers import TimedRotatingFileHandler
import os

def ensure_log_directory():
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

class Logger(logging.Logger):
    """Класс, обеспечивающий настройку логгирования с ротацией логов по времени."""
    def __init__(self, name, log_file, level=logging.INFO):
        """
            Инициализация класса Logger.

            :param name: Имя логгера.
            :param log_file: Имя файла для сохранения логов.
            :param level: Уровень логгирования.
            """
        super().__init__(name, level)

        # Форматтер
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Обработчик для ротации файлов по времени
        handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, encoding='utf-8')
        handler.suffix = "%Y-%m-%d"
        handler.setFormatter(formatter)

        # Добавление обработчика в логгер
        self.addHandler(handler)