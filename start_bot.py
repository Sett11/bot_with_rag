import asyncio
from bot.bot import main
from utils.mylogger import Logger

# Настройка логов
logger = Logger("project_assistant", "logs/rag.log")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}") 