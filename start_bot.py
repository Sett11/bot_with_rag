import asyncio
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.utils.markdown import hbold, hitalic
import nest_asyncio
from typing import Dict, List, Set
from setting.setting_rag import query_llm
from config import MAX_HISTORY, BOT_TOKEN
from utils.mylogger import Logger


# избегаем ошибки вложенния асинхронных event loops
nest_asyncio.apply()
# Настройка логов
logger = Logger("philosoph", "logs/rag.log")

# Инициализация бота
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Хранилище истории сообщений для каждого пользователя
# Структура: {chat_id: [(message_id, text), ...]}
message_history: Dict[int, List[tuple[int, str]]] = {}

# Множество пользователей, ожидающих ввода вопроса
# Структура: {chat_id}
waiting_for_question: Set[int] = set()

# Клавиатуры
def get_main_keyboard():
    """
    Создает и возвращает основную клавиатуру бота.
    
    Returns:
        ReplyKeyboardMarkup: Клавиатура с основными командами бота
    """
    logger.debug("Создание основной клавиатуры")
    return types.ReplyKeyboardMarkup(
        keyboard=[
            [types.KeyboardButton(text="/help"),
             types.KeyboardButton(text="/ask"),
             types.KeyboardButton(text="/clear_history"),
             ]
        ],
        resize_keyboard=True,
        input_field_placeholder="Выберите действие..."
    )

def update_message_history(chat_id: int, message_id: int, text: str) -> None:
    """
    Обновляет историю сообщений для указанного чата.
    
    Args:
        chat_id: ID чата
        message_id: ID сообщения
        text: Текст сообщения
    """
    if chat_id not in message_history:
        message_history[chat_id] = []
    
    # Добавляем новое сообщение
    message_history[chat_id].append((message_id, text))
    
    # Ограничиваем размер истории
    if len(message_history[chat_id]) > MAX_HISTORY:
        message_history[chat_id] = message_history[chat_id][-MAX_HISTORY:]
    
    logger.debug(f"История сообщений обновлена для чата {chat_id}. Текущий размер: {len(message_history[chat_id])}")

def remove_from_history(chat_id: int, message_id: int) -> None:
    """
    Удаляет сообщение из истории по его ID.
    
    Args:
        chat_id: ID чата
        message_id: ID сообщения для удаления
    """
    if chat_id in message_history:
        message_history[chat_id] = [(mid, text) for mid, text in message_history[chat_id] if mid != message_id]
        logger.debug(f"Сообщение {message_id} удалено из истории чата {chat_id}")

# Команда /start
@dp.message(Command("start"))
async def start(message: types.Message):
    """
    Обработчик команды /start.
    Отправляет приветственное сообщение и показывает основную клавиатуру.
    """
    logger.info(f"Пользователь {message.from_user.id} запустил бота")
    welcome_text = (
        f"Привет, {hbold(message.from_user.full_name)}! 👋\n"
        "Я философ, который поможет тебе разобраться в своих мыслях.\n\n"
        f"{hitalic('Основные команды:')}\n"
        "/help - помощь и инструкции\n"
        "/ask - задать вопрос\n"
        "/clear_history - очистить историю диалога\n"
    )
    await message.answer(
        welcome_text,
        reply_markup=get_main_keyboard(),
        parse_mode="HTML"
    )
    logger.debug(f"Приветственное сообщение отправлено пользователю {message.from_user.id}")

# Команда /help
@dp.message(Command("help"))
async def help_command(message: types.Message):
    """
    Обработчик команды /help.
    Отправляет справочную информацию о боте и его возможностях.
    """
    logger.info(f"Пользователь {message.from_user.id} запросил помощь")
    help_text = (
        f"{hbold('Помощь по боту:')}\n\n"
        "Задай вопрос по философии /ask 🤔\n"
        f"{hitalic('Пример:')}\n"
        "/ask Всё - тлен..? 💭\n\n"
        "Если твой вопрос не связан с философией, то я не смогу ответить на него. 🚫\n"
        f"Я сохраняю историю последних {MAX_HISTORY} сообщений для контекста. 📚\n"
        "Ты можешь очистить историю командой /clear_history 🧹"
    )
    await message.answer(
        help_text,
        reply_markup=get_main_keyboard(),
        parse_mode="HTML"
    )
    logger.debug(f"Справка отправлена пользователю {message.from_user.id}")

# Команда /ask - начало ожидания вопроса
@dp.message(Command("ask"))
async def ask_command(message: types.Message):
    """
    Обработчик команды /ask.
    Включает режим ожидания вопроса от пользователя.
    """
    chat_id = message.chat.id
    user_id = message.from_user.id
    
    logger.info(f"Пользователь {user_id} запросил режим ожидания вопроса")
    
    # Добавляем пользователя в список ожидающих
    waiting_for_question.add(chat_id)
    
    await message.answer(
        "Ну давай, задай мне вопрос... 🤔",
        reply_markup=get_main_keyboard(),
        parse_mode="HTML"
    )
    logger.debug(f"Пользователь {user_id} переведен в режим ожидания вопроса")

# Команда /clear_history - очистка истории
@dp.message(Command("clear_history"))
async def clear_history(message: types.Message):
    """
    Обработчик команды /clear_history.
    Очищает историю сообщений для текущего чата.
    """
    chat_id = message.chat.id
    logger.info(f"Запрос на очистку истории от пользователя {message.from_user.id}")
    
    try:
        if chat_id in message_history:
            message_history[chat_id] = []
            logger.info(f"История чата {chat_id} успешно очищена")
            await message.answer("✅ История диалога очищена")
        else:
            logger.debug(f"История для чата {chat_id} уже пуста")
            await message.answer("История диалога уже пуста")
    except Exception as e:
        error_msg = f"Ошибка при очистке истории: {str(e)}"
        logger.error(error_msg)
        await message.answer("⚠️ Произошла ошибка при очистке истории.")

# Обработка удаления сообщений
@dp.message(F.delete_message)
async def handle_message_deletion(message: types.Message):
    """
    Обработчик удаления сообщений.
    Удаляет сообщение из истории при его удалении пользователем.
    """
    try:
        chat_id = message.chat.id
        message_id = message.message_id
        logger.info(f"Сообщение {message_id} удалено в чате {chat_id}")
        remove_from_history(chat_id, message_id)
    except Exception as e:
        logger.error(f"Ошибка при обработке удаления сообщения: {str(e)}")

# Обработка обычных сообщений
@dp.message(F.text)
async def handle_message(message: types.Message):
    """
    Обработчик текстовых сообщений.
    Обрабатывает вопрос пользователя и сохраняет его в истории.
    """
    try:
        user_id = message.from_user.id
        chat_id = message.chat.id
        question = message.text
        
        # Проверяем, ожидаем ли мы вопрос от этого пользователя
        if chat_id in waiting_for_question:
            logger.info(f"Получен вопрос от пользователя {user_id} в режиме ожидания: {question[:100]}...")
            
            # Удаляем пользователя из списка ожидающих
            waiting_for_question.remove(chat_id)
            
            # Сохраняем вопрос в истории
            update_message_history(chat_id, message.message_id, question)
            
            # Получаем ответ от LLM
            response = query_llm(question)
            
            # Отправляем ответ
            sent_message = await message.answer(response)
            
            # Сохраняем ответ в истории
            update_message_history(chat_id, sent_message.message_id, response)
            
            logger.debug(f"Ответ отправлен пользователю {user_id}")
        else:
            # Если пользователь не в режиме ожидания, отправляем подсказку
            logger.info(f"Получено сообщение от пользователя {user_id} вне режима ожидания")
            await message.answer(
                "⚠️ Пожалуйста, используйте команду /ask, чтобы задать вопрос.",
                reply_markup=get_main_keyboard(),
                parse_mode="HTML"
            )
    except Exception as e:
        error_msg = f"Ошибка при обработке сообщения: {str(e)}"
        logger.error(error_msg)
        await message.answer("⚠️ Произошла ошибка при обработке вашего вопроса.")

# Обработка ошибок
@dp.errors()
async def errors_handler(update: types.Update, exception: Exception):
    """
    Глобальный обработчик ошибок.
    Логирует все необработанные исключения.
    """
    error_msg = f"Необработанное исключение: {str(exception)}\nUpdate: {update}"
    logger.error(error_msg)
    return True

# Запуск бота
async def main():
    """
    Основная функция запуска бота.
    Инициализирует необходимые компоненты и запускает поллинг.
    """
    try:
        logger.info("Инициализация бота...Запуск поллинга...")
        await dp.start_polling(bot)
    except Exception as e:
        error_msg = f"Ошибка при запуске бота: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

if __name__ == "__main__":
    try:
        logger.info("Запуск приложения...")
        asyncio.run(main())
    except Exception as e:
        error_msg = f"Критическая ошибка: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)