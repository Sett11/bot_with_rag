import asyncio
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.utils.markdown import hbold, hitalic, hcode
import nest_asyncio
from typing import Dict, List, Set
from setting.setting_rag import query_llm, docs_loader
from config import MAX_HISTORY, BOT_TOKEN, DOCS_DIR
from utils.mylogger import Logger
from .database import Database
from datetime import date, datetime
from telegram import Update
from telegram.ext import ContextTypes, CommandHandler, MessageHandler, filters
import os


# избегаем ошибки вложенния асинхронных event loops
nest_asyncio.apply()
# Настройка логов
logger = Logger("project_assistant", "logs/rag.log")

# Инициализация бота и базы данных
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
db = Database()

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
             types.KeyboardButton(text="/ask")],
            [types.KeyboardButton(text="/clear_history"),
             types.KeyboardButton(text="/project_info")],
            [types.KeyboardButton(text="/system_type"),
             types.KeyboardButton(text="/requirements")],
            [types.KeyboardButton(text="/stats"),
             types.KeyboardButton(text="/upgrade")]
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
    user_id = message.from_user.id
    username = message.from_user.username or ""
    first_name = message.from_user.first_name or ""
    last_name = message.from_user.last_name or ""
    
    logger.info(f"Пользователь {user_id} запустил бота")
    
    # Регистрация или получение пользователя
    is_new_user, is_premium = await db.get_or_create_user(user_id, username, first_name, last_name)
    
    welcome_text = (
        f"Привет, {hbold(message.from_user.full_name)}! 👋\n\n"
        "Я ваш ассистент по проектированию инженерных систем.\n"
        "Я помогу вам с вопросами по:\n"
        "• Вентиляционным системам\n"
        "• Трубопроводам\n"
        "• Системам кондиционирования\n"
        "• И другим инженерным системам\n\n"
    )
    
    if is_new_user:
        welcome_text += (
            f"{hitalic('Вы новый пользователь!')}\n"
            "У вас есть 10 бесплатных запросов в день.\n"
            "Для безлимитного доступа используйте команду /upgrade\n\n"
        )
    elif not is_premium:
        welcome_text += (
            f"{hitalic('У вас осталось 10 бесплатных запросов в день.')}\n"
            "Для безлимитного доступа используйте команду /upgrade\n\n"
        )
    else:
        welcome_text += f"{hitalic('У вас премиум доступ!')}\n\n"
    
    welcome_text += (
        f"{hitalic('Основные команды:')}\n"
        "/help - помощь и инструкции\n"
        "/ask - задать вопрос\n"
        "/project_info - информация о проекте\n"
        "/system_type - тип системы\n"
        "/requirements - требования к системе\n"
        "/stats - ваша статистика\n"
        "/upgrade - улучшить до премиум\n"
        "/clear_history - очистить историю диалога\n"
    )
    
    await message.answer(
        welcome_text,
        reply_markup=get_main_keyboard(),
        parse_mode="HTML"
    )
    logger.debug(f"Приветственное сообщение отправлено пользователю {user_id}")

# Команда /help
@dp.message(Command("help"))
async def help_command(message: types.Message):
    """
    Обработчик команды /help.
    Отправляет справочную информацию о боте и его возможностях.
    """
    user_id = message.from_user.id
    logger.info(f"Пользователь {user_id} запросил помощь")
    
    # Получаем статус пользователя
    stats = await db.get_user_stats(user_id)
    is_premium = stats["is_premium"] if stats else False
    
    help_text = (
        f"{hbold('Помощь по боту:')}\n\n"
        "Я могу помочь вам с вопросами по проектированию инженерных систем.\n\n"
        f"{hitalic('Основные команды:')}\n"
        "/ask - задать вопрос по проекту\n"
        "/project_info - получить общую информацию о проекте\n"
        "/system_type - узнать тип системы\n"
        "/requirements - узнать требования к системе\n"
        "/stats - ваша статистика\n"
        "/upgrade - улучшить до премиум\n"
        "/clear_history - очистить историю диалога\n\n"
        f"{hitalic('Примеры вопросов:')}\n"
        "• Какие требования к вентиляционной системе?\n"
        "• Какой тип трубопровода используется?\n"
        "• Какие параметры кондиционера?\n\n"
    )
    
    if not is_premium:
        help_text += (
            f"{hitalic('Лимиты:')}\n"
            "• 10 бесплатных запросов в день\n"
            "• Для безлимитного доступа используйте /upgrade\n\n"
        )
    
    help_text += (
        f"Я сохраняю историю последних {MAX_HISTORY} сообщений для контекста. 📚\n"
        "Вы можете очистить историю командой /clear_history 🧹"
    )
    
    await message.answer(
        help_text,
        reply_markup=get_main_keyboard(),
        parse_mode="HTML"
    )
    logger.debug(f"Справка отправлена пользователю {user_id}")

# Команда /stats
@dp.message(Command("stats"))
async def stats_command(message: types.Message):
    """
    Обработчик команды /stats.
    Показывает статистику пользователя.
    """
    user_id = message.from_user.id
    logger.info(f"Пользователь {user_id} запросил статистику")
    
    try:
        stats = await db.get_user_stats(user_id)
        if not stats:
            await message.answer(
                "❌ Не удалось получить статистику. Попробуйте позже.",
                reply_markup=get_main_keyboard()
            )
            return
        
        stats_text = (
            f"{hbold('Ваша статистика:')}\n\n"
            f"👤 Пользователь: {stats['username'] or 'Не указан'}\n"
            f"📅 Дата регистрации: {stats['created_at']}\n"
            f"🔄 Последняя активность: {stats['last_active']}\n"
            f"⭐ Статус: {'Премиум' if stats['is_premium'] else 'Бесплатный'}\n\n"
        )
        
        if not stats['is_premium']:
            today = str(date.today())
            today_requests = stats['requests'].get(today, 0)
            stats_text += (
                f"{hitalic('Использование запросов:')}\n"
                f"• Сегодня: {today_requests}/10\n"
                f"• Осталось: {10 - today_requests}\n\n"
            )
        
        stats_text += (
            f"{hitalic('История запросов (последние 7 дней):')}\n"
        )
        
        for date_str, count in stats['requests'].items():
            stats_text += f"• {date_str}: {count} запросов\n"
        
        await message.answer(
            stats_text,
            reply_markup=get_main_keyboard(),
            parse_mode="HTML"
        )
        logger.debug(f"Статистика отправлена пользователю {user_id}")
    except Exception as e:
        error_msg = f"Ошибка при получении статистики: {str(e)}"
        logger.error(error_msg)
        await message.answer(
            "⚠️ Произошла ошибка при получении статистики. Попробуйте позже.",
            reply_markup=get_main_keyboard()
        )

# Команда /upgrade
@dp.message(Command("upgrade"))
async def upgrade_command(message: types.Message):
    """
    Обработчик команды /upgrade.
    Показывает информацию о премиум подписке.
    """
    user_id = message.from_user.id
    logger.info(f"Пользователь {user_id} запросил информацию о премиум подписке")
    
    upgrade_text = (
        f"{hbold('Премиум подписка')}\n\n"
        f"{hitalic('Преимущества:')}\n"
        "• Безлимитные запросы\n"
        "• Приоритетная поддержка\n"
        "• Расширенная статистика\n\n"
        f"{hitalic('Стоимость:')}\n"
        "• 299 рублей в месяц\n\n"
        "Для оплаты свяжитесь с администратором: @admin"
    )
    
    await message.answer(
        upgrade_text,
        reply_markup=get_main_keyboard(),
        parse_mode="HTML"
    )
    logger.debug(f"Информация о премиум подписке отправлена пользователю {user_id}")

# Команда /project_info
@dp.message(Command("project_info"))
async def project_info_command(message: types.Message):
    """
    Обработчик команды /project_info.
    Запрашивает общую информацию о проекте.
    """
    user_id = message.from_user.id
    logger.info(f"Пользователь {user_id} запросил информацию о проекте")
    
    # Проверяем лимит запросов
    can_make_request, remaining = await db.check_request_limit(user_id)
    if not can_make_request:
        await message.answer(
            "⚠️ Вы достигли лимита бесплатных запросов.\n"
            "Для продолжения работы используйте команду /upgrade",
            reply_markup=get_main_keyboard()
        )
        return
    
    try:
        response = query_llm("Расскажи общую информацию о проекте: тип здания, основные системы, ключевые параметры.")
        
        # Увеличиваем счетчик запросов
        await db.increment_request_count(user_id)
        
        await message.answer(
            response,
            reply_markup=get_main_keyboard(),
            parse_mode="HTML"
        )
        logger.debug(f"Информация о проекте отправлена пользователю {user_id}")
    except Exception as e:
        error_msg = f"Ошибка при получении информации о проекте: {str(e)}"
        logger.error(error_msg)
        await message.answer(
            "⚠️ Произошла ошибка при получении информации о проекте. Попробуйте позже.",
            reply_markup=get_main_keyboard()
        )

# Команда /system_type
@dp.message(Command("system_type"))
async def system_type_command(message: types.Message):
    """
    Обработчик команды /system_type.
    Запрашивает информацию о типе системы.
    """
    user_id = message.from_user.id
    logger.info(f"Пользователь {user_id} запросил информацию о типе системы")
    
    # Проверяем лимит запросов
    can_make_request, remaining = await db.check_request_limit(user_id)
    if not can_make_request:
        await message.answer(
            "⚠️ Вы достигли лимита бесплатных запросов.\n"
            "Для продолжения работы используйте команду /upgrade",
            reply_markup=get_main_keyboard()
        )
        return
    
    try:
        response = query_llm("Какой тип инженерной системы используется в проекте? Опиши основные характеристики.")
        
        # Увеличиваем счетчик запросов
        await db.increment_request_count(user_id)
        
        await message.answer(
            response,
            reply_markup=get_main_keyboard(),
            parse_mode="HTML"
        )
        logger.debug(f"Информация о типе системы отправлена пользователю {user_id}")
    except Exception as e:
        error_msg = f"Ошибка при получении информации о типе системы: {str(e)}"
        logger.error(error_msg)
        await message.answer(
            "⚠️ Произошла ошибка при получении информации о типе системы. Попробуйте позже.",
            reply_markup=get_main_keyboard()
        )

# Команда /requirements
@dp.message(Command("requirements"))
async def requirements_command(message: types.Message):
    """
    Обработчик команды /requirements.
    Запрашивает информацию о требованиях к системе.
    """
    user_id = message.from_user.id
    logger.info(f"Пользователь {user_id} запросил информацию о требованиях")
    
    # Проверяем лимит запросов
    can_make_request, remaining = await db.check_request_limit(user_id)
    if not can_make_request:
        await message.answer(
            "⚠️ Вы достигли лимита бесплатных запросов.\n"
            "Для продолжения работы используйте команду /upgrade",
            reply_markup=get_main_keyboard()
        )
        return
    
    try:
        response = query_llm("Какие основные требования к инженерной системе в проекте? Опиши технические параметры и нормативы.")
        
        # Увеличиваем счетчик запросов
        await db.increment_request_count(user_id)
        
        await message.answer(
            response,
            reply_markup=get_main_keyboard(),
            parse_mode="HTML"
        )
        logger.debug(f"Информация о требованиях отправлена пользователю {user_id}")
    except Exception as e:
        error_msg = f"Ошибка при получении информации о требованиях: {str(e)}"
        logger.error(error_msg)
        await message.answer(
            "⚠️ Произошла ошибка при получении информации о требованиях. Попробуйте позже.",
            reply_markup=get_main_keyboard()
        )

# Команда /ask - начало ожидания вопроса
@dp.message(Command("ask"))
async def ask_command(message: types.Message):
    """
    Обработчик команды /ask.
    Включает режим ожидания вопроса от пользователя.
    """
    user_id = message.from_user.id
    chat_id = message.chat.id
    
    logger.info(f"Пользователь {user_id} запросил режим ожидания вопроса")
    
    # Проверяем лимит запросов
    can_make_request, remaining = await db.check_request_limit(user_id)
    if not can_make_request:
        await message.answer(
            "⚠️ Вы достигли лимита бесплатных запросов.\n"
            "Для продолжения работы используйте команду /upgrade",
            reply_markup=get_main_keyboard()
        )
        return
    
    # Добавляем пользователя в список ожидающих
    waiting_for_question.add(chat_id)
    
    await message.answer(
        "Задайте ваш вопрос по проекту... 🤔\n"
        "Например:\n"
        "• Какие требования к вентиляционной системе?\n"
        "• Какой тип трубопровода используется?\n"
        "• Какие параметры кондиционера?",
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
    user_id = message.from_user.id
    logger.info(f"Запрос на очистку истории от пользователя {user_id}")
    
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
        if chat_id not in waiting_for_question:
            logger.debug(f"Сообщение от пользователя {user_id} проигнорировано (не в режиме ожидания вопроса)")
            return
        
        # Проверяем лимит запросов
        can_make_request, remaining = await db.check_request_limit(user_id)
        if not can_make_request:
            waiting_for_question.remove(chat_id)
            await message.answer(
                "⚠️ Вы достигли лимита бесплатных запросов.\n"
                "Для продолжения работы используйте команду /upgrade",
                reply_markup=get_main_keyboard()
            )
            return
        
        # Удаляем пользователя из списка ожидающих
        waiting_for_question.remove(chat_id)
        
        # Сохраняем вопрос в истории
        update_message_history(chat_id, message.message_id, question)
        
        # Отправляем сообщение о том, что обрабатываем вопрос
        processing_message = await message.answer("🤔 Анализирую документацию...")
        
        try:
            # Получаем ответ от LLM
            response = query_llm(question)
            
            # Увеличиваем счетчик запросов
            await db.increment_request_count(user_id)
            
            # Сохраняем ответ в истории
            update_message_history(chat_id, processing_message.message_id, response)
            
            # Отправляем ответ пользователю
            await processing_message.edit_text(response)
            logger.info(f"Ответ успешно отправлен пользователю {user_id}")
            
        except Exception as e:
            error_msg = f"Ошибка при обработке вопроса: {str(e)}"
            logger.error(error_msg)
            await processing_message.edit_text("⚠️ Произошла ошибка при обработке вопроса. Попробуйте позже.")
            
    except Exception as e:
        logger.error(f"Ошибка в обработчике сообщений: {str(e)}")

@dp.errors()
async def errors_handler(update: types.Update, exception: Exception):
    """
    Обработчик ошибок бота.
    Логирует ошибки и отправляет уведомление пользователю.
    """
    logger.error(f"Ошибка при обработке обновления {update}: {str(exception)}")
    if update and update.message:
        await update.message.answer(
            "⚠️ Произошла ошибка при обработке запроса. Попробуйте позже."
        )

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик загрузки документов"""
    try:
        # Получаем информацию о файле
        file = await context.bot.get_file(update.message.document)
        file_name = update.message.document.file_name
        
        # Проверяем расширение файла
        allowed_extensions = ['.pdf', '.docx', '.txt']
        if not any(file_name.lower().endswith(ext) for ext in allowed_extensions):
            await update.message.reply_text(
                "Извините, поддерживаются только файлы форматов: PDF, DOCX, TXT"
            )
            return

        # Создаем уникальное имя файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_file_name = f"{timestamp}_{file_name}"
        file_path = os.path.join(DOCS_DIR, new_file_name)

        # Скачиваем файл
        await update.message.reply_text("Загружаю файл...")
        await file.download_to_drive(file_path)

        # Обрабатываем файл через docs_loader
        await update.message.reply_text("Обрабатываю файл...")
        try:
            docs_loader(DOCS_DIR)
            await update.message.reply_text(
                f"Файл {file_name} успешно загружен и обработан! Теперь вы можете задавать вопросы по его содержимому."
            )
        except Exception as e:
            await update.message.reply_text(
                f"Произошла ошибка при обработке файла: {str(e)}"
            )
            # Удаляем файл в случае ошибки
            if os.path.exists(file_path):
                os.remove(file_path)

    except Exception as e:
        await update.message.reply_text(
            f"Произошла ошибка при загрузке файла: {str(e)}"
        )

def setup_handlers(application):
    """Настройка обработчиков команд"""
    # Добавляем обработчик документов
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))

async def main():
    """
    Основная функция запуска бота.
    Инициализирует базу данных и запускает бота.
    """
    try:
        # Инициализация базы данных
        await db.init_db()
        logger.info("База данных успешно инициализирована")
        
        # Запуск бота
        logger.info("Запуск бота...")
        await dp.start_polling(bot)
        
    except Exception as e:
        error_msg = f"Ошибка при запуске бота: {str(e)}"
        logger.error(error_msg)
        raise

if __name__ == "__main__":
    asyncio.run(main())