import aiosqlite
from datetime import datetime, date
from typing import Optional, Tuple, List, Dict
from utils.mylogger import Logger
import json

logger = Logger("database", "logs/rag.log")

class Database:
    def __init__(self, db_path: str = "bot/users.db"):
        """
        Инициализация базы данных.
        
        Args:
            db_path: Путь к файлу базы данных
        """
        self.db_path = db_path
        logger.info(f"Инициализация базы данных: {db_path}")

    async def init_db(self):
        """
        Инициализация таблиц базы данных.
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id INTEGER PRIMARY KEY,
                        username TEXT,
                        first_name TEXT,
                        last_name TEXT,
                        is_premium BOOLEAN DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS requests (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        request_date DATE,
                        request_count INTEGER DEFAULT 0,
                        FOREIGN KEY (user_id) REFERENCES users (user_id),
                        UNIQUE(user_id, request_date)
                    )
                """)
                
                await db.commit()
                logger.info("База данных успешно инициализирована")
        except Exception as e:
            error_msg = f"Ошибка при инициализации базы данных: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def get_or_create_user(self, user_id: int, username: str, first_name: str, last_name: str) -> Tuple[bool, bool]:
        """
        Получение или создание пользователя.
        
        Args:
            user_id: ID пользователя в Telegram
            username: Имя пользователя в Telegram
            first_name: Имя пользователя
            last_name: Фамилия пользователя
            
        Returns:
            Tuple[bool, bool]: (is_new_user, is_premium)
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Проверяем существование пользователя
                async with db.execute(
                    "SELECT is_premium FROM users WHERE user_id = ?",
                    (user_id,)
                ) as cursor:
                    user = await cursor.fetchone()
                
                if user is None:
                    # Создаем нового пользователя
                    await db.execute(
                        """
                        INSERT INTO users (user_id, username, first_name, last_name)
                        VALUES (?, ?, ?, ?)
                        """,
                        (user_id, username, first_name, last_name)
                    )
                    await db.commit()
                    logger.info(f"Создан новый пользователь: {user_id}")
                    return True, False
                
                # Обновляем время последней активности
                await db.execute(
                    "UPDATE users SET last_active = CURRENT_TIMESTAMP WHERE user_id = ?",
                    (user_id,)
                )
                await db.commit()
                
                return False, bool(user[0])
        except Exception as e:
            error_msg = f"Ошибка при работе с пользователем {user_id}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def check_request_limit(self, user_id: int) -> Tuple[bool, int]:
        """
        Проверка лимита запросов для пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Tuple[bool, int]: (can_make_request, remaining_requests)
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Проверяем статус премиум
                async with db.execute(
                    "SELECT is_premium FROM users WHERE user_id = ?",
                    (user_id,)
                ) as cursor:
                    user = await cursor.fetchone()
                
                if not user or not user[0]:  # Если пользователь не премиум
                    today = date.today()
                    
                    # Получаем количество запросов за сегодня
                    async with db.execute(
                        """
                        SELECT request_count FROM requests 
                        WHERE user_id = ? AND request_date = ?
                        """,
                        (user_id, today)
                    ) as cursor:
                        result = await cursor.fetchone()
                    
                    if result is None:
                        # Создаем запись для нового дня
                        await db.execute(
                            """
                            INSERT INTO requests (user_id, request_date, request_count)
                            VALUES (?, ?, 0)
                            """,
                            (user_id, today)
                        )
                        await db.commit()
                        return True, 10  # Новый день, полный лимит
                    
                    request_count = result[0]
                    if request_count >= 10:  # Лимит бесплатных запросов
                        return False, 0
                    
                    return True, 10 - request_count
                
                return True, -1  # Премиум пользователь, безлимитный доступ
        except Exception as e:
            error_msg = f"Ошибка при проверке лимита запросов для пользователя {user_id}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def increment_request_count(self, user_id: int):
        """
        Увеличивает счетчик запросов для пользователя.
        
        Args:
            user_id: ID пользователя
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                today = date.today()
                
                # Проверяем статус премиум
                async with db.execute(
                    "SELECT is_premium FROM users WHERE user_id = ?",
                    (user_id,)
                ) as cursor:
                    user = await cursor.fetchone()
                
                if not user or not user[0]:  # Если пользователь не премиум
                    await db.execute(
                        """
                        UPDATE requests 
                        SET request_count = request_count + 1
                        WHERE user_id = ? AND request_date = ?
                        """,
                        (user_id, today)
                    )
                    await db.commit()
                    logger.debug(f"Увеличен счетчик запросов для пользователя {user_id}")
        except Exception as e:
            error_msg = f"Ошибка при увеличении счетчика запросов для пользователя {user_id}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    async def upgrade_to_premium(self, user_id: int) -> bool:
        """
        Обновление пользователя до премиум статуса.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            bool: True если обновление успешно
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    "UPDATE users SET is_premium = 1 WHERE user_id = ?",
                    (user_id,)
                )
                await db.commit()
                logger.info(f"Пользователь {user_id} обновлен до премиум статуса")
                return True
        except Exception as e:
            error_msg = f"Ошибка при обновлении пользователя {user_id} до премиум статуса: {str(e)}"
            logger.error(error_msg)
            return False

    async def get_user_stats(self, user_id: int) -> Optional[dict]:
        """
        Получение статистики пользователя.
        
        Args:
            user_id: ID пользователя
            
        Returns:
            Optional[dict]: Статистика пользователя или None
        """
        try:
            async with aiosqlite.connect(self.db_path) as db:
                # Получаем информацию о пользователе
                async with db.execute(
                    """
                    SELECT username, first_name, last_name, is_premium, created_at, last_active
                    FROM users WHERE user_id = ?
                    """,
                    (user_id,)
                ) as cursor:
                    user = await cursor.fetchone()
                
                if not user:
                    return None
                
                # Получаем статистику запросов
                async with db.execute(
                    """
                    SELECT request_date, request_count
                    FROM requests
                    WHERE user_id = ?
                    ORDER BY request_date DESC
                    LIMIT 7
                    """,
                    (user_id,)
                ) as cursor:
                    requests = await cursor.fetchall()
                
                return {
                    "username": user[0],
                    "first_name": user[1],
                    "last_name": user[2],
                    "is_premium": bool(user[3]),
                    "created_at": user[4],
                    "last_active": user[5],
                    "requests": {str(date): count for date, count in requests}
                }
        except Exception as e:
            error_msg = f"Ошибка при получении статистики пользователя {user_id}: {str(e)}"
            logger.error(error_msg)
            return None 