from gbd2 import generate
import asyncio
import logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token="6999052431:AAEqCw05fu5mNIcj4l2ewn6VLRmw1H3abJE")
# Диспетчер
dp = Dispatcher()


# Хэндлер на команду /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Приветствую, на связи бот Олег, задайте интересующий вас вопрос(без приветствия)!")


@dp.message(F.text)
async def answer_message(message: types.Message):
    msg_txt = message.text
    a = generate(msg_txt)
    await message.reply(text=a)


# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())