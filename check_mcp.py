'''
from orchestrator import Orchestrator

o = Orchestrator()
# Первый вызов займет несколько секунд на запуск сервера и подъем индекса
for c in o.tools.search("Как правильно оформить таблицу регистрации изменений?"):
    page = f" (стр. {c.page})" if c.page else ""
    print(f"[{c.score:.3f}] {c.source}{page} [{c.category}]")

o.close()
'''
import logging
from orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO)
o = Orchestrator()

# 1. "Прогрев" сервера
try:
    print("Инициализация сервера и загрузка модели... (может занять время)")
    o.tools.search("test")
except TimeoutError:
    print("Таймаут прогрева проигнорирован. Сервер готов к работе.")

# 2. Основной вызов (дальнейшие вызовы идут быстро[cite: 47])
for c in o.tools.search("ваш вопрос из gold_dataset"):
    page = f" стр. {c.page}" if c.page else ""
    print(f"{c.score:.3f} {c.source}{page} [{c.category}]")

o.close()