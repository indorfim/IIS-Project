import logging
logging.basicConfig(level=logging.INFO)
from orchestrator import Orchestrator

o = Orchestrator()
phrases = [
    "Какова последовательность приёмочных испытаний по ГОСТ 15.309?", # question -> analyst
    "Что делать при отрицательном результате?", # situation -> clarifier
    "Деталь не прошла проверку", # situation -> clarifier
    "абвгд 12345", # непонятная фраза -> уйдет в question (по умолчанию)
]

for p in phrases:
    intent = o.detect_intent(p)
    role = o.route("test_session", intent)
    print(f"{p[:40]: <42} интент: {intent: <12} роль: {role}")
o.close()