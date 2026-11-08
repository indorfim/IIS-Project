import logging
import uuid
import prompts
from orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO)

original = prompts.build_messages
captured = {}

def spy(role, question, chunks, history):
    msgs = original(role, question, chunks, history)
    captured["messages"] = msgs
    captured["role"] = role
    captured["question"] = question
    return msgs

prompts.build_messages = spy

o = Orchestrator()
sid = f"demo:{uuid.uuid4()}"

# Вопросы 
questions = [
    "Какие документы необходимо оформить по завершении приемо-сдаточных испытаний?",
    "А если в процессе был получен отрицательный результат?",
    "Кто именно должен подписывать этот акт?"
]

for q in questions:
    print(f"- {q}")
    print(o.handle(sid, q), "\n")

print("--- Дамп build_messages() для третьей реплики ---")
for i, m in enumerate(captured["messages"], 1):
    print(f"{i}. [{m['role']}] {m['content'][:120]}...")

o.close()