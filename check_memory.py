import config
import memory

# Проверка создания схемы
memory.create_schema()
assert config.DB_PATH.exists(), "Файл БД не создан"
print(f"БД создана: {config.DB_PATH}")

# Проверка идемпотентности
memory.create_schema()
print("Повторный вызов create_schema не упал - ок")

# Проверка сохранения и очистки
memory.clear("test")
memory.save("test", "user", "реплика 1")
memory.save("test", "assistant", "ответ 1")

conn = memory.connect()
try:
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
        ("test",)
    ).fetchall()
finally:
    conn.close()

assert len(rows) == 2, f"Ожидалось 2 реплики, получено {len(rows)}"
assert rows[0] == ("user", "реплика 1"), rows[0]
assert rows[1] == ("assistant", "ответ 1"), rows[1]
print("save: реплики записаны в правильном порядке - ок")

memory.clear("test")
conn = memory.connect()
try:
    count = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE session_id = ?",
        ("test",)
    ).fetchone()[0]
finally:
    conn.close()
assert count == 0, "clear не удалил реплики"
print("clear: реплики сессии удалены - ок")

# Проверка скользящего окна
memory.clear("t1")
for i in range(1, 11):
    role = "user" if i % 2 else "assistant"
    memory.save("t1", role, f"сообщение {i}")

w = memory.window("t1", 4)
assert [m.content for m in w] == ["сообщение 7", "сообщение 8", "сообщение 9", "сообщение 10"], "Окно вернуло не те сообщения или не в том порядке"
print("окно: 4 последних в хронологическом порядке - ок")