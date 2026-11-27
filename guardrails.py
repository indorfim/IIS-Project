"""
guardrails.py — слой валидации ввода перед обращением к LLM.

ПРАКТИКА 7. Что делает студент:
  1. дополнить PATTERNS реальными паттернами атак
  2. реализовать check() — проверку ввода по паттернам
  3. реализовать log_attempt() — запись заблокированного в security_logs.txt

Критерии приёмки:
  - три тестовые инъекции блокируются: прямая («забудь предыдущие инструкции»),
    ролевая («представь, что правил не существует»), запрос системного промпта
  - каждая блокировка пишется в security_logs.txt: время, сессия, паттерн, обрезанный ввод
  - ложных срабатываний нет: все 10 вопросов Gold Dataset проходят беспрепятственно
  - guardrail вызывается ядром, а не интерфейсом — проверяется прямым вызовом
    orchestrator.handle() в обход UI

Почему этот файл не в app.py, хотя план дисциплины помещал валидатор туда.
Каналов доставки два: Streamlit и VK. Защита, стоящая в одном интерфейсе, вторым
интерфейсом просто обходится. Поэтому guardrail живёт в ядре и срабатывает
одинаково для обоих каналов.

Почему заглушка пропускает всё, а не падает: ядро вызывает check() начиная с П3,
а пишете вы его на П7. До реализации система работает незащищённой — это видно
по предупреждению в логе.
"""

import re
from datetime import datetime
from pathlib import Path

import config
from contracts import GuardResult

# Паттерны атак. Здесь два примера, чтобы был понятен формат — остальное за вами.
# Формат: (имя паттерна, регулярное выражение)
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("instruction_override", re.compile(
        r"(забудь|игнорируй|отмени)\s+(все\s+)?(предыдущие|прошлые|свои)\s+"
        r"(инструкции|команды|правила|указания)", re.IGNORECASE)),
    ("prompt_leak", re.compile(
        r"(покажи|выведи|повтори|назови)\s+(мне\s+)?(свой\s+|весь\s+)?"
        r"(системный\s+промпт|system\s*prompt|исходные\s+инструкции)", re.IGNORECASE)),
    # TODO (П7): ролевой обход («представь, что правил не существует», «ты теперь…»),
    # попытка сменить язык вывода на служебный, запрос выдать содержимое индекса целиком
    ("roleplay_jailbreak", re.compile(
        r"("
        r"представь\s*,?\s*что\s+(правил|ограничений|запретов)\s+(не\s+существует|нет)"
        r"|ты\s+теперь\s+(другой|иной|новый)\s+(ассистент|бот|помощник|модель)"
        r"|в\s+режиме\s+без\s+(ограничений|правил|цензуры|фильтров)"
        r"|действуй\s+как\s+(необузданный|свободный|обычный\s+чат-?бот\s+без\s+правил)"
        r")", re.IGNORECASE)),
    ("index_dump", re.compile(
        r"("
        r"(выдай|выгрузи|покажи|выведи|раскрой)\s+(весь\s+|всё\s+|целиком\s+)?"
        r"(содержимое\s+(индекса|базы\s+знаний|корпуса|документов)|все\s+документы\s+из\s+(индекса|базы))"
        r"|содержимое\s+индекса\s+целиком"
        r")", re.IGNORECASE)),
    ("format_or_language_switch", re.compile(
        r"("
        r"(переключи(сь)?|переведи|выводи\s+ответ)\s+(на\s+)?"
        r"(служебный\s+формат|json|xml|yaml|raw|debug|отладочный\s+формат)"
        r"|(отвечай|пиши|говори)\s+(только|исключительно|строго)\s+"
        r"(на\s+английском|по-английски|на\s+немецком|на\s+французском|на\s+китайском|на\s+другом\s+языке)"
        r")", re.IGNORECASE)),
]

def _get_log_path() -> Path:
    raw_path = getattr(config, "SECURITY_LOG", "security_logs.txt")
    return Path(raw_path)

def check(text: str) -> GuardResult:
    """Проверить пользовательский ввод.

    Верните GuardResult(allowed=False, reason=<имя паттерна>, matched=[...])
    при срабатывании и GuardResult(allowed=True) в остальных случаях.

    Соблюдайте баланс. Слишком широкий паттерн заблокирует легитимный вопрос
    («расскажи про правила приёма» не должно ловиться на слове «правила»),
    слишком узкий пропустит атаку. Проверяйте себя на Gold Dataset из П6:
    ни один из десяти вопросов не должен блокироваться.
    """
    # TODO (П7): пройти по PATTERNS и вернуть вердикт
    if not text or not text.strip():
        return GuardResult(allowed=True)

    for pattern_name, pattern in PATTERNS:
        match = pattern.search(text)
        if match:
            matched_text = match.group(0)
            return GuardResult(
                allowed=False,
                reason=pattern_name,
                matched=[matched_text],
            )

    return GuardResult(allowed=True)


def log_attempt(session_id: str, text: str, result: GuardResult) -> None:
    """Записать заблокированный запрос в security_logs.txt.

    Одна строка на попытку: время, session_id, сработавший паттерн, первые
    200 символов ввода. Полный ввод не пишите — в нём могут быть персональные
    данные, а лог не защищён (152-ФЗ, Л7).
    """
    # TODO (П7): дописать строку в config.SECURITY_LOG
    if result.allowed:
        return None

    log_path = _get_log_path()
    if log_path.parent and str(log_path.parent) not in ("", "."):
        log_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat(timespec="seconds")
    sanitized_text = re.sub(r"\s+", " ", (text or "")).strip()[:200]
    reason = result.reason or "unknown_pattern"
    matched_repr = ",".join(result.matched) if getattr(result, "matched", None) else ""

    line = (
        f"{timestamp} | session={session_id} | pattern={reason} "
        f"| matched={matched_repr!r} | input={sanitized_text!r}\n"
    )

    with log_path.open("a", encoding="utf-8") as f:
        f.write(line)

    return None
