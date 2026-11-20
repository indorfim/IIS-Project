"""
test_suite.py — прогон Gold Dataset и оценка ответов промптом-судьёй.

ПРАКТИКА 6. Что делает студент:
  1. расширить gold_dataset.json с пяти вопросов, написанных на П2, до десяти,
     добавив к каждому эталонный ответ в поле expected
  2. написать JUDGE_PROMPT — промпт-судью по триаде RAG-метрик
  3. реализовать judge() — вызов судьи и разбор его вердикта

Готово: загрузка датасета, прогон через ядро, сводный отчёт.

Критерии приёмки:
  - в gold_dataset.json ровно 10 вопросов, среди них есть заведомо сложные:
    вне исходных данных, на стыке разделов, с неоднозначной формулировкой
  - пять вопросов с П2 сохранены; замена вопроса допускается только с письменным
    объяснением, чем он оказался негоден
  - прогон проходит все 10 без падений
  - судья возвращает три оценки триады на каждый вопрос, а не одну общую
  - в отчёте зафиксирована хотя бы одна пойманная галлюцинация
    (или обоснованно показано, что их нет)
  - обе метрики измерены и разведены: hit-rate@5 (механически, ingest.py, с П2)
    и Context Precision из триады (судьёй, здесь). В отчёте видно, что это
    разные числа, а не одно под двумя именами

Почему судьёй выступает LLM. Классические Accuracy и F1 на свободном тексте
не работают: верный ответ можно сформулировать десятком способов, и посимвольное
сравнение забракует их все. Сильная модель в роли эксперта-оценщика сравнивает
смысл — это и есть LLM-as-a-judge (Л6).

Запуск:
    python test_suite.py
"""

from copy import copy
import json
import re
import statistics
import uuid
from pathlib import Path

from llm_client import build_client
from orchestrator import Orchestrator
from prompts import format_chunks

DATASET = Path(__file__).parent / "gold_dataset.json"

# TODO (П6): промпт-судья.
# Он должен вернуть СТРОГО JSON с тремя оценками от 0 до 1 и коротким обоснованием:
#   faithfulness     — ответ не противоречит найденным фрагментам и ничего не выдумывает
#   answer_relevance — ответ отвечает именно на заданный вопрос
#   context_precision— найденные фрагменты действительно относятся к вопросу
# Требуйте JSON явно и покажите пример вывода — иначе разбирать будет нечего.
JUDGE_PROMPT = """Ты оцениваешь ответ RAG-системы по нормативной документации (ГОСТ, испытания, постановка продукции на производство).
Верни ТОЛЬКО один JSON-объект без markdown-ограждений и пояснений вокруг него.

Оцени три метрики от 0.0 до 1.0 раздельно:
- faithfulness: не противоречит ли ответ найденным фрагментам и не добавлены ли выдуманные факты (1.0 — строго по фрагментам или корректный отказ, 0.5 — есть неточности, 0.0 — противоречит фрагментам / выдумка).
- answer_relevance: отвечает ли ответ по существу на заданный вопрос (1.0 — по делу, 0.5 — частично, 0.0 — не по теме).
- context_precision: относятся ли найденные фрагменты к вопросу (1.0 — относятся, 0.5 — частично, 0.0 — не относятся).

Если case_type == "out_of_scope": эталон — отказ, context_precision ставь 0.0 и напиши в comment, что контекст для out_of_scope не определён (0.0); faithfulness ставь 1.0 при честном отказе без выдумок.

Пример вывода:
{"faithfulness": 1.0, "answer_relevance": 1.0, "context_precision": 0.8, "comment": "Краткое пояснение."}
"""


def _parse_verdict(raw: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("JSON не найден в ответе судьи")
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError("Ответ судьи не является JSON-объектом")

    for field in ("faithfulness", "answer_relevance", "context_precision"):
        if field not in data or isinstance(data[field], bool):
            raise ValueError(f"Некорректное или отсутствующее поле {field}")

    return {
        "faithfulness": round(max(0.0, min(1.0, float(data["faithfulness"]))), 4),
        "answer_relevance": round(max(0.0, min(1.0, float(data["answer_relevance"]))), 4),
        "context_precision": round(max(0.0, min(1.0, float(data["context_precision"]))), 4),
        "comment": str(data.get("comment", "")).strip(),
    }


MAX_CHARS_PER_CHUNK = 400


def _trim_chunks(chunks: list, max_chars: int = MAX_CHARS_PER_CHUNK) -> list:
    trimmed = []
    for ch in chunks or []:
        c = copy(ch)
        if isinstance(c, dict):
            for key in ("text", "content"):
                if isinstance(c.get(key), str) and len(c[key]) > max_chars:
                    c[key] = c[key][:max_chars].rstrip() + "…"
        else:
            for attr in ("text", "content"):
                val = getattr(c, attr, None)
                if isinstance(val, str) and len(val) > max_chars:
                    setattr(c, attr, val[:max_chars].rstrip() + "…")
        trimmed.append(c)
    return trimmed


def judge(
    llm,
    question: str,
    answer: str,
    expected: str,
    chunks: list | None = None,
    case_type: str = "direct",
) -> dict:
    """Оценить один ответ с урезанным по объёму текстом найденных фрагментов."""
    compact_chunks = _trim_chunks(chunks or [], MAX_CHARS_PER_CHUNK)
    chunks_str = format_chunks(compact_chunks)

    user_msg = (
        f"Тип вопроса: {case_type}\n"
        f"Вопрос: {question}\n"
        f"Эталон: {expected}\n"
        f"Ответ: {answer}\n\n"
        f"Найденные фрагменты:\n{chunks_str}"
    )
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    try:
        raw = llm.generate(messages, temperature=0.0)
        if not isinstance(raw, str) or raw.startswith("Ошибка при обращении"):
            raise RuntimeError(raw)
        verdict = _parse_verdict(raw)
        if case_type == "out_of_scope":
            verdict["context_precision"] = 0.0
            if "out_of_scope" not in verdict["comment"].lower():
                verdict["comment"] = f"[out_of_scope: context_precision=0.0] {verdict['comment']}"
        return verdict
    except Exception as exc:
        return {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_precision": 0.0,
            "comment": f"разбор: {exc}",
        }


def run() -> None:
    """Прогнать датасет и напечатать отчёт."""
    cases = json.loads(DATASET.read_text(encoding="utf-8"))
    if len(cases) < 10:
        print(f"ВНИМАНИЕ: в датасете {len(cases)} вопросов, по критериям П6 нужно 10.\n")

    orchestrator = Orchestrator()
    judge_llm = build_client()

    all_scores = {"faithfulness": [], "answer_relevance": [], "context_precision": []}
    in_scope_scores = {"faithfulness": [], "answer_relevance": [], "context_precision": []}
    out_scope_scores = {"faithfulness": [], "answer_relevance": [], "context_precision": []}
    failed_ids = []

    try:
        for case in cases:
            case_id = str(case.get("id", "?"))
            case_type = case.get("type", "direct")
            session_id = f"test:{uuid.uuid4()}"

            answer = orchestrator.handle(session_id, case["question"])
            chunks = getattr(orchestrator, "_context", {}).get(session_id, [])
            verdict = judge(
                judge_llm,
                question=case["question"],
                answer=answer,
                expected=case.get("expected", ""),
                chunks=chunks,
                case_type=case_type,
            )

            comment = verdict.get("comment", "")
            if comment.startswith(("разбор:", "сбой")):
                failed_ids.append(case_id)
                flag = "СБОЙ РАЗБОРА"
            else:
                for m in all_scores:
                    val = float(verdict.get(m, 0.0))
                    all_scores[m].append(val)
                    if case_type == "out_of_scope":
                        out_scope_scores[m].append(val)
                    else:
                        in_scope_scores[m].append(val)
                flag = "ГАЛЛЮЦИНАЦИЯ" if float(verdict.get("faithfulness", 1.0)) < 0.5 else "ok"

            print(
                f"[{case_id:>2}] ({case_type:<13}) {flag:<13} "
                f"F={verdict.get('faithfulness')} "
                f"AR={verdict.get('answer_relevance')} "
                f"CP={verdict.get('context_precision')}"
            )
            print(f"     вопрос: {case['question']}")
            print(f"     ответ:  {answer[:160]}")
            print(f"     судья:  {comment}\n")
    finally:
        orchestrator.close()

    print("--- Сводка ---")
    for label, bucket in [
        ("Все валидные", all_scores),
        ("In-scope", in_scope_scores),
        ("Out-of-scope", out_scope_scores),
    ]:
        print(f"{label}:")
        for metric, values in bucket.items():
            mean_str = f"{statistics.mean(values):.2f}" if values else "н/д"
            print(f"  {metric:<18} {mean_str}")

    print(f"Сбоев разбора: {len(failed_ids)} (id: {', '.join(failed_ids) if failed_ids else 'нет'})")
    hallucinations = sum(1 for v in all_scores["faithfulness"] if v < 0.5)
    print(f"Галлюцинаций: {hallucinations} из {len(all_scores['faithfulness'])}")


if __name__ == "__main__":
    run()