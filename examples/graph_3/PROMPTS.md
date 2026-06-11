# PROMPTS: полная постановка взаимодействия с LLM (graph_3)

Все промпты, форматы вопросов/ответов, параметры генерации и то, что попадает в
обучающие датасеты после self-study. Примеры — реальные, из сгенерированных данных
и ночных прогонов (10–11.06.2026).

> **Про thinking.** Ночные прогоны шли с `enable_thinking=False` — после анализа
> (ICL рухнул до 12%, модель не ищет по графу) режим переключён: все три команды
> идут с `--thinking` (`THINKING=0` возвращает старое). При thinking ассистентский
> текст начинается с `<think>…</think>`, парсер всегда читает зону **после**
> `</think>`. Примеры трейсов ниже — из ночного no-think прогона.

---

## 0. Карта стадий

| Стадия | system prompt | user | Кто/что отвечает | Парсинг/фильтр | Куда идёт |
|---|---|---|---|---|---|
| qagen (gold) | — | — | **не модель**: BFS по графу | — | `train/test_handshake.parquet` + meta |
| Exp 0 · ICL eval | инструкция + корпус | вопрос из test | модель (HF, greedy) | `extract_answer/path`, без фильтра | `exp0_icl/results.json` |
| Exp 1 · фаза 1 | инструкция + корпус | вопрос из train | модель (Tokasaurus, greedy) | верно → в датасет; неверно → фаза 2 | `exp1_adaptive/artifact/dataset.parquet` |
| Exp 1 · фаза 2 | инструкция + корпус + **worked scratchpad** | тот же вопрос | модель (greedy) | **без фильтра** (correct — флаг) | туда же |
| Exp 2 | инструкция + корпус | вопрос из train | модель (greedy; доп. пассы temp=0.7) | **без фильтра** (correct — флаг) | `exp2_plain/artifact/dataset.parquet` |
| Обучение (KL) | **пустой** (граф в картридже) | вопрос из parquet | — (дистилляция top-k logprobs ассистента) | — | `cache-step*.pt` |
| Train-time eval | пустой + картридж | вопрос из test | модель+картридж | `extract_answer` → acc | wandb |
| Cartridge eval | `None` + картридж | вопрос из test | модель+картридж (greedy) | как Exp 0 | `exp*/eval/results.json` |

Ключевая симметрия: **system prompt синтеза и ICL-эвала — один и тот же текст**
(`evaluation/eval.py::_SYSTEM_INSTRUCTION`, импортируется синтезом), а при обучении
и cartridge-эвале граф уезжает из промпта в картридж. Рассогласования форматов
train↔eval (как в graph_2) нет.

---

## 1. Корпус (граф как текст)

3145 токенов Qwen3-токенизатором; 392 строки — по предложению на ребро, каждое
ребро один раз, порядок строк перемешан (seed), чтобы не кодировать структуру
компонент порядком:

```
Stephen and Gerald know each other.
Brennan and Morales know each other.
Abigail and Teresa know each other.
Peterson and Thomas know each other.
... (всего 392 строки)
```

---

## 2. QA-датасет: вопрос и gold-ответ

### Шаблон вопроса (один на весь проект — train, test, все эксперименты)

```
How many handshakes apart are {X} and {Y}? If they are not connected, say so.
End your reply with "path: {X} - ... - {Y}" and "Answer: <number>", or
"Answer: not connected".
```

### Gold-ответ (assistant в QA-parquet; используется только как разметка)

```
path: {X} - {A} - {B} - {Y}        # позитив: единственный путь в дереве
Answer: {d}
```
```
Answer: not connected               # негатив (разные компоненты)
```

### Реальные примеры всех типов

**train, hop=1** (ребро есть строкой в корпусе):
```
Q:    How many handshakes apart are Goodman and Bond? If they are not connected, say so.
      End your reply with "path: Goodman - ... - Bond" and "Answer: <number>", or "Answer: not connected".
gold: path: Goodman - Bond
      Answer: 1
meta: {"answer": "1", "true_distance": 1, "n_bucket": "1", "x": "Goodman", "y": "Bond"}
```

**train, hop=6** (максимальная глубина в train):
```
Q:    How many handshakes apart are Janet and Hudson? ...
gold: path: Janet - Johnston - Hale - Jordan - Schmidt - Murphy - Hudson
      Answer: 6
```

**train, none** (12% train; разные компоненты):
```
Q:    How many handshakes apart are Brittany and Faulkner? ...
gold: Answer: not connected
meta: {"answer": "not connected", "true_distance": null, "n_bucket": "none"}
```

**test, hop=8** (test-only: проверка обобщения, в train хопов 7–8 нет):
```
Q:    How many handshakes apart are Cunningham and Burgess? ...
gold: path: Cunningham - Linda - Paul - Wheeler - Graham - Hooper - Barker - Fowler - Burgess
      Answer: 8
```

Поля meta-записи (train/test_meta.json): `question, answer, true_distance,
n_bucket ("1".."8"|"none"), x, y, path (list|null)`.

---

## 3. System prompt — инструкция + корпус (Exp 0 / Exp 1 фаза 1 / Exp 2)

Дословно (`evaluation/eval.py::_SYSTEM_INSTRUCTION`):

```
You will be asked how many handshakes apart two people are. Use the friendship
list below. Two people are 1 handshake apart if they know each other directly;
people in different friend groups are not connected.

Friendships:
{corpus}
```

Сообщения, уходящие в модель:
```json
[
  {"role": "system", "content": "<инструкция + 392 строки корпуса>"},
  {"role": "user",    "content": "How many handshakes apart are Christina and Brooks? ..."}
]
```

---

## 4. Hint-промпт Exp 1 (фаза 2 — только для вопросов, где фаза 1 ошиблась)

Шаблон (`synthesis/exp1_synthesize.py::_HINT_TMPL`):

```
{инструкция + корпус — как в §3}

A worked breadth-first search for this exact question:
{scratchpad}

Use it to answer the question correctly, showing the search the same way.
```

`{scratchpad}` генерируется детерминированно из графа
(`GraphIndex.scratchpad(x, y, rng)`, rng сидируется `"{seed}:{x}:{y}"`,
порядок соседей внутри уровня рандомизирован). Полный реальный пример
(Christina ↔ Brooks, gold=4):

```
queue: [Christina(0)]
pop Christina(0) -> visit, push Kyle(1), Harvey(1), Maria(1), Dickson(1), Olivia(1) -> queue: [Kyle(1), Harvey(1), Maria(1), Dickson(1), Olivia(1)]
pop Kyle(1) -> visit, push Garza(2) -> queue: [Harvey(1), Maria(1), Dickson(1), Olivia(1), Garza(2)]
pop Harvey(1) -> visit, push Hodge(2) -> queue: [Maria(1), Dickson(1), Olivia(1), Garza(2), Hodge(2)]
pop Maria(1) -> visit, push Karen(2) -> queue: [Dickson(1), Olivia(1), Garza(2), Hodge(2), Karen(2)]
pop Dickson(1) -> visit -> queue: [Olivia(1), Garza(2), Hodge(2), Karen(2)]
pop Olivia(1) -> visit, push Emerson(2), Snyder(2), Andrews(2) -> queue: [Garza(2), Hodge(2), Karen(2), Emerson(2), Snyder(2), Andrews(2)]
pop Garza(2) -> visit -> queue: [Hodge(2), Karen(2), Emerson(2), Snyder(2), Andrews(2)]
pop Hodge(2) -> visit, push Mendoza(3) -> queue: [Karen(2), Emerson(2), Snyder(2), Andrews(2), Mendoza(3)]
pop Karen(2) -> visit, push Brian(3), Juan(3), Andrew(3), Carol(3), Gonzales(3), Dudley(3) -> queue: [...]
... (по строке на каждого посещённого)
pop Carol(3) -> visit, push Lisa(4), Brooks(4) -> target Brooks reached at distance 4
path: Christina - Maria - Karen - Carol - Brooks
Answer: 4
```

Для не-связанных пар scratchpad обходит всю компоненту и заканчивается:
```
queue: [] -> frontier exhausted, {Y} not reached
Answer: not connected
```

Длины подсказки (медианы): hop1 ≈ 45 ток., hop3 ≈ 182, hop6 ≈ 818, none ≈ 2200
(max 2711). User-сообщение в фазе 2 — тот же вопрос, без изменений.

---

## 5. Параметры генерации (все стадии)

| Стадия | temp | max tokens | top_logprobs | thinking (сейчас) |
|---|---|---|---|---|
| Exp 0 ICL eval | 0.0 (greedy) | 4096 | — | `--thinking` (вкл. по умолч. в скрипте) |
| Exp 1 фаза 1 | 0.0 | 4096 | 20 (min_prob_mass 0.99) | вкл. |
| Exp 1 фаза 2 | 0.0 | 4096 | 20 | вкл. |
| Exp 2 пасс 0 | 0.0 | 4096 | 20 | вкл. |
| Exp 2 пассы 1+ (`--samples-per-question`>1) | 0.7 | 4096 | 20 | вкл. |
| Cartridge eval | 0.0 | 4096 | — | флаг `--thinking` |
| Train-time eval | 0.0 | 4096 | — | `cot` в конфиге |

Сервер: Tokasaurus, `model=Qwen/Qwen3-1.7B`, `max_top_logprobs=20`. Exp 0 и
cartridge-eval — HF-модель локально на GPU, без сервера.

---

## 6. Что попадает в датасет после self-study

Каждый train-вопрос даёт **ровно один** `Conversation` в обоих экспериментах
(датасеты size-matched: 2045 vs 2045). Схема:

```python
Conversation(
    system_prompt="",            # ПУСТОЙ: при обучении граф живёт в картридже
    type="handshake_selfstudy",
    metadata={
        "question", "answer",            # вопрос и gold ("3" | "not connected")
        "true_distance", "n_bucket",     # int|null, "1".."8"|"none"
        "x", "y",
        "source",                        # "no_hint" | "with_hint" (exp2 — всегда no_hint)
        "correct",                       # финальный ответ трейса == gold (флаг, НЕ фильтр)
        "temp",
    },
    messages=[
        Message(role="user",      content=<вопрос>,        token_ids=None, top_logprobs=None),
        Message(role="assistant", content=<ТРЕЙС МОДЕЛИ>,  token_ids=[...], top_logprobs=<top-20>),
    ],
)
```

Обучение (`targets="logits"`): KL-дистилляция top-20 logprobs ассистентского
сообщения; модель-студент видит картридж + пустой system + user-вопрос и учится
воспроизводить распределение учителя (той же модели с графом/подсказкой в
контексте) на токенах трейса.

### Реальные трейсы из ночного прогона (no-think)

**Exp 1, `source="no_hint"`, correct (none-бакет; таких 260, из них 241 none):**
```
USER:      How many handshakes apart are Brittany and Faulkner? ...
ASSISTANT: To determine how many handshakes apart Brittany and Faulkner are, we need to
           check if they are connected by a friendship (direct handshake) or if they are
           in different friend groups.

           After examining the list of friendships, **Brittany and Faulkner are not connected**.

           **Answer: not connected**
           path: Brittany - ... - Faulkner
```

**Exp 1, `source="with_hint"`, correct (hop4; таких 1749 — 97.9% случаев подсказка
исправляет; модель воспроизводит scratchpad-формат):**
```
USER:      How many handshakes apart are Christina and Brooks? ...
ASSISTANT: queue: [Christina(0)]
           pop Christina(0) -> visit, push Kyle(1), Harvey(1), Maria(1), Dickson(1), Olivia(1) -> queue: [...]
           pop Kyle(1) -> visit, push Garza(2) -> queue: [...]
           ... (полный лог)
           pop Carol(3) -> visit, push Lisa(4), Brooks(4) -> target Brooks reached at distance 4
           path: Christina - Maria - Karen - Carol - Brooks
           Answer: 4
```

**Exp 1, `source="with_hint"`, WRONG (таких 36 — оставлены с `correct=false`;
модель пересказывает scratchpad маркдауном и сбивается):**
```
USER:      How many handshakes apart are Dalton and Price? ...   gold=5
ASSISTANT: We will perform a breadth-first search to determine the number of handshakes
           between Dalton and Price.
           **BFS Steps:**
           1. **Start with Dalton (0)**
              - Visit Dalton (0)
              - Push Amber (1) to queue
           ... (уходит в свободный пересказ и ошибается)
```

**Exp 2, WRONG (типичный: 87% датасета; gold=3, модель даже не ищет):**
```
ASSISTANT: To determine how many handshakes apart Clayton and Jordan are, we need to check
           if they are connected by a direct handshake. ...
           After analyzing the friendship list, Clayton and Jordan are **not connected** by
           a direct handshake.
           path: Clayton - not connected - Jordan
           Answer: not connected
```

**Exp 2, correct (hop1; формат «грязный», но финальный Answer верен — парсер
берёт последнее вхождение):**
```
ASSISTANT: ... From the given list of friendships, **Thomas and Peterson** are directly connected.
           **path: Thomas - Peterson**
           Answer: Thomas - Peterson
           Answer: 1
```

### Состав ночных датасетов

| | exp1_adaptive | exp2_plain |
|---|---|---|
| трейсов | 2045 | 2045 |
| no_hint / with_hint | 260 / 1785 | 2045 / 0 |
| верных | 2009 (98.2%) | 261 (12.8%) |
| медиана токенов трейса | 205 | 73 |
| logprobs | 100% | 100% |

---

## 7. Эвал: что видит модель и как скорится

**ICL (Exp 0):** system из §3, user-вопрос, greedy. KV system-префикса кэшируется
один раз и переиспользуется на все 900 вопросов.

**Cartridge:** system **отсутствует**, граф — в обученном KV-кэше (629 токенов);
user-вопрос тот же. Генерация `flex_generate` батчами.

**Парсинг (общий для всех):** зона после последнего `</think>` (незакрытый
`<think>` → unparsed); regex `Answer:\s*(not connected|\d+)` — последнее
вхождение; `path:` — последняя строка, сплит по «-». Метрики на строку:
`correct` (exact match), `fidelity` (correct И путь == gold-путь; для none =
correct), `path_valid` (все соседние пары — рёбра), `abs_err`. Unparsed
добивается `--rerun-unparsed`.
