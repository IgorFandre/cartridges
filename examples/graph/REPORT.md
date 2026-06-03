# Отчёт: KV-cartridges для семейного графа

> Cartridges — small trainable KV caches that encode a corpus, attached to a frozen base model. We study whether a cartridge captures relational structure (a family tree) as well as in-context learning does, using controlled name-swap variants to isolate structural vs surface memorization.

---

## 1. Постановка

Обучаем cartridge — обучаемый KV-cache, накладываемый поверх замороженной модели Qwen3-1.7B — кодировать структуру семейного графа на 45 человек / 6 поколений и отвечать на вопросы о родстве. Три эксперимента:

- **Exp 1.** Сравнение **инициализированных** caches между 4 вариантами графа (без обучения)
- **Exp 2.** 4 cartridges от **общего** Alex-init, каждый обучен на данных своего варианта → init **идентичен**, post-train различия идут **только** от train data. Два среза: trained-KV diff (где осело различие) + cross-variant eval (own-acc vs cross-acc как индикатор структурного обобщения)
- **Exp 3.** ICL baseline (Qwen3-1.7B + corpus в prompt, без cartridge), 2 формата corpus
- **Exp 4.** Stability / noise floor (32-slot картриджи): 5 alex-seed прогонов = пол шума; type-compare = train-identity vs init-source

---

## 2. Инициализация: чем отличаются `graph-variant-masked-{alex,ben,carl,dan}`

**Init — идентичен.** Все 4 cartridges инициализируются одним и тем же:

```python
INIT_CORPUS = variants/alex/family_tree_corpus.txt    # hardcoded в training/train.py
kv_cache_initializer = KVFromText.Config(text_source=INIT_CORPUS, max_tokens=None)
```

`KVFromText` пропускает corpus через Qwen3-1.7B forward pass, собирает K/V tensors по всем layers/heads/slots. Результат — стартовый cache, **identical** для всех 4 runs. Это критично: разница после train объясняется **только** различиями в training data.

**Что различается между runs:**

| Компонент | alex run | ben run | carl run | dan run |
|---|---|---|---|---|
| Init corpus | `variants/alex/family_tree_corpus.txt` | (то же) | (то же) | (то же) |
| Init KV cache | identical (bit-for-bit) | (то же) | (то же) | (то же) |
| Train data | `variants/alex/train_mc.parquet` | `variants/ben/...` | `variants/carl/...` | `variants/dan/...` |
| Test data | `variants/alex/test_mc.parquet` | `variants/ben/...` | `variants/carl/...` | `variants/dan/...` |
| Seed, hyperparams | identical across all 4 | | | |

Граф **изоморфен** — только имя одного из основателей переименовано (`Jason → Alex/Ben/Carl/Dan`). 21 пара супругов, 46 parent-child edges, остальные 44 имени общие.

**Контраст с Exp 1:** там init **различается** между вариантами (каждый init из своего corpus), чтобы измерить эффект одной замены имени до обучения.

---

## 3. Датасет

### 3.1. Семейное дерево

`generate_tree.py`, параметры:
- `n_people=45, max_depth=6, founders=1, min_kids=1, max_kids=2, spouse_prob=0.95`
- Одна пара-основатель → каждое поколение 1-2 ребёнка → ~95% детей женятся
- Результат: 45 человек, 6 поколений, 21 пара супругов, 46 parent-child edges

### 3.2. Генерация QA

`data_gen/qagen.py` строит 9 категорий вопросов с помощью BFS-обхода (`FamilyTree.find_path_reasoning`):

| Cat | Описание | Options | Кол-во | Пример (вопрос → ответ) |
|-----|---|--:|--:|---|
| 1   | direct 1-hop single: father/mother/husband/wife (gender-gated) | 5 (A-E) | 135 | "Who is Mary's father?" → "John." |
| 1m  | direct 1-hop multi: sons/daughters | 5 | 90 | "Who are John's sons?" → "Paul, Steven." |
| 1w  | whose-style: "Whose son/father/husband is X?" | 5 | 225 | "Whose son is Paul?" → "John." |
| 2   | multi-hop single: grandfather/grandmother | 5 | 90 | "Who is Mary's grandfather?" → "George." |
| 2m  | multi-hop multi: brothers/sisters/uncles/aunts/grandsons/granddaughters/cousins | 5 | 315 | "Who are Mary's cousins?" → "Anna, Lucy." |
| 3   | counting: children/sons/daughters/siblings/grandchildren | 5 | 225 | "How many children does John have?" → "3." |
| 4   | verification: "Is A B's rel?" mix true/false | **3 (Yes/No/Unknown)** | 8100 | "Is John Mary's father?" → "Yes." |
| 5   | existence: "Does X have any rel?" | **3 (Yes/No/Unknown)** | 405 | "Does Mary have any sisters?" → "No." |
| 6   | disambig: "Name one of X's rel" — correct = lex-first valid | 5 | 40 | "Name one of John's sons." → "Paul." |

**Сырьё: 9625 QA** (Cat 4 доминирует ~84%). По умолчанию данные **ребалансируются**
(`MIX_DEFAULT`, флаг `--rebalance`/`--no-rebalance`) к целевой смеси: reasoning-категории
≈ натуральные доли, verification+existence суммарно **25%** → **1333 QA** (1067 train / 266 test).

⚠️ **Фикс (качество MC-опций).** Раньше диспетчеризация по категории в `build_mc_record`
сравнивала строковую `category` с int → **все** категории падали в name-multi ветку, и
Cat 4/5 получали мусорные опции (`"Is A B's father?"` → `["Emily.","Mary.","None.","No.","Dorothy, No."]`).
Исправлено → Cat 4/5 теперь Yes/No/Unknown (3 опции), Cat 3 — числа. **Все прежние данные/accuracy на диске были на битых опциях.**

Дополнительно: каждый record несёт `hops` (1/2/3 — hop-класс отношения, для per-hop eval);
буква правильного ответа **равномерна** (5-опц: 20% на A–E; 3-опц: 33% на A–C) через `balance_letters`.

Формат MC (`_format_mc`): вопрос + строки `A. <opt>`, ответ модели = одна буква. Примеры (имена иллюстративные):

```
Cat 1  — Who is Mary's father?          Cat 2m — Who are Mary's cousins?
  A. Steven                               A. Anna, Lucy
  B. John          → B                    B. None.            → A
  C. Paul                                 C. Steven, Paul
  D. George                               D. John
  E. None.                                E. George, Anna

Cat 1m — Who are John's sons?           Cat 3  — How many children does John have?
  A. Mary, Anna                           A. 1
  B. None.                                B. 4
  C. Paul, Steven  → C                    C. 3                → C
  D. Steven                               D. 0
  E. George                               E. 2

Cat 1w — Whose son is Paul?             Cat 4  — Is John Mary's father?
  A. George                               A. No
  B. John          → B                    B. Yes              → B
  C. Mary                                 C. Unknown
  D. Steven
  E. None.                              Cat 5  — Does Mary have any sisters?
                                          A. No               → A
Cat 2  — Who is Mary's grandfather?       B. Yes
  A. Paul                                 C. Unknown
  B. John
  C. George        → C                  Cat 6  — Name one of John's sons.
  D. Steven                               A. Paul             → A
  E. None.                                B. Anna
                                          C. George
                                          D. Mary
                                          E. Steven
```

- Cat 4/5 — 3 опции (Yes/No/Unknown), остальные — 5 (A-E)
- "None." / "Unknown" — distractor'ы, корректным ответом не бывают

Дизайн-решения:
- Опция "Unknown" — всегда distractor (никогда не правильный ответ)
- nephew/niece исключены (избыточны с uncle/aunt)
- Spousal вопросы gender-gated (нет "Who is Steven's husband?")
- Собственное имя person'a запрещено как distractor
- Cat 6 distractors **не входят** в valid set (чисто disambiguation)

### 3.3. Train/test split

Режим выбирается флагом `--split-mode` (в `qagen.py` и `generate_variants.py`),
20% в test, фиксированный seed → один и тот же hold-out во всех вариантах:

- **`question`** (текущий **дефолт**): рандомный hold-out 20% **вопросов**. Test
  может спрашивать про людей, **виденных** в train → ближе к стандартному random
  split / проверке recall, чем к обобщению.
- **`person`** (прежний дефолт): hold-out целых **людей** (36 train / 9 test).
  Гарантирует, что test — про **никогда не виденных** в train людей → проверяет
  структурное обобщение. Cross-references разрешены (ответ может быть train-человеком).

⚠️ Замена дефолта на `question` **меняет смысл** замеров §5.4 (cross-variant как
индикатор структуры): при question-сплите test уже не изолирует невиданных людей.
Для структурного теста запускать с `--split-mode person`.

### 3.4. Формат train-примера

Каждый QA → `Conversation`:
- `user`: question + 3 или 5 опций A-E
- `assistant`: **только letter** + точка (`"C."`). Никакого reasoning, никакого `<think>`.

```
user:      Who is Steven's wife?
           A. Kevin.
           B. None.
           C. Lisa.
           D. Karen.
           E. Jennifer.
assistant: D.
```

Metadata несёт `correct_letter`, `n_options`, `category`, `rel`, `person`, `options`.

---

## 4. Обучение

### 4.1. Loss spec: masked letter

`MaskedAnswerTrainDataset` (custom dataset в `masked_answer_dataset.py`) переопределяет `_prepare_elements`:

1. Tokenize всю conversation (user + assistant) стандартным converter Qwen3
2. Найти **позицию letter token** в assistant message (`_find_letter_position`)
3. Установить `topk_token_idxs = [letter_pos]` — единственная позиция, попадающая в loss
4. Установить `topk_token_ids = [letter_id]`

Loss в `train.py:407` использует `topk_token_idxs` — берёт логиты только в этих позициях и cross-entropy с `topk_token_ids`. Все остальные позиции (контекст, options, decimal points) **не дают градиента**.

**Почему так:**
- Модель не видит teacher-forced reasoning → не может skipать рассуждение
- На eval (`cot=True`) модель свободно генерирует `<think>...</think>` + letter. Distribution shift отсутствует
- Сигнал в каждом примере крошечный (1 токен), но cartridge обучается через десятки тысяч повторений
- Чистый изолированный тест: справится ли cartridge сам нести структуру?

### 4.2. Что обучается

**Frozen:** все веса Qwen3-1.7B  
**Trainable:** только `cache.trainable_keys[layer]` и `cache.trainable_values[layer]` — shape `(1, n_kv_heads=8, n_tokens_in_cache=282, head_dim=128)` для каждого из 28 слоёв

`TrainableCache` использует sentinel `CARTRIDGE_SEQ_ID = -1` чтобы FlexAttention понимал, что cartridge-токены доступны всем запросам.

### 4.3. Гиперпараметры (`training/train.py`)

Все env-знобы: `EPOCHS`, `LR`, `MAX_STEPS`, `SAVE_EVERY`.

| Параметр | Значение (дефолт) |
|---|---|
| Model | Qwen3-1.7B (FlexQwen3ForCausalLM) |
| Optimizer | AdamW (default) |
| LR | 2e-2 (`LR`) |
| Epochs | 10 (`EPOCHS`) — верхняя граница |
| **Max optimizer steps** | **100** (`MAX_STEPS`; `-1` = без лимита, весь EPOCHS) |
| **Checkpoint every** | **20 шагов** (`SAVE_EVERY`) → cache-step{20,40,60,80,100}.pt для dynamics |
| Global batch | 32 |
| Packed seq length | 1024 |
| Packing mode | pad |
| KV init | KVFromText на `variants/alex/family_tree_corpus.txt` (фикс для exp 2) |
| Seed | 42 |
| DDP backend | gloo |

Inline eval: `generate_eval_every_n_steps=150` (при `MAX_STEPS=100` срабатывает только финальный eval после обучения), `cot=True`, `max_new_tokens=256`, batch 8, temperature 0.

---

## 5. Валидация

### 5.1. Inline eval (во время train)

`GraphMCEvalDataset` (в `training/mc_eval.py`) на каждом 150-м шаге:
1. Берёт `test_mc.parquet` варианта
2. Применяет chat-template с `enable_thinking=True` (модель сама генерирует reasoning)
3. Генерирует до 256 новых токенов
4. Из выхода извлекает letter regex (`A-E` для 5-option, `A-C` для 3-option)
5. Сравнивает с `correct_letter` из metadata
6. Логирует `acc` в W&B

### 5.2. Финальная eval (после train)

`evaluation/eval.py --mode cartridge-cot --checkpoint cache_last.pt --variant-dir variants/<v>`:
- Прогон по всему test set (1925 QA)
- Per-question запись: question, options, correct_letter, predicted_text, predicted_letter, correct, n_options, category, rel, person
- Output: `results.json`

### 5.3. Post-hoc анализ

`evaluation/analyze.py results.json`:
- Overall accuracy
- Per-category accuracy
- Per-relation accuracy
- Distribution предсказанных vs корректных букв (sanity check на bias)
- Топ-N ошибок

### 5.4. Ключевой тест: cross-variant generalization

Запускаем cartridge `alex` на test set `ben` (и наоборот):

```bash
evaluation/eval.py --mode cartridge-cot \
  --checkpoint <alex_cartridge>/cache_last.pt \
  --variant-dir variants/ben \
  --output .../alex/eval_on_ben/results.json
```

Интерпретация:
- **Высокая cross-accuracy** → cartridge выучил **структуру** графа, имя не критично
- **Низкая cross, высокая own** → cartridge запомнил **поверхностное** соответствие "Jason → Alex"
- **Обе низкие** → train не сошёлся / loss spec слишком слабый

### 5.5. Baseline (Exp 3, без cartridge)

ICL eval тем же `evaluation/eval.py --mode icl-cot --n-shot 0 --corpus-path <path>`:
- corpus формат 1: structured (`Alex is married to Karen. Alex and Karen have children: ...`)
- corpus формат 2: prose narrative (`Alex and Karen built a life together. The couple had ...`)
- Без few-shot
- Cache system-prefix через `past_key_values` один раз → reuse на все 1925 вопросов

Сравнение через `evaluation/analyze.py cartridge.json icl_corpus.json icl_narrative.json` — показывает стоила ли тренировка ICL baseline.

---

## 6. Анализ KV-cache (без accuracy)

### 6.1. Exp 1: init KV diff (`comparison/compare.py --source init`)

Для каждой пары вариантов:
1. Build init cache → K/V shape `(1, H=8, T=282, D=128)` per layer
2. **Head-mean direction:** усреднить по головам → `(T, D)` per layer (user-requested aggregation)
3. Stack → `(L=28, T=282, D=128)`
4. Pairwise метрики per (layer, slot):
   - `cos` = `<d_A, d_B> / (|d_A|·|d_B|)`
   - `angle_deg` = `arccos(cos)`
   - `l2_shift` = `|d_A - d_B|`
   - `norm_ratio` = `|d_A| / |d_B|`
5. Output: heatmaps (layer × slot) для K/V × 4 метрики, summary JSON с top-K diverging slots

Подсветка позиций swapped name token (красные vlines) на heatmap → показывает локализацию изменения.

### 6.2. Exp 2: trained KV diff (`comparison/compare.py --source trained`)

Та же логика на trained caches. Сравнение init vs trained показывает где обучение **расплыло** изменение.

---

## 7. Структура результатов

> Канонічная карта путей (чекпоинты, compare, dynamics, env-vars и хелперы
> `paths.py`) — в [`OUTPUTS.md`](OUTPUTS.md). Ниже — обзорный срез.

```
outputs_graph/
├── exp1_init_kv/          compare_summary.json, localization.json + heatmaps
├── exp2_train/
│   ├── <variant>/         <launch_id>/<run_id>/cache-step*.pt (+ cache_last.pt)
│   │   ├── eval/          accuracy results.json
│   │   └── dynamics/      rotation analysis (dynamics.py)
│   └── compare/           trained-KV diff (+ spectra при --spectra)
├── exp3_icl/
│   ├── base/              corpus/ + narrative/ + compare.txt
│   └── alex/  ben/  …     то же per variant
└── exp4_stability/        *_run{i}/ checkpoints + alex_stability_compare/ + type_compare/
```

---

# ЧАСТЬ II. Результаты

> ⚠️ **Все ранее посчитанные результаты УСТАРЕЛИ и удалены из отчёта.** Причины:
> 1. **Баг MC-опций** (исправлен): `build_mc_record` сравнивал строковую `category`
>    с int → все категории попадали в name-multi ветку, Cat 4/5 (84% данных) имели
>    мусорные опции вместо Yes/No/Unknown. Любая прежняя accuracy недействительна.
> 2. **Датасет пересобран**: целевая смесь (verif+exist ≈ 25% вместо 88%), стратифицированный
>    сплит по вопросам (дефолт), равномерные буквы ответа, поле `hops`.
> 3. **Фикс кэша**: `TrainableCache.from_pretrained` ранее ронял первые H=8 слотов
>    (читал число frozen-токенов из размерности голов) → прежние trained-сравнения
>    (§ Exp 2/Exp 4) считались на усечённых картриджах.
> 4. На диске **нет** trained-чекпоинтов — обучение надо запустить заново.
>
> Эксперименты пересчитываются с нуля. Команды запуска — в [`README.md`](README.md)
> («Recompute everything»). Карта выходов — в [`OUTPUTS.md`](OUTPUTS.md).

## 8. Статус — всё к пересчёту

| Эксп | Что меряет | Драйвер | Статус |
|---|---|---|---|
| Exp 1 — init KV diff | что кодирует init (до обучения) | `compare.py --source init --spectra` | ⏳ recompute |
| Exp 2 — train (4 варианта) | картриджи от общего alex-init | `MODE=variants train.py` | ⏳ recompute |
| Exp 2 — trained KV diff | где осело различие после train | `compare.py --source trained --spectra` | ⏳ recompute |
| Exp 2 — cartridge accuracy | own-acc по категориям/хопам | `eval.py --mode cartridge-cot` + `analyze.py` | ⏳ recompute |
| Exp 2 — cross-variant | own-acc vs cross-acc (структура vs surface) | `eval.py … --variant-dir <other>` | ⏳ recompute |
| Exp 3 — ICL baseline | corpus vs narrative, per-hop vs cartridge | `eval.py --mode icl-cot` + `analyze.py` | ⏳ recompute |
| Exp 4 — stability | noise floor (seed-вариация) | `MODE=stability train.py` → `compare.py --run-prefix` | ⏳ recompute |
| Dynamics | вращение K/V по ходу обучения | `dynamics.py` (на `cache-step*.pt`) | ⏳ recompute |

После пересчёта сюда возвращаются: таблицы accuracy × {категория, hop} × {cartridge,
ICL-corpus, ICL-narrative}, cross-variant матрица, init/trained KV-diff с noise floor,
spectra и кривые вращения.

## 9. Что измерять при пересчёте (чек-лист)

- **Accuracy по hop-классу** (1/2/3) на test — где предел «хождения по графу»; сравнить
  cartridge vs ICL по хопам (`analyze.py results_cartridge.json results_icl.json`).
- **Per-category** accuracy против chance (5-опц = 20%, 3-опц = 33%) — баланс делает
  агрегат осмысленным.
- **Cross-variant**: разница own-acc − cross-acc как индикатор структурного обобщения
  (только при `--split-mode person` это честный тест на невиданных людей).
- **Init vs trained KV diff** против **noise floor** (Exp 4) — слот значим, если angle ≫ θ₀.
- **K vs V**: расходятся ли направления и спектры по-разному (Paper 2: keys-роутеры, values-контент).
- **Dynamics**: крутятся ли values сильнее keys и дольше по ходу обучения.

## 10. Открытые вопросы (дизайн, не зависят от старых чисел)

1. **Loss spec.** Letter-only выбран для чистоты. Если accuracy низкая — обсудить
   teacher-forced BFS-reasoning (с маскированием) или RL.
2. **Целевая смесь.** `MIX_DEFAULT` даёт verif+exist ≈ 25%; подобрать доли под цель
   (retrieval/reasoning) — стоит ли менять.
3. **Curriculum.** Подавать ли hop-1 раньше мультихопов (сейчас глобальный шафл; нужен
   отдельный режим). Гипотеза слабая — проверяемо.
4. **Размер картриджа.** Сейчас `max_tokens=None` (≈ длина корпуса). Варьировать и
   смотреть accuracy/размер.
5. **Cousin = бакет «3+»** (реальная цепочка 3–6). Хранить ли точный `chain_length`.
6. **Split mode.** `question` (дефолт) меряет recall; `person` — структурное обобщение.
   Какой считать основным для тезиса о «структуре».
