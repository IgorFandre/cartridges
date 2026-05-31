# Отчёт: KV-cartridges для семейного графа

> Cartridges — small trainable KV caches that encode a corpus, attached to a frozen base model. We study whether a cartridge captures relational structure (a family tree) as well as in-context learning does, using controlled name-swap variants to isolate structural vs surface memorization.

---

## 1. Постановка

Обучаем cartridge — обучаемый KV-cache, накладываемый поверх замороженной модели Qwen3-1.7B — кодировать структуру семейного графа на 45 человек / 6 поколений и отвечать на вопросы о родстве. Три эксперимента:

- **Exp 1.** Сравнение **инициализированных** caches между 4 вариантами графа (без обучения)
- **Exp 2.** Обучение 4 cartridges от **общего** init на разных вариантах графа, сравнение обученных
- **Exp 3.** ICL baseline (Qwen3-1.7B + corpus в prompt, без cartridge), 2 формата corpus

---

## 2. Инициализация: чем отличаются `graph-variant-masked-{alex,ben,carl,dan}`

**Init — идентичен.** Все 4 cartridges инициализируются одним и тем же:

```python
INIT_CORPUS = variants/alex/family_tree_corpus.txt    # hardcoded в graph_train_variants.py
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

`graph_qagen.py` строит 8 категорий вопросов с помощью BFS-обхода (`FamilyTree.find_path_reasoning`):

| Cat | Описание | Options | Кол-во |
|-----|---|--:|--:|
| 1   | direct 1-hop single: father/mother/husband/wife (gender-gated) | 5 (A-E) | 135 |
| 1m  | direct 1-hop multi: sons/daughters | 5 | 90 |
| 1w  | whose-style: "Whose son/father/husband is X?" | 5 | 225 |
| 2   | multi-hop single: grandfather/grandmother | 5 | 90 |
| 2m  | multi-hop multi: brothers/sisters/uncles/aunts/grandsons/granddaughters/cousins | 5 | 315 |
| 3   | counting: children/sons/daughters/siblings/grandchildren | 5 | 225 |
| 4   | verification: "Is A B's rel?" mix true/false | **3 (Yes/No/Unknown)** | 8100 |
| 5   | existence: "Does X have any rel?" | **3 (Yes/No/Unknown)** | 405 |
| 6   | disambig: "Name one of X's rel" — correct = lex-first valid | 5 | 40 |

**Итого: 9625 QA, ~214 на человека.** Cat 4 доминирует (~84%) и масштабируется через `--n-verif-per-rel`.

Дизайн-решения:
- Опция "Unknown" — всегда distractor (никогда не правильный ответ)
- nephew/niece исключены (избыточны с uncle/aunt)
- Spousal вопросы gender-gated (нет "Who is Steven's husband?")
- Собственное имя person'a запрещено как distractor
- Cat 6 distractors **не входят** в valid set (чисто disambiguation)

### 3.3. Train/test split

**By-person, 20% test:**
- 36 people train, 9 people test
- Train: 7700 QA, test: 1925 QA
- Cross-references **разрешены**: вопрос про test-person может иметь train-person в ответе

Преимущество перед random split: гарантирует что test содержит вопросы про **никогда не виденных** в train людей, проверяя структурное обобщение.

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

### 4.3. Гиперпараметры (`graph_train_variants.py`)

| Параметр | Значение |
|---|---|
| Model | Qwen3-1.7B (FlexQwen3ForCausalLM) |
| Optimizer | AdamW (default) |
| LR | 2e-2 |
| Epochs | 10 |
| Global batch | 32 |
| Packed seq length | 1024 |
| Packing mode | pad |
| KV init | KVFromText на `variants/alex/family_tree_corpus.txt` (фикс для exp 2) |
| Seed | 42 |
| DDP backend | gloo |

Inline eval: `generate_eval_every_n_steps=150`, `cot=True`, `max_new_tokens=256`, batch 8, temperature 0.

---

## 5. Валидация

### 5.1. Inline eval (во время train)

`GraphMCEvalDataset` (в `graph_mc_eval.py`) на каждом 150-м шаге:
1. Берёт `test_mc.parquet` варианта
2. Применяет chat-template с `enable_thinking=True` (модель сама генерирует reasoning)
3. Генерирует до 256 новых токенов
4. Из выхода извлекает letter regex (`A-E` для 5-option, `A-C` для 3-option)
5. Сравнивает с `correct_letter` из metadata
6. Логирует `acc` в W&B

### 5.2. Финальная eval (после train)

`graph_eval.py --mode cartridge-cot --checkpoint cache_last.pt --variant-dir variants/<v>`:
- Прогон по всему test set (1925 QA)
- Per-question запись: question, options, correct_letter, predicted_text, predicted_letter, correct, n_options, category, rel, person
- Output: `results.json`

### 5.3. Post-hoc анализ

`analyze_results.py results.json`:
- Overall accuracy
- Per-category accuracy
- Per-relation accuracy
- Distribution предсказанных vs корректных букв (sanity check на bias)
- Топ-N ошибок

### 5.4. Ключевой тест: cross-variant generalization

Запускаем cartridge `alex` на test set `ben` (и наоборот):

```bash
graph_eval.py --mode cartridge-cot \
  --checkpoint <alex_cartridge>/cache_last.pt \
  --variant-dir variants/ben \
  --output .../alex/eval_on_ben/results.json
```

Интерпретация:
- **Высокая cross-accuracy** → cartridge выучил **структуру** графа, имя не критично
- **Низкая cross, высокая own** → cartridge запомнил **поверхностное** соответствие "Jason → Alex"
- **Обе низкие** → train не сошёлся / loss spec слишком слабый

### 5.5. Baseline (Exp 3, без cartridge)

ICL eval тем же `graph_eval.py --mode icl-cot --n-shot 0 --corpus-path <path>`:
- corpus формат 1: structured (`Alex is married to Karen. Alex and Karen have children: ...`)
- corpus формат 2: prose narrative (`Alex and Karen built a life together. The couple had ...`)
- Без few-shot
- Cache system-prefix через `past_key_values` один раз → reuse на все 1925 вопросов

Сравнение через `analyze_results.py cartridge.json icl_corpus.json icl_narrative.json` — показывает стоила ли тренировка ICL baseline.

---

## 6. Анализ KV-cache (без accuracy)

### 6.1. Exp 1: init KV diff (`compare_init_kv.py`)

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

### 6.2. Exp 2: trained KV diff (`compare_kv.py`)

Та же логика на trained caches. Сравнение init vs trained показывает где обучение **расплыло** изменение.

---

## 7. Структура результатов

```
outputs_graph/
├── exp1_init_kv/          init_summary.json + heatmaps
├── exp2_train/
│   ├── alex/  ben/  …     cache_last.pt + eval/ + eval_on_<other>/
│   └── compare/           trained-KV diff
└── exp3_icl/
    ├── base/              corpus/ + narrative/ + compare.txt
    └── alex/  ben/  …     то же per variant
```

---

# ЧАСТЬ II. Уже посчитанные эксперименты

В `outputs_graph/` уже посчитаны: **Exp 1** (init KV diff, все 4 варианта, 6 пар) и **Exp 2 compare** (trained KV diff, только пара alex|ben — carl/dan ещё не обучены). ICL baseline и cartridge accuracy ещё не запущены.

---

## 8. Конфигурация замеров

- **Model:** Qwen3-1.7B
- **Cache shape:** 28 layers × 8 KV heads × **282 tokens** × 128 head_dim
- **Corpus tokens с swapped name:** позиции **3 и 9** (anchor founder появляется дважды в structured listing)
- **Метрики** (head-averaged direction per (layer, slot)):
  - `cos` — direction similarity (1.0 = identical)
  - `angle_deg` = arccos(cos)
  - `rel_l2` = `|A − B| / |A|`
  - `norm_ratio` = `|A| / |B|`

---

## 9. Exp 1 — Init KV diff (без обучения, все 6 пар)

### 9.1. Aggregate per pair (mean over layers, slots)

| Pair | K angle° | V angle° | K rel-l2 | V rel-l2 | K norm_ratio | V norm_ratio |
|---|--:|--:|--:|--:|--:|--:|
| alex \| ben  | 1.73 | 3.02 | 0.32 | 1.16 | 1.000 | 1.000 |
| alex \| carl | 2.14 | 3.52 | 0.40 | 1.33 | 1.000 | 1.000 |
| alex \| dan  | 2.70 | 4.39 | 0.49 | 1.67 | 1.000 | 1.000 |
| ben \| carl  | 2.03 | 3.37 | 0.38 | 1.29 | 1.000 | 1.000 |
| ben \| dan   | 2.61 | 4.15 | 0.47 | 1.61 | 1.000 | 1.000 |
| carl \| dan  | 2.95 | 4.54 | 0.53 | 1.78 | 1.000 | 1.000 |

**Наблюдения:**
- Углы малые (1.7°–4.5°), но **систематические** — замена одного имени двигает direction в каждом slot'е.
- **norm preserved** (`norm_ratio ≈ 1.0` всюду) → меняется **направление** ключей/значений, не магнитуда.
- V каналы расходятся **в 1.5–2× сильнее** чем K. Объяснение: V несёт «контент» (имена, факты), K — «адресацию» (позиционно-структурную).
- Расстояние между парами **зависит от выбора имён**: alex|dan расходится сильнее всего, alex|ben — слабее. Гипотеза: tokenization-зависимо (длина имени в BPE, частотность токена).

### 9.2. Name-slots vs other-slots

| Pair | K@name° | K@other° | ratio | V@name° | V@other° | ratio |
|---|--:|--:|--:|--:|--:|--:|
| alex \| ben  | **3.03** | 1.72 | **1.76** | **6.33** | 3.00 | **2.11** |
| alex \| carl | 1.99 | 2.14 | 0.93 | 4.00 | 3.51 | 1.14 |
| alex \| dan  | 2.58 | 2.70 | 0.95 | 5.00 | 4.39 | 1.14 |
| ben \| carl  | 2.75 | 2.03 | 1.36 | 5.76 | 3.36 | **1.72** |
| ben \| dan   | 2.12 | 2.61 | 0.81 | 4.76 | 4.14 | 1.15 |
| carl \| dan  | 2.11 | 2.95 | 0.72 | 4.18 | 4.54 | 0.92 |

**Ключевой результат:** ratio name/other **непостоянен** по парам. Для `alex|ben` divergence сконцентрирован в name slots (×2 для V), но для `carl|dan` — наоборот, name slots дивиржируют **меньше** среднего.

Это значит: **замена имени не локализована в name slots на уровне init KV.** Effect расплывается по контексту. Возможные причины:
- BPE: разные имена могут токенизироваться в разное число sub-tokens → сдвигает позиционные кодировки последующих слов
- Attention head mixing: даже до обучения, K/V одного слота — сумма attention-weighted contributions всех предшествующих токенов

### 9.3. Top диверджирующие slots (K)

Через все 6 пар топ-5 slots по mean-over-layers angle стабильно содержит: **5, 6, 11, 12, 13** (и иногда **190**).

Name positions 3, 9 — **НЕ** в топе. То есть наибольшее расхождение между init caches лежит в slots сразу **после** позиции имени (контекстное распространение), а не в самих name slots.

Slot 190 — обособленная точка ближе к концу corpus. Возможно совпадает с другим именем-токеном в later sentences (нужна проверка).

---

## 10. Exp 2 — Trained KV diff (alex vs ben, оба обучены)

Checkpoint paths:
```
checkpoints_variants/alex/2026-05-27-12-05-02-.../cache_last.pt
checkpoints_variants/ben/2026-05-27-12-05-44-.../cache_last.pt
```

### 10.1. Aggregate

| Pair | K cos | K angle° | K rel-l2 | V cos | V angle° | V rel-l2 |
|---|--:|--:|--:|--:|--:|--:|
| alex \| ben (init)    | 0.9996 | 1.73° | 0.32 | 0.9986 | 3.02° | 1.16 |
| alex \| ben (trained) | **0.9992** | **2.29°** | **0.037** | **0.9694** | **14.2°** | **0.150** |

**Парадокс масштабов** (отдельный effect — нужно проверить normalization!): rel-l2 у trained сильно **меньше** init (0.037 vs 0.32 для K). Возможные объяснения:
- Init values имеют большую magnitude (KVFromText проходит base model — высокие активации). После train cache scaled / regularized → меньшая norm.
- Норма `|A|` в знаменателе rel-l2 различается между init и trained.

**Направление (cos / angle):** training **усиливает** дивергенцию между alex и ben.
- K: 1.73° → 2.29° (×1.3)
- V: 3.02° → **14.2°** (×4.7)

Это ожидаемо: training специализирует cartridge под свою train data. Но **V расходится в 4.7× сильнее** чем при init, тогда как K только в 1.3×. → **Контентная составляющая (V) — главный носитель различий после train**, structural (K) почти не меняется относительно init.

### 10.2. Артефакты

Сохранены heatmaps (28×282 layer × slot):
- `heatmap_{K,V}_{cos,rel_l2}.png` — global per-layer
- `slot_alex_ben_{K,V}_{cos,rel_l2}.png` — per-slot detail для пары alex|ben

---

## 11. Status и что осталось

| Эксп | Status | Output |
|---|---|---|
| Exp 1 init KV (all 4 variants, 6 pairs) | ✅ done | `outputs_graph/exp1_init_kv/init_summary.json` |
| Exp 2 train alex                         | ✅ done | `checkpoints_variants/alex/.../cache_last.pt` |
| Exp 2 train ben                          | ✅ done | `checkpoints_variants/ben/.../cache_last.pt` |
| Exp 2 train carl, dan                    | ⏳ pending | — |
| Exp 2 compare alex \| ben                | ✅ done | `exp2_train/compare/compare_summary.json` + 8 heatmap PNG |
| Exp 2 compare всех 4 пар (после train carl/dan) | ⏳ pending | — |
| Exp 2 cartridge accuracy (cartridge-cot eval)   | ⏳ pending | — |
| Exp 2 cross-variant eval (alex→ben, ben→alex)   | ⏳ pending | — |
| Exp 3 ICL base corpus + narrative               | ⏳ pending | — |
| Exp 3 ICL per-variant                           | ⏳ pending | — |

---

## 12. Главные тезисы для доклада

> **Tезис 1.** Замена одного имени в corpus уже двигает init KV cache, но не локализованно: divergence затрагивает целиком окружающий контекст (top slots = post-name positions, не сами name slots).

> **Tезис 2.** Magnitude direction сохранена при init (`norm_ratio ≈ 1.0`) — изменяется только direction в head-mean space.

> **Tезис 3.** Train усиливает разделение между variants преимущественно в **V** (×4.7), почти не трогая **K** (×1.3). Cartridge учится разносить **содержимое** ответов, не **структуру** обращения.

> **Tезис 4.** ratio name-slot/other-slot **не стабилен по парам** — пара (alex, ben) показывает чистую локализацию в name slots, но (carl, dan) нет. Гипотеза: BPE-токенизация конкретных имён влияет неоднородно.

---

## 13. Ключевые вопросы для обсуждения с лабораторией

1. **Loss spec.** Letter-only выбран сознательно для чистоты — но если accuracy не растёт, обсудить teacher-forced BFS reasoning (с маскированием) или RL.
2. **Cat 4 disbalance.** Verification = 84% датасета. Можно ли это считать честным или нужна явная балансировка по категориям (sub-sample)?
3. **Init choice.** Alex как anchor — произвольный выбор. Альтернатива: усреднённый init из всех 4 corpus.
4. **Cross-variant как primary metric.** Я бы предложил считать **разницу** own-acc vs cross-acc главным индикатором "structural understanding".
5. **Размер cartridge.** Сейчас `max_tokens=None` (size = длина corpus, ~282 токена). Стоит варьировать.
6. **Eval cost.** 256 new tokens × 1925 вопросов × 4 variants × N inline evals — основная статья расходов. Можно subsample test set или сократить max_new.
7. **Норма rel-l2 у trained << init** — physical или artifact метрики? Проверить через absolute |A−B| без нормировки.
8. **K почти не меняется при train, V радикально** — это хорошо (cartridge не ломает structural attention) или сигнал что K cache слишком «жёсткий» и его нужно тоже учить интенсивнее?
9. **Top-divergence на slot 190** — есть второй name token там? Декодировать корпус и проверить.

---

## 14. Что подготовить для презентации

- [ ] Tree visualization (graphviz из `family_tree.json`) + corpus.txt пример
- [ ] Bar chart распределения QA по 9 категориям
- [ ] Pipeline diagram (generate_tree → qagen → train → eval × 3 ветки)
- [ ] Loss-mask схема ("user — full visible; assistant — only letter graded")
- [ ] Heatmap из exp1 (cos/angle layer × slot, с подсвеченным name slot)
- [ ] Heatmap из exp2 compare alex|ben (`heatmap_V_rel_l2.png`, `slot_alex_ben_V_cos.png`)
- [ ] Таблица из §9.1 (init divergence aggregate)
- [ ] Таблица из §10.1 (init vs trained)
- [ ] **(после доп. прогонов)** Таблица accuracy: per-cat × {cartridge alex/ben/carl/dan, ICL corpus, ICL narrative}
- [ ] **(после доп. прогонов)** Cross-variant matrix: train variant × test variant
- [ ] **(после доп. прогонов)** Cartridge-vs-ICL barplot (где cartridge wins / loses по категориям)
- [ ] Open questions slide (§13)

---

## 15. Что нужно дозапустить **до доклада**

- carl + dan training → закроет полную матрицу 6 trained пар
- cartridge-cot eval всех 4 → primary accuracy результат
- cross-variant eval (alex↔ben↔carl↔dan, 12 cross runs) → структура vs surface signal
- ICL baseline (base + 4 variants, × 2 corpus formats) → нижняя граница

После этого таблица accuracy × категория × {cartridge_per_variant, ICL_corpus, ICL_narrative} + cross-variant matrix станут centerpiece доклада.
