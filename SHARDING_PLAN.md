# План: шардинг пула и микстура экспертов вместо committee-из-клонов

Статус: реализовано 2026-09-13. Код, тесты, demo, API-контракт и dashboard обновлены.

Принятые решения: `n_partitions = len(node_ids)` фиксировано на всю кампанию; оператор задаёт
`train_mode=fresh|continue`, сервер превращает его в явную `task.operation`, поэтому клиент больше не
угадывает, переобучаться с нуля или делать warm-start; финальный merge режима `experts` использует
простое среднее.

## 1. Контекст: как мы сюда пришли

Проект прошёл через два архитектурных видения:

1. **Suverенные ноды** (README/CONTRACT v0.4.0, ещё в коде, не удалено): у каждой ноды свои приватные
   данные, сырые данные с ноды не уходят. `pooling.py` — механизм, который склеивает scores от разных
   нод с разными приватными корпусами в один общий ранжированный список (z-score/calibrated
   нормализация, Reliability Gate по held-out precision).
2. **Server-master** (решено в этом же разговоре позже): сервер держит все данные и оракула, ноды —
   вычислительные мощности. Реализовано как `campaigns.py`: кампания раздаёт **одну и ту же** задачу
   (один и тот же шард целиком) N нодам ("committee"), каждая учит свою модель с разным seed, сервер
   усредняет их scores (`mean`) и меряет разброс (`std`) для активного обучения (query-by-committee).

**Committee-из-клонов оказался пустышкой.** Два независимых наблюдения:

- В `tools/demo_campaign.py` и во всех живых прогонах `campaigns.py` до сих пор нода — это
  `mock_score(seed, cid) = hash(f"{seed}:{cid}")`, функция, которая **не смотрит ни на текст, ни на
  метки оракула**. Даже когда оракул был настоящим (`StaticOracle` на реальных Llama-3-метках из
  FineWeb-Edu, см. `tools/load_golden_shard.py`), score всё равно оставался хэшем. Это значит: все
  живые демо `campaigns.py` доказывают только то, что оркестрация не падает и воспроизводится —
  ничего не доказывают про качество отбора, потому что вход был шумом по построению.
- Более фундаментально: даже с настоящим обучением (fastText, реальные метки — это делалось отдельно
  в `tools/al_committee_sim.py`, не в `campaigns.py`), committee из N копий одной и той же задачи с
  разным seed даёт выигрыш, но некрупный (committee-cutoff/qbc против single-random — единицы процентов
  сверху того, что и так даёт cutoff-стратегия одной модели). Не то, ради чего стоит городить
  распределённые вычисления.

**Ценностное предложение, которое реально стоит строить** (сформулировано пользователем):

> вся наша ценность в том что мы можем дробить большие сеты на маленькие вычислительные задачи и
> отдавать их нодам

Два разных, оба честных источника выигрыша от децентрализации:

- **(A) Шардинг ради throughput.** Пул слишком большой для одной ноды → сервер режет его на N частей →
  нода N считает только свой кусок → сервер склеивает N кусков в один общий ранжированный список.
  Выигрыш — параллелизм, работа физически не влезла бы в одну машину за вменяемое время.
- **(B) Микстура экспертов ради качества.** M нод, каждая учится на своём куске (домене), но потом
  **все** скорят один и тот же общий пул целиком; сервер мержит M мнений на каждый документ в одно.
  Выигрыш — не throughput, а то, что несколько разных точек зрения точнее одной.

Committee-режим не даёт throughput (работы не убавилось — все делают одно и то же) и не даёт
экспертизу (это не разные точки зрения, это шум одного и того же). Отсюда и решение его заменить.

## 2. Словарь (чтобы не путаться снова)

| Термин | Значение в этом плане |
|---|---|
| **пул** (pool) | Множество chunk_id одного шарда — `store.chunk_ids_for_shard(shard_id)` |
| **партиция** (partition) | Непересекающееся подмножество пула, полученное детерминированной хэш-функцией от chunk_id. Не отдельная сущность в БД — вычисляется на лету |
| **режим A / sharded** | Нода видит и скорит **только свою партицию**. Merge = склейка непересекающихся списков (`pooling.select_topk`) |
| **режим B / experts** | Нода **учится** на своей партиции (домене), но **скорит весь пул**. Merge = усреднение по chunk_id между M submissions, которые покрывают один и тот же набор id |
| **committee** (старое, out) | Нода видит и скорит весь пул, учится на тех же метках, что и остальные, отличается только seed. Заменяется на (A)/(B) |

## 3. Общий примитив: `partition_of()`

`chunk_id` уже гарантированно нижний hex-хэш 16-128 символов (`CHUNK_ID_RE` в `schemas.py`), так что
партиционирование не требует лишнего хэширования — берём срез самого chunk_id:

```python
# control_plane/partitioning.py (новый файл)
def partition_of(chunk_id: str, n_partitions: int) -> int:
    """Детерминированно и без состояния: та же chunk_id всегда попадает в ту же партицию при том же N."""
    return int(chunk_id[:8], 16) % n_partitions
```

Ключевое свойство: **партиция не хранится в БД**. Она всегда пересчитывается из `chunk_id` и `N`.
Значит, ничего не нужно добавлять в `chunks`-таблицу, не нужна миграция при смене N между кампаниями.

Оба режима (A и B) используют этот примитив, но по-разному:

| | что нода ТРЕНИРУЕТ (получает метки/labels на) | что нода СКОРИТ (обязана вернуть score) |
|---|---|---|
| Режим A | своя партиция | своя партиция (то же самое) |
| Режим B | своя партиция (= "домен") | **весь пул** |

## 4. Что переиспользуется как есть

- `control_plane/oracle.py` — `Oracle`/`MockOracle`/`StaticOracle`, без изменений.
- `control_plane/acquisition.py` — `acquire()`, без изменений. В обоих новых режимах вызывается
  **на партицию**, а не на весь пул: `mean`/`std` — это scores по индексам партиции, не всего пула.
  Для (A) `std` всегда нулевой (одна нода = одна партиция, committee из одного). Для (B) `std` тоже
  нулевой на этапе выбора новых меток внутри домена (один эксперт на домен) — если захочется committee
  *внутри* домена, это отдельное расширение поверх этого плана, не в этой итерации.
- `control_plane/pooling.py` — `select_topk()`, `NodeSubmission`, без изменений сигнатур.
  Для режима A `NodeSubmission.chunk_ids` — это партиция каждой ноды (непересекающиеся множества),
  `curve=None` (held-out gate в mock-режиме не задействован, `enforce_gate=False`).
  Годится буквально как есть — это тот же код, что был написан для суверенных нод с приватными
  данными, просто теперь партиции физически на сервере, а не на ноде. `normalize="zscore"` или
  `"rank"` (не `"calibrated"` — она требует `HeldoutCurve`).
- `control_plane/storage.py` — `chunks`/`labels` таблицы без изменений (labels ключуются по
  chunk_id глобально, не по партиции — это уже работает правильно для обоих режимов).
- `GET/POST /shards/{shard_id}/chunks`, `GET /shards/{shard_id}/labels` — без изменений по сути,
  см. §6 про новый query-параметр.
- Механика раундов (`store.create_round`, `store.get_task`, ack/submit транспорт, `set_round_status`)
  — без изменений. Раунд как был "один round_id → несколько participants", так и остаётся; меняется
  только что каждый participant обязан заскорить (params передают, что именно).

## 5. Что меняется

### 5.1 Новый файл `control_plane/partitioning.py`

```python
def partition_of(chunk_id: str, n_partitions: int) -> int:
    return int(chunk_id[:8], 16) % n_partitions

def chunk_ids_for_partition(pool: list[str], partition: int, n_partitions: int) -> list[str]:
    return [c for c in pool if partition_of(c, n_partitions) == partition]
```

### 5.2 Новая ручка: партиционированный доступ к чанкам

`GET /shards/{shard_id}/chunks` уже существует и отдаёт всё. Добавить query-параметры:

```
GET /shards/{shard_id}/chunks?partition=2&n_partitions=5
```
→ возвращает только чанки этой партиции. Без параметров — как сейчас, весь шард (нужно для режима B,
где нода обязана видеть весь пул для скоринга).

### 5.3 `control_plane/campaigns.py` — переписать `CampaignService`

Текущий `create()`/`_create_round()`/`advance()`/`_finalize()` целиком построены вокруг "все участники
видят один и тот же шард". Это не патчится точечно, это переписывается. План по методам:

#### `create()` — два варианта вместо одного

```python
def create_sharded(self, campaign_id, shard_id, model, metrics, node_ids, schedule, train_mode, strategy,
                   k_frac, good_min, seed) -> dict:
    """Режим A. node_ids[i] получает партицию i из len(node_ids) партиций. schedule/strategy/k_frac
    применяются одинаково к КАЖДОЙ партиции независимо (у каждой своя схема разметки, свой budget_k)."""

def create_experts(self, campaign_id, pool_shard_id, model, metrics, node_ids, schedule, train_mode, strategy,
                   k_frac, good_min, seed) -> dict:
    """Режим B. node_ids[i] учится на партиции i (своём домене) пула pool_shard_id, но каждый раунд
    скорит ВЕСЬ pool_shard_id."""
```

Общая часть (вынести в приватный `_create_common`):
- проверка регистрации нод и `model_kinds`, как сейчас;
- `spec = {..., "mode": "sharded" | "experts", "n_partitions": len(node_ids), ...}` — режим и число
  партиций пишутся в spec, чтобы `advance()` знал, как читать текущий раунд;
- сид первых меток на партицию/домен: `seed_idx` теперь считается **на каждую партицию отдельно**,
  а не один раз на весь пул — иначе домены/партиции окажутся размечены неравномерно.

#### `_create_round()` — params на партицию, а не на "seed"

Сейчас: `participants = [{"node_id": nid, "dataset_id": shard_id, "params": {"seed": i}} ...]`.

Новое: `params` несёт партицию явно, чтобы клиент коллеги знал, что скорить:

```python
{"node_id": nid, "dataset_id": shard_id,
 "params": {"partition": i, "n_partitions": len(node_ids), "mode": spec["mode"]}}
```

Это уже поддержано контрактом — `participants[].params` существует и мержится в `task["params"]`
(см. `task_view()` в `app.py`, не менять). Клиент коллеги видит в задаче `params.partition`,
`params.n_partitions`, `params.mode` и по ним решает: тянуть ли `?partition=i&n_partitions=N`
(режим A) или весь `GET /shards/{id}/chunks` без параметров (режим B).

#### `advance()` — разное ожидаемое покрытие по режиму

Ключевая точка расхождения. Сейчас `advance()` строит один `runs: np.zeros((N, len(pool)))` на весь
пул и требует от каждой ноды покрыть весь пул. Новая версия:

```python
def advance(self, campaign_id: str) -> AdvanceResult:
    ...
    spec = campaign["spec"]
    mode, n_part = spec["mode"], spec["n_partitions"]
    pool = self.store.chunk_ids_for_shard(campaign["shard_id"])

    per_node_scores = {}   # node_id -> {chunk_id: score}, только то, что нода реально прислала
    for i, nid in enumerate(spec["node_ids"]):
        rows = self.store.scores_for_submission(subs[nid]["id"])
        got = {r["chunk_id"]: r["score"] for r in rows}
        expected = (chunk_ids_for_partition(pool, i, n_part) if mode == "sharded" else pool)
        absent = [c for c in expected if c not in got]
        if absent:
            raise CampaignError(f"node {nid!r} did not score {len(absent)} expected chunks, e.g. {absent[:3]}")
        per_node_scores[nid] = got

    # для acquisition — каждая партиция/домен обрабатывается независимо:
    for i, nid in enumerate(spec["node_ids"]):
        domain_ids = chunk_ids_for_partition(pool, i, n_part)   # партиция ИЛИ домен — одна и та же
                                                                  # функция, разница только в том,
                                                                  # что подаётся на скоринг
        domain_scores = np.array([per_node_scores[nid][c] for c in domain_ids])
        labeled_here = {c for c in domain_ids if c in labels}
        ...acquisition.acquire(...) на domain_ids/domain_scores...
        self._label(shard_id, новые chunk_id из ЭТОЙ партиции/домена)

    if всё расписание пройдено:
        result = self._finalize(mode, pool, per_node_scores, spec, labels)
        ...
```

`_finalize()` — тоже разный merge по режиму:

```python
def _finalize(self, mode, pool, per_node_scores, spec, labels):
    if mode == "sharded":
        nodes = [NodeSubmission(nid, list(scores), list(scores.values()), curve=None)
                for nid, scores in per_node_scores.items()]
        topk = select_topk(nodes, k, normalize="zscore", enforce_gate=False)
        ranked_unlabeled = [s["chunk_id"] for s in topk["selected"] if s["chunk_id"] not in labels]
    else:  # experts — все per_node_scores покрывают один и тот же pool, мержим по chunk_id
        merged = {c: statistics.fmean(per_node_scores[nid][c] for nid in per_node_scores) for c in pool}
        ranked_unlabeled = sorted((c for c in pool if c not in labels), key=lambda c: -merged[c])
    # дальше то же самое: known_good вперёд, потом ranked_unlabeled, до k
```

### 5.4 `control_plane/schemas.py` — новая ручка/поля

Вариант А (проще): один `POST /campaigns`, добавить обязательное поле `mode: Literal["sharded", "experts"]`
и переименовать `shard_id` → `pool_shard_id` (или оставить `shard_id`, работает для обоих режимов
одинаково — это просто "откуда брать чанки"). Оставить `node_ids` как список — партиция/домен ноды i
= индекс i в списке, отдельно передавать не нужно.

```python
class CreateCampaign(BaseModel):
    campaign_id: str
    shard_id: str
    mode: Literal["sharded", "experts"] = "sharded"
    model: ModelRef
    metrics: List[str] = Field(default_factory=list, max_length=MAX_AGG_KEYS)
    node_ids: List[str] = Field(min_length=1, max_length=64,
                                description="node_ids[i] владеет партицией/доменом i")
    schedule: List[int]      # как сейчас — метки НА ПАРТИЦИЮ, не суммарно (важно явно задокументировать)
    strategy: Literal["random", "cutoff", "qbc"] = "cutoff"
    k_frac: float = 0.1
    good_min: Optional[int] = None
    seed: int = 0
```

Убрать `committee` из формулировок в docstring/описаниях полей (сейчас `node_ids` описан как
"committee: same task, every round, one proxy run each" — переписать под новый смысл).

### 5.5 `control_plane/app.py`

- `POST /campaigns` — роутить на `campaigns.create_sharded(...)` или `campaigns.create_experts(...)`
  по `body.mode`.
- `GET /shards/{shard_id}/chunks` — добавить `partition: Optional[int] = Query(None)`,
  `n_partitions: Optional[int] = Query(None)`; если оба заданы — фильтровать через
  `chunk_ids_for_partition`; 422 если задан только один из двух.

### 5.6 `control_plane/metrics.py` / `dashboard.html`

Минимально: в `snapshot()`/карточке кампании показывать `mode` и, для sharded, сколько партиций уже
прошли текущий раунд (частичный прогресс), не только общий `rounds_done`. Не критично для первой
итерации — можно оставить как есть и добавить отдельным шагом.

## 6. Что явно НЕ делаем в этой итерации (mock, не глубина)

- Никакого реального обучения в control plane — как и раньше, скоринг ноды остаётся зоной клиента
  коллеги. Этот план меняет только то, **какую часть пула** нода видит и **как сервер мержит**
  результаты — не то, как нода считает score.
- Домены в режиме B пока = партиции того же пула по хэшу chunk_id, не смысловые домены (banking vs
  telecom). Осмысленная доменная нарезка — отдельная задача (нужны метаданные документа или
  кластеризация), не блокирует эту итерацию.
- Held-out curve / Reliability Gate — не подключаем к новым режимам сейчас (`enforce_gate=False`
  везде). Это отдельная задача про верификацию честности ноды (канареечные чанки, дублирующее
  задание одной партиции двум нодам), см. `pooling.py`'s docstring про MIN_HELDOUT_AT_CUTOFF — тот же
  механизм годится, но его переиспользование не входит в этот план.
- Committee (старый режим) физически не удаляем из `campaigns.py` историю — просто заменяем логику
  `create()`/`advance()`. Если где-то в тестах/демо-скриптах остались вызовы старой сигнатуры
  (`campaigns.create(campaign_id, shard_id, model, metrics, node_ids, schedule, strategy, k_frac,
  good_min, seed)` — 10 позиционных/именованных аргументов без `mode`) — их нужно поправить на
  `create_sharded(...)`, это ожидаемо ломающее изменение, не баг.

## 7. Пошаговый план реализации

1. `control_plane/partitioning.py` — `partition_of()`, `chunk_ids_for_partition()`. Юнит-тесты:
   детерминированность, равномерность распределения на реальных chunk_id (можно взять уже
   скачанные `data/fineweb_edu_ann/*.parquet`, посчитать `hashlib.sha256(text).hexdigest()` и
   проверить, что `partition_of` не даёт сильно неравных партиций при разумном N).
2. `control_plane/schemas.py` — `CreateCampaign.mode`, обновить docstring/examples.
3. `control_plane/campaigns.py` — переписать `create()` на `create_sharded()`/`create_experts()`
   через общий `_create_common()`; переписать `_create_round()` (params: partition/n_partitions/mode
   вместо seed); переписать `advance()` и `_finalize()` под разное ожидаемое покрытие и разный merge.
4. `control_plane/app.py` — роутинг `POST /campaigns` по `mode`; query-параметры на
   `GET /shards/{shard_id}/chunks`.
5. `tools/demo_campaign.py` — обновить под новый контракт: mock-нода должна уметь читать
   `task["params"]["mode"]`/`["partition"]`/`["n_partitions"]` и соответственно либо тянуть только
   свою партицию (`?partition=&n_partitions=`), либо весь пул. **Здесь же наконец имеет смысл сделать
   mock_score зависимым хоть от чего-то настоящего** — например, для режима B: эксперт домена i
   даёт более высокий score документам, СЕМАНТИЧЕСКИ похожим на его домен (можно грубо: похожим по
   длине/языку/наличию ключевых слов домена в тексте), чтобы дашборд и финальный отбор наконец
   показывали не чистый шум. Не обязательно к первому проходу, но обозначено как следующий логичный
   шаг, иначе problem из §1 (проверяем только воспроизводимость) повторится в новых режимах.
6. `tests/test_campaigns.py` — переписать существующие тесты кампаний под `create_sharded`/
   `create_experts` (текущие вызовы `campaigns.create(...)` больше не скомпилируются как есть);
   добавить: тест, что режим A действительно даёт непересекающиеся партиции и merge не теряет и не
   дублирует чанки; тест, что режим B требует от каждой ноды покрытия ВСЕГО пула, а не только домена;
   тест на `partition_of` детерминированность через API (одна и та же нода в одном и том же раунде
   всегда получает один и тот же список чанков при повторном GET).
7. `TESTING.md`, `CONTRACT.md`, `README.md` — актуализировать под новый контракт кампаний (сейчас там
   описан committee-режим, после этого плана описание будет враньём).

## 8. Открытые вопросы — не гадать, спросить пользователя

- Нужно ли поддерживать **разное** число партиций/экспертов на разных раундах одной кампании
  (сейчас предполагается фиксированное `n_partitions = len(node_ids)` на всю жизнь кампании), или
  это преждевременная гибкость?
- Для режима B: если домен-эксперт i получил новые метки в своём домене, а раунд i+1 просит его
  заново заскорить **весь** пул — предполагаем, что нода переобучает модель с нуля на всех метках
  своего домена каждый раунд (как сейчас в `bake_local()` из `tools/proxy_rounds_sim.py`), или нужен
  warm-start? Это решение клиента коллеги, не control plane, но стоит явно проговорить в TESTING.md.
- Что считать финальным score в режиме B — среднее (как в плане выше), медиана, или взвешенное
  среднее (эксперт с более узким/сильным доменом весит больше)? Среднее — самый простой старт,
  но стоит подтвердить, что это то, что имелось в виду под "мержим N сигналов".
