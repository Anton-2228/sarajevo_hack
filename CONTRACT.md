# Proxy Mesh — контракт нода ↔ control plane

Версия: 0.11.0 (зафиксировано 2026-09-13). Источник истины — [control_plane/schemas.py](control_plane/schemas.py)
и [control_plane/app.py](control_plane/app.py); при расхождении документа с кодом прав код, документ правится.

## 1. Принципы

- **Нода только ходит наружу.** Control plane (CP) никогда не обращается к ноде. Задачи нода забирает сама.
- **Разделение ответственности.**
  - CP: реестр нод, раунды, server-held пулы кампаний и их разбиение, oracle labels, merge, Reliability Gate, top-k.
  - Нода: вычислительные ресурсы, обучение proxy с параметрами задачи, скоринг и метрики. Все входные данные назначает
    CP; приватных или принадлежащих ноде датасетов в контракте нет.
- **Egress Gate.** Наружу из ноды уходят только: метаданные ноды (хендшейк), хэши чанков, числовые scores, числовые метрики.
  Сервер отвергает всё, что похоже на текст, там где ожидаются числа или хэши.
- **Разделение путей.** Lifecycle API ноды — `/nodes/…` и `/tasks/…`; назначенные тексты и labels
  нода читает через `GET /shards/…`. API оператора — `/rounds/…`, `/campaigns/…`, `GET /nodes`
  и `POST /shards/…`. Наблюдаемость — `/health`, `/metrics*`, `/dashboard`.

## 2. Общее

- Транспорт: HTTP(S), тела JSON, UTF-8.
- Время: unix time в секундах, `float`.
- Форматы строк:

| Имя | Регэксп | Где |
|---|---|---|
| `ID` | `^[A-Za-z0-9_.:-]{1,64}$` | `node_id`, `round_id`, `dataset_id`, `model.id`, ключи `metrics` / `params` / `load` / `agg_stats` / `software` |
| `LABEL` | `^[A-Za-z0-9_.:\-+/() ]{1,64}$` | `gpus[].model`, значения `software` |
| `HASH` | `^[a-f0-9]{16,128}$` | `chunk_id` |

- Ошибки:
  - прикладные: `{"detail": "<текст>"}`;
  - валидация (422): `{"detail": [{"loc": [...], "msg": "...", "type": "..."}]}`. В `submit` значения полей в ошибке не возвращаются.
- Лишние поля в теле запроса на любом уровне → 422 (кроме `agg_stats`, где разрешены дополнительные числовые ключи).

### 2.1 Авторизация

Нет (сборка для хакатона): токена на регистрацию нет, секрета у ноды нет, запросы не подписываются.
Любой вызывающий может действовать от имени любого `node_id` — знания `node_id` достаточно, а он публичен через
`GET /nodes`. Приемлемо в доверенной локальной/демо-сети; выкладывать этот CP в недоверенную сеть без возврата
авторизации не стоит. Операторские ручки (раздел 4) тоже без авторизации.

## 3. Ручки ноды

Жизненный цикл: `handshake` → `heartbeat` каждые `heartbeat_interval_s` → `GET tasks` →
`GET chunks/labels` → `ack` → пайплайн → `submit`.

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/nodes/handshake` | регистрация / перерегистрация |
| POST | `/nodes/{node_id}/heartbeat` | пульс |
| GET | `/tasks/{node_id}` | задачи ноды |
| GET | `/shards/{shard_id}/chunks` | тексты всего пула или партиции |
| GET | `/shards/{shard_id}/labels` | текущие oracle labels для обучения |
| POST | `/tasks/{node_id}/{round_id}/ack` | «приступила» |
| POST | `/tasks/{node_id}/{round_id}/submit` | результат (Egress Gate) |

### 3.1 `POST /nodes/handshake` — регистрация

Запрос:

```json
{
  "name": "bank-a",
  "node_id": null,
  "hardware": {"cpu_cores": 16, "ram_gb": 64, "gpus": [{"model": "A100", "vram_gb": 80}], "disk_free_gb": 500},
  "software": {"agent_version": "0.11.0", "python": "3.12"},
  "model_kinds": ["classifier", "lora"]
}
```

| Поле | Тип | Обяз. | Ограничения |
|---|---|---|---|
| `name` | ID | да | ≤ 48 символов; из него строится `node_id` |
| `node_id` | ID | нет | только для перерегистрации уже известной ноды |
| `hardware.cpu_cores` | int | да | ≥ 1 |
| `hardware.ram_gb` | float | да | > 0 |
| `hardware.gpus[]` | `{model: LABEL, vram_gb: float ≥ 0}` | нет | ≤ 64 |
| `hardware.disk_free_gb` | float | нет | ≥ 0 |
| `software` | `{ID: LABEL}` | нет | ≤ 32 ключей |
| `model_kinds` | `["lora" \| "classifier"]` | да | ≥ 1 |

Ответ `201`:

```json
{"node_id": "bank-a-3f9c1d", "heartbeat_interval_s": 3.0, "tasks_url": "/tasks/bank-a-3f9c1d"}
```

- Новая нода: `node_id` = `name` + `-` + 6 hex. Нода обязана сохранить его.
- Перерегистрация (`node_id` передан): id сохраняется, `name` и характеристики заменяются. Авторизации нет (2.1) —
  переданный `node_id` принимается от любого вызывающего, если он уже зарегистрирован.

Ошибки: `404` переданный `node_id` не найден; `422` валидация.

### 3.2 `POST /nodes/{node_id}/heartbeat` — пульс

Запрос (все поля необязательны, `{}` валиден):

```json
{"status": "busy", "stage": "training", "round_id": "r1", "load": {"progress_pct": 42.5, "docs_processed": 8500,
 "docs_total": 20000, "docs_per_sec": 127.4, "eta_s": 90, "train_loss": 0.31,
 "cpu_pct": 73.5, "ram_pct": 61.0, "gpu_util_pct": 92.0, "gpu_mem_pct": 78.0}}
```

| Поле | Тип | По умолчанию | Ограничения |
|---|---|---|---|
| `status` | `"idle" \| "busy"` | `"idle"` | |
| `stage` | `"downloading" \| "training" \| "scoring" \| "uploading"` | — | текущая фаза задачи |
| `round_id` | ID | — | раунд в работе |
| `load` | `{ID: number}` | `{}` | ≤ 32 ключей, только конечные числа |

Стабильные имена telemetry, которые понимает dashboard:

| Ключ | Смысл |
|---|---|
| `progress_pct` | прогресс задачи, 0…100; если нет, UI вычисляет из `docs_processed / docs_total` |
| `docs_processed`, `docs_total` | обработано и всего документов |
| `docs_per_sec` | текущая пропускная способность |
| `eta_s` | оценка оставшегося времени, секунды |
| `train_loss` | текущий training loss |
| `cpu_pct`, `ram_pct` | загрузка CPU/RAM, 0…100 |
| `gpu_util_pct`, `gpu_mem_pct` | загрузка GPU/VRAM, 0…100 |

Другие числовые ключи допустимы и попадают в
`proxy_mesh_node_load{node_id, name, metric}` на `GET /metrics`.

Ответ `200`:

```json
{"ok": true, "next_heartbeat_s": 3.0, "pending_tasks": 1}
```

`pending_tasks` — число задач из `GET /tasks/{node_id}`. Нода `online`, пока последний пульс не старше 3 × `heartbeat_interval_s`.

Ошибки: `404` нода не зарегистрирована, `422`.

### 3.3 `GET /tasks/{node_id}` — задачи ноды

Возвращает раунды со статусом `open`, где нода участник и ещё не отправила результат (`assigned` или `accepted`), по времени назначения.

Ответ `200`:

```json
{
  "node_id": "bank-a-3f9c1d",
  "tasks": [{
    "round_id": "r1",
    "status": "assigned",
    "model": {"kind": "classifier", "id": "quality-clf-v1"},
    "operation": {"train": "fresh", "score": true, "input_checkpoint_id": null,
                  "output_checkpoint_id": "r1"},
    "dataset_id": "support-tickets-2025",
    "metrics": ["eval_spearman", "n_dedup_dropped"],
    "params": {"proxy_lr": 1e-5},
    "budget_k": 1000,
    "assigned_at": 1789245600.12,
    "accepted_at": null,
    "ack_url": "/tasks/bank-a-3f9c1d/r1/ack",
    "submit_url": "/tasks/bank-a-3f9c1d/r1/submit"
  }]
}
```

| Поле | Смысл |
|---|---|
| `status` | `assigned` — назначена; `accepted` — нода подтвердила |
| `model` | recipe/family модели: `kind` (`lora` / `classifier`) + стабильный `id`; конкретные веса адресуются checkpoint-полями |
| `operation.train` | `fresh` — новая модель из recipe; `continue` — продолжить входной checkpoint; `skip` — не обучать, только score |
| `operation.score` | сейчас всегда `true`: каждая задача обязана вернуть scores |
| `operation.input_checkpoint_id` | checkpoint в локальном хранилище ноды; обязателен для `continue`/`skip`, запрещён для `fresh` |
| `operation.output_checkpoint_id` | куда сохранить результат обучения; для автоматически созданных задач равен `round_id` |
| `dataset_id` | server-held shard или workload, назначенный control plane |
| `metrics` | ключи, обязательные в `agg_stats` при `submit` |
| `params` | параметры раунда поверх `participants[].params` этой ноды (4.1); в кампании также строковый `mode` и целые `partition`, `n_partitions`, `n_labels` |
| `budget_k` | общий top-k раунда |
| `ack_url`, `submit_url` | справочно: пути 3.4 и 3.5 для этой задачи; однозначно выводятся из `node_id` и `round_id` |

`operation` — единственный источник истины о lifecycle модели. Нода не выводит необходимость обучения
из `model.id`, номера раунда или количества labels. Checkpoint namespace локален для ноды: логический
ключ равен `(node_id, checkpoint_id)`, перенос весов через control plane не выполняется. Повторно
полученный тот же `round_id` означает resume/idempotent retry: нода проверяет локальное состояние и не
повторяет уже завершённый train/submit вслепую.

Ошибки: `404`.

#### 3.3.1 Получение workload кампании

Задача не содержит тексты. `dataset_id` задаёт `shard_id`, а `params.partition` и
`params.n_partitions` — маршрутизацию ноды. Для `mode: "sharded"` нода получает свою
партицию:

```text
GET /shards/{dataset_id}/chunks?partition={partition}&n_partitions={n_partitions}
GET /shards/{dataset_id}/labels
```

Карта labels общая для шарда; нода соединяет её с chunks по `chunk_id`. В `sharded` она обучается
и скорит только эту партицию. В `mode: "experts"` она так же фильтрует labels до своего домена
для обучения, но забирает без query и скорит весь шард. Точные форматы ответов — в 4.7.

### 3.4 `POST /tasks/{node_id}/{round_id}/ack` — «приступила»

Без тела. Переводит участника `assigned` → `accepted`. Идемпотентна: повтор или вызов после `submitted` статус не меняет.

Ответ `200`: одна задача в формате элемента `tasks[]` из 3.3.

Ошибки: `404` нода не найдена или нет такой задачи; `409` раунд закрыт.

### 3.5 `POST /tasks/{node_id}/{round_id}/submit` — результат (Egress Gate)

Запрос:

```json
{
  "node_id": "bank-a-3f9c1d",
  "round_id": "r1",
  "scores": [{"chunk_id": "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08", "score": 0.91}],
  "agg_stats": {"n_chunks": 5000, "eval_spearman": 0.96, "proxy_lr": 1e-5, "n_dedup_dropped": 12,
                "ho_n": 2000, "ho_good": 190, "ho_n_q01": 20, "ho_good_q01": 15, "ho_n_q02": 40, "ho_good_q02": 27,
                "ho_n_q05": 100, "ho_good_q05": 58, "ho_n_q10": 200, "ho_good_q10": 96, "ho_n_q20": 400,
                "ho_good_q20": 140, "ho_n_q30": 600, "ho_good_q30": 160, "ho_n_q50": 1000, "ho_good_q50": 178}
}
```

| Поле | Тип | Ограничения |
|---|---|---|
| `node_id`, `round_id` | ID | совпадают с путём |
| `scores[]` | `{chunk_id: HASH, score: float}` | 1…200 000, `chunk_id` уникальны, `score` конечный, лишних полей нет |
| `agg_stats.n_chunks` | int | обязателен, ≥ `len(scores)` |
| `agg_stats.eval_spearman` | float | необязателен, [-1, 1]; информационная метрика, в gate не участвует |
| `agg_stats.proxy_lr` | float | необязателен, > 0 |
| `agg_stats.ho_*` | int | кривая точности на held-out (3.5.1); без неё нода не попадает в top-k |
| `agg_stats.<прочие>` | number | ключ ID, всего ≤ 64 ключей, только конечные числа |

Подписи нет (2.1) — используйте `node_sdk.client.build_payload` для сборки тела, а не для подписи.

Порядок проверок и ошибки:

| Код | Причина |
|---|---|
| `413` | тело > `PM_MAX_BODY_BYTES` (32 MiB) |
| `400` | невалидный JSON |
| `422` | нарушение схемы выше |
| `400` | `node_id` / `round_id` в пути и теле различаются |
| `404` | нода не зарегистрирована |
| `404` | нет такой задачи: раунда нет или нода не его участник |
| `409` | раунд закрыт |
| `422` | в `agg_stats` нет метрик, запрошенных раундом (`type: missing_metrics`) |

Ответ `201`:

```json
{"accepted": true, "round_id": "r1", "node_id": "bank-a-3f9c1d", "revision": 1, "n_scores": 5000,
 "trust": "pending", "gate_min_precision": 0.2, "payload_sha256": "<64 hex>"}
```

- Участник переходит в `submitted`, задача пропадает из `GET /tasks`. `ack` перед `submit` не обязателен.
- Повторная отправка в открытый раунд заменяет предыдущую, `revision` + 1.
- `trust`: `pending` — кривая held-out есть, gate решается при отборе (4.4), потому что зависит от распределения бюджета
  между нодами; `unknown` — кривой нет.

### 3.5.1 Кривая точности на held-out

Если workflow оператора/ноды заранее подготовил отдельный labelled evaluation split, нода
ранжирует его той же моделью, что и основную часть задачи. В submit уходят только счётчики.
Текущий API сам не выделяет и не маркирует held-out среди `/shards/.../labels`; CP доверяет присланной
нодой кривой:

| Ключ | Смысл |
|---|---|
| `ho_n`, `ho_good` | размер held-out и число хороших в нём |
| `ho_n_qXX`, `ho_good_qXX` | для q ∈ {0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5} (`XX` = `01`, `02`, `05`, `10`, `20`, `30`, `50`): число документов в top-`ceil(q·ho_n)` held-out по score и число хороших среди них |

Правила (`422` при нарушении, значения в ошибке не возвращаются): все 16 ключей или ни одного; целые ≥ 0; `ho_n ≥ 1`,
`ho_good ≤ ho_n`; с ростом q оба счётчика не убывают, `ho_good_qXX ≤ ho_n_qXX`, и оба не больше итогов.
Считать через `node_sdk.client.heldout_curve(scores, good)`.

Кривая — вход калибровки и Reliability Gate (4.4). Отдельных документов, их scores и меток она не содержит; при очень
маленьком held-out верхние квантили состоят из 1–2 документов.

## 4. Ручки оператора и общие read-only ручки

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/rounds` | создать раунд |
| GET | `/rounds/{round_id}` | раунд: спецификация, участники, отправки |
| POST | `/rounds/{round_id}/close` | закрыть раунд |
| GET | `/rounds/{round_id}/topk` | общий отбор |
| GET | `/nodes` | реестр нод |
| GET | `/shards` | инвентарь server-held датасетов |
| POST | `/shards/{shard_id}/chunks` | загрузить чанки server-held пула |
| POST | `/imports/parquet/inspect` | временно загрузить Parquet и прочитать схему |
| POST | `/shards/{shard_id}/imports/parquet` | импортировать выбранные Parquet-колонки |
| GET | `/shards/{shard_id}/preview` | постраничный просмотр и поиск документов с labels |
| GET | `/shards/{shard_id}/chunks` | нода/оператор: весь пул или детерминированная партиция |
| GET | `/shards/{shard_id}/labels` | нода/оператор: текущие oracle labels пула |
| POST | `/campaigns` | создать sharded/experts кампанию |
| GET | `/campaigns/{campaign_id}` | состояние и результат кампании |
| POST | `/campaigns/{campaign_id}/advance` | обработать текущий раунд |
| GET | `/health` | жив ли сервис |
| GET | `/metrics` | Prometheus text exposition |
| GET | `/metrics.json` | JSON snapshot для dashboard |
| GET | `/telemetry/history` | сохранённая история heartbeat для графиков |
| GET | `/dashboard` | встроенный control-room UI |

### 4.1 `POST /rounds` — создать раунд

```json
{
  "round_id": "r1",
  "budget_k": 1000,
  "note": "first calibration round",
  "model": {"kind": "classifier", "id": "quality-clf-v1"},
  "operation": {"train": "fresh", "score": true, "output_checkpoint_id": "r1"},
  "metrics": ["eval_spearman", "n_dedup_dropped"],
  "params": {"proxy_lr": 1e-5},
  "participants": [{"node_id": "worker-a-3f9c1d", "dataset_id": "wave1", "params": {"cutoff_q": 0.08}}]
}
```

| Поле | Тип | Обяз. | Ограничения |
|---|---|---|---|
| `round_id` | ID | да | уникален |
| `budget_k` | int | нет | ≥ 1, по умолчанию 1000 |
| `note` | string | нет | ≤ 280 |
| `model` | `{kind: "lora" \| "classifier", id: ID}` | да | |
| `operation` | объект lifecycle | нет | default: `train=fresh`, `score=true`, output checkpoint = `round_id` |
| `metrics` | `[ID]` | да | 1…64, уникальны |
| `params` | `{ID: number}` | нет | ≤ 64 ключей |
| `participants[]` | `{node_id: ID, dataset_id: ID, params?: {ID: number}}` | да | ≥ 1, `node_id` уникальны; `dataset_id` задаёт CP; `params` ≤ 64 ключей и перекрывают параметры раунда |

Проверки участников (`422`): нода зарегистрирована; `model.kind` есть в её `model_kinds`. Нода не
декларирует датасеты и ownership-проверки для `dataset_id` нет.

Для `operation.train=continue|skip` обязателен `input_checkpoint_id`; для `fresh` он запрещён.
`score` в текущей версии может быть только `true`. Если `fresh`/`continue` не задаёт
`output_checkpoint_id`, сервер подставляет `round_id`.

Ответ `201`: представление раунда (4.2). Ошибки: `409` раунд уже есть, `422`.

### 4.2 `GET /rounds/{round_id}` — раунд

```json
{
  "round_id": "r1", "budget_k": 1000, "note": null, "status": "open", "created_at": 1789245600.1,
  "spec": {"model": {"kind": "classifier", "id": "quality-clf-v1"},
           "operation": {"train": "fresh", "score": true, "input_checkpoint_id": null,
                         "output_checkpoint_id": "r1"}, "metrics": ["eval_spearman"],
           "params": {"proxy_lr": 1e-5}},
  "participants": [{"node_id": "bank-a-3f9c1d", "dataset_id": "support-tickets-2025", "status": "submitted",
                    "assigned_at": 1789245600.1, "accepted_at": 1789245601.3, "submitted_at": 1789245609.8,
                    "params": {"cutoff_q": 0.08}}],
  "nodes": [{"node_id": "bank-a-3f9c1d", "revision": 1, "received_at": 1789245609.8, "n_scores": 5000,
             "eval_spearman": 0.96, "trust": "pending", "agg_stats": {"n_chunks": 5000, "…": "…"}}],
  "nodes_with_heldout": 1
}
```

`participants` — назначения, их статусы и параметры ноды, `nodes` — принятые отправки (`trust` как в 3.5),
`nodes_with_heldout` — сколько отправок пришло с кривой held-out. Ошибки: `404`.

### 4.3 `POST /rounds/{round_id}/close` — закрыть раунд

Без тела. `open` → `closed`: задачи исчезают, `ack` и `submit` получают `409`. Обратного перехода нет. Ответ `200`: представление раунда. Ошибки: `404`.

### 4.4 `GET /rounds/{round_id}/topk` — общий отбор

| Query | По умолчанию | Смысл |
|---|---|---|
| `k` | `budget_k` раунда | ≥ 1 |
| `normalize` | `calibrated` | как сделать scores разных нод сравнимыми, см. ниже |
| `include_untrusted` | `false` | считать и показывать gate, но не исключать ноды |
| `include_unknown` | `false` | брать ноды без кривой held-out; с `calibrated` не действует |

Нормировка:

- `calibrated` — чанк в верхней доле f своей ноды получает ожидаемую долю хороших документов в этой точке по кривой ноды:
  точность held-out по полосам квантилей, сглаженная к базовой доле ноды (5 псевдодокументов), невозрастающая, с линейной
  интерполяцией между серединами полос. Бюджет идёт туда, где ожидается больше хороших документов: ноде с более богатым
  корпусом или более точным прокси достаётся больше;
- `zscore` / `rank` — z-оценка / перцентильный ранг внутри submission: каждая нода получает примерно одинаковую долю бюджета;
- `none` — сырые scores.

Reliability Gate: после распределения у ноды отсечка q = выбрано / `n_scores`. Нода `trusted`, если нижняя граница
Вильсона (95%) точности held-out выше отсечки не меньше `PM_GATE_MIN_PRECISION` (0.2). Точность берётся по кривой, линейно
между квантилями, и не меньше чем по 30 документам held-out. Непрошедшие ноды исключаются, бюджет перераспределяется, пока
все оставшиеся не пройдут. Ноды без кривой — `unknown`.

```json
{"round_id": "r1", "k": 500, "normalize": "calibrated", "gate_min_precision": 0.2,
 "nodes_used": ["bank-a-3f9c1d"],
 "nodes_skipped": [
   {"node_id": "gov-c-a83a89", "trust": "untrusted", "selected": 40, "n_scores": 3000, "cutoff_q": 0.013333,
    "heldout_at_cutoff": 30.0, "precision_at_cutoff": 0.2, "precision_lower_bound": 0.0951, "heldout_base_rate": 0.18,
    "reason": "held-out precision at cutoff below gate"},
   {"node_id": "telco-d-19be04", "trust": "unknown", "reason": "no held-out curve"}],
 "gate": [{"node_id": "bank-a-3f9c1d", "trust": "trusted", "selected": 500, "n_scores": 5000, "cutoff_q": 0.1,
           "heldout_at_cutoff": 200.0, "precision_at_cutoff": 0.48, "precision_lower_bound": 0.4118,
           "heldout_base_rate": 0.095}],
 "pool_size": 5000, "selected_per_node": {"bank-a-3f9c1d": 500},
 "selected": [{"node_id": "bank-a-3f9c1d", "chunk_id": "<hash>", "raw_score": 1.42, "norm_score": 0.61}]}
```

`gate` — ноды последнего распределения; `nodes_skipped` — исключённые gate (с его данными на момент исключения) и ноды без
кривой. Ошибки: `404`, `422` (некорректные query).

### 4.5 `GET /nodes` — реестр

```json
[{"node_id": "bank-a-3f9c1d", "name": "bank-a",
  "specs": {"hardware": {...}, "software": {...}, "model_kinds": [...]},
  "registered_at": 1789245500.0, "updated_at": 1789245500.0,
  "last_heartbeat": 1789245612.0, "heartbeat": {"status": "idle", "load": {}}, "online": true}]
```

### 4.6 Наблюдаемость

- `GET /health` → `{"status": "ok"}`.
- `GET /metrics` возвращает Prometheus text format: ноды, heartbeat load, раунды, submissions,
  campaigns и shards. Каждый числовой `heartbeat.load` экспортируется как
  `proxy_mesh_node_load{node_id, name, metric}`; текущая фаза — как
  `proxy_mesh_node_stage{node_id, name, stage} 1`.
- `GET /metrics.json` возвращает operational snapshot в JSON-форме, включая последние
  `status`, `stage`, `round_id`, `load` и `heartbeat_age_s` каждой ноды.
- `GET /telemetry/history?node_id=&since=&limit=` возвращает сохранённые heartbeat-сэмплы. `node_id`
  и Unix timestamp `since` необязательны; `limit` — 1…5000 последних точек отдельно для каждой ноды,
  по умолчанию `PM_TELEMETRY_HISTORY_LIMIT`. Внутри ноды точки отсортированы от старой к новой:

```json
{"generated_at": 1789245700.0, "retention_s": 604800.0, "limit_per_node": 240,
 "nodes": {"bank-a-3f9c1d": [
   {"received_at": 1789245612.0, "status": "busy", "stage": "training", "round_id": "r1",
    "load": {"progress_pct": 42.5, "train_loss": 0.31}}
 ]}}
```

- `GET /dashboard` содержит вкладки `Control room` и `Datasets`. Control room опрашивает
  `/metrics.json`, один раз загружает `/telemetry/history` и продолжает
  графики с новых heartbeat. Heartbeat атомарно сохраняет последний снимок и history в SQLite;
  старые точки удаляются по `PM_TELEMETRY_RETENTION_S` при поступлении новых.

Практический код клиента, частота и семантика ключей: [METRICS_GUIDE.md](METRICS_GUIDE.md).

### 4.7 Server-held шарды

`GET /shards` возвращает инвентарь без текстов:

```json
{"shards": [{"shard_id": "wave1", "n_chunks": 20000, "n_labels": 500,
             "created_at": 1789245600.0, "updated_at": 1789245600.0}]}
```

`POST /shards/{shard_id}/chunks` принимает `{"chunks": [{"chunk_id": HASH, "text": string}]}`. Текст
непустой, до 20 000 символов; в одном запросе 1…5 000 чанков с уникальными id. Уже известный
`chunk_id` не перезаписывается. Ответ `201` содержит `shard_id`, `received`, `new`, `total`.

`GET /shards/{shard_id}/chunks` без query возвращает весь шард:

```json
{"shard_id": "wave1", "chunks": [{"chunk_id": "<hash>", "text": "..."}]}
```

Пара `partition` (целое ≥ 0) и `n_partitions` (целое ≥ 1) фильтрует ответ. Оба параметра обязательны
вместе, `partition < n_partitions`. Принадлежность вычисляется без состояния:
`int(chunk_id[:8], 16) % n_partitions`. Поэтому повторный GET с теми же параметрами стабилен, а все
партиции непересекаются и в объединении дают полный шард.

`GET /shards/{shard_id}/labels` возвращает `{"shard_id": "wave1", "labels": {HASH: int}}`. Labels
привязаны к chunk, общие для всех кампаний над этим шардом и не помечены как train/held-out.

`GET /shards/{shard_id}/preview?offset=0&limit=25&q=needle` предназначен для операторского UI.
`offset` ≥ 0, `limit` 1…100, `q` — необязательная case-insensitive подстрока chunk ID или текста
длиной до 200 символов. Неизвестный shard даёт `404`. Ответ соединяет chunk с optional oracle label:

```json
{"shard_id": "wave1", "offset": 0, "limit": 25, "query": "needle", "total": 1,
 "chunks": [{"chunk_id": "<hash>", "text": "...", "added_at": 1789245600.0,
             "label": 4, "source": "static:fineweb", "labeled_at": 1789245610.0}]}
```

Вкладка `Datasets` в `/dashboard` использует эти две GET-ручки. Импорт JSON/JSONL/TXT вычисляет
SHA-256 для строк без `chunk_id`, удаляет дубликаты и вызывает существующий POST батчами ≤ 5000.

Для Parquet GUI сначала отправляет сырой файл в `POST /imports/parquet/inspect` с
`Content-Type: application/vnd.apache.parquet` и URL-encoded именем в `X-Filename`. Ответ содержит
одноразовый `upload_token`, число строк, размер, список колонок с Arrow-типами и рекомендованный
mapping. По умолчанию один файл ограничен 512 MiB и 500 000 строк, token живёт один час.

Затем `POST /shards/{shard_id}/imports/parquet` принимает:

```json
{"upload_token": "<32 hex>", "text_column": "text", "id_column": null,
 "label_column": "score", "label_threshold": 3}
```

`text_column` обязан быть строковым. При `id_column=null` ID равен SHA-256 текста; выбранная ID-колонка
должна содержать lowercase hex hash длиной 16…128. `label_column=null` не импортирует labels.
С label-колонкой `label_threshold=null` допустимы только готовые `0/1`; числовой threshold преобразует
значение по правилу `value >= threshold`. Импорт chunks и labels атомарный, token удаляется после
успеха. Пустые и слишком длинные тексты пропускаются, дубли внутри файла и уже существующие строки
считаются в ответе. Источник label хранится как `parquet:<filename>:<column>`.

### 4.8 Кампании

Запрос `POST /campaigns`:

```json
{
  "campaign_id": "c1", "shard_id": "wave1", "mode": "sharded",
  "model": {"kind": "classifier", "id": "quality-clf-v1"},
  "metrics": [], "node_ids": ["node-a-3f9c1d", "node-b-a83a89"],
  "schedule": [500, 1000, 1500], "train_mode": "fresh", "strategy": "cutoff",
  "k_frac": 0.1, "good_min": 1, "seed": 0
}
```

| Поле | Смысл |
|---|---|
| `mode` | `sharded` (по умолчанию) или `experts` |
| `node_ids` | 1…64 уникальных нод; `node_ids[i]` владеет партицией/доменом `i` |
| `schedule` | 1…20 строго возрастающих положительных cumulative label counts **на партицию** |
| `train_mode` | `fresh` (default: новый классификатор на всех текущих labels каждый раунд) или `continue` (warm-start с checkpoint прошлого раунда) |
| `strategy` | `random`, `cutoff` (по умолчанию) или `qbc` |
| `k_frac` | доля полного пула в финальном top-k, `(0, 1]` |
| `good_min` | необязательно: label ≥ порога считается заведомо хорошим при finalize |
| `seed` | seed детерминированного выбора oracle labels |

Число партиций фиксировано на `len(node_ids)` на всю кампанию. Каждая партиция должна содержать не
меньше `schedule[-1]` чанков. Все ноды должны быть зарегистрированы и поддерживать `model.kind`.
Создание ставит первые `schedule[0]` labels в каждом домене и создаёт раунд 1.

Campaign преобразует `train_mode` в явную task `operation`. При `fresh` каждый раунд получает
`train=fresh`, без input checkpoint, и уникальный output checkpoint, равный `round_id`. При `continue`
первый раунд всё равно `fresh`, а каждый следующий получает `train=continue`, input checkpoint
предыдущего `round_id` и output checkpoint текущего `round_id`.

В каждой задаче `params` содержит `mode`, индекс `partition`, `n_partitions`, `n_labels`. В `sharded`
нода фильтрует общий ответ `/labels` той же функцией, обучается и возвращает score ровно для своей
партиции. В `experts` она так же обучается только на labels своего домена, но обязана вернуть score для
каждого chunk полного пула. Лишние и отсутствующие chunk ids обнаруживаются при `advance`.

`POST /campaigns/{campaign_id}/advance` возвращает `status: waiting`, пока не отправились все ноды;
затем независимо пополняет labels каждого домена и создаёт следующий раунд (`advanced`) либо завершает
кампанию (`done`). Финальный merge: z-score + объединение непересекающихся списков для `sharded`, среднее
по экспертам для `experts`. Reliability Gate в этих двух server-master campaign-режимах пока не применяется.
Источник labels ортогонален режиму: это может быть как `MockOracle`, так и `StaticOracle` с реальной
предварительной разметкой.

`GET /campaigns/{campaign_id}` возвращает сохранённые `spec`, `rounds_done`, `status`, `current_round` и
после завершения `result.selected`, `n_labels`, `n_known_good`, `generated_at`.

## 5. Состояния

```
нода:       (нет) --handshake--> зарегистрирована;  online ⇔ пульс младше 3 × interval
раунд:      open --close--> closed
участник:   assigned --ack--> accepted --submit--> submitted
            assigned ----------------submit------> submitted
```

## 6. Вне контракта (известные пробелы)

- Операторские ручки без авторизации.
- Нет списка раундов: оператор обращается к раунду по известному `round_id`.
- Дедлайнов у раундов нет: раунд живёт, пока его не закроют.
- Нода не может отказаться от задачи или сообщить об ошибке выполнения.
- `params.proxy_lr` раунда не сверяется с `agg_stats.proxy_lr` из отправки.
- Калибровка и gate пока верят присланным счётчикам held-out: CP не проверяет вычисление curve независимо.
- API не выделяет отдельный held-out split из shard labels; его должен задать внешний workflow.
- Авторизации нет (2.1): любой вызывающий действует от имени любого `node_id`.

## 7. Конфигурация сервера

| Env | По умолчанию |
|---|---|
| `PM_HEARTBEAT_S` | `3` |
| `PM_TELEMETRY_RETENTION_S` | `604800` — 7 дней истории heartbeat в SQLite |
| `PM_TELEMETRY_HISTORY_LIMIT` | `240` — default `limit` для `/telemetry/history` |
| `PM_GATE_MIN_PRECISION` | `0.2` — нижняя граница точности held-out у отсечки ноды (4.4) |
| `PM_MAX_BODY_BYTES` | 32 MiB |
| `PM_MAX_PARQUET_BYTES` | 512 MiB на staged-файл |
| `PM_MAX_PARQUET_ROWS` | 500 000 строк на staged-файл |
| `PM_PARQUET_UPLOAD_TTL_S` | `3600` — срок жизни неиспользованного upload token |
| `PM_DB_PATH` | `data/control_plane.sqlite3` |
| `PM_AUDIT_LOG` | `data/submissions.jsonl` |
| `PM_GOLDEN_LABELS_PATH` | не задан; JSON-карта `chunk_id -> label`, читается при старте и включает `StaticOracle` |
| `PM_MOCK_ORACLE_GOOD_RATE` | `0.1` — доля положительных labels в fallback `MockOracle` |
| `PM_HOST` / `PM_PORT` | `0.0.0.0` / `8100` |
