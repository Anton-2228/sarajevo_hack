# Тестовые раунды и таски — гайд для отладки клиента (ноды)

Для тебя и твоего агента: как самому наспавнить раунд/кампанию на control plane и погонять по нему
своего клиента, без реальных данных и без токенов (авторизации в этой сборке нет вообще, см.
CONTRACT.md §2.1 — просто зови ручки).

Источник истины — [CONTRACT.md](CONTRACT.md). Этот файл — шпаргалка поверх него для двух конкретных
сценариев отладки.

## 0. Поднять сервер

```bash
./run.sh                    # слушает 0.0.0.0:8100, база — data/control_plane.sqlite3
curl -s localhost:8100/health
```

Swagger: http://127.0.0.1:8100/docs — там же можно жать ручки руками.
Живой дашборд: http://127.0.0.1:8100/dashboard — обновляется каждые 2 сек, видно ноды/раунды/кампании.

---

## Путь A — один тестовый раунд, максимально быстро (без чанков, без кампании)

Для отладки голого цикла клиента: handshake → heartbeat → poll tasks → ack → submit. Ничего кроме
curl не нужно.

```bash
# 1. регистрируешь ноду
curl -s -X POST localhost:8100/nodes/handshake -H 'Content-Type: application/json' -d '{
  "name": "test-client",
  "hardware": {"cpu_cores": 4, "ram_gb": 8},
  "model_kinds": ["classifier"]
}'
# -> {"node_id": "test-client-XXXXXX", "heartbeat_interval_s": 3.0, "tasks_url": "/tasks/test-client-XXXXXX"}
# сохрани node_id, дальше он везде

# 2. создаёшь раунд с этой нодой как единственным участником
curl -s -X POST localhost:8100/rounds -H 'Content-Type: application/json' -d '{
  "round_id": "t1",
  "model": {"kind": "classifier", "id": "test-clf"},
  "metrics": ["n_chunks"],
  "participants": [{"node_id": "test-client-XXXXXX", "dataset_id": "ds1"}]
}'

# 3. твой клиент видит задачу
curl -s localhost:8100/tasks/test-client-XXXXXX
# -> {"tasks": [{"round_id": "t1", "status": "assigned", "model": {...}, "dataset_id": "ds1",
#               "metrics": ["n_chunks"], "params": {}, "ack_url": "...", "submit_url": "..."}]}

# 4. ack (необязателен перед submit, но проверь что не падает)
curl -s -X POST localhost:8100/tasks/test-client-XXXXXX/t1/ack

# 5. submit — Egress Gate: только chunk_id (hex-хэш, 16-128 симв.) + score, и agg_stats
curl -s -X POST localhost:8100/tasks/test-client-XXXXXX/t1/submit -H 'Content-Type: application/json' -d '{
  "node_id": "test-client-XXXXXX",
  "round_id": "t1",
  "scores": [{"chunk_id": "aaaaaaaaaaaaaaaa", "score": 0.9},
             {"chunk_id": "bbbbbbbbbbbbbbbb", "score": 0.1}],
  "agg_stats": {"n_chunks": 2}
}'
# -> {"accepted": true, "revision": 1, "trust": "unknown", ...}

# 6. смотришь что получилось
curl -s localhost:8100/rounds/t1 | python3 -m json.tool
```

**Замечания:**
- Подписи в `submit` больше нет — поле `signature` теперь запрещено (`extra="forbid"`), убери его,
  если копировал старый пример.
- `metrics` в `POST /rounds` не может быть пустым (нужен хотя бы один ключ) — `["n_chunks"]` самый
  простой вариант, `n_chunks` и так обязателен в `agg_stats`, так что требование выполнится само собой.
- `trust` будет `unknown`, пока не пришлёшь кривую held-out (`ho_*` поля, см. CONTRACT.md §3.5.1) —
  для голой отладки клиента это не важно.
- `dataset_id` задаёт control plane в самой задаче. Нода не объявляет и не выбирает датасеты при
  handshake; для тестового fake-node это может быть произвольный id workload.

---

## Путь B — полный цикл кампании (ближе к реальности)

Если отлаживаешь не только submit, а весь цикл active learning (несколько раундов подряд, разметка,
финальный отбор) — используй кампанию. Тут уже нужны реальные чанки на сервере.

```bash
# 1. грузишь чанки в шард (сервер сам их хранит и потом отдаёт ноде)
curl -s -X POST localhost:8100/shards/t1shard/chunks -H 'Content-Type: application/json' -d '{
  "chunks": [{"chunk_id": "'"$(echo -n doc0 | shasum -a 256 | cut -d" " -f1)"'", "text": "test document 0"},
             {"chunk_id": "'"$(echo -n doc1 | shasum -a 256 | cut -d" " -f1)"'", "text": "test document 1"}]
}'
# для реального объёма — см. tools/load_golden_shard.py, он же грузит настоящие тексты и метки

# 2. регистрируешь свою(и) ноду(ы) — handshake описывает только вычислительные возможности
curl -s -X POST localhost:8100/nodes/handshake -H 'Content-Type: application/json' -d '{
  "name": "test-client", "hardware": {"cpu_cores": 4, "ram_gb": 8}, "model_kinds": ["classifier"]}'

# 3. создаёшь кампанию; mode выбирает шардинг throughput или смесь экспертов
curl -s -X POST localhost:8100/campaigns -H 'Content-Type: application/json' -d '{
  "campaign_id": "camp-test", "shard_id": "t1shard", "mode": "sharded",
  "model": {"kind": "classifier", "id": "test-clf"}, "metrics": [],
  "node_ids": ["test-client-XXXXXX"], "schedule": [1], "train_mode": "fresh",
  "strategy": "random", "k_frac": 0.5
}'
# schedule=[1]: сервер размечает 1 чанк НА ПАРТИЦИЮ и создаёт раунд
# train_mode=fresh: task явно потребует новый classifier; continue свяжет следующие раунды checkpoint-ами

# 4. сначала читаешь task.params: mode, partition, n_partitions, n_labels
curl -s localhost:8100/tasks/test-client-XXXXXX
curl -s localhost:8100/shards/t1shard/labels   # общий map; оставь labels своей partition из task.params
# sharded: тренируешься и скоришь только назначенную партицию
curl -s 'localhost:8100/shards/t1shard/chunks?partition=0&n_partitions=1'
# experts: тренируешься на labels своей партиции, но скоришь весь пул
curl -s localhost:8100/shards/t1shard/chunks

# 5. submit обязан покрывать ровно назначенный scope, иначе advance() вернёт 422
curl -s -X POST localhost:8100/tasks/test-client-XXXXXX/camp-test-r1/submit -H 'Content-Type: application/json' -d '...'

# 6. продвигаешь кампанию сам (авто-триггера по последнему submit пока нет)
curl -s -X POST localhost:8100/campaigns/camp-test/advance
curl -s localhost:8100/campaigns/camp-test | python3 -m json.tool
```

**Оракул сейчас — `MockOracle`**, детерминированная заглушка (hash от chunk_id → 0/1), не настоящая
LLM. Если нужен предсказуемый набор с реальными текстами и реальными метками — смотри
`tools/load_golden_shard.py` (тянет из FineWeb-Edu, метки реальные, вызов оракула — mock).

С несколькими нодами `node_ids[i]` всегда получает partition/domain `i`, где
`partition = int(chunk_id[:8], 16) % len(node_ids)`. Число нод фиксируется на всю кампанию. В режиме
`experts` клиент на каждом раунде переобучает модель на всех доступных labels своего домена; warm-start
остаётся внутренним решением клиента. Финальный expert score сейчас является простым средним.

---

## Готовые скрипты, если не хочется руками

| Скрипт | Что делает |
|---|---|
| [tools/fake_node.py](tools/fake_node.py) | Одна нода полного цикла (handshake→heartbeat→poll→ack→submit) на синтетике. Читай как референс, если непонятно, что должен делать твой клиент |
| [tools/demo_campaign.py](tools/demo_campaign.py) | Грузит шард, поднимает N mock-нод и прогоняет `sharded` или `experts` кампанию до конца |

```bash
.venv/bin/python tools/fake_node.py --name test-client --noise 0.3 --once
.venv/bin/python tools/demo_campaign.py --mode sharded --nodes 3 --n-pool 500 --schedule 30,60
.venv/bin/python tools/demo_campaign.py --mode experts --nodes 3 --n-pool 500 --schedule 30,60
```

`demo_campaign.py` — это буквально пример твоего клиента в миниатюре: смотри в нём, как строится
`payload` для submit и как читается `GET /shards/{id}/chunks`.

---

## Как посмотреть, что вообще происходит

- `GET /rounds/{round_id}` — статус раунда, участники, отправки
- `GET /campaigns/{campaign_id}` — статус кампании, текущий раунд, финальный отбор (когда `done`)
- `GET /metrics.json` — то же самое одним снимком, структурировано
- `GET /telemetry/history` — сохранённые heartbeat-сэмплы; графики переживают reload/restart
- `GET /dashboard` — то же самое, но глазами, обновляется само каждые 2 сек
- `GET /nodes` — реестр всех зарегистрированных нод и их online-статус
- `GET /shards` — список датасетов и покрытие labels
- `GET /shards/{shard_id}/preview?offset=0&limit=25&q=...` — страницы документов с labels

Во вкладке `Datasets` на `/dashboard` те же данные можно искать, листать и загружать из
JSON/JSONL/TXT/Parquet без curl. Для Parquet после загрузки видны Arrow-колонки: выбери text, optional
chunk ID, optional oracle label и binary/threshold-преобразование. На FineWeb-Edu ожидаемый mapping —
`text`, generated SHA-256, `score >= 3`.

Как инструментировать реальную ноду: [METRICS_GUIDE.md](METRICS_GUIDE.md).

## Частые ошибки (422/404/409) и что они значат

| Код | Причина |
|---|---|
| `422` при `/nodes/handshake` | лишнее поле в теле, или `model_kinds`/`hardware` не тех типов |
| `422` при `/rounds` | нода не зарегистрирована, не умеет заявленный `model.kind` или нарушена схема раунда |
| `422` при `/campaigns` | нода неизвестна/не поддерживает model kind, либо хотя бы одна hash-партиция меньше `schedule[-1]` |
| `404` при `/tasks/{id}/{round}/submit` | раунда нет или нода — не участник именно этого раунда |
| `409` при submit/ack | раунд уже `closed` — для кампаний это нормально, раунд закрывается сразу после того, как его отправки обработаны |
| `422` `missing_metrics` при submit | в `agg_stats` нет ключа, который раунд требует в `metrics` |
| `422` `chunk_id must be...` | `chunk_id` не хэш: только `[a-f0-9]`, 16–128 символов, никаких других символов |
| `422` при `/campaigns/{id}/advance` | submission не совпадает с task scope: своя партиция для `sharded`, 100% пула для `experts` |

Полный список полей, лимитов и кодов ошибок — [CONTRACT.md](CONTRACT.md).
