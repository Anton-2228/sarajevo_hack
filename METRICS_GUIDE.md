# Метрики ноды: практический гайд

Этот документ описывает realtime-телеметрию Proxy Mesh: что должна отправлять нода, как часто,
как данные хранятся и где их смотреть. Формальный wire-контракт находится в [CONTRACT.md](CONTRACT.md).

## Коротко

- После `handshake` отправляй heartbeat с интервалом `heartbeat_interval_s`, который вернул сервер.
  По умолчанию это 15 секунд.
- Во время задачи передавай `status="busy"`, `round_id`, текущий `stage` и числовой `load`.
- После задачи сразу отправь `status="idle"` без `round_id`, `stage` и старого `load`.
- Не отправляй тексты, пути, имена пользователей, hostname или другие приватные данные. `load` принимает
  только конечные числа.
- Смотреть: `/dashboard` для людей, `/metrics` для Prometheus/Grafana,
  `/metrics.json` для последнего состояния, `/telemetry/history` для сохранённой истории.

## Жизненный цикл клиента

`handshake` сообщает редко меняющиеся возможности ноды: CPU, RAM, GPU, версии ПО и поддерживаемые
типы моделей. Прогресс задачи туда не входит. Heartbeat сообщает живость и текущий снимок работы.
`submit` отправляется один раз по завершении задачи и содержит scores и финальные агрегаты, а не
realtime-состояние.

Готовый шаблон на SDK:

```python
import threading

from node_sdk.client import ControlPlane

cp = ControlPlane("http://control-plane:8100", state_file=".pm_node.json")
cp.handshake(
    name="worker-a",
    hardware={
        "cpu_cores": 16,
        "ram_gb": 64,
        "gpus": [{"model": "A100", "vram_gb": 80}],
    },
    model_kinds=["classifier"],
    software={"agent_version": "0.11.0", "python": "3.12"},
)

state_lock = threading.Lock()
state = {"status": "idle", "load": {}}


def heartbeat_snapshot():
    # Возвращаем копии: рабочий поток может обновлять state одновременно с heartbeat-потоком.
    with state_lock:
        return {**state, "load": dict(state.get("load", {}))}


def report(*, status, round_id=None, stage=None, **load):
    next_state = {"status": status, "load": load}
    if round_id is not None:
        next_state["round_id"] = round_id
    if stage is not None:
        next_state["stage"] = stage
    with state_lock:
        state.clear()
        state.update(next_state)


stop_heartbeat = cp.start_heartbeat(heartbeat_snapshot)

try:
    for task in cp.tasks():
        round_id = task["round_id"]
        cp.ack(round_id)

        report(status="busy", round_id=round_id, stage="downloading", progress_pct=0)
        documents, labels = fetch_task_data(task)
        operation = task["operation"]

        if operation["train"] == "fresh":
            model = create_model_from_recipe(task["model"])
        else:  # continue and skip both name the checkpoint to load
            model = load_local_checkpoint(operation["input_checkpoint_id"])

        total = len(documents)
        if operation["train"] != "skip":
            for step, batch in enumerate(train_batches(documents, labels), start=1):
                loss = train(model, batch)
                processed = min(step * len(batch), total)
                report(
                    status="busy",
                    round_id=round_id,
                    stage="training",
                    progress_pct=100 * processed / total,
                    docs_processed=processed,
                    docs_total=total,
                    docs_per_sec=current_docs_per_second(),
                    eta_s=estimated_seconds_left(),
                    train_loss=loss,
                    cpu_pct=current_cpu_percent(),
                    ram_pct=current_ram_percent(),
                    gpu_util_pct=current_gpu_percent(),
                    gpu_mem_pct=current_gpu_memory_percent(),
                )
            save_local_checkpoint(operation["output_checkpoint_id"], model)

        report(status="busy", round_id=round_id, stage="scoring", progress_pct=0)
        scores, agg_stats = score_documents(model, documents)
        report(status="busy", round_id=round_id, stage="uploading", progress_pct=100)
        cp.submit(round_id, scores, agg_stats)
        report(status="idle")
finally:
    stop_heartbeat.set()
```

Функции работы с данными и ресурсами в примере условные: подключи существующий trainer и системный
мониторинг проекта. Важно, что тяжёлый цикл только обновляет локальный снимок, а SDK сам отправляет
его в отдельном daemon-потоке.

## Как понять, нужно ли обучать

Не делай эвристику по `round_id`, `model.id`, `n_labels` или наличию локального файла. Читай
`task.operation`:

| `operation.train` | Действие ноды |
|---|---|
| `fresh` | Создать новую модель из recipe `task.model`; input checkpoint отсутствует; обучить и сохранить в `output_checkpoint_id` |
| `continue` | Загрузить `input_checkpoint_id`, продолжить обучение на данных текущей задачи и сохранить в `output_checkpoint_id` |
| `skip` | Загрузить `input_checkpoint_id`, не запускать trainer, сразу выполнить scoring |

`operation.score` сейчас всегда `true`. Checkpoint хранится самой нодой и не загружается в control
plane; его namespace — `(node_id, checkpoint_id)`. Вместе с checkpoint сохраняй manifest как минимум
с `round_id`, `model.id`, набором использованных `chunk_id` и статусом train/score/submit. Если после
рестарта приходит уже знакомый `round_id`, восстанавливайся по manifest вместо повторного обучения
или двойного submit.

Для campaign поле `train_mode` задаёт политику всех раундов. `fresh` означает новый классификатор на
полном актуальном наборе labels в каждом раунде. `continue` делает первый раунд fresh, затем связывает
каждый новый раунд с checkpoint предыдущего. Клиент всё равно ориентируется только на task
`operation`, а не повторяет эту серверную логику у себя.

## Частота heartbeat

Используй значение `heartbeat_interval_s` из ответа `POST /nodes/handshake`, а не захардкоженное
число. `ControlPlane.start_heartbeat()` уже делает это правильно. Сервер считает ноду offline после
трёх пропущенных heartbeat: при стандартном интервале примерно через 9 секунд.

Обновлять локальный `state` можно после каждого batch, но HTTP heartbeat всё равно уйдёт не чаще
серверного интервала. Не отправляй отдельный запрос на каждый документ: это создаёт нагрузку и не
делает график полезнее. После смены важной фазы или завершения задачи допустимо вызвать
`cp.heartbeat(...)` один раз немедленно, не дожидаясь следующего тика.

## Поля heartbeat

```json
{
  "status": "busy",
  "stage": "training",
  "round_id": "campaign-r1",
  "load": {
    "progress_pct": 42.5,
    "docs_processed": 8500,
    "docs_total": 20000,
    "docs_per_sec": 127.4,
    "eta_s": 90,
    "train_loss": 0.31,
    "cpu_pct": 73.5,
    "ram_pct": 61.0,
    "gpu_util_pct": 92.0,
    "gpu_mem_pct": 78.0
  }
}
```

| Поле | Как заполнять |
|---|---|
| `status` | `idle` или `busy` |
| `stage` | `downloading`, `training`, `scoring` или `uploading`; только для активной задачи |
| `round_id` | Точный `round_id` текущей задачи |
| `load` | До 32 числовых gauge; строки, bool, NaN и Infinity запрещены |

Стабильные ключи `load`, которые понимает встроенный dashboard:

| Ключ | Единица и семантика |
|---|---|
| `progress_pct` | 0…100 внутри текущей фазы/задачи; на следующей задаче сбрасывается |
| `docs_processed` | Количество уже обработанных документов |
| `docs_total` | Полный объём текущей работы |
| `docs_per_sec` | Текущая или сглаженная пропускная способность, документов/с |
| `eta_s` | Оценка оставшегося времени, секунды |
| `train_loss` | Последний или сглаженный training loss |
| `cpu_pct` | Использование CPU, 0…100 |
| `ram_pct` | Использование RAM, 0…100 |
| `gpu_util_pct` | Использование GPU, 0…100; для нескольких GPU лучше среднее |
| `gpu_mem_pct` | Использование VRAM, 0…100; для нескольких GPU лучше максимум |

Если `progress_pct` отсутствует, dashboard вычислит его как
`docs_processed / docs_total * 100`. Дополнительные числовые ключи разрешены и автоматически
появятся в Prometheus как `proxy_mesh_node_load{metric="имя"}`, но встроенный UI не рисует для них
отдельный график.

## Где смотреть

| Адрес | Для чего |
|---|---|
| `GET /dashboard` | Realtime control room: online/busy, этап, прогресс, throughput, loss, ресурсы |
| `GET /metrics.json` | Последний снимок состояния для собственного UI/интеграции |
| `GET /telemetry/history` | Сохранённые heartbeat-сэмплы, сгруппированные по `node_id` |
| `GET /metrics` | Prometheus text exposition для долгосрочного мониторинга и Grafana |
| `GET /nodes` | Реестр нод и необработанное тело последнего heartbeat |

Примеры:

```bash
# Последние 240 сэмплов каждой ноды, в хронологическом порядке.
curl 'http://127.0.0.1:8100/telemetry/history?limit=240'

# Только одна нода и только с заданного Unix timestamp.
curl 'http://127.0.0.1:8100/telemetry/history?node_id=worker-a-3f9c1d&since=1789245600&limit=500'

# Текущий Prometheus-снимок.
curl 'http://127.0.0.1:8100/metrics'
```

`limit` применяется отдельно к каждой ноде и допускает 1…5000 записей. Ответ history содержит
`received_at`, `status`, `stage`, `round_id` и `load`; внутри каждой ноды записи идут от старой к
новой, чтобы их можно было сразу рисовать.

## Хранение

Каждый принятый heartbeat атомарно обновляет последний снимок ноды и добавляет запись в таблицу
SQLite `heartbeat_samples`. Поэтому графики восстанавливаются после перезагрузки страницы и после
рестарта control plane. По умолчанию история хранится 7 дней (`PM_TELEMETRY_RETENTION_S=604800`),
а dashboard загружает последние 240 точек на ноду. При интервале 15 секунд это примерно один час.

Очистка старых записей выполняется при приёме очередного heartbeat. Для хранения за пределами окна
SQLite подключи Prometheus к `/metrics`: этот endpoint отдаёт последний gauge, а частоту и срок
хранения уже определяет Prometheus.

## Приватность и кардинальность

Heartbeat не предназначен для логов датасета. Не клади в имена ключей или значения тексты
документов, URL, локальные пути, hostname, user/email, tenant id, stack trace и сообщения ошибок.
Для дискретной фазы используй `stage`, а не динамические ключи вроде `processing_file_123`.
Это одновременно сохраняет приватность и не взрывает cardinality в Prometheus.

## Если график пустой

1. Проверь, что `POST /nodes/{node_id}/heartbeat` отвечает `200`, а не `404`/`422`.
2. Убедись, что нода использует `node_id`, полученный от этого control plane.
3. Проверь числовые типы: `"73.5"` строкой и `NaN` будут отклонены.
4. Посмотри последний снимок в `/nodes` и историю в `/telemetry/history?node_id=...`.
5. Если нода offline, heartbeat не приходил более трёх интервалов; смотри stderr SDK с сообщениями
   `[proxy-mesh] heartbeat failed`.
6. Если после обновления сервера нет старых линий, это ожидаемо: heartbeat, принятые до появления
   таблицы history, восстановить нельзя. Последний снимок остаётся видимым, новая история копится сразу.
