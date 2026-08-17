# HHK 接入说明

Integration 可以直接接收 HHK `HeartbeatManager::build_report_json()` 的 JSON。

必须注意原 HHK `edge-gui/app.cpp` 目前给 heartbeat 写入的 RTT 是：

```cpp
_cloud_client.is_reachable() ? 0 : -1
```

而真正的 `measure_rtt_ms()` 在后面才调用。请应用 `app_rtt_fix.patch`，让 heartbeat 中的 `network.rtt_ms` 变成真实 RTT。

另外，原代码传给 heartbeat 的 `task_queue_len` 是 `_engine.is_running() ? 1 : 0`。Integration 不把这个字段当 EDF queue 使用；真实 queue 由 `RealTaskManager` 管理。
