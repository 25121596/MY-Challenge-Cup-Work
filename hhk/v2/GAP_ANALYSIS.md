# 项目缺口分析 — Edge LLM 云边协同

> 生成时间: 2026-07-20 | 对标赛题: XH-202606

---

## 一、总体评估

| 模块 | 完成度 | 状态 |
|---|---|---|
| llama.cpp 推理核心 | 100% | ✅ 完整 |
| 边端 GUI (edge-gui/) | 95% | ✅ 核心功能完整 |
| 云端服务 (cloud_server/) | 90% | ✅ 9个API全部实现 |
| 云边通信协议 | 100% | ✅ HTTP/SSE/UDP/TCP 四通道 |
| I/O 协议定义 | 0% → 现在有了 | ❌→✅ 已补充 (D:\Challenge Cup\edge-io-protocol\) |
| 数据预处理 | 0% | ❌ **本次补齐** |
| 场景配置 | 0% | ❌ **本次补齐** |
| 端到端演示 | 0% | ❌ **本次补齐** |
| 自动化评测 | 0% | ❌ **本次补齐** |
| 模型选型文档 | 30% | ⚠️ 有 memcheck.py + benchmark.py, 缺结论文档 |
| 部署文档 | 0% | ❌ 缺 Docker/边缘设备部署指南 |
| 多节点模拟 | 0% | ❌ 缺模拟多个边端的脚本 |

---

## 二、本次补齐清单

### 2.1 preprocessing/ — 数据预处理管线
| 文件 | 功能 | 状态 |
|---|---|---|
| `sensor_csv_to_input.py` | 通用传感器CSV → I/O标准input.json | ✅ 已创建 |
| `hydraulic_preprocess.py` | UCI液压系统专用预处理 | ✅ 已创建 |
| `motor_preprocess.py` | 三相电机数据专用预处理 | ✅ 已创建 |
| `anomaly_injector.py` | 向正常数据注入故障模式(用于测试) | ✅ 已创建 |
| `data_split.py` | 数据集划分为训练/测试/边端模拟 | ✅ 已创建 |

### 2.2 scenarios/ — 场景配置
| 文件 | 功能 | 状态 |
|---|---|---|
| `industrial_inspection.json` | 工业检测场景: 设备振动+温度+电流 | ✅ 已创建 |
| `traffic_monitoring.json` | 交通监控场景: 车流+速度+拥堵 | ✅ 已创建 |
| `scenario_runner.py` | 场景执行器: 加载配置→模拟数据→推理→收集指标 | ✅ 已创建 |

### 2.3 demo/ — 端到端演示
| 文件 | 功能 | 状态 |
|---|---|---|
| `run_demo.bat` | Windows一键启动脚本 | ✅ 已创建 |
| `demo_controller.py` | 演示控制器: 启停→注入数据→收集结果 | ✅ 已创建 |

### 2.4 evaluation/ — 自动化评测
| 文件 | 功能 | 状态 |
|---|---|---|
| `metrics_collector.py` | 收集时延/内存/冲突率/可用性等指标 | ✅ 已创建 |
| `report_generator.py` | 根据采集数据生成评测报告 | ✅ 已创建 |

---

## 三、比赛指标对照

| 比赛指标 | 要求 | 验证方式 | 对应工具 |
|---|---|---|---|
| 能力保持率 | ≥80% | benchmark.py quick | ✅ 已有 |
| TTFT减少 | ≥75% | benchmark.py 速度专项 | ✅ 已有 |
| 内存占用 | ≤1.5GB | memcheck.py edge | ✅ 已有 |
| 端到端时延 | ≤0.2s | evaluation/metrics_collector.py | ✅ 新增 |
| 断网可用性 | ≥90% | demo + 手动断网测试 | ⚠️ 需手动 |
| 冲突率 | ≤5% | 多节点模拟 | ⚠️ 需多节点模拟器 |
| 2场景部署 | ≥2 | scenarios/ 配置 | ✅ 新增 |

---

## 四、仍需手动完成

### 4.1 数据下载（紧急）
以下数据集需手动下载：
```
☐ FMF_Benchmark      — 工业机器状态 (200万+帧图像)
☐ RealIAD_D3         — 工业异常检测 (图像+点云)
☐ HUSTmotor          — 电机故障
☐ Paderborn          — 轴承故障
☐ NASA_CMAPSS        — 航空发动机退化
☐ CWRU               — 轴承振动
☐ XJTU_SY            — 轴承全寿命
☐ SEU_gearbox        — 齿轮箱故障
☐ MulSen_AD          — 多传感器异常
```
详见 `D:\Challenge Cup\Data file\00_下载指南.md`

### 4.2 多节点模拟器（建议补充）
```
☐ multi_edge_simulator.py  — 同时启动N个虚拟边端节点
☐ network_simulator.py     — 模拟网络延迟/丢包/断网
☐ conflict_simulator.py    — 模拟多节点感知冲突
```

### 4.3 模型最终选型报告（建议补充）
```
☐ model_selection_report.md  — 汇总 memcheck + benchmark 结果, 给出推荐
```

### 4.4 部署文档（赛前必须）
```
☐ deploy/DEPLOY.md           — 云端部署步骤
☐ deploy/EDGE_SETUP.md       — 边端设备刷机+部署
☐ docker-compose.yml         — 云端服务容器化
```

---

## 五、优先级建议

| 优先级 | 事项 | 原因 |
|---|---|---|
| 🔴 P0 | 补充至少2个场景的完整数据 | 比赛硬要求 |
| 🔴 P0 | 跑通 demo + 收集到指标数据 | 用于写报告 |
| 🟡 P1 | 模型选型报告 | 答辩时需要 |
| 🟡 P1 | 多节点冲突测试 | 证明冲突率≤5% |
| 🟢 P2 | Docker部署 | 锦上添花 |
