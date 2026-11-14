# D:\llama.cpp 整理计划

> llama.cpp 原项目所有内容保留不动，只清理冗余副本和中间产物。

---

## 1. 删除 `competition/`（省 3.2 GB）

**原因：** 这是 `supernova701/competition` 仓库的完整 git clone，里面只有 `hhk/` 目录。而 `D:\llama.cpp` 本身就是 `hhk/` 的内容——这是一个套娃副本。代码已经通过 git push 同步到 GitHub，本地不需要保留这个 clone。

```bash
rm -rf D:\llama.cpp\competition
```

## 2. 删除多余的模型文件（省 ~2 GB）

**保留：** `models/Qwen_Qwen3-1.7B-IQ4_XS.gguf`（1.1 GB，边端主模型）

**删除：**

| 文件 | 大小 | 原因 |
|---|---|---|
| `Qwen3-1.7B-IQ4_NL.gguf` | 1.0 GB | 备用模型，与 IQ4_XS 功能重复 |
| `Qwen3-1.7B-Q3_K_M.gguf` | 897 MB | 备用模型，精度更低 |
| `ggml-vocab-*.gguf` (22个) | ~71 MB | llama.cpp 分词器测试桩，边端项目用不到 |
| `ggml-vocab-*.inp` / `*.out` (19个) | ~50 KB | 测试输入输出 |

```bash
cd D:\llama.cpp\models
rm Qwen3-1.7B-IQ4_NL.gguf
rm Qwen3-1.7B-Q3_K_M.gguf
rm ggml-vocab-*.gguf
rm *.inp *.out
```

## 3. 删除中间产物 JSON（省 ~5 MB）

这些是之前运行 `edge_run.py` / `run_all.py` / demo 时产出的中间文件和结果，已经不需要了：

```bash
cd D:\llama.cpp
rm -r results/
rm processed_*.json
rm demo_results.json
rm edge_run_results.json
rm edge_run_hydraulic_inputs.json
rm stream_results.json
rm test_inputs_industrial_inspection_v1.json
```

## 4. 删除 Python 缓存

```bash
rm -rf D:\llama.cpp\.pytest_cache
find D:\llama.cpp -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
```

## 5. 保留清单（不动）

| 类别 | 内容 |
|---|---|
| llama.cpp 原项目 | `ggml/` `src/` `include/` `common/` `cmake/` `vendor/` `examples/` `tests/` `docs/` `tools/` `benches/` `scripts/` `ci/` `pocs/` `media/` `licenses/` `grammars/` `gguf-py/` `conversion/` `app/` `convert_*.py` `CMakeLists.txt` `Makefile` `flake.nix` 等 |
| 竞赛边端 | `edge-gui/` `edge-io-protocol/` `preprocessing/` `edge_preprocess/` `scenarios/` `evaluation/` `demo/` `cloud_server/` `run_all.py` `edge_run.py` `stream_adapter.py` |
| 编译产物 | `build/`（可保留 `build/bin/Release/*.exe` + `*.dll`） |
| 模型 | `models/Qwen_Qwen3-1.7B-IQ4_XS.gguf` |
| 文档 | `*.md` `AGENTS.md` `CLAUDE.md` 等 |

---

## 预计效果

| 项目 | 清理前 | 清理后 |
|---|---|---|
| 总大小 | ~10 GB | **~5 GB** |
| 节省 | — | **~5 GB** |
| 文件整洁度 | 散落 15+ 中间产物 JSON | 清爽 |
