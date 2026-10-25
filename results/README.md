# 运行产物

该目录存放生成的实验数据，默认不提交大文件。

- `cpu-smoke/`、`cpu-profile/`、`cpu-correctness.json`：本地 toy CPU 验证，不是实际 SmolVLA 结果。
- `server_environment.json`、`server-requirements.lock.txt`：服务器实际环境。
- `profile_caches/`：Stage-A 原始缓存与更新前快照。
- `runs/`：三方法 pilot/正式 run、manifest、groups、failures、sessions、checkpoints。
- `evaluations/`：每个 checkpoint 的逐 episode ODE 评估。
- `report/`：逐seed结果、逐任务结果、同缓存成本和图。

正式结果应保留原始日志、失败记录和所有种子，缺测值保持缺测，不填造示例数据。
