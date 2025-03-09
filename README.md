# LLM
## 使用vllm部署Qwen2.5-7B-Instruct
```
vllm serve model/Qwen2.5-VL-7B-Instruct --trust-remote-code --served-model-name gpt-4 --gpu-memory-utilization 0.90 --tensor-parallel-size 2 --port 8000
```
* `model/Qwen2.5-VL-7B-Instruct` 是文件夹本地位置
* `--trust-remote-code` 允许执行远程代码（还没弄懂）
* `--served-model-name gpt-4` 将模型名称设置为gpt-4，调用时不对会提醒
* `--gpu-memory-utilization 0.90` 设置GPU内存利用率为90%
* `--tensor-parallel-size 2` 设置张量并行处理的大小为2，（即部署2张GPU）
* `--port 8000` 在端口8000上启动
