# waifu-nl2tags

把自然语言(**中文 / 英文**)翻译成 Illustrious / NoobAI-XL 能识别的 Danbooru tag 提示词。
一行 `pip` 安装,自带 `nl2tags` 命令。

```
"画一个金发双马尾女孩，蓝眼睛，校服，樱花树下微笑"
  ->  1girl, solo, blonde hair, twintails, blue eyes, school uniform, smile,
      cherry blossoms, looking at viewer, masterpiece, best quality,
      amazing quality, very aesthetic, absurdres
```

## 一行安装(在你那台强机器上)

先发布到你自己的 GitHub(见 `PUBLISH.md`,一条 `gh` 命令),然后在**任何别的电脑**:

```bash
# 全量(含微调/推理所需的 torch/transformers/…),训练即用
pip install "waifu-nl2tags[train] @ git+https://github.com/<你的用户名>/waifu-nl2tags.git"
```

```powershell
# Windows 一样,或用自带脚本:
irm https://raw.githubusercontent.com/<你的用户名>/waifu-nl2tags/main/install.ps1 | iex
```

只想要 CLI / 数据工具 / 零样本 `serve --proxy`?去掉 `[train]` 即可(极轻)。

装完验证:

```bash
nl2tags doctor        # 认出你的双卡 + 推荐档位
nl2tags quickstart    # 造演示数据并打印训练命令
```

## 选“厉害的模型”(你的双 RTX PRO 6000 = 192 GB)

```bash
nl2tags presets
```

| preset | base | 方式 | 显存 |
|---|---|---|---|
| balanced | Qwen3-8B | QLoRA | 24 GB |
| strong | Qwen3-14B | QLoRA | 48 GB |
| **max**(默认) | **Qwen3-32B** | QLoRA | 单卡 96 GB |
| full-8b | Qwen3-8B | 全量微调 | 双卡(accelerate) |

零样本(不训练)可直接跑 **Qwen3-32B** 双卡张量并行,甚至 **Qwen3-235B-A22B MoE**(4bit,192 GB 放得下)——这是最强的一档。

## 三步用起来

```bash
# 1 数据:合成(中英混)+ 挖你自己的卡片(PNG 里已带 tag)
nl2tags gen --n 20000 --lang mix
nl2tags cards --cards E:/waifumaster/cards --strip-quality
nl2tags dataset --inputs data/synth.jsonl data/cards.jsonl
# 2 训练(默认 Qwen3-32B QLoRA;单卡即可)
nl2tags train --preset max
# 全量微调走双卡:accelerate launch -m nl2tags.train_qlora --preset full-8b
# 3 推理
nl2tags infer --adapter out/adapter "银发猫娘女仆，红眼睛，室内"
```

## 不想训练?零样本跑大模型

```bash
# 双卡拉起 Qwen3-32B(或换成 235B MoE)
vllm serve Qwen/Qwen3-32B --tensor-parallel-size 2
export OAI_BASE_URL=http://localhost:8000/v1 OAI_API_KEY=x OAI_MODEL=Qwen/Qwen3-32B
nl2tags baseline "两个女孩在海滩，比基尼，夏天"     # 立即出结果
```

## 网页界面（左输入 · 右输出）

`nl2tags serve` 自带一个网页:左边输自然语言,右边出 prompt,支持一键复制(含负面词模板)、示例、`Ctrl+Enter` 翻译。

```bash
nl2tags serve --adapter out/adapter     # 或 --proxy 接本地 vLLM/Ollama
# 浏览器打开 http://localhost:8000
```

## 接到游戏后端

```bash
nl2tags serve --adapter out/adapter --port 8000      # 微调模型
nl2tags serve --proxy                                # 或代理到本地 vLLM/Ollama
# POST /translate  {"text":"..."}  ->  {"prompt":"1girl, solo, ..."}
```

```python
from nl2tags import load_model, translate
load_model("Qwen/Qwen3-32B", "out/adapter")   # 启动时加载一次
prompt = translate(user_text)                  # 喂给你的 SDXL 管线
```

## 设计要点

模型只学**核心内容 tag**;质量词、rating、排序、去下划线/括号转义由 `nl2tags/illustrious.py`
确定性后处理添加,所以质量模板可换、无需重训。数据 = 反向合成(已知 tag 造用户会打的 NL)
+ 挖你真实卡片(与游戏分布完全一致)。想更准可下游再接 TIPO/DanTagGen 做 tag 扩写。

发布到 GitHub:见 `PUBLISH.md`。
