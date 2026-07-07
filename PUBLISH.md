# 发布到 GitHub(一次搞定)

前提:装了 [GitHub CLI](https://cli.github.com/) 并登录过(`gh auth login`)。
在本文件夹里运行 —— 一条命令建库并推送:

### Linux / Mac
```bash
rm -rf .git
git init && git add -A && git commit -m "init: waifu-nl2tags"
gh repo create waifu-nl2tags --public --source=. --remote=origin --push
```

### Windows PowerShell
```powershell
Remove-Item -Recurse -Force .git -ErrorAction SilentlyContinue
git init; git add -A; git commit -m "init: waifu-nl2tags"
gh repo create waifu-nl2tags --public --source=. --remote=origin --push
```

推送完成后,`gh` 会告诉你仓库地址。**别的机器一行安装**:

```bash
pip install "waifu-nl2tags[train] @ git+https://github.com/<你的用户名>/waifu-nl2tags.git"
```

想私有仓库就把 `--public` 换成 `--private`(私有仓库安装时 git 需要你的凭据)。

---

## 没有 gh 的手动方式

先在 github.com 建一个空仓库 `waifu-nl2tags`(不要勾 README),然后:

```bash
rm -rf .git
git init && git add -A && git commit -m "init: waifu-nl2tags"
git branch -M main
git remote add origin https://github.com/<你的用户名>/waifu-nl2tags.git
git push -u origin main
```

> 注:`data/`、`out/`、`*.jsonl`、`.venv/`、`__pycache__/` 已在 `.gitignore` 里,
> 不会被提交;`nl2tags/data/tag_ontology.json` 会随包发布。
