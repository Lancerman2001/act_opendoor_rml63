# 发布到 GitHub

源码包与 GitHub 仓库发布范围为训练脚本、配置、依赖、测试和说明。数据、训练模型、下载的第三方 wheel/权重和运行日志不在其中；需要分享模型时可另行准备有明确授权的模型发布包。

## 创建可审查的源码包

```bash
source env.sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/export_release.py --check-only
.venv/bin/python scripts/export_release.py
```

产物位于 `dist/`，ZIP/TAR.GZ 中包含文件级 SHA256 manifest，外部另有压缩包校验和。重新生成使用 `--force`。打包器会拒绝用户 home 绝对路径、符号链接和非白名单文件。

## 建立仓库

本项目仓库是 [Lancerman2001/act_opendoor_rml63](https://github.com/Lancerman2001/act_opendoor_rml63)。如果你维护的是自己的 fork，请改用自己的远程地址。首次建立本地 Git 仓库时可执行：

```bash
git init -b main
git add README.md LICENSE THIRD_PARTY.md requirements.txt requirements.lock.txt \
  .gitignore .gitattributes .python-version .github configs docs scripts tests \
  setup.sh env.sh run.sh train.sh evaluate.sh benchmark.sh
git diff --cached --stat
git commit -m "Prepare reproducible LeRobot ACT training project"
git remote add origin https://github.com/Lancerman2001/act_opendoor_rml63.git
git push -u origin main
```

如果项目已是 Git 仓库，沿用现有仓库和分支；远程已有提交时应先同步历史。打包脚本只生成本地源码包，不自动推送远程仓库。

`.github/workflows/checks.yml` 使用 CPU 依赖执行单元测试与发布白名单检查，不运行 CUDA 训练。GPU环境与短训练检查需在具有 NVIDIA GPU 的机器上执行。
