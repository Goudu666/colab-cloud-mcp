# colab-cloud-mcp

让 **Claude 云端对话（网页 / 手机 App / Desktop Custom Connectors）**、**Google Antigravity** 以及 **Claude Code** 通过公网 HTTPS (443) **Streamable HTTP** 协议直接连接 Google Colab GPU 运行时的远程 MCP Server。

## 为什么需要这个项目？

本地 `stdio` 模式的 Colab MCP 依赖本机 CLI 进程与本机浏览器标签页桥接——一旦**合上笔记本电脑、休眠或本地网络闪断**，连接就会立即断开，且无法在网页版或手机端云对话中使用。

`colab-cloud-mcp` 将 MCP 服务端**直接运行在 Google Colab 虚拟机内部**，并通过 HTTPS 隧道（Cloudflare Tunnel 或 ngrok）暴露符合最新规范的 `Streamable HTTP` 端点：

- **真正脱离本机**：云端对话客户端直接访问 Colab 端服务，本机合盖、断网或切换到手机端对话均不断连。
- **持久化 Python 命名空间 + 自动捕获图表**：`execute_python` 跨轮次保留所有变量、PyTorch 模型与显存张量；Matplotlib 绘图自动转为 PNG 图像块回传给对话界面。
- **异步后台长任务管理**：针对云端单次工具调用超时限制（Claude 240s / Antigravity 默认 180s），内置 `start_background_job` / `get_job_status` / `cancel_job`，支持数小时模型训练或大权重下载至 Google Drive，随时在对话中拉取最新日志。
- **会话导出为 `.ipynb`**：随时调用 `export_session_notebook` 将执行过的所有代码与输出图像保存为标准 Jupyter Notebook。
- **三重鉴权兼容所有客户端**：
  1. **零请求头加密路径模式**（`<TUNNEL_URL>/s/<MCP_TOKEN>/mcp`）：适配 Claude Custom Connectors 的 `No sign in` 模式，粘贴单条 URL 即可安全鉴权。
  2. **标准 Bearer Header 模式**（`Authorization: Bearer <MCP_TOKEN>`）：适配 Google Antigravity `mcp_config.json`、Claude Code HTTP MCP 与 Claude Request Headers。
  3. **内置 OAuth 2.1 (PKCE + DCR + CIMD)**：完整实现 `/.well-known/oauth-protected-resource`、`/.well-known/oauth-authorization-server`、动态客户端注册与 PKCE 授权码流程。

## 快速开始（在 Google Colab 中运行）

打开本仓库的 `notebooks/colab_cloud_mcp.ipynb` 按顺序执行即可启动服务并打印所有客户端接入配置。

合盖前在最后一个单元格运行保活监控：

```python
mcp_server.keep_alive(status_interval_seconds=60)
```

> **Colab 实例生命周期提示**：MCP 连接本身已完全脱离本机，但 Google Colab 对免费版/Pro 版在关闭全部浏览器标签页后仍有后台空闲回收时间窗口；如需合盖后持续运行数小时，建议配合 `keep_alive()` 或 Colab Pro+ 后台执行，并将日志与权重保存至 `/content/drive/MyDrive`。

## 固定域名配置（重启 Colab 免改客户端 URL）

在 Colab 左侧 **Secrets** 中配置以下密钥并以 `provider="auto"` 启动：

| 方案 | Colab Secrets 变量名 | 说明 |
| :--- | :--- | :--- |
| **固定鉴权密钥** | `MCP_TOKEN` | 自定义高强度字符串，重启免改 Token |
| **Cloudflare Named Tunnel** | `CLOUDFLARE_TUNNEL_TOKEN`、`CLOUDFLARE_HOSTNAME` | 在 Cloudflare Zero Trust Tunnel 将子域名指向本地端口 `8765` |
| **ngrok 免费静态域名** | `NGROK_AUTHTOKEN`、`NGROK_DOMAIN` | 使用 ngrok 账号免费静态域名（需安装 `pyngrok`） |

## 客户端接入指南

### 1. Claude 云端（Custom Connectors）

- 进入 **Customize -> Connectors -> + Add -> Add custom connector**
- **最简方式（免配 Header）**：URL 填入启动器打印的 `<TUNNEL_URL>/s/<MCP_TOKEN>/mcp`，Authentication 选择 `No sign in`
- **Header / OAuth 方式**：URL 填入 `<TUNNEL_URL>/mcp`，在 Request headers 添加 `Authorization: Bearer <MCP_TOKEN>` 或选择 OAuth 授权

### 2. Google Antigravity（App / IDE / CLI）

编辑 `~/.gemini/config/mcp_config.json` 或工作区 `.agents/mcp_config.json`：

```json
{
  "mcpServers": {
    "colab-remote": {
      "serverUrl": "<TUNNEL_URL>/mcp",
      "headers": {
        "Authorization": "Bearer <MCP_TOKEN>"
      },
      "timeoutSeconds": 600
    }
  }
}
```

### 3. Claude Code CLI

```bash
claude mcp add --transport http colab-remote <TUNNEL_URL>/mcp \
  --header "Authorization: Bearer <MCP_TOKEN>"
```

## 内置 MCP 工具一览

| 工具名称 | 功能说明 |
| :--- | :--- |
| `get_runtime_status` | 查看 GPU 型号与显存 (`nvidia-smi`)、CPU/内存/磁盘、PyTorch CUDA 状态、Google Drive 挂载情况与后台任务数 |
| `execute_python` | 在持久化命名空间执行 Python 代码，自动捕获 `stdout`/`stderr`、返回值及 Matplotlib PNG 图像 |
| `reset_python_namespace` | 清空持久化变量并触发 `gc.collect()` 与 `torch.cuda.empty_cache()` 释放显存 |
| `run_shell` | 执行 Bash 命令（支持工作目录与超时进程组清理） |
| `install_packages` | 通过 `pip` 安装或升级 Python 依赖包 |
| `start_background_job` | 异步启动长时间后台训练或大权重下载任务，日志实时落盘 |
| `get_job_status` / `list_jobs` / `cancel_job` | 查询后台任务状态与末尾日志、列出或终止后台任务 |
| `list_files` / `read_file` / `write_file` | 浏览、按行读取或写入 Colab 工作区与 Google Drive 文件 |
| `export_session_notebook` | 将当前会话执行过的所有代码与图表导出为 `.ipynb` 文件 |

## 本地测试

```bash
python3 tests/test_server.py
```

## License

MIT
