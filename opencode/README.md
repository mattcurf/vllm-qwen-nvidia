# OpenCode CLI

AI-powered coding assistant that works with local and remote LLM providers.

## Installation

### macOS

**Option 1: Homebrew (Recommended)**
```bash
brew install opencode
```

**Option 2: npm**
```bash
npm install -g @opencode/cli
```

**Option 3: Binary Download**
```bash
curl -fsSL https://opencode.ai/install | bash
```

### Windows

**Option 1: npm**
```bash
npm install -g @opencode/cli
```

**Option 2: PowerShell**
```powershell
iwr https://opencode.ai/install.ps1 -useb | iex
```

**Option 3: Scoop**
```bash
scoop install opencode
```

### Linux

**Option 1: npm**
```bash
npm install -g @opencode/cli
```

**Option 2: Binary Download**
```bash
curl -fsSL https://opencode.ai/install | bash
```

**Option 3: Debian/Ubuntu**
```bash
curl -fsSL https://opencode.ai/install.sh | sudo bash
```

## Configuration

OpenCode uses a `.opencode.json` configuration file in your project root to define providers, models, agents, and permissions.

### Setup

1. **Copy the config file** to your project root:
   ```bash
   cp opencode.json .opencode.json
   ```

2. **Verify the configuration:**
   ```bash
   opencode config validate
   ```

### Configuration Structure

The `.opencode.json` file supports the following sections:

- **`provider`** — Define one or more LLM providers (e.g., vLLM, OpenAI, Anthropic)
- **`model`** — Set the default model for all agents
- **`agent`** — Configure named agents with custom prompts, temperatures, and tool permissions
- **`permission`** — Control what actions the agent can perform (edit, bash, webfetch)
- **`instructions`** — Reference external instruction files (e.g., `AGENTS.md`)
- **`watcher`** — Configure file watching and ignore patterns
- **`disabled_providers`** — Explicitly disable certain providers

### Example: vLLM Provider

```json
{
  "provider": {
    "vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "vLLM — Qwen3.6-27B-INT4-AutoRound",
      "options": {
        "baseURL": "http://localhost:8888/v1"
      },
      "models": {
        "qwen3.6-27b-int4-autoround": {
          "name": "Qwen3.6 27B Dense INT4 AutoRound",
          "limit": { "context": 262144, "output": 32768 }
        }
      }
    }
  },
  "model": "vllm/qwen3.6-27b-int4-autoround"
}
```

### Agents

The config defines three built-in agents:

| Agent    | Purpose                          | Thinking | Permissions          |
|----------|----------------------------------|----------|----------------------|
| `build`  | Code edits, refactors, tool loops| Disabled | Full (with safeguards) |
| `general`| Conversational Q&A               | Disabled | Read-only            |
| `plan`   | Architecture & planning          | Enabled  | Read-only + safe bash |

### Usage

```bash
# Start with default agent
opencode

# Use a specific agent
opencode --agent build
opencode --agent plan

# Run in a specific directory
opencode --dir ./my-project

# Pass a prompt directly
opencode "Explain the architecture of this project"
```

## Security

The included configuration enforces strict permissions:

- **Bash commands** requiring confirmation for dangerous operations (`rm -rf`, `sudo`, `git push --force`)
- **External directory access** restricted to `~/projects/`, `~/code/`, `~/work/`
- **System paths** (`/etc/`, `/var/`, `/usr/`) and sensitive files (`~/.ssh/`, `~/.aws/`) are denied
- **Pipelines to shell** (`curl | sh`, `wget | bash`) are explicitly denied

## Schema

Configuration schema is available at: https://opencode.ai/config.json
