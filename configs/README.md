# Configs

This directory holds all configuration files for the project.

## Files

| File | Description |
|---|---|
| `config.example.yaml` | Template configuration — copy to `config.yaml` before use |
| `config.yaml` | Active project configuration (**gitignored**) |
| `logging.yaml` | Python logging configuration (handlers, formatters, levels) |

## Usage

```bash
cp configs/config.example.yaml configs/config.yaml
# Edit configs/config.yaml with your local settings
```

> **Note:** Never commit `config.yaml` — it may contain environment-specific paths.
