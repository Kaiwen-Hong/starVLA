# Flash Attention Configuration

The `attn_implementation` parameter is now configurable via YAML config instead of being hardcoded.

## Configuration

In your training YAML config:

```yaml
framework:
  qwenvl:
    attn_implementation: flash_attention_2  # or flash_attention_3
```

## Environment-Specific Setup

| GPU | Attention Implementation | CUDA Requirement |
|-----|--------------------------|------------------|
| RTX 4090 (Ada Lovelace) | `flash_attention_2` | CUDA 11.x+ |
| H100/H200 (Hopper) | `flash_attention_3` | CUDA 12.3+ (12.8 recommended) |

### Local Machine (4090)

```yaml
framework:
  qwenvl:
    attn_implementation: flash_attention_2
```

### Cluster (H200)

```yaml
framework:
  qwenvl:
    attn_implementation: flash_attention_3
```

## Installation

### Flash Attention 2

```bash
pip install flash-attn --no-build-isolation
```

### Flash Attention 3 (Hopper GPUs only)

```bash
pip install "git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=hopper"
```

Note: FA3 requires Hopper GPU (H100/H200) with compute capability 9.0+.

## Files Modified

- `starVLA/model/modules/vlm/QWen2_5.py` - reads `attn_implementation` from config
- `starVLA/model/modules/vlm/QWen3.py` - reads `attn_implementation` from config

Default is `flash_attention_2` if not specified.
