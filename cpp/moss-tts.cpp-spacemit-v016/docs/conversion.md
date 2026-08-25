# Converting MOSS-Audio-Tokenizer to GGUF

`scripts/convert_audio_tokenizer_to_gguf.py` converts the upstream
[`OpenMOSS-Team/MOSS-Audio-Tokenizer`](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer)
safetensors checkpoint into a single f32 GGUF that the C++ loader reads.

```bash
. .venv/bin/activate
python scripts/convert_audio_tokenizer_to_gguf.py \
    --model OpenMOSS-Team/MOSS-Audio-Tokenizer \
    --out models/moss-audio-tokenizer-f32.gguf [--strict]
```

`--model` accepts a HF repo id (downloaded via `huggingface_hub.snapshot_download`,
~7 GB) or a local directory containing `config.json` and the `*.safetensors`
shards. The output is always f32; quantization is a separate later step.
`--strict` makes the converter fail if any source key is not mapped to an output
tensor (otherwise it only warns).

## GGUF metadata keys

Read from `config.json` and written as KV metadata:

| key | type | source | value (this checkpoint) |
| --- | --- | --- | --- |
| `moss.at.sample_rate` | u32 | `sample_rate` | 24000 |
| `moss.at.downsample` | u32 | `downsample_rate` | 1920 |
| `moss.at.num_quantizers` | u32 | `quantizer_kwargs.num_quantizers` | 32 |
| `moss.at.codebook_size` | u32 | `quantizer_kwargs.codebook_size` | 1024 |
| `moss.at.codebook_dim` | u32 | `quantizer_kwargs.codebook_dim` | 8 |
| `moss.at.rvq_dim` | u32 | `quantizer_kwargs.rvq_dim` | 512 |
| `moss.at.context_seconds` | f32 | `causal_transformer_context_duration` | 10 |
| `moss.at.{enc,dec}.n_stages` | u32 | derived from `*_kwargs` | per-tower stage count |
| `moss.at.{enc,dec}.{kind,patch,in_dim,d_model,n_heads,n_layers,d_ff,out_dim}` | i32[] | derived from `*_kwargs` | per-stage arrays |
| `moss.at.encoder_kwargs` | str | `encoder_kwargs` (JSON) | provenance only |
| `moss.at.decoder_kwargs` | str | `decoder_kwargs` (JSON) | provenance only |

The C++ loader builds the encoder/decoder towers **from the explicit
`moss.at.{enc,dec}.*` int32 stage arrays** — `n_stages` plus the parallel
per-stage arrays `kind`, `patch`, `in_dim`, `d_model`, `n_heads`, `n_layers`,
`d_ff`, `out_dim` (read via `get_i32_array`). `encoder_kwargs` /
`decoder_kwargs` are the JSON-serialized *lists* of per-stage configs (the
towers alternate `PatchedPretransform` and `Transformer` modules) emitted for
provenance / debugging only — the loader does **not** parse them.

## WNConv fusion rule

The encoder/decoder block projections and all quantizer projections are stored
as weight-normalized 1x1 convs (`torch.nn.utils.parametrizations.weight_norm`,
`dim=0`):

```
<name>.parametrizations.weight.original0   # g, shape (out, 1, 1)
<name>.parametrizations.weight.original1   # v, shape (out, in, 1)
<name>.bias                                # passes through unchanged
```

torch reconstructs the weight as `W = g * v / ||v||`, where the norm is taken
over every dimension except dim 0 (the output channel). For a 1x1 conv the
kernel dim is 1, so after squeezing the trailing kernel axis the norm reduces to
a per-output-row sum over the input dim:

```
v2d  = v.reshape(out, in)                  # squeeze kernel dim
norm = sqrt( sum_in v2d**2 )  -> (out, 1)  # per output channel
W    = g.reshape(out, 1) * v2d / norm      # dense (out, in)
```

The converter emits the fused dense `<name>.weight` of shape `(out, in)` and
keeps the stored `<name>.bias`. Verified bit-exact against torch's
`weight_norm(dim=0)` reconstruction.

## Identity-projection drop rule

Block projections are only present in the checkpoint when the dims actually
differ. When `in == d_model == out` the projection is the identity and is
**absent**; the converter does not synthesize it, and the C++ loader treats a
missing `input_proj`/`output_proj` as identity. In this checkpoint the present
block projections are exactly:

- `encoder.1.input_proj.weight`
- `encoder.{1,3,5,7}.output_proj.weight`
- `decoder.{0,2,4,6}.input_proj.weight`
- `decoder.6.output_proj.weight`

## Per-stage block configs

The tower is a sequence of patchify (`PatchedPretransform`) and causal
`Transformer` stages. Transformer stages live at the indices the C++ loader
iterates (`encoder` blocks 1,3,5,7 and `decoder` blocks 0,2,4,6).

- **Encoder Transformers** (small): `d_model=768`, `num_heads=12`,
  `num_layers=12`, `dim_feedforward=3072`, `layer_scale=0.01`, RoPE
  (`max_period=10000`), `norm=layer_norm`, causal.
- **Decoder Transformers** (large): `d_model=1280`, `num_heads=20`,
  `num_layers=32`, `dim_feedforward=5120`, otherwise as above.
- **Encoder patch ratios**: `240, 2, 2, 2` (product = 1920 = downsample).
- **Decoder patch ratios**: mirror — `2, 2, 2, 240`.
- **Quantizer**: 32 quantizers, `codebook_size=1024`, `codebook_dim=8`,
  `rvq_dim=512`, `input_dim=output_dim=768`.

## Tensor-name inventory (what the C++ loader expects)

Per Transformer block `{B}` (encoder 1/3/5/7, decoder 0/2/4/6), per layer `L`:

```
{B}.transformer.layers.{L}.norm1.weight   .norm1.bias
{B}.transformer.layers.{L}.norm2.weight   .norm2.bias
{B}.transformer.layers.{L}.self_attn.in_projs.0.weight     # fused QKV
{B}.transformer.layers.{L}.self_attn.out_projs.0.weight
{B}.transformer.layers.{L}.linear1.weight
{B}.transformer.layers.{L}.linear2.weight
{B}.transformer.layers.{L}.layer_scale_1.scale
{B}.transformer.layers.{L}.layer_scale_2.scale
```

Block projections (only the non-identity ones listed above, fused from WNConv).

Quantizer:

```
quantizer.input_proj.weight   .input_proj.bias        # WNConv-fused
quantizer.output_proj.weight  .output_proj.bias       # WNConv-fused
quantizer.quantizers.{i}.in_proj.weight   .in_proj.bias    # WNConv-fused
quantizer.quantizers.{i}.out_proj.weight  .out_proj.bias   # WNConv-fused
quantizer.quantizers.{i}.codebook.weight                   # plain, for i in 0..31
```

## Upstream key naming (verified)

The upstream `model.safetensors.index.json` `weight_map` keys were inspected and
match these names **verbatim** — the encoder/decoder/quantizer prefixes are
top-level with **no extra `model.` prefix**, so the converter passes plain
`.weight`/`.bias`/`.scale`/`codebook.weight` tensors through unchanged and only
transforms the `.parametrizations.weight.original{0,1}` pairs. Against this
checkpoint the converter reports **0 unmapped keys**.

If a future checkpoint revision nests these under a different prefix, re-verify
with:

```bash
python -c "from huggingface_hub import hf_hub_download; import json; \
print(list(json.load(open(hf_hub_download( \
'OpenMOSS-Team/MOSS-Audio-Tokenizer','model.safetensors.index.json' \
)))['weight_map'].keys())[:40])"
```

and add a `strip_prefix` step to the pass-through in
`convert_audio_tokenizer_to_gguf.py` so emitted names still match the loader.

## Nano "Cat" codec (48 kHz stereo) — `convert_audio_tokenizer_nano_to_gguf.py`

The V4 Nano codec
([`OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano`](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano))
is the SAME architectural family as the 24 kHz codec (PatchedPretransform +
causal ProjectedTransformer stages with LayerScale + RoPE + ResidualLFQ) but is
48 kHz, **stereo** (`number_channels=2`, `enable_channel_interleave=true` → the
2 channels are folded into the codec sequence, factor 2), multi-stage,
**RVQ-16**, `code_dim=768`, `downsample_rate=3840`. It is a **separate model
file** from the 24 kHz codec.

```bash
python scripts/convert_audio_tokenizer_nano_to_gguf.py \
    --model OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano \
    --out models/moss-audio-tokenizer-nano-f32.gguf --strict
```

Differences from the 24 kHz converter:

- **Stereo metadata**: emits `moss.at.channels=2` and
  `moss.at.channel_interleave=1` (plus `moss.at.code_dim`). The C++
  `AudioTokenizer` reads these; when `channels==2 && channel_interleave`, it
  applies the channel-interleave factor (2) to the encode padding / latent frame
  count and to the per-transformer sliding-window context (the codec runs at
  `sample_rate*2`). The decode output is the interleaved-stereo waveform of
  length `n_frames * downsample * channels`. `channels==1` is byte-identical to
  the 24 kHz path.
- **Multi-stage stage tables**: `moss.at.{enc,dec}.*` are built generically from
  `config.encoder_kwargs` / `decoder_kwargs` (9 stages each: interleaved
  PatchedPretransform reshapes and ProjectedTransformer blocks). The encoder
  patch-size product equals `downsample_rate * number_channels` (7680), which the
  decoder inverts — this invariant is asserted by the upstream modeling.
- **Weight-norm fusion**: only the **quantizer** projections are WNConv1d
  (`quantizer.input_proj` / `output_proj` and each
  `quantizer.quantizers.{i}.in_proj` / `out_proj`, since `rvq_dim=512 !=
  output_dim=768` and `input_dim=768 != codebook_dim=8`); these are fused via
  `fuse_wn` exactly as the 24 kHz converter. The ProjectedTransformer
  `input_proj` / `output_proj` are **plain `nn.Linear(bias=False)`** and pass
  through verbatim (no fusion).

