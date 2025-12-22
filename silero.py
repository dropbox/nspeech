# export_silero_vad_split.py
import json
import torch
from safetensors.torch import save_file

def pick_attr(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default

@torch.no_grad()
def export_one(core, prefix: str, out_st: str, out_cfg: str):
    """
    prefix is '_model.' or '_model_8k.'
    """
    sd = core.state_dict()

    # --- Gather tensors for this variant ---
    def g(k): return sd[prefix + k]

    out = {}

    # STFT basis (conv1d filters)
    out["stft.forward_basis_buffer"] = g("stft.forward_basis_buffer")

    # Encoder convs
    for i in range(4):
        out[f"enc.{i}.weight"] = g(f"encoder.{i}.reparam_conv.weight")
        out[f"enc.{i}.bias"]   = g(f"encoder.{i}.reparam_conv.bias")

    # RNN (LSTM cell style: weight_ih/weight_hh/bias_ih/bias_hh)
    out["rnn.weight_ih"] = g("decoder.rnn.weight_ih")
    out["rnn.weight_hh"] = g("decoder.rnn.weight_hh")
    out["rnn.bias_ih"]   = g("decoder.rnn.bias_ih")
    out["rnn.bias_hh"]   = g("decoder.rnn.bias_hh")

    # Head conv (usually 1x1)
    out["head.weight"] = g("decoder.decoder.2.weight")
    out["head.bias"]   = g("decoder.decoder.2.bias")

    save_file(out, out_st)
    print("wrote", out_st)

    # --- Export config (best-effort: attribute names differ across versions) ---
    # We introspect the underlying module for STFT params.
    m = core
    # For the hub model, core is likely already the wrapper with _model/_model_8k attributes:
    # So we select the submodule by prefix.
    sub = getattr(m, prefix[:-1])  # "_model" or "_model_8k"
    stft = sub.stft

    cfg = {
        "sample_rate": 16000 if "8k" not in prefix else 8000,
        "hop_length": int(pick_attr(stft, ["hop_length", "hop"], default=128)),
        "win_length": int(pick_attr(stft, ["win_length", "win"], default=256)),
        "n_fft": int(pick_attr(stft, ["n_fft", "filter_length", "fft_size"], default=256)),
        # STFT configuration: The official model uses ReflectionPad1d([0, 64])
        # No left padding, 64 samples of reflection padding on right
        "stft_right_padding": 64,
        # Encoder configuration: All encoder layers use padding=1 with kernel=3
        "encoder_padding": 1,
        # Context buffer: 64 samples from previous chunk are prepended
        "context_size": 64,
        # Chunk size: Model processes 512-sample chunks (32ms at 16kHz)
        "chunk_size": 512,
    }
    with open(out_cfg, "w") as f:
        json.dump(cfg, f, indent=2)
    print("wrote", out_cfg, cfg)

def main():
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True
    )
    model.eval()

    # Many hub models wrap the real module in attributes; yours has keys with "_model" already,
    # so the returned model likely has attributes: model._model and model._model_8k.
    core = model

    export_one(core, "_model.", "vad16.safetensors", "vad16.config.json")
    export_one(core, "_model_8k.", "vad8.safetensors", "vad8.config.json")

if __name__ == "__main__":
    main()

