#!/usr/bin/env python3
"""TCG-VoxCPM2 inference: text with non-verbal vocalization tags -> 48 kHz speech.

Single sentence, the model's own voice (the setting of the paper's results):
    python infer.py --text "That's hilarious! [laugh] I can't believe he actually did it." --output laugh.wav

In a given voice (VoxCPM2 reference-voice cloning; the reference clip is isolated from the text):
    python infer.py --text "[sigh] 好吧，那我们明天再试一次。" --ref_audio voice.wav --output sigh.wav

Batch: one JSON object per line with "id" and "text" (optional "ref_audio"); writes <out_dir>/<id>.wav and seeds each
line with seed + crc32(id) % 1e6, as in the paper:
    python infer.py --input examples/examples.jsonl --out_dir outputs
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import soundfile as sf
import torch

from nv_tags import TAG_TOKENS, to_model_text

DEFAULT_MODEL = "airguoqiang/TCG-VoxCPM2"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id or local model directory")
    ap.add_argument("--text", help="text to synthesize, with tags such as [laugh] or [笑声]")
    ap.add_argument("--output", default="output.wav")
    ap.add_argument("--input", help="JSONL file for batch synthesis")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--ref_audio", help="reference clip for the voice (reference-voice cloning)")
    ap.add_argument("--prompt_audio", help="prompt clip to continue from (needs --prompt_text)")
    ap.add_argument("--prompt_text", help="transcript of --prompt_audio")
    ap.add_argument("--cfg", type=float, default=2.0, help="classifier-free guidance scale (paper: 2.0)")
    ap.add_argument("--steps", type=int, default=10, help="flow-matching steps per patch (paper: 10)")
    ap.add_argument("--max_len", type=int, default=600, help="maximum generated patches (paper: 600)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cuda, cpu, ... (default: automatic)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (faster after warm-up)")
    ap.add_argument("--list_tags", action="store_true", help="print the supported tags and exit")
    a = ap.parse_args()
    if a.list_tags:
        return a
    if bool(a.text) == bool(a.input):
        ap.error("give exactly one of --text or --input")
    if bool(a.prompt_audio) != bool(a.prompt_text):
        ap.error("--prompt_audio and --prompt_text go together")
    return a


def synthesize(model, text, seed, ref_audio=None, prompt_audio=None, prompt_text=None, a=None):
    model_text, unknown = to_model_text(text)
    if unknown:
        print(f"warning: {unknown} are not among the 16 supported tags (see --list_tags)", file=sys.stderr)
    torch.manual_seed(seed)
    return model.generate(text=model_text, reference_wav_path=ref_audio, prompt_wav_path=prompt_audio,
                          prompt_text=prompt_text, cfg_value=a.cfg, inference_timesteps=a.steps, max_len=a.max_len,
                          normalize=False, denoise=False)


def main():
    a = parse_args()
    if a.list_tags:
        for name, tok in TAG_TOKENS.items():
            print(f"[{name}]  ->  {tok}")
        return
    from voxcpm import VoxCPM

    model = VoxCPM.from_pretrained(hf_model_id=a.model, load_denoiser=False, optimize=a.compile, device=a.device)
    sr = model.tts_model.sample_rate
    if a.text:
        wav = synthesize(model, a.text, a.seed, a.ref_audio, a.prompt_audio, a.prompt_text, a)
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        sf.write(a.output, wav, sr)
        print(f"{a.output}: {len(wav) / sr:.2f} s")
        return
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(a.input, encoding="utf-8") if l.strip()]
    for r in rows:
        seed = a.seed + zlib.crc32(str(r["id"]).encode()) % 1_000_000
        wav = synthesize(model, r["text"], seed, r.get("ref_audio") or a.ref_audio, a.prompt_audio, a.prompt_text, a)
        sf.write(out / f"{r['id']}.wav", wav, sr)
        print(f"{r['id']}: {len(wav) / sr:.2f} s")


if __name__ == "__main__":
    main()
