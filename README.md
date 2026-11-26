<div align="center">

# TCG-VoxCPM2

Text-to-speech with non-verbal vocalization tags, in Mandarin and English

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b)](#citation)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/airguoqiang/TCG-VoxCPM2)
[![Demo](https://img.shields.io/badge/Demo-Audio%20samples-0d6b63)](https://coding-sharks.github.io/TCG-VoxCPM2-demo/)
[![License](https://img.shields.io/badge/License-Non--commercial-blue)](LICENSE)

**English** | [中文](README_zh.md)

</div>

TCG-VoxCPM2 is the model of the paper *Where More Than Which: Distilling Tag-Contrastive Guidance for Non-Verbal
Vocalizations*. Write tags such as `[laugh]`, `[sigh]` or `[cough]` in the text, and the model voices them where they
stand. It is fine-tuned from the
[NVVSpeech Challenge Track-2 baseline](https://huggingface.co/NVVSpeech-Challenge/NVVSpeech-Challenge-Track2-Baseline)
(VoxCPM2, 2B) and decodes like any VoxCPM2 model. Audio samples are on the [demo page](https://coding-sharks.github.io/TCG-VoxCPM2-demo/).

## Release

- [x] Inference code
- [x] [Model weights](https://huggingface.co/airguoqiang/TCG-VoxCPM2)
- [ ] Training code

## Installation

```bash
git clone https://github.com/coding-sharks/TCG-VoxCPM2.git
cd TCG-VoxCPM2
pip install -r requirements.txt
```

Python 3.10+ and a CUDA GPU are recommended. The model weights are downloaded from Hugging Face on first use.

## Usage

```bash
# a single sentence
python infer.py --text "Sorry, [cough] it's the dust in here. Where were we?" --output cough.wav

# voice cloning: ref.wav is a short clip of the target voice that you provide
python infer.py --text "[sigh] 又下雨了，今天的野餐看来只能取消了。" --ref_audio ref.wav --output sigh.wav

# batch synthesis, one {"id": ..., "text": ...} per line
python infer.py --input examples/examples.jsonl --out_dir outputs
```

See `python infer.py --help` for all options. From Python:

```python
from voxcpm import VoxCPM
from nv_tags import to_model_text

model = VoxCPM.from_pretrained(hf_model_id="airguoqiang/TCG-VoxCPM2", load_denoiser=False, optimize=False)
text, _ = to_model_text("Wait, you actually wore that? [laugh] I wish I'd been there.")
wav = model.generate(text=text, cfg_value=2.0, inference_timesteps=10, max_len=600)  # 48 kHz
```

## Supported tags

`[breath]` `[sniff]` `[laugh]` `[cry]` `[cough]` `[throat clearing]` `[sneeze]` `[sigh]` `[gasp]` `[snore]` `[yawn]`
`[hum]` `[moan]` `[hiss]` `[lipsmack]` `[burp]`

The model's own Chinese tokens for these 16 tags, such as `[笑声]`, also work; `nv_tags.py` gives the mapping.

## Citation

The paper will be available on arXiv soon; the BibTeX entry will be added here.

## License

The code and model weights are released under the [NTUITIVE license](LICENSE) for non-commercial use only. For a
commercial license, contact guoqiang002@e.ntu.edu.sg. Third-party components are listed in [NOTICE](NOTICE).

## Acknowledgements

This work builds on [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) and the
[NVVSpeech Challenge Track-2 baseline](https://huggingface.co/NVVSpeech-Challenge/NVVSpeech-Challenge-Track2-Baseline).
