<div align="center">

# TCG-VoxCPM2

支持非语言声音标签的中英文语音合成

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b)](#引用)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/airguoqiang/TCG-VoxCPM2)
[![Demo](https://img.shields.io/badge/Demo-Audio%20samples-0d6b63)](https://coding-sharks.github.io/TCG-VoxCPM2-demo/)
[![License](https://img.shields.io/badge/License-Non--commercial-blue)](LICENSE)

[English](README.md) | **中文**

</div>

TCG-VoxCPM2 是论文 *Where More Than Which: Distilling Tag-Contrastive Guidance for Non-Verbal Vocalizations* 的模型。
在文本中写入 `[laugh]`、`[sigh]`、`[cough]` 等标签，模型会在标签位置发出对应的声音。模型基于
[NVVSpeech 挑战赛 Track 2 基线](https://huggingface.co/NVVSpeech-Challenge/NVVSpeech-Challenge-Track2-Baseline)（VoxCPM2，2B）微调，推理方式与 VoxCPM2 相同。试听样例见 [demo 页面](https://coding-sharks.github.io/TCG-VoxCPM2-demo/)。

## 发布计划

- [x] 推理代码
- [x] [模型权重](https://huggingface.co/airguoqiang/TCG-VoxCPM2)
- [ ] 训练代码

## 安装

```bash
git clone https://github.com/coding-sharks/TCG-VoxCPM2.git
cd TCG-VoxCPM2
pip install -r requirements.txt
```

建议使用 Python 3.10+ 和 CUDA GPU。首次运行时会自动从 Hugging Face 下载模型权重。

## 使用

```bash
# 单句合成
python infer.py --text "Sorry, [cough] it's the dust in here. Where were we?" --output cough.wav

# 用参考音频克隆音色：ref.wav 换成你自己准备的目标音色短录音
python infer.py --text "[sigh] 又下雨了，今天的野餐看来只能取消了。" --ref_audio ref.wav --output sigh.wav

# 批量合成，每行一个 {"id": ..., "text": ...}
python infer.py --input examples/examples.jsonl --out_dir outputs
```

全部参数见 `python infer.py --help`。在 Python 中调用：

```python
from voxcpm import VoxCPM
from nv_tags import to_model_text

model = VoxCPM.from_pretrained(hf_model_id="airguoqiang/TCG-VoxCPM2", load_denoiser=False, optimize=False)
text, _ = to_model_text("Wait, you actually wore that? [laugh] I wish I'd been there.")
wav = model.generate(text=text, cfg_value=2.0, inference_timesteps=10, max_len=600)  # 48 kHz
```

## 支持的标签

`[breath]` `[sniff]` `[laugh]` `[cry]` `[cough]` `[throat clearing]` `[sneeze]` `[sigh]` `[gasp]` `[snore]` `[yawn]`
`[hum]` `[moan]` `[hiss]` `[lipsmack]` `[burp]`

也可以直接使用这 16 个标签在模型词表中的中文标签，例如 `[笑声]`，对应关系见 `nv_tags.py`。

## 引用

论文即将发布在 arXiv，届时在这里补充 BibTeX。

## 许可

代码和模型权重采用 [NTUITIVE 许可](LICENSE)，仅限非商业用途。如需商业授权，请联系 guoqiang002@e.ntu.edu.sg。第三方组件见 [NOTICE](NOTICE)。

## 致谢

本工作基于 [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) 和
[NVVSpeech 挑战赛 Track 2 基线](https://huggingface.co/NVVSpeech-Challenge/NVVSpeech-Challenge-Track2-Baseline)。
