"""Non-verbal vocalization (NV) tags for TCG-VoxCPM2.

The model's vocabulary holds the 16 NV classes of the NVVSpeech Challenge Track 2 as Chinese tag tokens. English tags
are mapped onto them before tokenization, for Chinese and English text alike (the challenge's own convention). The
mapping is the one used for every number in the paper; note that gasp maps to [惊讶] and burp to [不满]."""
import re

TAG_TOKENS = {
    "breath": "[呼吸]", "sniff": "[吸鼻]", "laugh": "[笑声]", "cry": "[哭声]", "cough": "[咳嗽]",
    "throat clearing": "[清嗓]", "sneeze": "[打喷嚏]", "sigh": "[叹气]", "gasp": "[惊讶]", "snore": "[打鼾]",
    "yawn": "[哈欠]", "hum": "[哼唱]", "moan": "[呻吟]", "hiss": "[嘶声]", "lipsmack": "[咂嘴]", "burp": "[不满]",
}
_ALIASES = {"throat_clearing": "throat clearing", "throat-clearing": "throat clearing", "lip smack": "lipsmack",
            "lip-smack": "lipsmack", "lip_smack": "lipsmack"}
_TOKENS = set(TAG_TOKENS.values())
_BRACKET = re.compile(r"\[([^\[\]]{1,32})\]")


def to_model_text(text: str):
    """Map English tags to the model's tag tokens. Returns (model_text, unknown_tags); an unknown tag is left as is
    and would be read out as words."""
    unknown = []

    def sub(m):
        tok = f"[{m.group(1).strip()}]"
        if tok in _TOKENS:
            return tok
        name = " ".join(m.group(1).strip().lower().split())
        name = _ALIASES.get(name, name)
        if name in TAG_TOKENS:
            return TAG_TOKENS[name]
        unknown.append(m.group(0))
        return m.group(0)

    return _BRACKET.sub(sub, text), unknown
