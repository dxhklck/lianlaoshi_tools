import random


class RandomPromptNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {                
                "multiline_text": ("STRING", {"default": "", "multiline": True, "placeholder": "每行一个提示词"}),
            },
            "optional": {
                "dedup": ("BOOLEAN", {"default": True, "tooltip": "对候选提示词去重"}),
                "source": (["auto", "list", "multiline"], {"default": "auto", "tooltip": "指定候选来源"}),
                "select_index": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1, "step": 1, "display": "number", "tooltip": "指定索引（-1 为随机）"}),
                "strings_list": ("*", {"tooltip": "列表输入（避免列表展开），优先作为候选来源"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1, "step": 1, "display": "number", "tooltip": "随机种子（-1 为系统随机）"}),
            },
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "pick"
    CATEGORY = "lianlaoshi/prompt"
    DESCRIPTION = "从列表或多行文本中抽取一条提示词（支持 seed、指定来源与索引；只去空行，保留空格；可选去重）"

    def pick(self, multiline_text, dedup=True, source="auto", select_index=-1, strings_list=None, seed=-1):
        s_list = []
        if strings_list is not None and isinstance(strings_list, list):
            for item in strings_list:
                if isinstance(item, str) and item.strip() != "":
                    s_list.append(item)
                elif isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, str) and sub.strip() != "":
                            s_list.append(sub)
        # 仅去空行，不改动空格
        m_lines = []
        if isinstance(multiline_text, str) and multiline_text:
            raw_lines = multiline_text.splitlines()
            m_lines = [line for line in raw_lines if line.strip() != ""]
        # 选择来源
        if source == "list":
            active = s_list
            supplement = m_lines
        elif source == "multiline":
            active = m_lines
            supplement = s_list
        else:  # auto
            if s_list:
                active = s_list
                supplement = m_lines
            else:
                active = m_lines
                supplement = s_list
        # 指定索引优先
        if isinstance(select_index, int) and select_index >= 0:
            if not active:
                return ("",)
            idx = max(0, min(select_index, len(active) - 1))
            return (active[idx],)
        # 随机模式：在选定来源基础上附加补充来源
        candidates = list(active)
        if supplement:
            candidates.extend(supplement)
        if dedup and candidates:
            seen = set(); unique = []
            for s in candidates:
                if s not in seen:
                    seen.add(s); unique.append(s)
            candidates = unique
        if not candidates:
            return ("",)
        rng = random.Random(seed) if isinstance(seed, int) and seed is not None and seed >= 0 else random.SystemRandom()
        return (rng.choice(candidates),)


NODE_CLASS_MAPPINGS = {
    "RandomPrompt": RandomPromptNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomPrompt": "lian 随机提示词抽取",
}