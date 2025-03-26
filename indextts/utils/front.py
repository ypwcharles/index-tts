# -*- coding: utf-8 -*-
import traceback
import os
import sys
import re
import re




class TextNormalizer:
    def __init__(self):
        # self.normalizer = Normalizer(cache_dir="textprocessing/tn")
        self.zh_normalizer = None
        self.en_normalizer = None
        self.char_rep_map = {
            "：": ",",
            "；": ",",
            ";": ",",
            "，": ",",
            "。": ".",
            "！": "!",
            "？": "?",
            "\n": ".",
            "·": ",",
            "、": ",",
            "...": "…",
            "……": "…",
            "$": ".",
            "“": "'",
            "”": "'",
            '"': "'",
            "‘": "'",
            "’": "'",
            "（": "'",
            "）": "'",
            "(": "'",
            ")": "'",
            "《": "'",
            "》": "'",
            "【": "'",
            "】": "'",
            "[": "'",
            "]": "'",
            "—": "-",
            "～": "-",
            "~": "-",
            "「": "'",
            "」": "'",
            ":": ",",
        }

    def match_email(self, email):
        # 正则表达式匹配邮箱格式：数字英文@数字英文.英文
        pattern = r'^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$'
        return re.match(pattern, email) is not None

    def use_chinese(self, s):
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', s))
        has_digit = bool(re.search(r'\d', s))
        has_alpha = bool(re.search(r'[a-zA-Z]', s))
        is_email = self.match_email(s)
        if has_chinese or not has_alpha or is_email:
            return True
        else:
            return False

    def load(self):
        # print(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        # sys.path.append(model_dir)
        import platform
        if platform.machine() == "aarch64":
            from wetext import Normalizer
            self.zh_normalizer = Normalizer(remove_erhua=False,lang="zh",operator="tn")
            self.en_normalizer = Normalizer(lang="en",operator="tn")
        else:
            from tn.chinese.normalizer import Normalizer as NormalizerZh
            from tn.english.normalizer import Normalizer as NormalizerEn
            self.zh_normalizer = NormalizerZh(remove_interjections=False, remove_erhua=False)
            self.en_normalizer = NormalizerEn()


    def infer(self, text):
        pattern = re.compile("|".join(re.escape(p) for p in self.char_rep_map.keys()))
        replaced_text = pattern.sub(lambda x: self.char_rep_map[x.group()], text)
        if not self.zh_normalizer or not self.en_normalizer:
            print("Error, text normalizer is not initialized !!!")
            return ""
        try:
            normalizer = self.zh_normalizer if self.use_chinese(replaced_text) else self.en_normalizer
            result = normalizer.normalize(replaced_text)
        except Exception:
            result = ""
            print(traceback.format_exc())
        return result


if __name__ == '__main__':
    # 测试程序
    text_normalizer = TextNormalizer()
    print(text_normalizer.infer("2.5平方电线"))