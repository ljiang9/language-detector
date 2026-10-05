#!/usr/bin/env python3
"""language_detector —— 基于字符 n-gram 轮廓的轻量语言检测。

做法：
  - 每种语言用一段样本文本统计「字符 trigram」频率轮廓；
  - 检测时对待测文本同样生成 trigram，按各语言轮廓的对数概率求和打分；
  - 输出最可能的语言与全部语言得分。
内置 4 种语言小样本：中文 zh、英文 en、日语 ja、韩语 ko。零第三方依赖。
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter

SAMPLES = {
    "zh": (
        "这是一段用于语言检测的中文示例文本。自然语言处理是计算机科学的一个分支，"
        "它研究如何让计算机理解和生成人类语言。中文以汉字书写，没有空格分词，"
        "句子之间用标点符号分隔。机器学习和深度学习在这个领域应用非常广泛。"
    ),
    "en": (
        "This is an English sample text used for language detection. Natural language "
        "processing is a branch of computer science. It studies how computers understand "
        "and generate human language. Machine learning and deep learning are widely used "
        "across this field of study."
    ),
    "ja": (
        "これは言語検出のための日本語のサンプルテキストです。自然言語処理はコンピュータ"
        "サイエンスの一分野であり、コンピュータが人間の言語を理解し生成する方法を研究します。"
        "日本語はひらがな、カタカナ、漢字を混ぜて書きます。"
    ),
    "ko": (
        "이것은 언어 감지를 위한 한국어 샘플 텍스트입니다. 자연어 처리는 컴퓨터 과학의 한 "
        "분야로 컴퓨터가 인간의 언어를 이해하고 생성하는 방법을 연구합니다. 한국어는 한글로 "
        "쓰며 음절 단위로 글자가 구성됩니다."
    ),
}

_LANG_NAME = {"zh": "中文", "en": "英文", "ja": "日语", "ko": "韩语"}
_NON_ALNUM = re.compile(r"[^a-z0-9\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af ]+")


def _normalize(text: str) -> str:
    t = text.lower()
    t = _NON_ALNUM.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def trigrams(text: str) -> list[str]:
    t = _normalize(text)
    padded = " " + t + " "
    return [padded[i:i + 3] for i in range(len(padded) - 2)]


class LanguageDetector:
    def __init__(self, samples: dict[str, str] | None = None, ngram: int = 3):
        self.ngram = ngram
        samples = samples or SAMPLES
        self.profiles: dict[str, dict[str, float]] = {}
        self.totals: dict[str, int] = {}
        self.vocab: set[str] = set()
        for lang, text in samples.items():
            self._build(lang, text)

    def _build(self, lang: str, text: str) -> None:
        cnt = Counter(trigrams(text))
        total = sum(cnt.values())
        self.profiles[lang] = dict(cnt)
        self.totals[lang] = total
        self.vocab.update(cnt.keys())

    def _score(self, text: str, lang: str) -> float:
        cnt = self.profiles[lang]
        total = self.totals[lang]
        n_vocab = len(self.vocab)
        score = 0.0
        for tri in trigrams(text):
            c = cnt.get(tri, 0)
            score += math.log((c + 1) / (total + n_vocab))
        return score

    def detect_scores(self, text: str) -> dict[str, float]:
        return {lang: self._score(text, lang) for lang in self.profiles}

    def detect(self, text: str) -> dict:
        scores = self.detect_scores(text)
        best = max(scores, key=scores.get)
        return {
            "language": best,
            "language_name": _LANG_NAME.get(best, best),
            "scores": {k: round(v, 3) for k, v in scores.items()},
        }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="字符 n-gram 语言检测")
    p.add_argument("text", nargs="?", help="待测文本；不传则读标准输入")
    args = p.parse_args(argv)
    text = args.text or sys.stdin.read()
    ld = LanguageDetector()
    r = ld.detect(text)
    print(f"检测语言：{r['language_name']}（{r['language']}）")
    for k, v in sorted(r["scores"].items(), key=lambda kv: -kv[1]):
        print(f"  {_LANG_NAME.get(k, k)}\t{v:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
