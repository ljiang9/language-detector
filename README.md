# language-detector

基于**字符 n-gram（trigram）轮廓**的轻量语言检测器。零第三方依赖，
内置中文 / 英文 / 日语 / 韩语 4 种语言的小样本轮廓，输出最可能语言与各语言得分。

## 功能简介

- 对待测文本生成字符 trigram；
- 与各语言轮廓比对（拉普拉斯平滑对数概率求和打分）；
- `detect(text)` 返回 `{language, language_name, scores}`；
- 可用自带样本，也可在构造时传入自定义语言样本。

## 快速开始

环境：Python 3.10+，零依赖。

```bash
python3 language_detector.py "自然语言处理非常有趣"
python3 language_detector.py "Natural language processing is fun"
python3 language_detector.py "これは日本語のテキストです"
```

作为库：

```python
from language_detector import LanguageDetector
ld = LanguageDetector()
print(ld.detect("这是一段测试文本")["language_name"])
```

## 使用示例（真实命令）

```bash
$ python3 language_detector.py "これは日本語のサンプルです"
检测语言：日语（ja）
  日语   -76.667
  中文   -83.246
```

## 无 API key 如何运行

纯本地统计，**不需要任何 API key**，不联网。

## 目录结构

```
language-detector/
├── language_detector.py          # 检测器 + 内置样本 + CLI
├── tests/
│   └── test_language_detector.py # unittest
├── README.md
├── LICENSE
└── .gitignore
```

## 运行测试

```bash
python3 -m unittest discover -s tests -v
```

## 许可证

[MIT](./LICENSE) © 2026 ljiang9
