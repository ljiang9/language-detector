import unittest

from language_detector import LanguageDetector, trigrams, SAMPLES


class TestTrigrams(unittest.TestCase):
    def test_trigram_count(self):
        t = trigrams("abc")
        self.assertTrue(any("abc" in x for x in t))


class TestDetector(unittest.TestCase):
    def setUp(self):
        self.ld = LanguageDetector()

    def test_detect_chinese(self):
        r = self.ld.detect("自然语言处理非常有趣，计算机理解人类语言。")
        self.assertEqual(r["language"], "zh")

    def test_detect_english(self):
        r = self.ld.detect("Natural language processing helps computers understand text.")
        self.assertEqual(r["language"], "en")

    def test_detect_japanese(self):
        r = self.ld.detect("これは日本語のテキストです。自然言語処理の研究をしています。")
        self.assertEqual(r["language"], "ja")

    def test_detect_korean(self):
        r = self.ld.detect("이것은 한국어 텍스트입니다. 자연어 처리를 연구합니다.")
        self.assertEqual(r["language"], "ko")

    def test_scores_have_all_langs(self):
        r = self.ld.detect("test")
        self.assertEqual(set(r["scores"]), {"zh", "en", "ja", "ko"})

    def test_self_consistency(self):
        for lang, text in SAMPLES.items():
            self.assertEqual(self.ld.detect(text)["language"], lang)


if __name__ == "__main__":
    unittest.main()
