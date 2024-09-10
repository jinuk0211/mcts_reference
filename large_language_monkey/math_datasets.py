import unittest
from datasets import load_dataset
from tqdm import tqdm
from llmonk.evaluate.math_datasets import is_correct as evaluate_correctness
import random


class TestCheckCompletions(unittest.TestCase):
    def setUp(self):
        datasets = ["GSM8K", "MATH"]
        models = ["Llama-3-8B-Instruct", "Llama-3-70B-Instruct"]
        self.gt_answers = {dataset: [] for dataset in datasets}
        self.samples = {dataset: [] for dataset in datasets}
        self.is_corrects = {dataset: [] for dataset in datasets}
        for dataset in datasets:
            for model in models:
                sample_dataset = load_dataset("ScalingIntelligence/monkey_business", f"{dataset}_{model}", download_mode="force_redownload")
                for problem in sample_dataset["test"]:
                    gt_answer = problem["gt_answer"]
                    for sample, is_correct in zip(problem["samples"], problem["is_corrects"]):
                        self.gt_answers[dataset].append(gt_answer)
                        self.samples[dataset].append(sample)
                        self.is_corrects[dataset].append(is_correct)
            
            # take 1000 randomly for faster eval
            random.seed(0)
            indices = random.sample(range(len(self.gt_answers[dataset])), 1000)
            self.gt_answers[dataset] = [self.gt_answers[dataset][i] for i in indices]
            self.samples[dataset] = [self.samples[dataset][i] for i in indices]
            self.is_corrects[dataset] = [self.is_corrects[dataset][i] for i in indices]
        
    def test_check_completions(self):
        for dataset in self.gt_answers.keys():
            gt_answers = self.gt_answers[dataset]
            samples = self.samples[dataset]
            is_corrects = self.is_corrects[dataset]
            for gt_answer,sample,is_correct in tqdm(zip(gt_answers,samples,is_corrects)):
                self.assertEqual(evaluate_correctness(sample, gt_answer, dataset.lower()), is_correct)


if __name__ == '__main__':
    unittest.main()

import unittest
from llmonk.evaluate.code_contests_utils.compare_results import (
    outputs_match,
    _split_and_lowercase,
    _values_match,
    _split_by_any_char,
)


class TestOutputsMatch(unittest.TestCase):
    def test_matching_strings(self):
        self.assertTrue(outputs_match("abc def", "abc def"))

    def test_differing_strings(self):
        self.assertFalse(outputs_match("abc deg", "abc def"))

    def test_differing_strings_prefix(self):
        self.assertFalse(outputs_match("abc def", "abc def 123"))

    def test_case_ignored(self):
        self.assertTrue(outputs_match("abc DEF", "abc def"))

    def test_close_floats_accepted(self):
        self.assertTrue(outputs_match("abc 123", "abc 123.000001"))
        self.assertTrue(outputs_match("abc 123.0", "abc 123.000001"))
        self.assertTrue(outputs_match("abc 123.000001", "abc 123.0"))

    def test_non_close_floats_not_accepted(self):
        self.assertFalse(outputs_match("abc 123", "abc 123.1"))
        self.assertFalse(outputs_match("abc 123.1", "abc 123"))

    def test_different_delimiters(self):
        self.assertTrue(outputs_match("abc\t\ndef\nGHI", "abc def ghi"))

    def test_extra_spaces(self):
        self.assertTrue(outputs_match("  abc   def  ", "abc def"))

    def test_mixed_types(self):
        self.assertTrue(outputs_match("abc 123 3.14", "ABC 123 3.14"))

    def test_empty_strings(self):
        self.assertTrue(outputs_match("", ""))

    def test_only_delimiters(self):
        self.assertTrue(outputs_match("   \t\n\r\v", ""))


class TestSplitAndLowercase(unittest.TestCase):
    def test_basic_split(self):
        self.assertEqual(_split_and_lowercase("abc def GHI"), ["abc", "def", "ghi"])

    def test_multiple_delimiters(self):
        self.assertEqual(
            _split_and_lowercase("abc\tdef\nGHI\r123\v456"),
            ["abc", "def", "ghi", "123", "456"],
        )

    def test_empty_input(self):
        self.assertEqual(_split_and_lowercase(""), [])

    def test_only_delimiters(self):
        self.assertEqual(_split_and_lowercase("   \t\n\r\v"), [])


class TestValuesMatch(unittest.TestCase):
    def test_identical_strings(self):
        self.assertTrue(_values_match("abc", "abc"))

    def test_different_strings(self):
        self.assertFalse(_values_match("abc", "def"))

    def test_identical_integers(self):
        self.assertTrue(_values_match("123", "123"))

    def test_different_integers(self):
        self.assertFalse(_values_match("123", "456"))

    def test_close_floats(self):
        self.assertTrue(_values_match("3.141593", "3.141592"))

    def test_non_close_floats(self):
        self.assertFalse(_values_match("3.14", "3.15"))

    def test_int_and_float(self):
        self.assertTrue(_values_match("3", "3.0"))

    def test_string_and_number(self):
        self.assertFalse(_values_match("3", "three"))


class TestSplitByAnyChar(unittest.TestCase):
    def test_basic_split(self):
        self.assertEqual(_split_by_any_char("a,b;c", {",", ";"}), ["a", "b", "c"])

    def test_consecutive_delimiters(self):
        self.assertEqual(
            _split_by_any_char("a,,b;;c", {",", ";"}), ["a", "", "b", "", "c"]
        )

    def test_no_delimiters_found(self):
        self.assertEqual(_split_by_any_char("abc", {",", ";"}), ["abc"])

    def test_empty_input(self):
        self.assertEqual(_split_by_any_char("", {",", ";"}), [""])

    def test_only_delimiters(self):
        self.assertEqual(_split_by_any_char(",,;;", {",", ";"}), ["", "", "", "", ""])

    def test_empty_delimiters(self):
        with self.assertRaises(ValueError):
            _split_by_any_char("abc", set())

    def test_multi_char_delimiter(self):
        with self.assertRaises(ValueError):
            _split_by_any_char("abc", {"a", "bc"})


if __name__ == "__main__":
    unittest.main()
