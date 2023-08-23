import os
import unittest
from unittest import mock

import pytest
from maeluma.utils import get_api_key

import maeluma

API_KEY = get_api_key()
ml = maeluma.Client(API_KEY)


class TestGenerate(unittest.TestCase):
    def test_success(self):
        prediction = ml.generate(model="medium", prompt="co:here", max_tokens=100)
        self.assertIsInstance(prediction.generations[0].text, str)
        self.assertIsInstance(prediction.meta, dict)

    def test_success_batched(self):
        _batch_size = 10
        predictions = ml.batch_generate(model="medium", prompts=["co:here"] * _batch_size, max_tokens=1)
        for prediction in predictions:
            self.assertIsInstance(prediction.generations[0].text, str)
        self.assertEqual(len(predictions), 10)

    def test_batch_return_exceptions(self):
        prompts = ["co:here", "x y z" * 3333, "co:here"]  # too long for 8192
        # test return_exceptions = False -> fails
        with self.assertRaises(maeluma.MaelumaError):
            predictions = ml.batch_generate(model="medium", prompts=prompts, max_tokens=1)
        # test return_exceptions = True
        predictions = ml.batch_generate(model="medium", prompts=prompts, max_tokens=1, return_exceptions=True)
        self.assertEqual(len(predictions), len(prompts))
        self.assertIsInstance(predictions[1], Exception)
        self.assertIsInstance(predictions[0][0].text, str)
        self.assertIsInstance(predictions[2][0].text, str)


    def test_invalid_temp(self):
        with self.assertRaises(maeluma.MaelumaError):
            ml.generate(model="medium", prompt="hi", max_tokens=1, temperature=-1, truncate='none').generations

    def test_no_version_works(self):
        maeluma.Client(API_KEY).generate(model="medium", prompt="co:here", max_tokens=1).generations
