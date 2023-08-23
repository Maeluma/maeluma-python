import os
import unittest
from unittest import mock
from typing import List

import pytest
from maeluma.utils import get_api_key

import maeluma

API_KEY = get_api_key()
ml = maeluma.Client(API_KEY)


class ImageGeneration(unittest.TestCase):
    def test_success(self):
        generated_image = ml.generate_image(model="medium", prompt="ultra realistic close up portrait ((beautiful pale cyberpunk female with heavy black eyeliner))")
        self.assertIsInstance(generated_image['output'], list)
        self.assertIsInstance(generated_image['meta'], dict)

    def test_success_batched(self):
        _batch_size = 4
        generations = ml.generate_image(model="medium", prompt='ultra realistic close up portrait', num_generations=_batch_size)
        print(generations)
        for generation in generations['output']:
            self.assertIsInstance(generation, str)
        self.assertEqual(len(generations), _batch_size)

    def test_batch_return_exceptions(self):
        prompts = ["co:here", "x y z" * 3333, "co:here"]  # too long for 8192
        # test return_exceptions = False -> fails
        with self.assertRaises(maeluma.MaelumaError):
            generations = ml.batch_generate_image(model="medium", prompts=prompts, return_exceptions=False, num_generations=2)
        # test return_exceptions = True
        generations = ml.batch_generate_image(model="medium", prompts=prompts, return_exceptions=True, num_generations=2)
        self.assertEqual(len(generations), len(prompts))
        self.assertIsInstance(generations[1], maeluma.MaelumaAPIError)
        self.assertIsInstance(generations[0]['output'], list)
        self.assertIsInstance(generations[2]['output'], list)
        self.assertIsInstance(generations[2]['output'][0], str)        


    def test_invalid_num_inference_steps(self):
        with self.assertRaises(maeluma.MaelumaError):
            ml.generate_image(model="medium", prompt="hi", num_inference_steps=23)

    def test_valid_num_inference_steps(self):
        generated_image = ml.generate_image(model="medium", prompt="hi", num_inference_steps=31)
        self.assertIsInstance(generated_image['output'], list)
        self.assertIsInstance(generated_image['output'][0], str)

    def test_returns_same_seed(self):
        generated_image = ml.generate_image(model="medium", prompt="cool woman ((glasses, ultra realistic close up portrait))", seed=1001)
        self.assertIsInstance(generated_image['output'][0], str)
        self.assertEqual(generated_image['meta']['seed'], 1001)

    def test_enhances_prompt(self):
        generated_image = ml.generate_image(model="medium", prompt="cool woman ((glasses, ultra realistic close up portrait))", enhance_prompt=True)
        self.assertIsInstance(generated_image['output'][0], str)
