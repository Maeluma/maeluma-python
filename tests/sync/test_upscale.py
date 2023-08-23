import os
import unittest
from unittest import mock

import io
import pytest
from maeluma.utils import get_api_key
from maeluma.responses import UpscaledImage

import maeluma

API_KEY = get_api_key()
ml = maeluma.Client(API_KEY)


class TestGenerate(unittest.TestCase):
    def test_success(self):
        img_response = ml.upscale_image(url="https://placeholder.com/250x250")
        self.assertIsInstance(img_response, UpscaledImage)
        self.assertIsInstance(img_response.url, str)

    def test_fails_with_no_url_and_no_file(self):
        with self.assertRaises(maeluma.MaelumaError):
            ml.upscale_image()
    
    def test_fails_with_file_io_memory(self):
        img_response = ml.upscale_image(image=io.BytesIO(b"\x00\x00\x00\x00\x00\x00\x00\x00\x01\x01\x01\x01\x01\x01"))
        self.assertIsInstance(img_response, UpscaledImage)
        self.assertIsInstance(img_response.url, str)

    def test_fails_with_file_io_on_disk(self):
        img_response = ml.upscale_image(image="tests/test_image.png")
        self.assertIsInstance(img_response, UpscaledImage)
        self.assertIsInstance(img_response.url, str)

    
