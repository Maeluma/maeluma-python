import os
import unittest
from unittest import mock

import io
import pytest
from maeluma.utils import get_api_key
from maeluma.responses import SummarizeResponse

import maeluma

API_KEY = get_api_key()
ml = maeluma.Client(API_KEY)


class TestGenerate(unittest.TestCase):
    def test_success(self):
        img_response = ml.summarize(text="Tell me how the DOW is doing",)
        self.assertIsInstance(img_response, SummarizeResponse)
        self.assertIsInstance(img_response.summary, str)

    def test_fails_with_no_text(self):
        with self.assertRaises(TypeError):
            ml.summarize()
    
    def test_success_w_length(self):
        response = ml.summarize(text="Tell me how the DOW is doing", length="long")
        self.assertIsInstance(response, SummarizeResponse)
        self.assertIsInstance(response.summary, str)
    
    def test_fails_with_invalid_length(self):
        with self.assertRaises(maeluma.MaelumaAPIError):
            ml.summarize(text="Tell me how the DOW is doing", length="extra-extra-long")
    
    def test_success_w_format(self):
        response = ml.summarize(text="Tell me how the DOW is doing", format="bullets")
        self.assertIsInstance(response, SummarizeResponse)
        self.assertIsInstance(response.summary, str)
    
    def test_fails_with_invalid_format(self):
        with self.assertRaises(maeluma.MaelumaAPIError):
            ml.summarize(text="Tell me how the DOW is doing", format="not-a-format")

    def test_success_w_extractiveness(self):
        response = ml.summarize(text="Tell me how the DOW is doing", extractiveness="high")
        self.assertIsInstance(response, SummarizeResponse)
        self.assertIsInstance(response.summary, str)
    
    def test_fails_with_invalid_extractiveness(self):
        with self.assertRaises(maeluma.MaelumaAPIError):
            ml.summarize(text="Tell me how the DOW is doing", extractiveness="way-too-high")
    

    def test_success_w_additional_command(self):
        response = ml.summarize(text="Tell me how the DOW is doing", additional_command="written by Chewy")
        self.assertIsInstance(response, SummarizeResponse)
        self.assertIsInstance(response.summary, str)
        
      

    
