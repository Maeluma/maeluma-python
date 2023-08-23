import unittest
import maeluma
from maeluma.utils import get_api_key

API_KEY = get_api_key()
ml = maeluma.Client(API_KEY)


class AudioGeneration(unittest.TestCase):
    def test_success(self):
        generated_speech = ml.generate_speech(voice="shirley", text="Abraham Lincoln was not born in 2005.")
        self.assertEqual(generated_speech.voice, 'shirley')
        self.assertIsInstance(generated_speech.generations, list)

    def test_return_exceptions(self):
        text = "This sentence repeats alot." * 3000  # too long for 10,000
        with self.assertRaises(maeluma.MaelumaAPIError, msg="Raises error for too long text"):
            ml.generate_speech(voice="shirley", text=text)
        with self.assertRaises(maeluma.MaelumaAPIError, msg="Raises error for incorrect voice"):
            ml.generate_speech(voice="not-a-voice", text="a normal sentence")