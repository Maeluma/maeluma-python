# Maeluma Client Library


## Installation

The package can be installed with `pip`:

```bash
pip install --upgrade maeluma
```

Install from source:

```bash
pip install .
```

### Requirements

- Python 3.7+


## Quick Start

To use this library, you must have an API key and specify it as a string when creating the `maeluma.Client` object. API keys can be created through the [platform](https://docs.maeluma.io). This is a basic example of the creating the client and using the `generate_text` endpoint.

```python
import maeluma

# initialize the Maeluma Client with an API Key
ml = maeluma.Client('YOUR_API_KEY')

# generate a prediction for a prompt
prediction = ml.generate_text(
            prompt='Tell me about the Richard Feynman',
            max_tokens=100,
            num_generations=1)

print('The model output: {}'.format(generations[0].text))
```


## Run Tests
`pytest tests/`