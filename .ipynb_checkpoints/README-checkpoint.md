# Chain of Thought (CoT) Decoding without Prompting

A simple Python implementation of Chain of Thought decoding without prompting for language models. The intuition is that through smart sampling techniques, we can unlock model reasoning without engineering prompts to elicit those behaviors. This implementation allows you to explore multiple decoding paths and aggregate results for more reliable outputs. A more thorough implementation can be found at [OptILLM](https://github.com/codelion/optillm/blob/main/optillm/cot_decoding.py), which served as inspiration for our replication.


## Installation

```bash
# Clone the repository
git clone [your-repo-url]
cd [your-repo-name]

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

To see the implementation in action, you can run the provided example:

```bash
python examples/example_usage.py
```

This will run a simple example using Qwen2.5-0.5B-Instruct model to answer a question. You can find the example code in `examples/example_usage.py` and modify it for your use case.

## Usage in Your Own Code

You can use the CoT Decoding in your own code by importing it:

```python
from src.decoding import CoT_Decoding

# Initialize your model and tokenizer
# ... (model initialization code)

answer, confidence = CoT_Decoding(
    messages=your_messages,
    k=5,
    model=model,
    tokenizer=tokenizer,
    device=device
)
```

## Parameters

- `messages`: List of conversation messages in the format `[{"role": str, "content": str}, ...]`
- `k`: Number of top tokens to consider (default: 5)
- `max_new_tokens`: Maximum length of generated response (default: 512)
- `agg_path`: Whether to aggregate similar paths (default: False)

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- CUDA (optional, for GPU acceleration)

This implementation is based on the following work:
- [Chain-of-Thought Reasoning wihtout Prompting](https://arxiv.org/abs/2402.10200) (Wang and Zhou, 2024)


Other implementations:
- [Optillm CoT Decoding Implementation](https://github.com/codelion/optillm/blob/main/optillm/cot_decoding.py)
- [Original Paper Replication](https://github.com/shirley-wu/cot_decoding)


## License

This project is licensed under the MIT License, which permits use, modification, and distribution subject to the license terms.