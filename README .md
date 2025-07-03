# Fine-tuning a GPT Model on Instruction Data

This project demonstrates the process of fine-tuning a pre-trained GPT-2 model on an instruction-following dataset to enhance its ability to understand and respond to various instructions and prompts.

## Project Overview

The project includes the following steps:

- **Data Loading and Preparation**: Downloads and loads the `instruction-data.json` dataset containing instruction, input, and output pairs.
- **Data Formatting**: Converts instruction data into the Alpaca format suitable for training.
- **Custom Dataset and DataLoader**: Implements a PyTorch `Dataset` and `DataLoader` with padding and masking for training.
- **Loading a Pre-trained GPT Model**: Uses the GPT-2 medium model and loads its pre-trained weights.
- **Fine-tuning the Model**: Trains the model on the instruction dataset using a simple training loop.
- **Model Evaluation**: Evaluates the fine-tuned model on a test set, comparing responses to expected outputs.
- **Saving the Fine-tuned Model**: Saves the fine-tuned model's state dictionary.

## Data

The project uses the `instruction-data.json` dataset, which includes pairs of instructions, optional inputs, and desired outputs.

## Model

The project utilizes the pre-trained **GPT-2 medium** model with the following architecture:

- Token and positional embeddings
- Multiple Transformer blocks with Multi-Head Attention and FeedForward layers
- Layer normalization
- A final linear layer for outputting logits over the vocabulary

## Custom Components

- `download_and_load_file`: Downloads and loads data from a URL.
- `format_input`: Combines instruction and input into a single string.
- `InstructionDataset`: A PyTorch `Dataset` for encoded instruction data.
- `custom_collate_fn`: A custom collate function for padding and masking sequences in the `DataLoader`.
- `CausalAttention` and `MultiHeadAttention`: Implementations of attention mechanisms.
- `LayerNorm`, `GELU`, and `FeedForward`: Standard neural network layer implementations.
- `TransformerBlock`: A single Transformer layer.
- `GPTModel`: The main GPT model architecture.
- `load_weights_into_gpt`: Loads pre-trained weights into the custom GPT model.
- `generate_text_simple` and `generate`: Functions for text generation with different sampling strategies.
- `text_to_token_ids` and `token_ids_to_text`: Functions for converting between text and token IDs.
- `evaluate_model`, `generate_and_print_sample`, `calc_loss_batch`, `calc_loss_loader`, `train_model_simple`: Functions for training and evaluation.
- `plot_losses`: Visualizes training and validation losses.

## Getting Started

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Install the required libraries:
   ```bash
   pip install torch numpy tiktoken tqdm
   ```

3. Run the main script to fine-tune and evaluate the model:
   ```bash
   python main.py
   ```

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Tiktoken
- TQDM

## Usage

1. Ensure the `instruction-data.json` dataset is available or downloadable.
2. Modify the training parameters in the script (e.g., learning rate, batch size) as needed.
3. Run the training script to fine-tune the model.
4. Use the provided evaluation functions to test the model's performance.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
