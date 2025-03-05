"""
This module defines the TransformerModel class, which loads the GPT-2 tokenizer,
the base GPT-2 model (for attention visualization), and the GPT2LMHeadModel for
computing next-token predictions.
"""

import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

class TransformerModel:
	"""
	TransformerModel loads and holds the GPT-2 models and tokenizer.
	
	It also provides a helper method to compute attention data for a given input.
	"""
	def __init__(self):
		# Load the GPT-2 tokenizer
		self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
		# Load the base GPT-2 model with attention outputs enabled
		self.model = GPT2Model.from_pretrained('gpt2', output_attentions=True)
		self.model.eval()  # We won't be training, so set it to eval mode
		# Load the GPT-2 LM Head model (used for next-token predictions)
		self.lm_model = GPT2LMHeadModel.from_pretrained('gpt2')
		self.lm_model.eval()  # Also in eval mode
		# Cache some config details
		self.num_layers = self.model.config.n_layer
		self.num_heads = self.model.config.n_head

	def get_attention_data(self, input_text, layer, head, threshold=0.0):
		"""
		Compute and return filtered attention data along with the token list.
		
		:param input_text: The input sentence as a string.
		:param layer: Layer index from which to retrieve attention.
		:param head: Head index within that layer.
		:param threshold: Minimum attention value; values below are zeroed out.
		:return: Tuple (filtered_attn_data, tokens)
		"""
		# Convert the input text into a tensor of token IDs
		input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
		
		# Get the raw tokens and clean them for display
		tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
		
		# Create display tokens that preserve uniqueness by adding position indicators
		# This ensures that repeated tokens like "the" are displayed uniquely
		display_tokens = []
		token_counts = {}  # To track number of occurrences of each token
		
		for i, token in enumerate(tokens):
			# Clean up the tokenizer's special prefixes for display
			if token.startswith('Ġ'):  # This is GPT-2's prefix for tokens that start with a space
				display_token = token[1:]  # Remove the 'Ġ' prefix
			else:
				display_token = token
				
			# Add position information for duplicate tokens to ensure uniqueness in visualization
			token_counts[display_token] = token_counts.get(display_token, 0) + 1
			if token_counts[display_token] > 1:
				display_token = f"{display_token}_{token_counts[display_token]}"
			
			display_tokens.append(display_token)
		
		# Run the model without gradient calculations
		with torch.no_grad():
			outputs = self.model(input_ids)
		
		# Retrieve the full attention stack (list of attention matrices per layer)
		attentions = outputs.attentions
		# Grab the attention matrix for the specified layer and head
		attn = attentions[layer][0, head, :, :]
		# Move to CPU and convert to a NumPy array for Plotly
		attn_data = attn.cpu().numpy()
		# Zero out any values below the threshold
		filtered_attn_data = attn_data * (attn_data >= threshold)
		
		# Add debug info to help users understand the tokenization
		print(f"Input text: {input_text}")
		print(f"Raw tokens: {tokens}")
		print(f"Display tokens: {display_tokens}")
		
		return filtered_attn_data, display_tokens
		
	def get_token_mapping(self, input_text):
		"""
		Returns the mapping between input text and tokenized output.
		Useful for debugging tokenization issues.
		
		:param input_text: The input text string.
		:return: A list of (token, original_text_span) tuples.
		"""
		# Tokenize the input text
		tokens = self.tokenizer.tokenize(input_text)
		
		# Print the tokens to help debug
		print(f"Input: {input_text}")
		print(f"Tokenized: {tokens}")
		
		# Return the tokens for inspection
		return tokens
