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
		# Convert those IDs back into tokens (for labeling in the heatmap)
		tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
		
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
		
		return filtered_attn_data, tokens
