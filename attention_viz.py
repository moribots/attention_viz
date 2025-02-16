import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import torch
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2', output_attentions=True)
model.eval()  

NUM_LAYERS = model.config.n_layer  
NUM_HEADS = model.config.n_head   

def get_attention_data(input_text, layer, head):
	"""
	Tokenizes the input text, runs it through the GPT-2 model,
	and extracts the attention weights for the specified layer and head.
	
	Args:
		input_text (str): The input sentence.
		layer (int): The index of the Transformer layer (0-indexed).
		head (int): The index of the attention head (0-indexed).
	
	Returns:
		attn_data (np.ndarray): A 2D NumPy array of shape (seq_len, seq_len) containing attention weights.
		tokens (list[str]): List of token strings corresponding to the input.
	"""
	input_ids = tokenizer.encode(input_text, return_tensors='pt')
	tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
	
	with torch.no_grad():
		outputs = model(input_ids)
	
	attentions = outputs.attentions
	
	attn = attentions[layer][0, head, :, :]
	attn_data = attn.cpu().numpy()
	
	return attn_data, tokens

app = dash.Dash(__name__)

app.layout = html.Div([
	html.H1("Interactive Attention Visualization"),
	
	html.Div([
		html.Label("Input Text:"),
		dcc.Input(
			id="input-text", 
			type="text", 
			value="The quick brown fox jumps over the lazy dog.",
			style={'width': '100%'}
		)
	], style={'marginBottom': '20px'}),
	
	html.Div([
		html.Label("Select Layer:"),
		dcc.Dropdown(
			id="layer-dropdown",
			options=[{'label': f"Layer {i}", 'value': i} for i in range(NUM_LAYERS)],
			value=0,
			clearable=False
		)
	], style={'width': '30%', 'display': 'inline-block', 'marginRight': '20px'}),
	
	html.Div([
		html.Label("Select Head:"),
		dcc.Dropdown(
			id="head-dropdown",
			options=[{'label': f"Head {i}", 'value': i} for i in range(NUM_HEADS)],
			value=0,
			clearable=False
		)
	], style={'width': '30%', 'display': 'inline-block'}),
	
	dcc.Graph(id="attention-heatmap")
])

@app.callback(
	Output("attention-heatmap", "figure"),
	[Input("input-text", "value"),
	 Input("layer-dropdown", "value"),
	 Input("head-dropdown", "value")]
)
def update_heatmap(input_text, selected_layer, selected_head):
	attn_data, tokens = get_attention_data(input_text, selected_layer, selected_head)
	
	fig = px.imshow(
		attn_data,
		labels=dict(x="Token Position", y="Token Position", color="Attention Weight"),
		x=tokens,
		y=tokens,
		color_continuous_scale='Viridis'
	)
	fig.update_layout(title=f"Attention Heatmap: Layer {selected_layer}, Head {selected_head}")
	
	return fig


if __name__ == '__main__':
	app.run_server(debug=True)
