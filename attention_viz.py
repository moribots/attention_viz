import dash                            # Dash for building web apps
from dash import dcc, html             # Core and HTML components
from dash.dependencies import Input, Output  # Interactive callbacks
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import torch
from transformers import GPT2Tokenizer, GPT2Model  # Hugging Face Transformers for pre-trained models
import numpy as np

# -------------------------------
# STEP 1: LOAD THE MODEL AND TOKENIZER
# -------------------------------
# Initialize the GPT-2 tokenizer to convert text into token IDs.
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Load the GPT-2 model with attention outputs enabled.
# The 'output_attentions=True' flag ensures that the model returns attention weights.
model = GPT2Model.from_pretrained('gpt2', output_attentions=True)
model.eval()  # Set model to evaluation mode to disable dropout and other training-specific layers.

# Retrieve configuration details: number of layers and heads.
NUM_LAYERS = model.config.n_layer  # e.g., GPT-2 base has 12 layers.
NUM_HEADS = model.config.n_head    # e.g., GPT-2 base has 12 attention heads per layer.

# -------------------------------
# STEP 2: DEFINE A FUNCTION TO EXTRACT ATTENTION DATA
# -------------------------------
def get_attention_data(input_text, layer, head):
	"""
	Tokenizes the input text, passes it through the GPT-2 model,
	and extracts the attention weights for the specified layer and head.
	
	Args:
		input_text (str): The raw text input.
		layer (int): The index (0-indexed) of the Transformer layer.
		head (int): The index (0-indexed) of the attention head within that layer.
	
	Returns:
		attn_data (np.ndarray): A 2D NumPy array containing the attention weights of shape 
								(seq_len, seq_len) where seq_len is the number of tokens.
		tokens (list[str]): List of token strings corresponding to the input text.
	"""
	# Tokenize the input text to obtain token IDs as a PyTorch tensor.
	# 'return_tensors="pt"' converts the list of token IDs into a PyTorch tensor.
	input_ids = tokenizer.encode(input_text, return_tensors='pt')
	
	# Convert token IDs back to token strings for labeling the axes in the heatmap.
	tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
	
	# Run the model with no gradient computation.
	with torch.no_grad():
		outputs = model(input_ids)
	
	# 'attentions' is a tuple with one tensor per layer.
	# Each tensor in 'attentions' has shape: (batch_size, num_heads, seq_len, seq_len)
	attentions = outputs.attentions
	
	# For simplicity, we take the attention weights from the specified layer and head.
	# Here, we assume batch_size=1 (only one example), so we index the first element in the batch.
	# 'attentions[layer]' gives the tensor for the selected layer.
	# '[0, head, :, :]' selects the first example in the batch and the specified head, 
	# resulting in a 2D tensor of shape (seq_len, seq_len).
	attn = attentions[layer][0, head, :, :]
	
	# Convert to numpy
	attn_data = attn.cpu().numpy()
	
	return attn_data, tokens

# -------------------------------
# STEP 3: CREATE THE DASH APP LAYOUT
# -------------------------------
app = dash.Dash(__name__)

# Create app layout
# Define dropdown options for layers with an option for selecting all layers.
layer_options = [{'label': "All Layers", 'value': 'all'}] + \
	[{'label': f"Layer {i}", 'value': i} for i in range(NUM_LAYERS)]
# Similarly for heads.
head_options = [{'label': "All Heads", 'value': 'all'}] + \
	[{'label': f"Head {i}", 'value': i} for i in range(NUM_HEADS)]

app.layout = html.Div([
	# Title
	html.H1("Attention Visualization"),
	
	# User input
	html.Div([
		html.Label("Input Text:"),
		dcc.Input(
			id="input-text",
			type="text",
			value="The quick brown fox jumps over the lazy dog.",
			style={'width': '100%'}
		)
	], style={'marginBottom': '20px'}),
	
	# Transformer layer selector.
	# 'multi=True' allows selecting multiple layers.
	html.Div([
		html.Label("Select Layers:"),
		dcc.Dropdown(
			id="layer-dropdown",
			options=layer_options,
			value=['all'],  # Default selection: All Layers.
			multi=True,
			clearable=False
		)
	], style={'width': '45%', 'display': 'inline-block', 'marginRight': '20px'}),
	
	# Transformer layer selector.
	# 'multi=True' allows selecting multiple layers.
	html.Div([
		html.Label("Select Heads:"),
		dcc.Dropdown(
			id="head-dropdown",
			options=head_options,
			value=['all'],  # Default selection: All Heads.
			multi=True,
			clearable=False
		)
	], style={'width': '45%', 'display': 'inline-block'}),
	
	# Graph component to display the attention heatmap(s).
	dcc.Graph(id="attention-heatmap")
])

# -------------------------------
# STEP 4: CREATE A CALLBACK TO UPDATE THE HEATMAP DYNAMICALLY
# -------------------------------
@app.callback(
	Output("attention-heatmap", "figure"),  # The callback outputs a Plotly figure to the Graph component.
	[Input("input-text", "value"),
	 Input("layer-dropdown", "value"),
	 Input("head-dropdown", "value")]
)
def update_heatmap(input_text, selected_layers, selected_heads):
	"""
	Callback that updates the attention heatmap based on user inputs:
	- Input text
	- Selected layers and heads
	
	Args:
		input_text (str): The sentence input by the user.
		selected_layers (list): List of selected Transformer layer indices or 'all'.
		selected_heads (list): List of selected attention head indices or 'all'.

	Returns:
		fig (plotly.graph_objects.Figure): The updated figure with a subplot grid showing
										   the attention heatmaps for each (layer, head) pair.
	"""
	# If "all" is selected in layers, set selected_layers to include all layer indices.
	if 'all' in selected_layers:
		selected_layers = list(range(NUM_LAYERS))
	
	# If "all" is selected in heads, set selected_heads to include all head indices.
	if 'all' in selected_heads:
		selected_heads = list(range(NUM_HEADS))
	
	# Determine the number of rows and columns for the subplot grid.
	rows = len(selected_layers)
	cols = len(selected_heads)
	
	# Create subplot titles for each (layer, head) combination.
	subplot_titles = [f"Layer {layer}, Head {head}" for layer in selected_layers for head in selected_heads]
	
	# Create a subplot grid using Plotly's make_subplots.
	fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
	
	# Loop over each selected layer and head to extract and plot the corresponding attention heatmap.
	for i, layer in enumerate(selected_layers):
		for j, head in enumerate(selected_heads):
			# Get the attention data and token labels.
			attn_data, tokens = get_attention_data(input_text, layer, head)
			
			# Create a heatmap trace using Plotly Graph Objects.
			heatmap = go.Heatmap(
				z=attn_data,             # The 2D attention matrix.
				x=tokens,                # Token labels for the x-axis.
				y=tokens,                # Token labels for the y-axis.
				colorscale='Viridis',    # Color scale for visualizing attention weights.
				colorbar=dict(title="Attention Weight")
			)
			
			# Add the heatmap trace to the appropriate subplot cell.
			fig.add_trace(heatmap, row=i+1, col=j+1)
	
	# Update the overall layout of the figure.
	fig.update_layout(
		height=300 * rows, 
		width=400 * cols,
		title_text="Interactive Multi-Head & Multi-Layer Attention Visualization"
	)
	
	return fig

# -------------------------------
# STEP 5: RUN THE DASH APP
# -------------------------------
if __name__ == '__main__':
	app.run_server(debug=True)
