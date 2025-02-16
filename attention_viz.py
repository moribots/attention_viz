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
# STEP 2: DEFINE A FUNCTION TO EXTRACT AND FILTER ATTENTION DATA
# -------------------------------
def get_attention_data(input_text, layer, head, threshold=0.0):
	"""
	Tokenizes the input text, passes it through the GPT-2 model,
	extracts the attention weights for the specified layer and head,
	and applies a threshold filter to zero-out values below the threshold.

	Args:
		input_text (str): The raw text input.
		layer (int): The index (0-indexed) of the Transformer layer.
		head (int): The index (0-indexed) of the attention head within that layer.
		threshold (float): The threshold value; attention weights below this value are set to zero.

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
	
	# Apply threshold filtering: set values below the threshold to zero.
	filtered_attn_data = np.where(attn_data >= threshold, attn_data, 0)
	
	return filtered_attn_data, tokens

# -------------------------------
# STEP 3: CREATE THE DASH APP LAYOUT WITH A THRESHOLD SLIDER
# -------------------------------
app = dash.Dash(__name__)

# Define dropdown options for layers (without "Select All" option).
layer_options = [{'label': f"Layer {i}", 'value': i} for i in range(NUM_LAYERS)]
# Define dropdown options for heads (without "Select All" option).
head_options = [{'label': f"Head {i}", 'value': i} for i in range(NUM_HEADS)]

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
	html.Div([
		html.Label("Select Layers:"),
		dcc.Dropdown(
			id="layer-dropdown",
			options=[{'label': f"Layer {i}", 'value': i} for i in range(NUM_LAYERS)],
			value=[0],
			multi=True,
			clearable=False
		)
	], style={'width': '45%', 'display': 'inline-block', 'marginRight': '20px'}),
	
	# Transformer layer selector.
	html.Div([
		html.Label("Select Heads:"),
		dcc.Dropdown(
			id="head-dropdown",
			options=[{'label': f"Head {i}", 'value': i} for i in range(NUM_HEADS)],
			value=[0],
			multi=True,
			clearable=False
		)
	], style={'width': '45%', 'display': 'inline-block'}),
	
	# Slider for dynamic filtering and thresholding.
	html.Div([
		html.Label("Attention Threshold (0.0 - 1.0):"),
		dcc.Slider(
			id="threshold-slider",
			min=0.0,
			max=1.0,
			step=0.01,
			value=0.0,
			marks={i/10: f"{i/10}" for i in range(0, 11)}
		)
	], style={'marginTop': '20px', 'marginBottom': '20px'}),
	
	# Graph to display the attention heatmap(s).
	dcc.Graph(id="attention-heatmap")
])

# -------------------------------
# STEP 4: CREATE A CALLBACK TO UPDATE THE HEATMAP WITH ANIMATED TRANSITIONS
# -------------------------------
@app.callback(
	Output("attention-heatmap", "figure"),  # The callback outputs a Plotly figure to the Graph component.
	[Input("input-text", "value"),
	 Input("layer-dropdown", "value"),
	 Input("head-dropdown", "value"),
	 Input("threshold-slider", "value")]
)
def update_heatmap(input_text, selected_layers, selected_heads, threshold):
	"""
	Callback that updates the attention heatmap based on user inputs:
	- Input text
	- Selected layers and heads
	- Threshold for filtering attention weights
	
	Args:
		input_text (str): The sentence input by the user.
		selected_layers (list): List of selected Transformer layer indices.
		selected_heads (list): List of selected attention head indices.

	Returns:
		fig (plotly.graph_objects.Figure): The updated subplot grid figure.
	"""
	# Ensure that selected_layers and selected_heads are lists.
	if not isinstance(selected_layers, list):
		selected_layers = [selected_layers]
	if not isinstance(selected_heads, list):
		selected_heads = [selected_heads]
	
	# Determine grid dimensions based on selected layers and heads.
	rows = len(selected_layers)
	cols = len(selected_heads)
	
	# Create subplot titles for each (layer, head) pair.
	subplot_titles = [f"Layer {layer}, Head {head}" for layer in selected_layers for head in selected_heads]
	
	# Create a subplot grid.
	fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles)
	
	# Loop over each selected layer and head.
	for i, layer in enumerate(selected_layers):
		for j, head in enumerate(selected_heads):
			# Get filtered attention data and tokens for the given parameters.
			attn_data, tokens = get_attention_data(input_text, layer, head, threshold)
			
			# Create a heatmap trace.
			heatmap = go.Heatmap(
				z=attn_data,
				x=tokens,
				y=tokens,
				colorscale='Viridis',
				colorbar=dict(title="Attention Weight")
			)
			
			# Add the trace to the appropriate subplot cell.
			fig.add_trace(heatmap, row=i+1, col=j+1)
	
	# Update layout with animated transition parameters.
	fig.update_layout(
		height=300 * rows, 
		width=400 * cols,
		title_text="Interactive Multi-Head & Multi-Layer Attention Visualization (Filtered)",
		transition={'duration': 500, 'easing': 'cubic-in-out'}  # Animate transitions smoothly.
	)
	
	return fig

# -------------------------------
# STEP 5: RUN THE DASH APP
# -------------------------------
if __name__ == '__main__':
	app.run_server(debug=True)
