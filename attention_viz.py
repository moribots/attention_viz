import dash                            # Dash for building web apps
from dash import dcc, html             # Core and HTML components
from dash.dependencies import Input, Output  # Interactive callbacks
import plotly.express as px
import torch
from transformers import GPT2Tokenizer, GPT2Model  # Hugging Face Transformers for pre-trained models
import numpy as np

# -------------------------------
# STEP 1: LOAD THE MODEL AND TOKENIZER
# -------------------------------
# Initialize the GPT-2 tokenizer to convert text into token IDs.
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Load the GPT-2 model with attention outputs enabled.
# The 'output_attentions=True' flag ensures the model returns attention weights.
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
	
	# Return the attention data and the corresponding tokens.
	return attn_data, tokens

# -------------------------------
# STEP 3: CREATE THE DASH APP LAYOUT
# -------------------------------
app = dash.Dash(__name__)

# Create app layout
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
	
	# Transformer layer selector
	html.Div([
		html.Label("Select Layer:"),
		dcc.Dropdown(
			id="layer-dropdown",           
			options=[{'label': f"Layer {i}", 'value': i} for i in range(NUM_LAYERS)],
			value=0,                
			clearable=False
		)
	], style={'width': '30%', 'display': 'inline-block', 'marginRight': '20px'}),
	
	# Transformer head selector (within chosen layer)
	html.Div([
		html.Label("Select Head:"),
		dcc.Dropdown(
			id="head-dropdown",            
			options=[{'label': f"Head {i}", 'value': i} for i in range(NUM_HEADS)],
			value=0,
			clearable=False
		)
	], style={'width': '30%', 'display': 'inline-block'}),
	
	# Graph component to display the attention heatmap.
	dcc.Graph(id="attention-heatmap")
])

# -------------------------------
# STEP 4: CREATE A CALLBACK TO UPDATE THE HEATMAP DYNAMICALLY
# -------------------------------
# The callback function links the UI components (input text, layer and head dropdowns)
# to the Graph component. It updates the attention heatmap based on user input.
@app.callback(
	Output("attention-heatmap", "figure"),  # The callback outputs a Plotly figure to the Graph component.
	[Input("input-text", "value"),
	 Input("layer-dropdown", "value"),
	 Input("head-dropdown", "value")]
)
def update_heatmap(input_text, selected_layer, selected_head):
	"""
	Callback that updates the attention heatmap based on user inputs:
	- Input text
	- Selected layers and heads
	
	Args:
		input_text (str): The text entered by the user.
		selected_layer (int): The index of the selected Transformer layer.
		selected_head (int): The index of the selected attention head.
	
	Returns:
		fig (plotly.graph_objs._figure.Figure): The updated heatmap figure.
	"""
	# Extract the attention weights and tokens for the provided input and selection.
	attn_data, tokens = get_attention_data(input_text, selected_layer, selected_head)
	
	# Create an interactive heatmap using Plotly Express.
	# 'px.imshow' automatically creates a heatmap from a 2D NumPy array.
	fig = px.imshow(
		attn_data,  # The 2D attention matrix.
		labels=dict(x="Token Position", y="Token Position", color="Attention Weight"),
		x=tokens,   # Label columns with token strings.
		y=tokens,   # Label rows with token strings.
		color_continuous_scale='Viridis'  # Color scale for visualizing attention weights.
	)
	
	# Update the layout of the figure with a title indicating the current layer and head.
	fig.update_layout(title=f"Attention Heatmap: Layer {selected_layer}, Head {selected_head}")
	
	# Return the updated figure to be displayed in the Dash app.
	return fig

# -------------------------------
# STEP 5: RUN THE DASH APP
# -------------------------------
if __name__ == '__main__':
	app.run_server(debug=True)
