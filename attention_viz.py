import dash                            # Dash for building web apps
from dash import dcc, html             # Core and HTML components
from dash.dependencies import Input, Output  # Interactive callbacks
import plotly.graph_objects as go
from plotly.subplots import make_subplots 
import torch
from transformers import GPT2Tokenizer, GPT2Model  # Hugging Face Transformers for pre-trained models
from transformers import GPT2LMHeadModel  # Added for next-token predictions
import numpy as np
import dash_bootstrap_components as dbc  # For improved UI styling with Bootstrap.
from dash.dependencies import State
from tqdm import tqdm

# Use Bootstrap stylesheet for better styling.
external_stylesheets = [dbc.themes.BOOTSTRAP]

# -------------------------------
# STEP 1: LOAD THE MODEL AND TOKENIZER
# -------------------------------
# Initialize the GPT-2 tokenizer to convert text into token IDs.
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Load the GPT-2 model with attention outputs enabled.
# The 'output_attentions=True' flag ensures that the model returns attention weights.
model = GPT2Model.from_pretrained('gpt2', output_attentions=True)
model.eval()  # Set model to evaluation mode to disable dropout and other training-specific layers.

# Added: Load the GPT-2 language model for next-token predictions.
lm_model = GPT2LMHeadModel.from_pretrained('gpt2')
lm_model.eval()

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
# STEP 3: CREATE THE DASH APP LAYOUT WITH A USER-FRIENDLY UI/UX
# -------------------------------
# Initialize the Dash app with external Bootstrap stylesheet.
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Define dropdown options for layers (without "Select All" option).
layer_options = [{'label': f"Layer {i}", 'value': i} for i in range(NUM_LAYERS)]
# Define dropdown options for heads (without "Select All" option).
head_options = [{'label': f"Head {i}", 'value': i} for i in range(NUM_HEADS)]

# Create a sidebar card with instructions.
sidebar = dbc.Card(
	[
		dbc.CardHeader("Instructions"),
		dbc.CardBody(
			[
				html.P("- The input text on the right is modifiable.", className="card-text"),
				html.P("- Select one or more layers and heads to view their attention heatmaps.", className="card-text"),
				html.P("- Adjust the attention threshold slider to filter weak connections.", className="card-text"),
				html.P("- Click on a cell in any heatmap to view token details.", className="card-text"),
				html.P("- Reading intuition: pick a token at any row and go through the columns to understand how much it attends"
				" to itself and prior tokens. E.g., in the default example, 'brown' @ Layer[0]Head[0] attends strongly"
				" to itself and 'the', but not 'quick'.", className="card-text"),
				html.P("- Check the ablation box to see how masking the selected Layer/Head combo changes the next prediction distribution."
		   		" This will also show KL divergence, which quantifies the change in probability distribution, where higher values indicate a larger change.", className="card-text"),

			]
		),
	],
	style={"width": "100%", "marginBottom": "20px"}
)

# Create the main content layout.
main_content = dbc.Card(
	dbc.CardBody(
		[
			# Input text component.
			html.Div([
				html.Label("Input Text:"),
				dcc.Input(
					id="input-text", 
					type="text", 
					value="The quick brown fox jumps over the lazy dog.",
					style={'width': '100%'}
				)
			], style={'marginBottom': '20px'}),
			# Dropdown for selecting layers.
			html.Div([
				html.Label("Select Layers:"),
				dcc.Dropdown(
					id="layer-dropdown",
					options=layer_options,
					value=[0],
					multi=True,
					clearable=False
				)
			], style={'width': '45%', 'display': 'inline-block', 'marginRight': '20px'}),
			# Dropdown for selecting heads.
			html.Div([
				html.Label("Select Heads:"),
				dcc.Dropdown(
					id="head-dropdown",
					options=head_options,
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
			# Checkbox for causal intervention (ablate selected heads)
			html.Div([
				dcc.Checklist(
					id="causal-intervention",
					options=[{'label': '\tEnable Causal Tracing (Ablate Selected Heads)', 'value': 'ablate'}],
					value=[]
				)
			], style={'marginTop': '20px', 'marginBottom': '20px'}),
			# Graph component wrapped in a Loading spinner.
			dcc.Loading(
				id="loading-graph",
				type="circle",
				children=dcc.Graph(id="attention-heatmap")
			),
			# Div to display enhanced token info on click.
			html.Div(id="token-info", style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ccc'}),

			# Button and Div for running the ablation study.
            html.Button("Run Ablation Study", id="run-ablation-study", n_clicks=0),
            html.Div(id="ablation-result", style={'marginTop': '20px', 'padding': '10px', 'border': '1px solid #ccc'})

		]
	),
	style={"width": "100%"}
)

# Use a responsive container with a row that has two columns: sidebar and main content.
app.layout = dbc.Container(
	[
		dbc.Row(
			dbc.Col(html.H1("Interactive Attention Visualization"), width=12),
			style={"marginTop": "20px", "marginBottom": "20px"}
		),
		dbc.Row(
			[
				dbc.Col(sidebar, width=3),
				dbc.Col(main_content, width=9)
			]
		)
	],
	fluid=True
)

# -------------------------------
# STEP 4: CREATE A CALLBACK TO UPDATE THE HEATMAP
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
	
	return fig

def make_ablate_hook(selected_head):
	"""
	Creates a forward hook that zeros out the output corresponding to the selected attention head.
	
	Args:
		selected_head (int): The index of the attention head to ablate.
	
	Returns:
		hook (function): A function to be registered as a forward hook.
	"""
	def hook(module, input, output):
		head_dim = lm_model.config.hidden_size // lm_model.config.n_head
		start = selected_head * head_dim
		end = start + head_dim
		# If output is a tuple, assume the first element is the hidden states.
		if isinstance(output, tuple):
			attn_output = output[0]
			attn_output_clone = attn_output.clone()
			attn_output_clone[:, :, start:end] = 0
			# Return the modified tuple with the rest of the outputs unchanged.
			return (attn_output_clone,) + output[1:]
		else:
			output_clone = output.clone()
			output_clone[:, :, start:end] = 0
			return output_clone
	return hook

# -------------------------------
# STEP 5: CREATE A CALLBACK TO UPDATE TOKEN INFO ON CLICK
# -------------------------------
@app.callback(
    Output("token-info", "children"),
    [Input("attention-heatmap", "clickData"),
     Input("input-text", "value"),
     Input("causal-intervention", "value"),
     Input("layer-dropdown", "value"),
     Input("head-dropdown", "value")]
)
def update_token_info(clickData, input_text, causal_intervention, layer_dropdown, head_dropdown):
    """
    Callback that updates the token info Div based on a click event on the heatmap.
    When a cell in the heatmap is clicked, it extracts the token from the 'x' value,
    computes its token ID and embedding norm from the model's embedding layer, and displays 
    this information along with deeper analysis metrics. The analysis includes baseline 
    next-token predictions, and if causal intervention is enabled, ablated predictions,
    KL divergence between the distributions, and whether the top prediction changed.
    """
    if clickData is None:
        return "Click on a cell in the heatmap to see token information."
    
    try:
        token_clicked = clickData["points"][0]["x"]
    except (KeyError, IndexError):
        return "Error retrieving token info from click data."
    
    token_id = tokenizer.convert_tokens_to_ids(token_clicked)
    embedding = model.wte.weight[token_id]
    embedding_norm = torch.norm(embedding).item()
    info = f"Token: {token_clicked}\nToken ID: {token_id}\nEmbedding Norm: {embedding_norm:.4f}"
    
    # Re-tokenize the full input and find the index of the clicked token.
    full_input_ids = tokenizer.encode(input_text, return_tensors='pt')
    full_tokens = tokenizer.convert_ids_to_tokens(full_input_ids[0])
    try:
        token_index = full_tokens.index(token_clicked)
    except ValueError:
        token_index = len(full_tokens) - 1
    
    # Use the context up to and including the clicked token.
    truncated_ids = full_input_ids[:, :token_index+1]
    
    # ----- Compute Baseline Predictions (without intervention) -----
    with torch.no_grad():
        baseline_outputs = lm_model(truncated_ids)
    baseline_logits = baseline_outputs.logits[0, -1, :]
    baseline_probs = torch.softmax(baseline_logits, dim=-1)
    baseline_topk = 5
    baseline_top_probs, baseline_top_indices = torch.topk(baseline_probs, baseline_topk)
    baseline_top_tokens = tokenizer.convert_ids_to_tokens(baseline_top_indices.tolist())
    
    baseline_info = "\n\nBaseline Next Token Predictions:\n"
    for token, prob in zip(baseline_top_tokens, baseline_top_probs.tolist()):
        baseline_info += f"{token}: {prob:.4f}\n"
    
    # ----- If Causal Intervention is Enabled, Compute Ablated Predictions -----
    if 'ablate' in causal_intervention:
        # Ensure layer_dropdown and head_dropdown are lists.
        layer_list = layer_dropdown if isinstance(layer_dropdown, list) else [layer_dropdown]
        head_list = head_dropdown if isinstance(head_dropdown, list) else [head_dropdown]
        hook_handles = []
        # Register hooks for every combination of selected layers and heads.
        for layer in layer_list:
            for head in head_list:
                hook_handle = lm_model.transformer.h[layer].attn.register_forward_hook(
                    make_ablate_hook(head)
                )
                hook_handles.append(hook_handle)
        with torch.no_grad():
            ablated_outputs = lm_model(truncated_ids)
        for handle in hook_handles:
            handle.remove()
            
        ablated_logits = ablated_outputs.logits[0, -1, :]
        ablated_probs = torch.softmax(ablated_logits, dim=-1)
        ablated_topk = 5
        ablated_top_probs, ablated_top_indices = torch.topk(ablated_probs, ablated_topk)
        ablated_top_tokens = tokenizer.convert_ids_to_tokens(ablated_top_indices.tolist())
        
        ablated_info = "\n\nAblated Next Token Predictions:\n"
        for token, prob in zip(ablated_top_tokens, ablated_top_probs.tolist()):
            ablated_info += f"{token}: {prob:.4f}\n"
        
        # ----- Compute Deeper Analysis Metrics -----
        # KL Divergence between baseline and ablated distributions.
        # Add a small epsilon to avoid log(0)
        epsilon = 1e-10
        kl_div = torch.sum(baseline_probs * torch.log((baseline_probs + epsilon) / (ablated_probs + epsilon))).item()
        # Compare top predictions.
        baseline_top_token = baseline_top_tokens[0]
        ablated_top_token = ablated_top_tokens[0]
        top_token_change = f"Top token changed from '{baseline_top_token}' to '{ablated_top_token}'" \
                           if baseline_top_token != ablated_top_token else "Top token remains unchanged"
        
        deeper_metrics = "\n\nDeeper Analysis Metrics:\n"
        deeper_metrics += f"KL Divergence: {kl_div:.4f}\n"
        deeper_metrics += f"{top_token_change}\n"
        
        info += baseline_info + ablated_info + deeper_metrics
    else:
        info += baseline_info
    
    return html.Pre(info)

def evaluate_ablation(truncated_ids, baseline_probs, ablation_set, epsilon=1e-10):
    """
    Applies all hooks corresponding to the ablation_set (a list of (layer, head) pairs),
    runs the LM model on the truncated_ids, and computes the KL divergence between the baseline
    probability distribution and the ablated one.
    """
    hook_handles = []
    for (layer, head) in ablation_set:
        hook_handle = lm_model.transformer.h[layer].attn.register_forward_hook(
            make_ablate_hook(head)
        )
        hook_handles.append(hook_handle)
    with torch.no_grad():
        ablated_logits = lm_model(truncated_ids).logits[0, -1, :]
    for handle in hook_handles:
        handle.remove()
    ablated_probs = torch.softmax(ablated_logits, dim=-1)
    kl_div = torch.sum(
         baseline_probs * torch.log((baseline_probs + epsilon) / (ablated_probs + epsilon))
    ).item()
    return kl_div

def find_best_ablation_combo(truncated_ids, baseline_probs, max_heads=3):
    """
    Performs a greedy search over all candidate (layer, head) pairs to find a set of ablations
    (up to max_heads in size) that maximizes the KL divergence between the baseline and ablated
    next-token prediction distributions.
    
    Returns:
        best_set (list): A list of (layer, head) pairs.
        best_kl (float): The corresponding KL divergence.
    """
    candidate_list = []
    for layer in range(lm_model.config.n_layer):
        for head in range(lm_model.config.n_head):
            candidate_list.append((layer, head))
    
    best_set = []  # start with no ablation; baseline divergence is 0.
    best_kl = evaluate_ablation(truncated_ids, baseline_probs, best_set)
    
    improved = True
    while improved and len(best_set) < max_heads:
        improved = False
        best_candidate = None
        candidate_kl = best_kl
        # Wrap the candidate iteration with tqdm to show progress in terminal.
        for candidate in tqdm(candidate_list, desc="Evaluating candidates", leave=False):
            if candidate in best_set:
                continue
            test_set = best_set + [candidate]
            kl = evaluate_ablation(truncated_ids, baseline_probs, test_set)
            if kl > candidate_kl:
                candidate_kl = kl
                best_candidate = candidate
                improved = True
        if best_candidate is not None:
            best_set.append(best_candidate)
            best_kl = candidate_kl
    return best_set, best_kl


# -------------------------------
# STEP 6: ABLATION STUDY CALLBACK
# -------------------------------
@app.callback(
    Output("ablation-result", "children"),
    [Input("run-ablation-study", "n_clicks")],
    [State("input-text", "value"),
     State("attention-heatmap", "clickData")]
)
def run_ablation_study(n_clicks, input_text, clickData):
    """
    When the "Run Ablation Study" button is clicked, this callback:
      - Re-tokenizes the input and finds the clicked token to define a truncated context.
      - Computes the baseline next-token probabilities.
      - Runs a greedy search over all (layer, head) candidates (up to a maximum number)
        to find a set of ablations that maximizes the KL divergence between baseline and ablated predictions.
      - Displays the resulting ablation set and the KL divergence.
    """
    if n_clicks is None or n_clicks == 0:
        return "Click the button to run the ablation study for the clicked token."
    
    if clickData is None:
        return "Click on a token in the heatmap before running the ablation study."
    
    try:
        token_clicked = clickData["points"][0]["x"]
    except (KeyError, IndexError):
        return "Error retrieving token info from click data."
    
    # Re-tokenize the full input and find the index of the clicked token.
    full_input_ids = tokenizer.encode(input_text, return_tensors="pt")
    full_tokens = tokenizer.convert_ids_to_tokens(full_input_ids[0])
    try:
        token_index = full_tokens.index(token_clicked)
    except ValueError:
        token_index = len(full_tokens) - 1
    truncated_ids = full_input_ids[:, :token_index+1]
    
    # Compute baseline next-token probabilities.
    with torch.no_grad():
        baseline_logits = lm_model(truncated_ids).logits[0, -1, :]
    baseline_probs = torch.softmax(baseline_logits, dim=-1)
    
    # Run greedy search for best ablation combination (e.g., up to 3 heads).
    best_set, best_kl = find_best_ablation_combo(truncated_ids, baseline_probs, max_heads=3)
    
    # Create a summary table for the selected ablations.
    table_rows = []
    for (layer, head) in best_set:
        table_rows.append(html.Tr([
            html.Td(layer, style={'border': '1px solid black', 'padding': '4px'}),
            html.Td(head, style={'border': '1px solid black', 'padding': '4px'})
        ]))
    table = html.Table(
        [html.Thead(html.Tr([html.Th("Layer", style={'border': '1px solid black', 'padding': '4px'}),
                              html.Th("Head", style={'border': '1px solid black', 'padding': '4px'})])),
         html.Tbody(table_rows)],
        style={'width': '100%', 'borderCollapse': 'collapse'}
    )
    
    result_text = f"Best ablation combo (simultaneously ablating {len(best_set)} heads):"
    result_text += f"\nKL Divergence: {best_kl:.4f}"
    
    return html.Div([html.Pre(result_text), table])

# -------------------------------
# STEP 7: RUN THE DASH APP
# -------------------------------
if __name__ == '__main__':
	app.run_server(debug=True)
