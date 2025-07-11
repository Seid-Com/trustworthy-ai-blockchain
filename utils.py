import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import random
from typing import Dict, List, Any, Tuple
import hashlib
import json

def generate_mnist_data(num_samples: int = 1000, client_id: str = "client_0") -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate MNIST-like synthetic data for federated learning simulation
    
    Args:
        num_samples: Number of samples to generate
        client_id: Client identifier for data distribution variation
    
    Returns:
        Tuple of (data, labels)
    """
    # Set seed based on client_id for consistent but different data distribution
    seed = hash(client_id) % 1000
    np.random.seed(seed)
    
    # Generate synthetic 28x28 images
    data = np.random.rand(num_samples, 28, 28, 1).astype(np.float32)
    labels = np.random.randint(0, 10, num_samples)
    
    # Add some structure to make it more realistic
    for i in range(num_samples):
        label = labels[i]
        
        # Add label-specific patterns
        if label < 5:
            # Add horizontal lines for digits 0-4
            data[i, 10:18, :, 0] += 0.3
        else:
            # Add vertical lines for digits 5-9
            data[i, :, 10:18, 0] += 0.3
        
        # Add some noise
        noise = np.random.normal(0, 0.1, (28, 28, 1))
        data[i] += noise
    
    # Normalize to [0, 1]
    data = np.clip(data, 0, 1)
    
    # Reset random seed
    np.random.seed(None)
    
    return data, labels

def create_cnn_model(input_shape: Tuple[int, int, int] = (28, 28, 1), num_classes: int = 10) -> MLPClassifier:
    """
    Create a neural network model for MNIST-like classification
    
    Args:
        input_shape: Input shape of the images
        num_classes: Number of output classes
    
    Returns:
        Configured MLPClassifier model
    """
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        random_state=42,
        warm_start=True
    )
    
    return model

def plot_network_graph(nodes: List[Dict[str, Any]], edges: List[Tuple[str, str]] = None, 
                      title: str = "Network Graph") -> go.Figure:
    """
    Create a network graph visualization using plotly
    
    Args:
        nodes: List of node dictionaries with 'id', 'label', and optional 'color', 'size'
        edges: List of edge tuples (source, target)
        title: Graph title
    
    Returns:
        Plotly figure object
    """
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for node in nodes:
        G.add_node(node['id'], **node)
    
    # Add edges
    if edges:
        G.add_edges_from(edges)
    else:
        # Create a fully connected graph if no edges specified
        node_ids = [node['id'] for node in nodes]
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                G.add_edge(node_ids[i], node_ids[j])
    
    # Generate layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Extract edge coordinates
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Extract node coordinates and attributes
    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []
    
    for node in nodes:
        node_id = node['id']
        x, y = pos[node_id]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node.get('label', node_id))
        node_colors.append(node.get('color', '#1f77b4'))
        node_sizes.append(node.get('size', 20))
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="middle center",
        hoverinfo='text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace])
    
    fig.update_layout(
        title=title,
        title_font_size=18,
        title_font_family="Arial, sans-serif",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        font=dict(family="Arial, sans-serif", size=14),
        plot_bgcolor='white',
        paper_bgcolor='white',
        width=800,
        height=600,
        annotations=[
            dict(
                text=f"Network with {len(nodes)} nodes and {len(G.edges())} edges",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor="left", yanchor="bottom",
                font=dict(color="#888", size=10)
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    
    return fig

def create_blockchain_visualization(blocks: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a blockchain visualization
    
    Args:
        blocks: List of block dictionaries
    
    Returns:
        Plotly figure object
    """
    if not blocks:
        # Return empty figure if no blocks
        fig = go.Figure()
        fig.update_layout(
            title="Blockchain Visualization",
            annotations=[
                dict(
                    text="No blocks to display",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=16)
                )
            ]
        )
        return fig
    
    # Create block chain visualization
    fig = go.Figure()
    
    # Add blocks
    for i, block in enumerate(blocks):
        x_pos = i * 2
        
        # Block rectangle
        fig.add_shape(
            type="rect",
            x0=x_pos, y0=0,
            x1=x_pos + 1.5, y1=1,
            fillcolor="lightblue",
            line=dict(color="darkblue", width=2)
        )
        
        # Block text
        fig.add_annotation(
            x=x_pos + 0.75, y=0.5,
            text=f"Block {block.get('index', i)}<br>{block.get('hash', 'N/A')[:8]}...",
            showarrow=False,
            font=dict(size=10)
        )
        
        # Arrow to next block
        if i < len(blocks) - 1:
            fig.add_annotation(
                x=x_pos + 1.75, y=0.5,
                text="→",
                showarrow=False,
                font=dict(size=20, color="darkgreen")
            )
    
    fig.update_layout(
        title="Blockchain Structure",
        title_font_size=18,
        title_font_family="Arial, sans-serif",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=250,
        width=900,
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family="Arial, sans-serif", size=14),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def calculate_model_similarity(model1_weights: List[np.ndarray], 
                              model2_weights: List[np.ndarray]) -> float:
    """
    Calculate similarity between two model weight sets
    
    Args:
        model1_weights: First model weights
        model2_weights: Second model weights
    
    Returns:
        Similarity score between 0 and 1
    """
    if len(model1_weights) != len(model2_weights):
        return 0.0
    
    similarities = []
    for w1, w2 in zip(model1_weights, model2_weights):
        if w1.shape != w2.shape:
            similarities.append(0.0)
            continue
        
        # Calculate cosine similarity
        w1_flat = w1.flatten()
        w2_flat = w2.flatten()
        
        norm1 = np.linalg.norm(w1_flat)
        norm2 = np.linalg.norm(w2_flat)
        
        if norm1 == 0 or norm2 == 0:
            similarities.append(0.0)
        else:
            similarity = np.dot(w1_flat, w2_flat) / (norm1 * norm2)
            similarities.append(max(0, similarity))  # Ensure non-negative
    
    return np.mean(similarities)

def generate_attack_visualization(attack_data: Dict[str, Any]) -> go.Figure:
    """
    Generate visualization for attack scenarios
    
    Args:
        attack_data: Attack data dictionary
    
    Returns:
        Plotly figure object
    """
    attack_type = attack_data.get('attack_type', 'unknown')
    
    if attack_type == 'data_poisoning':
        # Create data poisoning visualization
        fig = go.Figure()
        
        # Normal data
        normal_x = np.random.normal(0, 1, 100)
        normal_y = np.random.normal(0, 1, 100)
        
        # Poisoned data
        poison_x = np.random.normal(2, 0.5, 20)
        poison_y = np.random.normal(2, 0.5, 20)
        
        fig.add_trace(go.Scatter(
            x=normal_x, y=normal_y,
            mode='markers',
            name='Normal Data',
            marker=dict(color='blue', size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=poison_x, y=poison_y,
            mode='markers',
            name='Poisoned Data',
            marker=dict(color='red', size=8)
        ))
        
        fig.update_layout(
            title='Data Poisoning Attack Visualization',
            xaxis_title='Feature 1',
            yaxis_title='Feature 2'
        )
        
    elif attack_type == 'model_poisoning':
        # Create model poisoning visualization
        rounds = list(range(1, 21))
        normal_accuracy = [85 + 5 * (1 - np.exp(-r/5)) + np.random.normal(0, 0.5) for r in rounds]
        poisoned_accuracy = [85 + 5 * (1 - np.exp(-r/5)) - r * 0.5 + np.random.normal(0, 0.5) for r in rounds]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=rounds, y=normal_accuracy,
            mode='lines+markers',
            name='Normal Training',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=rounds, y=poisoned_accuracy,
            mode='lines+markers',
            name='Poisoned Training',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title='Model Poisoning Attack Impact',
            xaxis_title='Training Round',
            yaxis_title='Model Accuracy (%)'
        )
        
    else:
        # Generic attack visualization
        fig = go.Figure()
        fig.update_layout(
            title=f'{attack_type.replace("_", " ").title()} Attack Visualization',
            annotations=[
                dict(
                    text=f"Attack Type: {attack_type}",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=16)
                )
            ]
        )
    
    return fig

def hash_data(data: Any) -> str:
    """
    Create a hash of any data structure
    
    Args:
        data: Data to hash
    
    Returns:
        SHA-256 hash string
    """
    if isinstance(data, (dict, list)):
        data_string = json.dumps(data, sort_keys=True)
    else:
        data_string = str(data)
    
    return hashlib.sha256(data_string.encode()).hexdigest()

def create_federated_learning_animation_data(num_rounds: int = 10, num_clients: int = 5) -> Dict[str, Any]:
    """
    Create data for federated learning animation
    
    Args:
        num_rounds: Number of training rounds
        num_clients: Number of client nodes
    
    Returns:
        Animation data dictionary
    """
    animation_data = {
        'rounds': [],
        'global_accuracy': [],
        'client_accuracies': {},
        'communication_events': []
    }
    
    # Initialize client accuracies
    for i in range(num_clients):
        client_id = f"client_{i+1}"
        animation_data['client_accuracies'][client_id] = []
    
    # Generate data for each round
    base_accuracy = 85.0
    for round_num in range(1, num_rounds + 1):
        animation_data['rounds'].append(round_num)
        
        # Calculate global accuracy with convergence
        global_acc = base_accuracy + 10 * (1 - np.exp(-round_num / 5)) + np.random.normal(0, 0.5)
        animation_data['global_accuracy'].append(min(97, max(85, global_acc)))
        
        # Calculate client accuracies
        for client_id in animation_data['client_accuracies']:
            client_variation = np.random.normal(0, 2)
            client_acc = global_acc + client_variation
            animation_data['client_accuracies'][client_id].append(min(98, max(82, client_acc)))
        
        # Generate communication events
        for client_id in animation_data['client_accuracies']:
            animation_data['communication_events'].append({
                'round': round_num,
                'client': client_id,
                'event': 'model_update',
                'timestamp': round_num * 10 + random.randint(0, 5)
            })
    
    return animation_data

def validate_model_weights(weights: List[np.ndarray]) -> Dict[str, Any]:
    """
    Validate model weights for anomalies
    
    Args:
        weights: Model weights list
    
    Returns:
        Validation results dictionary
    """
    validation_results = {
        'is_valid': True,
        'anomalies': [],
        'statistics': {}
    }
    
    try:
        # Calculate weight statistics
        all_weights = np.concatenate([w.flatten() for w in weights])
        
        validation_results['statistics'] = {
            'mean': float(np.mean(all_weights)),
            'std': float(np.std(all_weights)),
            'min': float(np.min(all_weights)),
            'max': float(np.max(all_weights)),
            'total_params': int(len(all_weights))
        }
        
        # Check for anomalies
        mean_val = validation_results['statistics']['mean']
        std_val = validation_results['statistics']['std']
        
        # Check for extreme values
        if abs(mean_val) > 10:
            validation_results['anomalies'].append('extreme_mean')
            validation_results['is_valid'] = False
        
        if std_val > 50:
            validation_results['anomalies'].append('extreme_variance')
            validation_results['is_valid'] = False
        
        # Check for NaN or infinity values
        if np.any(np.isnan(all_weights)) or np.any(np.isinf(all_weights)):
            validation_results['anomalies'].append('invalid_values')
            validation_results['is_valid'] = False
        
    except Exception as e:
        validation_results['is_valid'] = False
        validation_results['anomalies'].append(f'validation_error: {str(e)}')
    
    return validation_results

def create_consensus_timeline(consensus_events: List[Dict[str, Any]]) -> go.Figure:
    """
    Create a timeline visualization for consensus events
    
    Args:
        consensus_events: List of consensus event dictionaries
    
    Returns:
        Plotly figure object
    """
    if not consensus_events:
        fig = go.Figure()
        fig.update_layout(
            title="Consensus Timeline",
            annotations=[
                dict(
                    text="No consensus events to display",
                    x=0.5, y=0.5,
                    xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=16)
                )
            ]
        )
        return fig
    
    # Prepare data for timeline
    timestamps = [event.get('timestamp', '') for event in consensus_events]
    event_types = [event.get('event_type', 'unknown') for event in consensus_events]
    descriptions = [event.get('description', '') for event in consensus_events]
    
    # Create timeline plot
    fig = go.Figure()
    
    # Add events as scatter points
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=event_types,
        mode='markers+text',
        marker=dict(
            size=12,
            color=['green' if 'success' in desc.lower() else 'red' for desc in descriptions]
        ),
        text=descriptions,
        textposition="top center",
        name="Consensus Events"
    ))
    
    fig.update_layout(
        title="Consensus Events Timeline",
        xaxis_title="Timestamp",
        yaxis_title="Event Type",
        hovermode='closest'
    )
    
    return fig
