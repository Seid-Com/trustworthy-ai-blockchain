"""
High-quality chart configuration for publication-ready visualizations
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Set default template for all charts
pio.templates.default = "plotly_white"

# Publication quality settings
PUBLICATION_CONFIG = {
    'width': 1000,
    'height': 700,
    'dpi': 300,
    'font_family': 'Arial, sans-serif',
    'font_size': 16,
    'title_font_size': 20,
    'axis_font_size': 16,
    'legend_font_size': 14,
    'line_width': 3,
    'marker_size': 10,
    'plot_bgcolor': 'white',
    'paper_bgcolor': 'white',
    'grid_color': '#E6E6E6',
    'axis_line_color': '#000000',
    'text_color': '#000000',
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
}

def get_base_layout(title: str = "", xaxis_title: str = "", yaxis_title: str = ""):
    """Get base layout configuration for high-quality charts"""
    return go.Layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(
                family=PUBLICATION_CONFIG['font_family'],
                size=PUBLICATION_CONFIG['title_font_size'],
                color=PUBLICATION_CONFIG['text_color']
            )
        ),
        xaxis=dict(
            title=dict(
                text=xaxis_title,
                font=dict(
                    family=PUBLICATION_CONFIG['font_family'],
                    size=PUBLICATION_CONFIG['axis_font_size'],
                    color='#333333'
                )
            ),
            tickfont=dict(
                family=PUBLICATION_CONFIG['font_family'],
                size=PUBLICATION_CONFIG['axis_font_size'],
                color='#333333'
            ),
            showgrid=True,
            gridwidth=1,
            gridcolor=PUBLICATION_CONFIG['grid_color'],
            showline=True,
            linewidth=1,
            linecolor=PUBLICATION_CONFIG['axis_line_color'],
            mirror=True
        ),
        yaxis=dict(
            title=dict(
                text=yaxis_title,
                font=dict(
                    family=PUBLICATION_CONFIG['font_family'],
                    size=PUBLICATION_CONFIG['axis_font_size'],
                    color='#333333'
                )
            ),
            tickfont=dict(
                family=PUBLICATION_CONFIG['font_family'],
                size=PUBLICATION_CONFIG['axis_font_size'],
                color='#333333'
            ),
            showgrid=True,
            gridwidth=1,
            gridcolor=PUBLICATION_CONFIG['grid_color'],
            showline=True,
            linewidth=1,
            linecolor=PUBLICATION_CONFIG['axis_line_color'],
            mirror=True
        ),
        plot_bgcolor=PUBLICATION_CONFIG['plot_bgcolor'],
        paper_bgcolor=PUBLICATION_CONFIG['paper_bgcolor'],
        font=dict(
            family=PUBLICATION_CONFIG['font_family'],
            size=PUBLICATION_CONFIG['font_size'],
            color='#333333'
        ),
        legend=dict(
            font=dict(
                family=PUBLICATION_CONFIG['font_family'],
                size=PUBLICATION_CONFIG['legend_font_size'],
                color='#333333'
            ),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#E6E6E6',
            borderwidth=1
        ),
        width=PUBLICATION_CONFIG['width'],
        height=PUBLICATION_CONFIG['height'],
        margin=dict(l=80, r=80, t=100, b=80)
    )

def create_enhanced_bar_chart(x_data, y_data, names, title, xaxis_title, yaxis_title, colors=None):
    """Create an enhanced bar chart with publication quality"""
    fig = go.Figure()
    
    if colors is None:
        colors = PUBLICATION_CONFIG['color_palette']
    
    for i, (name, y_vals) in enumerate(zip(names, y_data)):
        fig.add_trace(go.Bar(
            x=x_data,
            y=y_vals,
            name=name,
            marker_color=colors[i % len(colors)],
            marker_line=dict(width=1, color='#333333'),
            opacity=0.8
        ))
    
    fig.update_layout(get_base_layout(title, xaxis_title, yaxis_title))
    fig.update_layout(barmode='group')
    
    return fig

def create_enhanced_line_chart(x_data, y_data, names, title, xaxis_title, yaxis_title, colors=None):
    """Create an enhanced line chart with publication quality"""
    fig = go.Figure()
    
    if colors is None:
        colors = PUBLICATION_CONFIG['color_palette']
    
    for i, (name, y_vals) in enumerate(zip(names, y_data)):
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_vals,
            mode='lines+markers',
            name=name,
            line=dict(
                color=colors[i % len(colors)],
                width=PUBLICATION_CONFIG['line_width']
            ),
            marker=dict(
                size=PUBLICATION_CONFIG['marker_size'],
                color=colors[i % len(colors)],
                line=dict(width=1, color='white')
            )
        ))
    
    fig.update_layout(get_base_layout(title, xaxis_title, yaxis_title))
    
    return fig

def create_enhanced_scatter_chart(x_data, y_data, names, title, xaxis_title, yaxis_title, colors=None):
    """Create an enhanced scatter chart with publication quality"""
    fig = go.Figure()
    
    if colors is None:
        colors = PUBLICATION_CONFIG['color_palette']
    
    for i, (name, x_vals, y_vals) in enumerate(zip(names, x_data, y_data)):
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            name=name,
            marker=dict(
                size=PUBLICATION_CONFIG['marker_size'],
                color=colors[i % len(colors)],
                line=dict(width=1, color='white'),
                opacity=0.8
            )
        ))
    
    fig.update_layout(get_base_layout(title, xaxis_title, yaxis_title))
    
    return fig

def create_enhanced_heatmap(z_data, x_labels, y_labels, title, colorscale='RdYlBu'):
    """Create an enhanced heatmap with publication quality"""
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Value",
                font=dict(
                    family=PUBLICATION_CONFIG['font_family'],
                    size=PUBLICATION_CONFIG['axis_font_size']
                )
            )
        )
    ))
    
    fig.update_layout(get_base_layout(title))
    
    return fig

def apply_high_quality_config(fig):
    """Apply high-quality configuration to existing figure"""
    fig.update_layout(
        font=dict(
            family=PUBLICATION_CONFIG['font_family'],
            size=PUBLICATION_CONFIG['font_size'],
            color='#333333'
        ),
        plot_bgcolor=PUBLICATION_CONFIG['plot_bgcolor'],
        paper_bgcolor=PUBLICATION_CONFIG['paper_bgcolor'],
        width=PUBLICATION_CONFIG['width'],
        height=PUBLICATION_CONFIG['height']
    )
    
    # Update all traces for better quality
    for trace in fig.data:
        if hasattr(trace, 'line') and trace.line:
            trace.line.width = PUBLICATION_CONFIG['line_width']
        if hasattr(trace, 'marker') and trace.marker:
            trace.marker.size = PUBLICATION_CONFIG['marker_size']
    
    return fig

def export_high_quality_image(fig, filename, format='png', width=1400, height=900):
    """Export figure as high-quality image"""
    fig.update_layout(
        width=width,
        height=height,
        font=dict(size=16)
    )
    
    return fig.to_image(
        format=format,
        width=width,
        height=height,
        scale=2  # Double resolution for high quality
    )