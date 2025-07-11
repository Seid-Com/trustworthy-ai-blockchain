import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any
import base64
from io import BytesIO
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

def create_high_quality_metrics_table(comparison_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a high-quality formatted metrics comparison table
    """
    # Restructure the data for better presentation
    metrics_table = {
        'Metric': [
            'Final Model Accuracy (%)',
            'Convergence Speed (rounds)', 
            'Communication Overhead (%)',
            'Energy Consumption (%)',
            'Security Incidents',
            'Anomaly Detection Speed (s)',
            'System Robustness (/10)',
            'Data Integrity (%)',
            'Attack Success Rate (%)',
            'Privacy Preservation Score (/10)'
        ],
        'Baseline Federated Learning': [
            97.1, 45, 0.0, 100.0, 8, 5.2, 7.5, 94.0, 24.0, 6.8
        ],
        'Blockchain-Integrated System': [
            97.5, 42, 6.0, 108.0, 2, 4.3, 8.8, 99.9, 6.0, 9.2
        ],
        'Improvement': [
            '+0.4%', '-3 rounds', '+6.0%', '+8.0%', '-6 incidents', 
            '-0.9s (18% faster)', '+1.3/10', '+5.9%', '-18% (75% reduction)', '+2.4/10'
        ]
    }
    
    return pd.DataFrame(metrics_table)

def create_publication_quality_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a publication-quality comparison chart
    """
    # Create subplots for different metric categories
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Model Performance', 'Security Metrics', 
            'Efficiency Metrics', 'System Quality'
        ),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Performance metrics
    performance_metrics = ['Final Model Accuracy (%)', 'Convergence Speed (rounds)']
    performance_baseline = [97.1, 45]
    performance_blockchain = [97.5, 42]
    
    fig.add_trace(
        go.Bar(
            name='Baseline FL',
            x=performance_metrics,
            y=performance_baseline,
            marker_color='#FF6B6B',
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            name='Blockchain FL',
            x=performance_metrics,
            y=performance_blockchain,
            marker_color='#4ECDC4',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Security metrics
    security_metrics = ['Security Incidents', 'Attack Success Rate (%)']
    security_baseline = [8, 24.0]
    security_blockchain = [2, 6.0]
    
    fig.add_trace(
        go.Bar(
            name='Baseline FL',
            x=security_metrics,
            y=security_baseline,
            marker_color='#FF6B6B',
            showlegend=False
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(
            name='Blockchain FL',
            x=security_metrics,
            y=security_blockchain,
            marker_color='#4ECDC4',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Efficiency metrics
    efficiency_metrics = ['Communication Overhead (%)', 'Energy Consumption (%)']
    efficiency_baseline = [0.0, 100.0]
    efficiency_blockchain = [6.0, 108.0]
    
    fig.add_trace(
        go.Bar(
            name='Baseline FL',
            x=efficiency_metrics,
            y=efficiency_baseline,
            marker_color='#FF6B6B',
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(
            name='Blockchain FL',
            x=efficiency_metrics,
            y=efficiency_blockchain,
            marker_color='#4ECDC4',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # System quality metrics
    quality_metrics = ['Data Integrity (%)', 'System Robustness (/10)']
    quality_baseline = [94.0, 7.5]
    quality_blockchain = [99.9, 8.8]
    
    fig.add_trace(
        go.Bar(
            name='Baseline FL',
            x=quality_metrics,
            y=quality_baseline,
            marker_color='#FF6B6B',
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Bar(
            name='Blockchain FL',
            x=quality_metrics,
            y=quality_blockchain,
            marker_color='#4ECDC4',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout for publication quality
    fig.update_layout(
        title={
            'text': 'Blockchain-Integrated vs Baseline Federated Learning: Performance Comparison',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'family': 'Arial, sans-serif'}
        },
        font={'family': 'Arial, sans-serif', 'size': 12},
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600,
        width=1000,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update all subplot axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(
                showgrid=False,
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
                row=i, col=j
            )
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                showline=True,
                linewidth=1,
                linecolor='black',
                mirror=True,
                row=i, col=j
            )
    
    return fig

def export_to_csv(df: pd.DataFrame) -> bytes:
    """
    Export DataFrame to CSV format
    """
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def export_to_excel(df: pd.DataFrame) -> bytes:
    """
    Export DataFrame to Excel format with formatting
    """
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Metrics_Comparison', index=False)
        
        # Get the xlsxwriter workbook and worksheet objects
        workbook = writer.book
        worksheet = writer.sheets['Metrics_Comparison']
        
        # Add formatting
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'valign': 'top',
            'fg_color': '#D7E4BC',
            'border': 1
        })
        
        # Apply header formatting
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num, value, header_format)
        
        # Auto-adjust column widths
        for i, col in enumerate(df.columns):
            column_len = max(df[col].astype(str).str.len().max(), len(col) + 2)
            worksheet.set_column(i, i, column_len)
    
    return excel_buffer.getvalue()

def export_chart_to_image(fig: go.Figure, format: str = 'png', width: int = 1200, height: int = 800) -> bytes:
    """
    Export plotly figure to high-quality image
    Note: Chrome dependency disabled for Replit compatibility
    """
    # Enhance figure for better export quality
    fig.update_layout(
        width=width,
        height=height,
        font={'size': 16, 'family': 'Arial, sans-serif'},
        title={'font': {'size': 20}},
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=80, r=80, t=100, b=80)
    )
    
    # Make text more visible
    fig.update_xaxes(
        tickfont={'size': 14, 'color': 'black'},
        title_font={'size': 16, 'color': 'black'},
        showgrid=True,
        gridcolor='lightgray',
        linecolor='black',
        linewidth=2
    )
    
    fig.update_yaxes(
        tickfont={'size': 14, 'color': 'black'},
        title_font={'size': 16, 'color': 'black'},
        showgrid=True,
        gridcolor='lightgray',
        linecolor='black',
        linewidth=2
    )
    
    try:
        if format.lower() == 'svg':
            # SVG export doesn't require Chrome
            return fig.to_image(format='svg', width=width, height=height)
        elif format.lower() == 'html':
            # HTML export as fallback
            return fig.to_html(include_plotlyjs='cdn').encode('utf-8')
        else:
            # Try PNG/PDF but fall back gracefully
            return fig.to_image(format='png', width=width, height=height, scale=2)
    except Exception as e:
        # Return empty bytes if image export fails (Chrome dependency issue)
        return b''

def export_chart_to_svg(fig: go.Figure, width: int = 1200, height: int = 800) -> str:
    """
    Export plotly figure to SVG format (Chrome-free)
    """
    fig.update_layout(
        width=width,
        height=height,
        font={'size': 14}
    )
    
    try:
        return fig.to_image(format='svg', width=width, height=height).decode('utf-8')
    except Exception as e:
        return f"<svg><text>SVG export failed: {str(e)}</text></svg>"

def create_latex_table(df: pd.DataFrame) -> str:
    """
    Create LaTeX table code for academic papers
    """
    latex_code = """
\\begin{table}[htbp]
\\centering
\\caption{Performance Comparison: Baseline vs Blockchain-Integrated Federated Learning}
\\label{tab:performance_comparison}
\\begin{tabular}{|l|c|c|c|}
\\hline
\\textbf{Metric} & \\textbf{Baseline FL} & \\textbf{Blockchain FL} & \\textbf{Improvement} \\\\
\\hline
"""
    
    for _, row in df.iterrows():
        latex_code += f"{row['Metric']} & {row['Baseline Federated Learning']} & {row['Blockchain-Integrated System']} & {row['Improvement']} \\\\\n"
        latex_code += "\\hline\n"
    
    latex_code += """\\end{tabular}
\\end{table}
"""
    
    return latex_code

def create_summary_statistics() -> Dict[str, Any]:
    """
    Create summary statistics for the comparison
    """
    return {
        'key_improvements': {
            'model_accuracy': '+0.4%',
            'security_incidents': '-75%',
            'attack_success_rate': '-75%',
            'detection_speed': '+18%',
            'data_integrity': '+5.9%'
        },
        'acceptable_overheads': {
            'communication': '+6.0%',
            'energy': '+8.0%'
        },
        'overall_assessment': {
            'security_enhancement': 'Significant',
            'performance_impact': 'Minimal',
            'reliability_improvement': 'Substantial'
        }
    }