from dash import Dash, dcc, html, Output, Input, no_update
import plotly.graph_objs as go
import psutil
import GPUtil
from collections import deque

from .device_utils import device_info

def device_monitor():

    devices = device_info()
    
    gpu_available = devices["GPU"]["Available"]
    tpu_available = devices["TPU"]["Available"]

    cpu_usage_buffer = deque(maxlen=20)
    gpu_usage_buffer = deque(maxlen=20) if gpu_available else None
    tpu_usage_buffer = deque(maxlen=20) if tpu_available else None

    stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
    app = Dash(__name__, external_stylesheets=stylesheets)
    app.title = "Harware Monitoring Dashboard"

    app.layout = html.Div([
        html.H1("Hardware Monitoring Dashboard", style={'textAlign': 'center', 'color': '#2F4F4F'}),
        
        html.Div([

            # CPU Monitoring Card
            html.Div([
                html.H2("CPU Usage Over Time", style={'textAlign': 'center', 'color': '#5F9EA0'}),
                dcc.Graph(id='cpu-graph')
            ], className="card", style={'width': '32%', 'padding': '10px'}),

            # GPU Monitoring Card
            html.Div([
                html.H2("GPU Monitoring", style={'textAlign': 'center', 'color': '#5F9EA0'}),
                dcc.Graph(id='gpu-graph') if gpu_available else html.P("GPU not available", style={'textAlign': 'center', 'color': 'red'})
            ], className="card", style={'width': '32%', 'padding': '10px'}),

            # TPU Monitoring Card
            html.Div([
                html.H2("TPU Monitoring", style={'textAlign': 'center', 'color': '#5F9EA0'}),
                dcc.Graph(id='tpu-graph') if tpu_available else html.P("TPU not available", style={'textAlign': 'center', 'color': 'red'})
            ], className="card", style={'width': '32%', 'padding': '10px'}),
            
        ], style={'display': 'flex', 'justify-content': 'center', 'gap': '20px'}),

        dcc.Interval(id='interval-component', interval=2000, n_intervals=0)
    ], style={'fontFamily': 'Arial', 'padding': '20px'})

    outputs = [Output('cpu-graph', 'figure')]
    if gpu_available:
        outputs.append(Output('gpu-graph', 'figure'))
    if tpu_available:
        outputs.append(Output('tpu-graph', 'figure'))

    @app.callback(
        outputs,
        [Input('interval-component', 'n_intervals')]
    )

    def update_graphs(n):
        cpu_usage = psutil.cpu_percent()
        cpu_usage_buffer.append(cpu_usage)

        cpu_fig = go.Figure()
        cpu_fig.add_trace(go.Scatter(
            x=list(range(len(cpu_usage_buffer))),
            y=list(cpu_usage_buffer),
            mode='lines+markers',
            line=dict(color='#4682B4'),
            marker=dict(size=6)
        ))
        cpu_fig.update_layout(
            xaxis_title="Time (2-second intervals)",
            yaxis_title="CPU Usage (%)",
            xaxis=dict(range=[0, len(cpu_usage_buffer) - 1]),
            yaxis=dict(range=[0, 100])
        )

        gpu_fig = no_update
        if gpu_available:
            gpu_data = GPUtil.getGPUs()[0].memoryUtil * 100
            gpu_usage_buffer.append(gpu_data)
            gpu_fig = go.Figure()
            gpu_fig.add_trace(go.Scatter(
                x=list(range(len(gpu_usage_buffer))),
                y=list(gpu_usage_buffer),
                mode='lines+markers',
                line=dict(color='#4682B4'),
                marker=dict(size=6)
            ))
            gpu_fig.update_layout(
                xaxis_title="Time (2-second intervals)",
                yaxis_title="Memory Usage (%)",
                xaxis=dict(range=[0, len(gpu_usage_buffer) - 1]),
                yaxis=dict(range=[0, 100])
            )

        tpu_fig = no_update
        if tpu_available:
            import torch
            import torch_xla.core.xla_model as xm

            tpu_mem_info = xm.get_memory_info(xm.xla_device())
            tpu_data = (tpu_mem_info["kb_used"] / tpu_mem_info["kb_total"]) * 100
            tpu_usage_buffer.append(tpu_data)
            tpu_fig = go.Figure()
            tpu_fig.add_trace(go.Scatter(
                x=list(range(len(tpu_usage_buffer))),
                y=list(tpu_usage_buffer),
                mode='lines+markers',
                line=dict(color='#4682B4'),
                marker=dict(size=6)
            ))
            tpu_fig.update_layout(
                xaxis_title="Time (2-second intervals)",
                yaxis_title="Memory Usage (%)",
                xaxis=dict(range=[0, len(tpu_usage_buffer) - 1]),
                yaxis=dict(range=[0, 100])
            )

        return [cpu_fig] + ([gpu_fig] if gpu_available else []) + ([tpu_fig] if tpu_available else [])

    app.run_server(debug=True)