from __future__ import annotations

from typing import Sequence

import plotly.graph_objects as go


def build_seismograph(
    step_indices: Sequence[int],
    lid_values: Sequence[float],
    pr_values: Sequence[float],
    entropy_values: Sequence[float],
    warning_steps: Sequence[int] | None = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=step_indices, y=lid_values, mode="lines+markers", name="LID"))
    fig.add_trace(go.Scatter(x=step_indices, y=pr_values, mode="lines+markers", name="PR"))
    fig.add_trace(
        go.Scatter(x=step_indices, y=entropy_values, mode="lines+markers", name="Entropy")
    )

    for ws in warning_steps or []:
        fig.add_vline(x=ws, line_width=1, line_dash="dash", line_color="red")

    fig.update_layout(
        title="Reasoning Seismograph",
        xaxis_title="Step Index",
        yaxis_title="Signal Value",
        template="plotly_white",
    )
    return fig
