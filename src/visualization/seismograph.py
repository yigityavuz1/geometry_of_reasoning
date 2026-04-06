from __future__ import annotations

from typing import Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def build_seismograph(
    step_indices: Sequence[int],
    lid_values: Sequence[float],
    pr_values: Sequence[float],
    entropy_values: Sequence[float],
    warning_scores: Sequence[float] | None = None,
    warning_threshold: float | None = None,
    warning_steps: Sequence[int] | None = None,
    alarm_step: int | None = None,
    incorrect_steps: Sequence[int] | None = None,
    parse_fail_steps: Sequence[int] | None = None,
    step_texts: Sequence[str] | None = None,
    final_correct: bool | None = None,
    title: str = "Reasoning Seismograph",
) -> go.Figure:
    customdata = list(step_texts) if step_texts is not None else None

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.62, 0.38],
        subplot_titles=("Geometric/Entropy Signals", "Warning Score"),
    )

    fig.add_trace(
        go.Scatter(
            x=step_indices,
            y=lid_values,
            mode="lines+markers",
            name="LID",
            customdata=customdata,
            hovertemplate="Step %{x}<br>LID=%{y:.3f}<br>%{customdata}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=step_indices,
            y=pr_values,
            mode="lines+markers",
            name="PR",
            customdata=customdata,
            hovertemplate="Step %{x}<br>PR=%{y:.3f}<br>%{customdata}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=step_indices,
            y=entropy_values,
            mode="lines+markers",
            name="Entropy",
            customdata=customdata,
            hovertemplate="Step %{x}<br>Entropy=%{y:.3f}<br>%{customdata}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if warning_scores is not None:
        fig.add_trace(
            go.Scatter(
                x=step_indices,
                y=warning_scores,
                mode="lines+markers",
                name="Warning Score",
                line=dict(color="#d62728"),
                customdata=customdata,
                hovertemplate="Step %{x}<br>Warning=%{y:.3f}<br>%{customdata}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    if warning_threshold is not None:
        fig.add_hline(
            y=float(warning_threshold),
            line_width=1,
            line_dash="dot",
            line_color="#d62728",
            row=2,
            col=1,
        )

    for ws in warning_steps or []:
        fig.add_vline(x=ws, line_width=1, line_dash="dash", line_color="#ff7f0e")
    if alarm_step is not None:
        fig.add_vline(x=alarm_step, line_width=2, line_dash="solid", line_color="red")

    if incorrect_steps:
        incorrect_x = list(incorrect_steps)
        incorrect_y = [lid_values[step_indices.index(step)] for step in incorrect_x if step in step_indices]
        if incorrect_y:
            fig.add_trace(
                go.Scatter(
                    x=incorrect_x,
                    y=incorrect_y,
                    mode="markers",
                    name="Incorrect Step",
                    marker=dict(symbol="x", size=10, color="black"),
                ),
                row=1,
                col=1,
            )
    if parse_fail_steps and warning_scores is not None:
        parse_x = list(parse_fail_steps)
        parse_y = [warning_scores[step_indices.index(step)] for step in parse_x if step in step_indices]
        if parse_y:
            fig.add_trace(
                go.Scatter(
                    x=parse_x,
                    y=parse_y,
                    mode="markers",
                    name="Parse Fail",
                    marker=dict(symbol="diamond", size=9, color="#9467bd"),
                ),
                row=2,
                col=1,
            )

    title_text = title
    if final_correct is not None:
        verdict = "final_correct" if final_correct else "final_failure"
        title_text = f"{title} ({verdict})"

    fig.update_layout(
        title=title_text,
        xaxis2_title="Step Index",
        yaxis_title="Signal Value",
        yaxis2_title="Warning Score",
        template="plotly_white",
        legend=dict(orientation="h"),
    )
    return fig
