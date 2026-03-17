import os
import uuid
import threading
import http.server
import functools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from mcp.server.fastmcp import FastMCP

# Create the FastMCP server
mcp = FastMCP(
    name="data-visualization",
    host="0.0.0.0",
    port=8003
)

# Register the tool with the FastMCP server
@mcp.tool()
def visualize_data(
    inputs: dict
) -> str:
    """
    Plot one or more time series with optional forecasts and return
    the URL of the interactive HTML chart.

    Parameters
    ===============================================================================
    inputs : dict
        A dictionary with the following keys:

        "data" (required) : dict
            A dictionary where each key is a series name and each value is the
            raw output of a ClickHouse query, with the following fields:
                - "columns" : list of strings, must contain "timestamp" and one value column
                - "rows"    : list of [timestamp_str, float] pairs

            Example:
            {
                "series_1": {
                    "columns": ["timestamp", "<VALUE>"],
                    "rows": [
                        ["2026-01-01", 1.0],
                        ["2026-01-02", 2.0]
                    ]
                },
                "series_2": {
                    "columns": ["timestamp", "<VALUE>"],
                    "rows": [
                        ["2026-01-03", 3.0],
                        ["2026-01-04", 4.0]
                    ]
                },
            }

        "forecasts" (optional) : dict
            Forecasts for the same time series in "data". Each key is a series
            name matching a key in "data", and each value is a dictionary with
            the following fields:
                - "timestamp"  : list of strings representing datetimes
                - "mean"       : list of floats (mean forecast)
                - "<quantile>" : list of floats for each quantile level, e.g.
                                 "0.05" and "0.95" for a 90% prediction interval.

            Example:
            {
                "series_1": {
                    "timestamp": ["2026-01-01", "2026-01-02"],
                    "mean": [1.0, 2.0],
                    "0.1": [0.5, 1.5],
                    "0.5": [1.0, 2.0],
                    "0.9": [1.5, 2.5],
                },
                "series_2": {
                    "timestamp": ["2026-01-03", "2026-01-04"],
                    "mean": [3.0, 4.0],
                    "0.1": [2.5, 3.5],
                    "0.5": [3.0, 4.0],
                    "0.9": [3.5, 4.5],
                },
            }

    Returns
    ===============================================================================
    str
        The URL of the interactive HTML chart.
    """
    # Extract the data and forecasts
    data = inputs["data"]
    forecasts = inputs.get("forecasts", {})
    
    # Parse the data
    parsed_data = {}
    for series, query_result in data.items():
        value_col = [c for c in query_result["columns"] if c != "timestamp"][0]
        idx = query_result["columns"].index
        parsed_data[series] = {
            "timestamp": [row[idx("timestamp")] for row in query_result["rows"]],
            "values": [row[idx(value_col)] for row in query_result["rows"]]
        }
    
    # Create the figure
    fig = make_subplots(
        rows=len(data),
        subplot_titles=list(data.keys())
    )
    
    # Update the figure layout
    fig.update_layout(
        height=250 * len(data),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(t=50, b=50, l=50, r=50),
        hovermode="x unified",
        hoverlabel=dict(
            namelength=-1
        ),
        legend=dict(
            font=dict(
                color="#24292f",
                size=12
            ),
        )
    )
    
    # Update the subplots titles
    fig.update_annotations(
        font=dict(
            color="#24292f",
            size=14
        ),
    )
    
    # Generate the subplots
    for i, series in enumerate(data):
        # Plot the forecasts
        if series in forecasts:
            # Extract the predicted quantiles
            q = sorted([float(k) for k in forecasts[series] if k not in ("mean", "timestamp")])
            
            # Extract the lower and upper bound of the prediction interval
            q_min, q_max = q[0], q[-1]
            
            # Plot the upper bound of the prediction interval
            fig.add_trace(
                go.Scatter(
                    x=forecasts[series]["timestamp"],
                    y=forecasts[series][str(q_max)],
                    name=f"Predicted Q{q_max:,.1%}",
                    hovertemplate="%{fullData.name}: %{y:,.0f}<extra></extra>",
                    showlegend=False,
                    mode="lines",
                    line=dict(
                        width=0.5,
                        color="#c2e5ff",
                    ),
                ),
                row=i + 1,
                col=1
            )
            
            # Plot the lower bound of the prediction interval
            fig.add_trace(
                go.Scatter(
                    x=forecasts[series]["timestamp"],
                    y=forecasts[series][str(q_min)],
                    name=f"Predicted Q{q_min:,.1%}",
                    hovertemplate="%{fullData.name}: %{y:,.0f}<extra></extra>",
                    showlegend=False,
                    mode="lines",
                    line=dict(
                        width=0.5,
                        color="#c2e5ff",
                    ),
                    fillcolor="#c2e5ff",
                    fill="tonexty",
                ),
                row=i + 1,
                col=1
            )
            
            # Plot the predicted median if available, otherwise fall back to the predicted mean
            fig.add_trace(
                go.Scatter(
                    x=forecasts[series]["timestamp"],
                    y=forecasts[series]["0.5" if 0.5 in q else "mean"],
                    name=f"Predicted {'Median' if 0.5 in q else 'Mean'}",
                    hovertemplate="%{fullData.name}: %{y:,.0f}<extra></extra>",
                    showlegend=i == 0,
                    mode="lines",
                    line=dict(
                        color="#0588f0",
                        width=1,
                        dash="dot"
                    )
                ),
                row=i + 1,
                col=1
            )
        
        # Plot the data
        fig.add_trace(
            go.Scatter(
                x=parsed_data[series]["timestamp"],
                y=parsed_data[series]["values"],
                name="Historical Data",
                hovertemplate="%{fullData.name}: %{y:,.0f}<extra></extra>",
                mode="lines",
                showlegend=i == 0,
                line=dict(
                    color="#838383",
                    width=1
                )
            ),
            row=i + 1,
            col=1
        )
        
        # Update the subplot's x-axis
        fig.update_xaxes(
            type="date",
            tickformat="%b %d %Y<br>(%a) %H:%M",
            tickangle=0,
            mirror=True,
            linecolor="#cecece",
            gridcolor="#e8e8e8",
            gridwidth=0.5,
            tickfont=dict(
                color="#24292f",
                size=10
            ),
            row=i + 1,
            col=1
        )
        
        # Update the subplot's y-axis
        fig.update_yaxes(
            tickformat=",.0f",
            mirror=True,
            linecolor="#cecece",
            gridcolor="#e8e8e8",
            gridwidth=0.5,
            tickfont=dict(
                color="#24292f",
                size=10
            ),
            row=i + 1,
            col=1
        )
    
    # Save the figure to an HTML file
    os.makedirs("/plots", exist_ok=True)
    filename = f"plot_{uuid.uuid4().hex}.html"
    fig.write_html(f"/plots/{filename}", full_html=True, include_plotlyjs="cdn")
    
    # Return the URL of the HTML file
    return f"http://localhost:8004/{filename}"


if __name__ == "__main__":
    # Serve the /plots directory over HTTP on port 8004
    def _serve_plots():
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory="/plots")
        with http.server.HTTPServer(("0.0.0.0", 8004), handler) as httpd:
            httpd.serve_forever()
    
    
    # Start the HTTP server
    threading.Thread(target=_serve_plots, daemon=True).start()
    
    # Run the FastMCP server with SSE transport
    mcp.run(transport="sse")
