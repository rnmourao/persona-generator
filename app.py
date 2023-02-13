import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, callback
from pages import content, analysis


app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.SKETCHY])
app.title = "Customer Segment Analysis"

@callback(Output("page-content", "children"), 
         Input("url", "pathname"),
        )
def render_page_content(pathname):
    if pathname == "/":
        return analysis.layout
    # If the user tries to reach a different page, return a 404 message
    return content.layout_404


app.layout = html.Div([dcc.Location(id="url"), 
                       content.layout,
                       ])


if __name__ == "__main__":
    app.run_server(port=5010, debug=True)