from dash import html, callback, Input, Output
import dash_bootstrap_components as dbc


@callback(
    Output("url", "pathname"),
    Input("home", "n_clicks")
)
def go_home(clicks):
    if clicks:
        return "/"


layout = html.Div(id="page-content",
                  style={
                      "margin-left": "2rem",
                      "margin-right": "2rem",
                      "padding": "2rem 1rem",
                  })


layout_404 = html.Div([
                        html.H1(f"404 - Ooops...", className="text-danger"),
                        dbc.Row(children=[
                                            dbc.Col([
                                                        html.H2(f"This isn't what you're looking for."),
                                                    ]),
                                            dbc.Col([
                                                        dbc.Button("Take me Home",          
                                                                    id="home", 
                                                                    color="primary", 
                                                                    className="me-1",
                                                                    n_clicks=0),
                                                    ], 
                                                    width="auto"),
                                        ],
                                        className="row mt-3", justify="end"),
                        html.Hr(),
                        html.Div([
                                    html.Img(src="assets/arab-4wheeler.gif", 
                                             width="75%", 
                                             height="75%"
                                            ),
                                ], 
                                style={"textAlign": "center"}
                                ),
                        html.Hr(),
                    ],)