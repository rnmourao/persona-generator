import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, callback, callback_context, get_asset_url
import pandas as pd
import numpy as np
import os
import random
import numpy as np
from dash.exceptions import PreventUpdate
import openai


def pie_chart(df):
    distribution = df.cluster.value_counts()
    labels = [f'Segment {i}' for i in distribution.index.tolist()]
    values = distribution.values.tolist()
    return {'labels': labels, 'values': values, 'type': 'pie'}


def clusters_combo(df):
    clusters = sorted(df.cluster.unique().tolist())
    ls = []
    for cluster in clusters:
        ls.append({"label": f"Segment {cluster}", "value": cluster})
    return ls


def retrieve_name(country, gender):
    ls = names.loc[(names['Country'] == country) & (names['Gender'] == gender), 'Name'].tolist()
    return random.choice(ls)


def retrieve_bio(name, country, emirate, gender, marital, occupation, income, education, coverage, remarks):

    text = f"""
        Given the data below of the fictional customer:

        Name: {name}
        Home country: {country}
        Currently lives: {emirate}
        Gender: {gender}
        Marital Status: {marital}
        Type of occupation: {occupation}
        Education: {education}
        {remarks}        

        Create a short biography and how the customer relates with our auto {coverage} insurance plan.

        Select one of the most common problems customers have with auto insurance plans and create a paragraph saying that the customer faces that problem.
        
        The text should have less than 200 words.

        Bio:
    """


# 1. Difficulty in understanding the policy terms and coverage options.
# 2. Confusion about the claims process and how to file a claim.
# 3. Premiums that are too high and not affordable.
# 4. Poor customer service and communication from the insurance company.
# 5. Unsatisfactory repair work done by the insurance company's preferred garages or mechanics.
# 6. Insufficient coverage that does not meet the customer's needs or expectations.
# 7. Difficulty in canceling a policy or making changes to coverage.

    # Call the OpenAI API to get the completion suggestions
    completions = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"{text}",
        max_tokens=500,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the completion suggestions from the API response
    return completions.choices[0].text


@callback(
    Output('txt-country', 'value'),
    Output('txt-emirate', 'value'),
    Output('txt-gender', 'value'),
    Output('txt-marital', 'value'),
    Output('txt-occupation', 'value'),
    Output('txt-income', 'value'),
    Output('txt-education', 'value'),
    Output('txt-coverage', 'value'),
    Output('btn-generate-persona', 'disabled'),
    Input('cbo-segments', 'value'),
)
def select_persona_data(segment):

    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate

    k = int(segment)
    temp = desc[k].copy()

    temp = temp.sort_values('support', ascending=False)

    # choosing one of the top subgroups
    chosen = temp.head(5).sample(n=1).to_dict(orient='records')[0]

    # filling blanks
    for k in chosen.keys():
        if chosen[k] is np.nan:
            chosen[k] = temp[k].dropna().sample(1).values[0]

    country = chosen['Country']
    if country == 'Other':
        country = random.choice(
            ['Lebanon', 'Egypt', 'Russia', 'Ukraine', 'Hungary', 'Lithuania', 'Spain'])

    emirate = chosen['Emirate']

    gender = 'Male' if chosen['Gender'] == 'M' else 'Female'

    marital = chosen['Marital Status']

    occupation = chosen['Occupation']

    positive_income = df.loc[df['Income'] > 0, 'Income']
    income = abs(random.gauss(positive_income.mean(), positive_income.std()))
    income = np.round(income, 2)

    education = chosen['Education']

    coverage = chosen['Coverage']

    return country, emirate, gender, marital, occupation, income, education, coverage, False


@callback(
    Output("modal-persona", "is_open"),
    Output('btn-generate-persona', 'n_clicks'),
    Output('btn-close-modal', 'n_clicks'),
    Output('div-persona-profile', 'children'),
    Input('btn-generate-persona', 'n_clicks'),
    Input('btn-close-modal', 'n_clicks'),
    State('txt-country', 'value'),
    State('txt-emirate', 'value'),
    State('txt-gender', 'value'),
    State('txt-marital', 'value'),
    State('txt-occupation', 'value'),
    State('txt-income', 'value'),
    State('txt-education', 'value'),
    State('txt-coverage', 'value'),
    State('txt-remarks', 'value'),
)
def show_persona(open, close, country, emirate, gender, marital, occupation, income, education, coverage, remarks):
    if close > open or (open == 0 and close == 0):
        return False, 0, 0, []

    photo = get_asset_url(f'{country}-{gender}.jpg'.lower())
    name = retrieve_name(country, gender)
    
    text = retrieve_bio(name, country, emirate, gender, marital,
                       occupation, income, education, coverage, remarks)
    text = text.replace('\n\n', '\n')
    text = text.strip('\n')
    rows = text.split('\n')
    bio = []
    for row in rows:
        bio.append(html.P(f"{row}", style={'font-size': '20px'}))
        bio.append(html.Br())
    bio = html.Div(bio)

    persona = [
        dbc.Row([
            dbc.Col([
                    dbc.Card(
                        html.Img(src=photo, width='300em', style={'object-fit': 'contain', 'width': '300px',
                                                                  'height': '300px'}), body=True)
                    ], width=4),

            dbc.Col([

                html.H2(f"{name}"),

                html.H4(f"Country: {country}"),
                html.H4(f"Emirate: {emirate}"),
                
                html.H4(f"Gender: {gender}"),
                html.H4(f"Marital Status: {marital}"),
                
                html.H4(f"Occupation: {occupation}"),
                html.H4(f"Income: {income}"),
                
                html.H4(f"Education: {education}"),

            ]),
            dbc.Col([
                html.H2(f"Bio"),
                bio                
            ])
        ], justify="center")
    ]

    return True, 0, 0, persona

# Initialize the OpenAI API client
openai.api_key = ""

# datasets
df = pd.read_csv("data/clustered.csv")

desc = dict()
for file in os.listdir('data/'):
    if 'cluster_' in file:
        n = int(file.split('_')[1].split('.')[0])
        desc[n] = pd.read_csv(f'data/{file}')

names = pd.read_csv('data/names.csv')


# modal layout
modal = dbc.Modal([
    dbc.ModalHeader(dbc.ModalTitle("Persona Profile")),
    dbc.ModalBody(html.Div([
        dbc.Row([html.Div(id='div-persona-profile')]),
        dbc.Row([
                dbc.Button(
                    "Close",
                    id="btn-close-modal",
                    color="primary",
                    className="me-1",
                    n_clicks=0
                ),
                ], className='row mt-3'),]))
], id='modal-persona', is_open=False,
    size="xl")

# page layout
layout = html.Div([
    dbc.Row([
        html.H1(f"Customer Segment Analysis")
    ], className="row mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        html.H2(f"Customer Segments")
                    ]),
                    dbc.Row([
                        dcc.Graph(
                            id='pie-chart',
                            figure={
                                'data': [pie_chart(df)],
                            }
                        ),
                    ], className="row mt-3 mb-4"),
                ]))
        ]),
        dbc.Col([
            dbc.Card(
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([html.H2(f"Generate a Persona")]),
                        dbc.Col([
                            dbc.Button("Go!",
                                       id="btn-generate-persona",
                                       color="success",
                                       className="me-1",
                                       disabled=True,
                                       n_clicks=0),
                        ], width="auto")
                    ]),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Select a segment"),
                            dbc.Select(
                                id="cbo-segments",
                                options=clusters_combo(df),
                                value=0,
                            )
                        ]),
                    ], className="row mt-3 mb-3"),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Country"),
                            dbc.Input(
                                id='txt-country', type="text", disabled=True),
                        ]),
                        dbc.Col([
                            dbc.Label("Emirate"),
                            dbc.Input(
                                id='txt-emirate', type="text", disabled=True),
                        ]),
                    ]),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Gender"),
                            dbc.Input(
                                id='txt-gender', type="text", disabled=True),
                        ]),
                        dbc.Col([
                            dbc.Label("Marital Status"),
                            dbc.Input(
                                id='txt-marital', type="text", disabled=True),
                        ]),
                    ]),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Occupation"),
                            dbc.Input(
                                id='txt-occupation', type="text", disabled=True),
                        ]),
                        dbc.Col([
                            dbc.Label("Income"),
                            dbc.Input(
                                id='txt-income', type="number", disabled=True),
                        ]),
                    ]),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Education"),
                            dbc.Input(
                                id='txt-education', type="text", disabled=True),
                        ]),
                        dbc.Col([
                            dbc.Label("Coverage"),
                            dbc.Input(
                                id='txt-coverage', type="text", disabled=True),
                        ]),
                    ]),

                    dbc.Row([
                        dbc.Textarea(
                            className="mb-3", placeholder="Add any other relevant information"),
                    ],
                        className="row mt-3", id='txt-remarks'),

                ]))
        ]),
    ]),
    modal
])
