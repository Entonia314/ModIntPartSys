from flask import Flask
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output, State
from scipy.optimize import fsolve
import SupplementaryFiles.dash_reusable_components as drc

t_start = 0
mass = [1, 1, 1]
g = 1


# Auxiliary function to multiply list with scalar
def mult(vector, scalar):
    newvector = [0] * len(vector)
    for i in range(len(vector)):
        newvector[i] = vector[i] * scalar
    return newvector


# Initial conditions for simplified models

[x0, y0], [v0, w0] = [-0.0347, 1.1856], [0.2495, -0.1076]
[x1, y1], [v1, w1] = [0.2693, -1.0020], [0.2059, -0.9396]
[x2, y2], [v2, w2] = [-0.2328, -0.5978], [-0.4553, 1.0471]
inits4 = np.array([[[x0, y0], [v0, w0]], [[x1, y1], [v1, w1]], [[x2, y2], [v2, w2]]])

inits1 = np.array([[[-0.602885898116520, 1.059162128863347 - 1], [0.122913546623784, 0.747443868604908]],
                   [[0.252709795391000, 1.058254872224370 - 1], [-0.019325586404545, 1.369241993562101]],
                   [[-0.355389016941814, 1.038323764315145 - 1], [-0.103587960218793, -2.116685862168820]]])

inits5 = np.array([[[0.716248295712871, 0.384288553041130], [1.245268230895990, 2.444311951776573]],
                   [[0.086172594591232, 1.342795868576616], [-0.675224323690062, -0.962879613630031]],
                   [[0.538777980807643, 0.481049882655556], [-0.570043907205925, -1.481432338146543]]])

r = 0.5
inits2 = np.array([[[1, 0], mult([0, 1], r)],
                   [[-0.5, 3 ** (1 / 2) / 2], mult([-3 ** (1 / 2) / 2, -0.5], r)],
                   [[-0.5, -3 ** (1 / 2) / 2], mult([3 ** (1 / 2) / 2, -0.5], r)]])

p51 = 0.347111
p52 = 0.532728
inits3 = np.array([[[-1, 0], [p51, p52]],
                   [[1, 0], [p51, p52]],
                   [[0, 0], [-2 * p51, -2 * p52]]])

p61 = 0.464445
p62 = 0.396060
inits6 = np.array([[[-1, 0], [p61, p62]],
                   [[1, 0], [p61, p62]],
                   [[0, 0], [-2 * p61, -2 * p62]]])

p71 = 0.080584
p72 = 0.588836
inits7 = np.array([[[-1, 0], [p71, p72]],
                   [[1, 0], [p71, p72]],
                   [[0, 0], [-2 * p71, -2 * p72]]])

# Initial conditions for real life models

Mear = 6e24  # Mass of Earth in kg
Msun = 2e30  # Mass of Sun in kg
Mjup = 1.9e27  # Mass of Jupiter
Msat = 5.7e26  # Mass of Saturn
Mmoo = 7.346e22  # Mass of Earth-Moon
Mven = 4.875e24  # Mass of Venus
Mnep = 1.024e26  # Mass of Neptun
Miss = 419725  # Mass of ISS

G = 6.673e-11  # Gravitational Constant

RR = 1.496e11  # Normalizing distance in km (= 1 AU)
MM = 6e24  # Normalizing mass (mass of earth)
TT = 365 * 24 * 60 * 60.0  # Normalizing time (1 year)

FF = (G * MM ** 2) / RR ** 2  # Unit force
EE = FF * RR  # Unit energy

GG = (MM * G * TT ** 2) / (RR ** 3)

Mear = Mear / MM  # Normalized mass of Earth
Msun = Msun / MM  # Normalized mass of Sun
Mjup = 500 * Mjup / MM  # Normalized mass of Jupiter/Super Jupiter
Msat = Msat / MM  # Normalized mass of Saturn
Mmoo = Mmoo / MM  # Normalized mass of Moon
Mven = Mven / MM  # Normalized mass of Venus
Mnep = Mnep / MM
Miss = Miss / MM

# Initialization

rear = [1, 0]  # initial position of earth
rjup = [5.2, 0]  # initial position of Jupiter
rsat = [9.582, 0]
rsun = [0, 0]
rmoo = [rear[0] + 0.00257, 0]
rven = [0.723, 0]
rnep = [30.047, 0]
riss = [rear[0] - 420e3 / RR, 0]

magear = np.sqrt(Msun * GG / rear[0])  # Magnitude of Earth's initial velocity
magjup = 13.06e3 * TT / RR  # Magnitude of Jupiter's initial velocity
magsat = 9.68e3 * TT / RR  # Magnitude of Saturn's initial velocity
magmoo = 1.022e3 * TT / RR  # Magnitude of Moon's initial velocity
magven = 35.02e3 * TT / RR  # Magnitude of Venus' initial velocity
magnep = 5.43e3 * TT / RR
magiss = 7.66e3 * TT / RR

vear = [0, magear * 1.0]  # Initial velocity vector for Earth.Taken to be along y direction as ri is on x axis.
vjup = [0, magjup * 1.0]  # Initial velocity vector for Jupiter
vsat = [0, magsat * 1.0]  # Initial velocity vector for Saturn
vsun = [0, 0]
vmoo = [0, magmoo]  # Initial velocity vector for Moon
vven = [0, magven]  # Initial velocity vector for Venus
vnep = [0, magnep]
viss = [0, magiss]

method_dict = {'forward_euler': 'Explizites Euler-Verfahren', 'backward_euler': 'Implizites Euler-Verfahren'}
init_dict = {1: inits1, 2: inits2, 3: inits3, 4: inits4, 5: inits5, 6: inits6, 7: inits7}
mass_dict = {'sun': Msun, 'sat': Msat, 'jup': Mjup, 'ear': Mear, 'moo': Mmoo, 'ven': Mven, 'nep': Mnep, 'iss': Miss}
r_dict = {'sun': rsun, 'sat': rsat, 'jup': rjup, 'ear': rear, 'moo': rmoo, 'ven': rven, 'nep': rnep, 'iss': riss}
v_dict = {'sun': vsun, 'sat': vsat, 'jup': vjup, 'ear': vear, 'moo': vmoo, 'ven': vven, 'nep': vnep, 'iss': viss}
colour_dict = {'sun': 'yellow', 'sat': 'grey', 'jup': 'orange', 'ear': '#4d7bd1', 'moo': 'darkgrey', 'ven': '#d1996f',
               'nep': '#2a1a82', 'iss': 'purple'}
name_dict = {'sun': 'Sun', 'sat': 'Saturn', 'jup': 'Jupiter', 'ear': 'Earth', 'moo': 'Moon', 'ven': 'Venus',
             'nep': 'Neptune', 'iss': 'ISS'}

# Here is were the dash app begins
# Standard css style sheet recommended by Dash
external_stylesheets = [dbc.themes.BOOTSTRAP]

# Generates the Dash app
server = Flask(__name__)
app = dash.Dash(__name__,
                server=server,
                external_stylesheets=external_stylesheets
                )
app.title = '3-Body Problem'

app.layout = dbc.Container(fluid=True, style={'background-color': '#333399'}, children=[
    # .container class is fixed, .container.scalable is scalable
    html.Div(className="banner", style={'background-color': '#333399'}, children=[
        html.Div(className='container scalable', children=[
            html.H2(html.A(
                'The Three-Body Problem - Gravitational Astronomy',
                href='',
                style={
                    'text-decoration': 'none',
                    'color': 'inherit',
                    'background-color': '#333399',
                    'text-align': 'center'
                }
            )),
        ]),
    ]),

    html.Div(
        [
            dbc.Row(style={'background-color': 'white', 'margin': '0px 0px 0px 0px'},
                    children=[
                        dbc.Col([
                            dbc.Card(
                                id='first-card',
                                style={'margin': '10px 10px 10px 10px'},
                                children=[
                                    html.H3('Parameters', style={'margin': '10px 10px 0px 0px'}, ),
                                    drc.NamedRadioItems(
                                        name='Model',
                                        id="radios",
                                        options=[
                                            {"label": "Simplified Model", "value": 1},
                                            {"label": "Sun System", "value": 2},
                                        ],
                                        value=1,
                                        style={'margin': '10px 10px 10px 10px'},
                                    ),
                                    drc.NamedDropdown(
                                        name='Steps',
                                        id='n-dropdown',
                                        options=[
                                            {
                                                'label': 'n = 100',
                                                'value': 100
                                            },
                                            {
                                                'label': 'n = 1000',
                                                'value': 1000
                                            },
                                            {
                                                'label': 'n = 5 000',
                                                'value': 5000
                                            },
                                            {
                                                'label': 'n = 10 000',
                                                'value': 10000
                                            },
                                        ],
                                        clearable=False,
                                        searchable=False,
                                        value=1000,
                                    ),
                                    drc.NamedSlider(
                                        name='Time',
                                        id='time',
                                        min=0,
                                        max=30,
                                        step=1,
                                        marks={0: '0', 5: '5', 10: '10',
                                               15: '15', 20: '20', 25: '25', 30: '30'},
                                        value=10
                                    ),
                                ]),
                            dbc.Tabs([
                                dbc.Tab(label="Simplified Model", children=[
                                    dbc.Card(
                                        id='second-card',
                                        style={'margin': '10px 10px 10px 10px'},
                                        children=[
                                            html.H5('Simplified Model', style={'margin': '10px 10px 10px 10px'}),
                                            html.Li('Model Assumption: g = 1', style={'margin': '5px 5px 5px 25px'}),
                                            drc.NamedDropdown(
                                                name='Starting Values',
                                                id='scenario-dropdown',
                                                options=[
                                                    {
                                                        'label': 'Scenario 1',
                                                        'value': 1
                                                    },
                                                    {
                                                        'label': 'Scenario 2',
                                                        'value': 2
                                                    },
                                                    {
                                                        'label': 'Scenario 3',
                                                        'value': 3
                                                    },
                                                    {
                                                        'label': 'Scenario 4',
                                                        'value': 4
                                                    },
                                                    {
                                                        'label': 'Scenario 5',
                                                        'value': 5
                                                    },
                                                    {
                                                        'label': 'Scenario 6',
                                                        'value': 6
                                                    },
                                                    {
                                                        'label': 'Scenario 7',
                                                        'value': 7
                                                    }
                                                ],
                                                clearable=False,
                                                searchable=False,
                                                value=1,
                                            ),
                                            drc.NamedSlider(
                                                name='Mass of Object 1',
                                                id='simple_mass1',
                                                min=0.01,
                                                max=3,
                                                step=0.00001,
                                                marks={0.00001: '0.00001', 0.5: '0.5', 1: '1',
                                                       1.5: '1.5', 2: '2', 2.5: '2.5', 3.0: '3.0'},
                                                value=1
                                            ),
                                            drc.NamedSlider(
                                                name='Mass of Object 2',
                                                id='simple_mass2',
                                                min=0.01,
                                                max=3,
                                                step=0.00001,
                                                marks={0.00001: '0.00001', 0.5: '0.5', 1: '1',
                                                       1.5: '1.5', 2: '2', 2.5: '2.5', 3.0: '3.0'},
                                                value=1
                                            ),
                                            drc.NamedSlider(
                                                name='Mass of Object 3',
                                                id='simple_mass3',
                                                min=0.01,
                                                max=3,
                                                step=0.00001,
                                                marks={0.00001: '0.00001', 0.5: '0.5', 1: '1',
                                                       1.5: '1.5', 2: '2', 2.5: '2.5', 3.0: '3.0'},
                                                value=1
                                            ),
                                        ])],
                                        ),
                                dbc.Tab(label="Sun System", children=[
                                    dbc.Card(
                                        id='third-card',
                                        style={'margin': '10px 10px 10px 10px'},
                                        children=[
                                            html.H5('Sun System', style={'margin': '10px 10px 10px 10px'}),
                                            drc.NamedDropdown(
                                                name='Object 1',
                                                id='object1-dropdown',
                                                options=[
                                                    {
                                                        'label': 'Sun',
                                                        'value': 'sun'
                                                    },
                                                    {
                                                        'label': 'Venus',
                                                        'value': 'ven'
                                                    },
                                                    {
                                                        'label': 'Earth',
                                                        'value': 'ear'
                                                    },
                                                    {
                                                        'label': 'ISS',
                                                        'value': 'iss'
                                                    },
                                                    {
                                                        'label': 'Moon',
                                                        'value': 'moo'
                                                    },
                                                    {
                                                        'label': 'Jupiter',
                                                        'value': 'jup'
                                                    },
                                                    {
                                                        'label': 'Saturn',
                                                        'value': 'sat'
                                                    },
                                                    {
                                                        'label': 'Neptune',
                                                        'value': 'nep'
                                                    },
                                                ],
                                                clearable=False,
                                                searchable=False,
                                                value='sun',
                                            ),
                                            drc.NamedDropdown(
                                                name='Object 2',
                                                id='object2-dropdown',
                                                options=[
                                                    {
                                                        'label': 'Sun',
                                                        'value': 'sun'
                                                    },
                                                    {
                                                        'label': 'Venus',
                                                        'value': 'ven'
                                                    },
                                                    {
                                                        'label': 'Earth',
                                                        'value': 'ear'
                                                    },
                                                    {
                                                        'label': 'ISS',
                                                        'value': 'iss'
                                                    },
                                                    {
                                                        'label': 'Moon',
                                                        'value': 'moo'
                                                    },
                                                    {
                                                        'label': 'Jupiter',
                                                        'value': 'jup'
                                                    },
                                                    {
                                                        'label': 'Saturn',
                                                        'value': 'sat'
                                                    },
                                                    {
                                                        'label': 'Neptune',
                                                        'value': 'nep'
                                                    },
                                                ],
                                                clearable=False,
                                                searchable=False,
                                                value='ear',
                                            ),
                                            drc.NamedDropdown(
                                                name='Object 3',
                                                id='object3-dropdown',
                                                options=[
                                                    {
                                                        'label': 'Sun',
                                                        'value': 'sun'
                                                    },
                                                    {
                                                        'label': 'Venus',
                                                        'value': 'ven'
                                                    },
                                                    {
                                                        'label': 'Earth',
                                                        'value': 'ear'
                                                    },
                                                    {
                                                        'label': 'ISS',
                                                        'value': 'iss'
                                                    },
                                                    {
                                                        'label': 'Moon',
                                                        'value': 'moo'
                                                    },
                                                    {
                                                        'label': 'Jupiter',
                                                        'value': 'jup'
                                                    },
                                                    {
                                                        'label': 'Saturn',
                                                        'value': 'sat'
                                                    },
                                                    {
                                                        'label': 'Neptune',
                                                        'value': 'nep'
                                                    },
                                                ],
                                                clearable=False,
                                                searchable=False,
                                                value='sat',
                                            ),
                                        ],
                                    ),
                                ])]),
                        ],
                            width=3),
                        dbc.Col(children=[
                            html.Div(
                                id='div-expl-euler',
                                children=[
                                    dcc.Graph(
                                        id='expl-euler'
                                    )],
                            ),
                            html.Div(
                                id='div-runge-kutta',
                                children=dcc.Graph(
                                    id='runge-kutta'
                                )
                            ),
                        ], width=4, align='center'),
                        dbc.Col(children=[
                            html.Div(
                                id='div-impl-euler',
                                children=dcc.Graph(
                                    id='impl-euler'
                                ),
                            ),
                            html.Div(
                                id='div-corr-pred',
                                children=[
                                    dcc.Graph(
                                        id='corr-pred'
                                    ),
                                    drc.NamedRadioItems(
                                        name='Adaptive Step-Size',
                                        id='ad_step pred-corr',
                                        options=[{"label": "Yes", "value": 1},
                                                 {"label": "No", "value": 0}],
                                        value=0,
                                    )
                                ]
                            ),
                        ], width=4, align='center'),
                    ]),
        ]
    ),
]
                           )


@app.callback(
    Output(component_id='expl-euler', component_property='figure'),
    Output(component_id='impl-euler', component_property='figure'),
    Output(component_id='runge-kutta', component_property='figure'),
    Output(component_id='corr-pred', component_property='figure'),
    Input(component_id='radios', component_property='value'),
    Input(component_id='n-dropdown', component_property='value'),
    Input(component_id='time', component_property='value'),
    Input(component_id='scenario-dropdown', component_property='value'),
    Input(component_id='simple_mass1', component_property='value'),
    Input(component_id='simple_mass2', component_property='value'),
    Input(component_id='simple_mass3', component_property='value'),
    Input(component_id='object1-dropdown', component_property='value'),
    Input(component_id='object2-dropdown', component_property='value'),
    Input(component_id='object3-dropdown', component_property='value'),
    Input(component_id='ad_step pred-corr', component_property='value')
)
def update_figure(model, n, t_end, scenario, m1, m2, m3, o1, o2, o3, ad_step):
    global mass
    global g
    if model == 1:
        names = ['Objekt 1', 'Objekt 2', 'Objekt 3']
        colours = ['green', 'blue', 'red']
        init_data = init_dict[scenario]
        mass = [m1, m2, m3]
        g = 1
        fig_expl = generate_figures(forward_euler(f, init_data, t_start, t_end, n), 'Explicit Euler Method',
                                    names,
                                    colours)
        fig_rk = generate_figures(runge_kutta_4(f, init_data, t_start, t_end, n, ad_step),
                                  'Runge-Kutta-Method of Order 3',
                                  names, colours)
        try:
            fig_impl = generate_figures(backward_euler(f, init_data, t_start, t_end, n), 'Implicit Euler Method',
                                        names, colours)
        except:
            fig_impl = fig_not_convergent('Implicit Euler Method')
        try:
            fig_precor = generate_figures(predictor_corrector(f, init_data, t_start, t_end, n, ad_step),
                                          'Predictor-Corrector Method',
                                          names, colours)
        except:
            fig_precor = fig_not_convergent('Predictor-Corrector Method')
        return fig_expl, fig_impl, fig_rk, fig_precor
    elif model == 2:
        g = GG
        names = [name_dict[o1], name_dict[o2], name_dict[o3]]
        colours = [colour_dict[o1], colour_dict[o2], colour_dict[o3]]
        mass = np.array([mass_dict[o1], mass_dict[o2], mass_dict[o3]])
        init_data = np.array([[r_dict[o1], v_dict[o1]], [r_dict[o2], v_dict[o2]], [r_dict[o3], v_dict[o3]]])
        fig_expl = generate_figures(forward_euler(f, init_data, t_start, t_end, n),
                                    'Explicit Euler Method', names, colours)
        fig_rk = generate_figures(runge_kutta_4(f, init_data, t_start, t_end, n),
                                  'Runge-Kutta-Method of Order 4 with adaptive step size', names, colours)
        try:
            fig_impl = generate_figures(backward_euler(f, init_data, t_start, t_end, n), 'Implicit Euler Method',
                                        names, colours)
        except:
            fig_impl = fig_not_convergent('Implicit Euler Method')
        try:
            fig_precor = generate_figures(predictor_corrector(f, init_data, t_start, t_end, n, ad_step_pc),
                                          'Predictor-Corrector Method',
                                          names, colours)
        except:
            fig_precor = fig_not_convergent('Predictor-Corrector Method')
        return fig_expl, fig_impl, fig_rk, fig_precor


def f(t, y):
    d0 = ((-g * mass[0] * mass[1] * (y[0] - y[1]) / np.linalg.norm(y[0] - y[1]) ** 3) +
          (-g * mass[0] * mass[2] * (y[0] - y[2]) / np.linalg.norm(y[0] - y[2]) ** 3)) / mass[0]
    d1 = ((-g * mass[1] * mass[2] * (y[1] - y[2]) / np.linalg.norm(y[1] - y[2]) ** 3) + (
            -g * mass[1] * mass[0] * (y[1] - y[0]) / np.linalg.norm(y[1] - y[0]) ** 3)) / mass[1]
    d2 = ((-g * mass[2] * mass[0] * (y[2] - y[0]) / np.linalg.norm(y[2] - y[0]) ** 3) + (
            -g * mass[2] * mass[1] * (y[2] - y[1]) / np.linalg.norm(y[2] - y[1]) ** 3)) / mass[2]
    return np.array([d0, d1, d2])


def flong(y):
    d00 = ((-g * mass[0] * mass[1] * (y[0] - y[2]) / np.linalg.norm([y[0] - y[2], y[1] - y[3]]) ** 3) +
           (-g * mass[0] * mass[2] * (y[0] - y[4]) / np.linalg.norm([y[0] - y[4], y[1] - y[5]]) ** 3)) / mass[0]
    d01 = ((-g * mass[0] * mass[1] * (y[1] - y[3]) / np.linalg.norm([y[0] - y[2], y[1] - y[3]]) ** 3) + (
            -g * mass[0] * mass[2] * (y[1] - y[5]) / np.linalg.norm([y[0] - y[4], y[1] - y[5]]) ** 3)) / mass[0]
    d10 = ((-g * mass[1] * mass[2] * (y[2] - y[4]) / np.linalg.norm([y[2] - y[4], y[3] - y[5]]) ** 3) + (
            -g * mass[1] * mass[0] * (y[2] - y[0]) / np.linalg.norm([y[0] - y[2], y[1] - y[3]]) ** 3)) / mass[1]
    d11 = ((-g * mass[1] * mass[2] * (y[3] - y[5]) / np.linalg.norm([y[2] - y[4], y[3] - y[5]]) ** 3) + (
            -g * mass[1] * mass[0] * (y[3] - y[1]) / np.linalg.norm([y[0] - y[2], y[1] - y[3]]) ** 3)) / mass[1]
    d20 = ((-g * mass[2] * mass[0] * (y[4] - y[0]) / np.linalg.norm([y[0] - y[4], y[1] - y[5]]) ** 3) + (
            -g * mass[2] * mass[1] * (y[4] - y[2]) / np.linalg.norm([y[2] - y[4], y[3] - y[5]]) ** 3)) / mass[2]
    d21 = ((-g * mass[2] * mass[0] * (y[5] - y[1]) / np.linalg.norm([y[0] - y[4], y[1] - y[5]]) ** 3) + (
            -g * mass[2] * mass[1] * (y[5] - y[3]) / np.linalg.norm([y[2] - y[4], y[3] - y[5]]) ** 3)) / mass[2]
    return np.array([d00, d01, d10, d11, d20, d21])


eps = 1e-15

'''def forward_euler(f, y0, t0, t1, N):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    h = (t1 - t0) / N
    t = t0
    v = np.zeros((len(y0), N + 1, 2))
    y = np.zeros((len(y0), N + 1, 2))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    for k in range(N):
        for i in range(len(y0)):
            y[i, k + 1, :] = y[i, k, :] + h * v[i, k, :]
            v[i, k + 1, :] = v[i, k, :] + h * f(t, y[:, k, :])[i]
        t = t + h
        t_list[k + 1] = t
    return y, t'''


def forward_euler(f, y0, t0, t1, N):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    h = (t1 - t0) / N
    h_min = h / 256
    h_max = h
    h_sum = 0
    k = 0
    t = t0
    v = np.zeros((len(y0), 1000000, 2))
    y = np.zeros((len(y0), 1000000, 2))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]

    while h_sum < t1 and k < 50000:
        for i in range(len(y0)):
            y[i, k + 1, :] = y[i, k, :] + h * v[i, k, :]
            v[i, k + 1, :] = v[i, k, :] + h * f(t, y[:, k, :])[i]
        energy = np.linalg.norm(
            mass[0] * f(t, y[:, k, :])[0] + mass[1] * f(t, y[:, k, :])[1] + mass[2] * f(t, y[:, k, :])[2])

        if abs(energy) < eps:
            k = k + 1
            h_sum = h_sum + h
            if abs(energy) < eps ** 2 and h < h_max:
                h = h * 2
        elif h > h_min:
            h = h * 0.5
        else:
            k = k + 1
            h_sum = h_sum + h
            #print('h zu klein')
        t = t + h

    y = y[:, :k + 1, :]
    print(k)
    return y, t


def runge_kutta(f, y0, t0, t1, N):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    h = (t1 - t0) / N
    t = t0
    v = np.zeros((len(y0), N + 1, 2))
    y = np.zeros((len(y0), N + 1, 2))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    for k in range(N):
        for i in range(len(y0)):
            k1 = f(t, y[:, k, :])[i]
            k2 = f(t + 0.5 * h, (y[:, k, :] + h * k1 / 2))[i]
            k3 = f(t + h, (y[:, k, :] + h * (-k1 + 2 * k2)))[i]
            v[i, k + 1] = v[i, k, :] + h / 6 * (k1 + 4 * k2 + k3)
            k12 = v[i, k, :]
            k22 = v[i, k, :] + h * k1 / 2
            k32 = v[i, k, :] + h * (-k1 + 2 * k2)
            y[i, k + 1] = y[i, k, :] + h / 6 * (k12 + 4 * k22 + k32)
        t = t + h
        t_list[k + 1] = t
    return y, t_list


A = np.array([0, 2 / 9, 1 / 3, 3 / 4, 1, 5 / 6])
B = np.matrix([[0, 0, 0, 0, 0], [2 / 9, 0, 0, 0, 0], [1 / 12, 1 / 4, 0, 0, 0], [69 / 128, -243 / 128, 135 / 64, 0, 0],
               [-17 / 12, 27 / 4, -27 / 5, 16 / 15, 0], [65 / 432, -5 / 16, 13 / 16, 4 / 27, 5 / 144]])
C = np.array([1 / 9, 0, 9 / 20, 16 / 45, 1 / 12])
CH = np.array([47 / 450, 0, 12 / 25, 32 / 225, 1 / 30, 6 / 25])
CT = np.array([-1 / 150, 0, 3 / 100, -16 / 75, -1 / 20, 6 / 25])


def runge_kutta_4(f, y0, t0, t1, N, ad_step):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    h = (t1 - t0) / N
    h_min = h / 256
    h_max = h
    h_sum = 0
    eps = 1e-4
    t = t0
    v = np.zeros((len(y0), 50000, 2))
    y = np.zeros((len(y0), 50000, 2))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    k = 0
    while k < 50000 and h_sum < t1:
        i = 0

        while i < (len(y0)):
            k1 = h * f(t + A[0] * h, y[:, k, :])[i]
            k2 = h * f(t + A[1] * h, (y[:, k, :] + B[1, 0] * h * k1))[i]
            k3 = h * f(t + A[2] * h, (y[:, k, :] + B[2, 0] * k1 + B[2, 1] * k2))[i]
            k4 = h * f(t + A[3] * h, (y[:, k, :] + B[3, 0] * k1 + B[3, 1] * k2 + B[3, 2] * k3))[i]
            k5 = h * f(t + A[4] * h, (y[:, k, :] + B[4, 0] * k1 + B[4, 1] * k2 + B[4, 2] * k3 + B[4, 3] * k4))[i]
            k6 = h * f(t + A[5] * h,
                       (y[:, k, :] + B[5, 0] * k1 + B[5, 1] * k2 + B[5, 2] * k3 + B[5, 3] * k4 + B[5, 4] * k5))[i]
            v[i, k + 1] = v[i, k] + CH[0] * k1 + CH[1] * k2 + CH[2] * k3 + CH[3] * k4 + CH[4] * k5 + CH[5] * k6

            l1 = v[i, k, :]
            l2 = v[i, k, :] + B[1, 0] * h * k1
            l3 = v[i, k, :] + B[2, 0] * k1 + B[2, 1] * k2
            l4 = v[i, k, :] + B[3, 0] * k1 + B[3, 1] * k2 + B[3, 2] * k3
            l5 = v[i, k, :] + B[4, 0] * k1 + B[4, 1] * k2 + B[4, 2] * k3 + B[4, 3] * k4
            l6 = v[i, k, :] + B[5, 0] * k1 + B[5, 1] * k2 + B[5, 2] * k3 + B[5, 3] * k4 + B[5, 4] * k5
            y[i, k + 1] = y[i, k] + h * CH[0] * l1 + h * CH[1] * l2 + h * CH[2] * l3 + h * CH[3] * l4 + h * CH[
                4] * l5 + h * CH[5] * l6

            TE = np.linalg.norm(CT[0] * l1 + CT[1] * l2 + CT[2] * l3 + CT[3] * l4 + CT[4] * l5 + CT[5] * l6)

            if ad_step == 1:

                if TE < eps:
                    i = i + 1
                    if TE < eps ** 2 and h < h_max:
                        h = h * 2
                elif TE > eps and h > h_min:
                    h = h * 0.5
                else:
                    i = i + 1
            else:
                i += 1

        k = k + 1
        h_sum = h_sum + h

    print('RK4 k: ', k)

    y = y[:, :k + 1, :]
    return y, t_list


def newton_raphson(f, g, x0, e, N):
    """
    Numerical solver of the equation f(x) = 0
    :param f: Function, left side of equation f(x) = 0 to solve
    :param g: Function, derivative of f
    :param x0: Float, initial guess
    :param e: Float, tolerable error
    :param N: Integer, maximal steps
    :return:
    """
    step = 1
    flag = 1
    condition = True
    while condition:
        if np.all(g(x0) == 0.0):
            print('Divide by zero error!')
            break
        x1 = x0 - f(x0) / g(x0)
        x0 = x1
        step = step + 1
        if step > N:
            flag = 0
            break
        condition = np.any(abs(f(x1)) > e)
    if flag == 1:
        return x1
    else:
        print('\nNot Convergent.')


def backward_euler(f, y0, t0, t1, N):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    h = (t1 - t0) / N
    h_min = h / 256
    h_max = h
    h_sum = 0
    k = 0
    v = np.zeros((len(y0), N+1, 2))
    y = np.zeros((len(y0), N+1, 2))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    t = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    for k in range(1, N + 1):

        for i in range(len(y0)):

            def fixpoint(x):
                terms = []
                for j in range(len(x)):
                    if j != i:
                        term = (-g * mass[0] * mass[1] * (x[i, :] - x[j, :]) / np.linalg.norm(x[i, :] - x[j, :]) ** 3) / \
                               mass[i]
                        terms.append(term)

                return v[i, k - 1, :] + h * (terms[0] + terms[1]) - x[i, :]

            def fixpoint_deriv(x):
                terms = []
                for j in range(len(x)):
                    if j != i:
                        term = (g * mass[0] * mass[1] * (
                                2 * x[i, :] ** 2 - 3 * x[i, :] * x[j, :] + x[j, :] ** 2) / np.linalg.norm(
                            x[i, :] - x[j, :]) ** (5 / 2)) / mass[i]
                        terms.append(term)
                return h * (terms[0] + terms[1]) - 1

            v[i, k, :] = newton_raphson(fixpoint, fixpoint_deriv, y[:, k - 1, :], 0.0001, 5)[i, :]
            y[i, k, :] = y[i, k - 1, :] + h * v[i, k - 1, :] + h ** 2 * 0.5 * f(t, y[:, k - 1, :])[i]
        t = t + h
        t_list[k] = t
    return y, t_list


def predictor_corrector(f, y0, t0, t1, N, ad_step):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :return: two lists of floats, approximation of y at interval t0-t1 in step-size h and interval list
    """
    h = (t1 - t0) / N
    v = np.zeros((len(y0), 1000000, 2))
    y = np.zeros((len(y0), 1000000, 2))
    t_list = [0] * (N + 2)
    t_list[0] = t0
    t = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    h_sum = 0
    h_min = h * 0.1
    k = 1
    while h_sum < t1 and k < 50000:

        vp = v[:, k - 1, :] + h * f(t, y[:, k - 1, :])
        yp = y[:, k - 1, :] + h * v[:, k - 1, :] + h ** 2 * 0.5 * f(t, y[:, k - 1, :])

        for i in range(len(y0)):

            def fixpoint(x):
                terms = []
                for j in range(len(x)):
                    if j != i:
                        term = (-g * mass[0] * mass[1] * (x[i, :] - x[j, :]) / np.linalg.norm(x[i, :] - x[j, :]) ** 3) / \
                               mass[i]
                        terms.append(term)
                return v[i, k - 1, :] + h * (terms[0] + terms[1]) - x[i, :]

            def fixpoint_deriv(x):
                terms = []
                for j in range(len(x)):
                    if j != i:
                        term = (g * mass[0] * mass[1] * (
                                2 * x[i, :] ** 2 - 3 * x[i, :] * x[j, :] + x[j, :] ** 2) / np.linalg.norm(
                            x[i, :] - x[j, :]) ** (5 / 2)) / mass[i]
                        terms.append(term)
                return h * (terms[0] + terms[1]) - 1

            vc = newton_raphson(fixpoint, fixpoint_deriv, yp, 0.01, 10)[i, :]

            v[i, k, :] = vc
            y[i, k, :] = y[i, k - 1, :] + h * v[i, k - 1, :] + h ** 2 * 0.5 * f(t, y[:, k - 1, :])[i]

        if ad_step == 1:
            err = np.linalg.norm(vp - v[:, k, :])

            if err > eps and h > h_min:
                h = h * 0.9
            elif err < eps ** 2:
                h_new = h * 1.1
                h = h_new
            else:
                k = k + 1
                h_sum += h
        else:
            k = k + 1
            h_sum += h

        t = t + h
        # t_list[k] = t
    y = y[:, :k + 1, :]
    return y, t_list


def fpi(f, t, yk, h, steps):
    x = yk
    while steps > 0:
        x = (1 + h) * x + h ** 2 * f(t, x)
        steps += -1
    return x


def backward_euler_0(f, y0, t0, t1, N):
    h = (t1 - t0) / N
    t = t0
    v = np.zeros((len(y0), N + 1, 2))
    y = np.zeros((len(y0), N + 1, 2))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    for k in range(N):
        v[:, k + 1, :] = fpi(f, t, y[:, k, :], h, 5)
        y[:, k + 1, :] = y[:, k, :] + h * v[:, k, :]
        t = t + h
        t_list[k + 1] = t
    return y, t_list


def backward_euler_equation(y, yn, h, f):
    ret = y - yn - h * f(y)
    return ret


def backward_euler_scipy(f, y0, t0, t1, N):
    h = (t1 - t0) / N
    t = t0
    v = np.zeros((N + 1, 6))
    y = np.zeros((N + 1, 6))
    y_start = y0[:, 0, :]
    v_start = y0[:, 1, :]
    y0_reshape = y_start.reshape(6, )
    v0_reshape = v_start.reshape(6, )
    t_list = [0] * (N + 1)
    t_list[0] = t0
    v[0, :] = v0_reshape
    y[0, :] = y0_reshape

    for k in range(N):
        v[k + 1, :] = fsolve(backward_euler_equation, y[k, :], (y[k, :], h, flong))
        y[k + 1, :] = y[k, :] + h * v[k, :]
        t = t + h
        t_list[k + 1] = t

    y_ret = np.zeros((len(y0), N + 1, 2))
    y_ret[:, :, 0] = y[:, ::2].transpose()
    y_ret[:, :, 1] = y[:, 1::2].transpose()
    return y_ret, t_list


def generate_figures(method, title, names, colours):
    y, t = method
    fig = go.Figure(
        data=[go.Scatter(x=y[0, :, 0], y=y[0, :, 1],
                         mode="lines", name=names[0],
                         line=dict(width=2, color=colours[0])),
              go.Scatter(x=y[1, :, 0], y=y[1, :, 1],
                         mode="lines", name=names[1],
                         line=dict(width=2, color=colours[1])),
              go.Scatter(x=y[2, :, 0], y=y[2, :, 1],
                         mode="lines", name=names[2],
                         line=dict(width=2, color=colours[2]))],
        layout=go.Layout(
            xaxis=dict(autorange=True, zeroline=False, range=[-2, 2]),
            yaxis=dict(autorange=True, zeroline=False, range=[-2, 2]),
            title=title, hovermode="closest"),
    )
    return fig


def generate_error_figures(method, title, names, colours):
    y_rk, t_rk = runge_kutta(f, inits1, 0, 10, 0.01)
    y_meth, t_meth = method
    y = np.absolute(y_rk - y_meth)
    fig = go.Figure(
        data=[go.Scatter(x=y[0, :, 0], y=y[0, :, 1],
                         mode="lines", name=names[0],
                         line=dict(width=2, color=colours[0])),
              go.Scatter(x=y[1, :, 0], y=y[1, :, 1],
                         mode="lines", name=names[1],
                         line=dict(width=2, color=colours[1])),
              go.Scatter(x=y[2, :, 0], y=y[2, :, 1],
                         mode="lines", name=names[2],
                         line=dict(width=2, color=colours[2]))],
        layout=go.Layout(
            xaxis=dict(autorange=True, zeroline=False, range=[-2, 2]),
            yaxis=dict(autorange=True, zeroline=False, range=[-2, 2]),
            title=title, hovermode="closest"),
    )
    return fig


def fig_not_convergent(title):
    fig = px.scatter(x=[1, 1, 1, 1, 1, 1, 1, 1], y=[3, 2.75, 2.5, 2.25, 2, 1.75, 1.5, 0.75],
                     title=str(title + ': not convergent'))
    return fig


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
