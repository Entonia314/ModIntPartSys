"""
Project for the course "Modelling interacting particle systems in science". Code by Verena Alton.
Layout of the code: global variables for the starting values of the particles, dashboard layout, dashboard functions,
numerical integration method functions, graph generating functions.
"""

from flask import Flask
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import numpy as np
from numpy.linalg import norm
import plotly.graph_objects as go
import plotly.express as px
from dash.dependencies import Input, Output, State
import SupplementaryFiles.dash_reusable_components as drc
import time

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
Miss = 440725  # Mass of ISS
Mtit = 1.345e23  # Mass of Titan

G = 6.673e-11  # Gravitational Constant

RR = 1.496e11  # Normalizing distance in km (= 1 AU)
MM = 6e24  # Normalizing mass (mass of earth)
TT = 365 * 24 * 60 * 60.0  # Normalizing time (1 year)

GG = (MM * G * TT ** 2) / (RR ** 3)

Mear = Mear / MM  # Normalized mass of Earth
Msun = Msun / MM  # Normalized mass of Sun
Mjup = Mjup / MM  # Normalized mass of Jupiter/Super Jupiter
Msat = Msat / MM  # Normalized mass of Saturn
Mmoo = Mmoo / MM  # Normalized mass of Moon
Mven = Mven / MM  # Normalized mass of Venus
Mnep = Mnep / MM
Miss = Miss / MM
Mtit = Mtit / MM

rear = [1, 0]  # initial position of earth
rjup = [5.2, 0]  # initial position of Jupiter
rsat = [9.582, 0]
rsun = [0, 0]
rmoo = [rear[0] + 0.00257, 0]
rven = [0.723, 0]
rnep = [30.047, 0]
riss = [rear[0] + (420e3 / RR), 0]
rtit = [rsat[0] + (1.221830e9 / RR), 0]

magear = np.sqrt(Msun * GG / rear[0])  # Magnitude of Earth's initial velocity
magjup = 13.06e3 * TT / RR  # Magnitude of Jupiter's initial velocity
magsat = 9.68e3 * TT / RR  # Magnitude of Saturn's initial velocity
magmoo = 1.022e3 * TT / RR  # Magnitude of Moon's initial velocity
magven = 35.02e3 * TT / RR  # Magnitude of Venus' initial velocity
magnep = 5.43e3 * TT / RR
magiss = 7.66e3 * TT / RR
magtit = 5.57e3 * TT / RR

vear = [0, magear * 1.0]  # Initial velocity vector for Earth.Taken to be along y direction as ri is on x axis.
vjup = [0, magjup * 1.0]  # Initial velocity vector for Jupiter
vsat = [0, magsat * 1.0]  # Initial velocity vector for Saturn
vsun = [0, 0]
vmoo = [0, magear + magmoo]  # Initial velocity vector for Moon
vven = [0, magven]  # Initial velocity vector for Venus
vnep = [0, magnep]
viss = [0, magiss + magear]
vtit = [0, magsat + magtit]

method_dict = {'forward_euler': 'Explizites Euler-Verfahren', 'backward_euler': 'Implizites Euler-Verfahren'}
init_dict = {1: inits1, 2: inits2, 3: inits3, 4: inits4, 5: inits5, 6: inits6, 7: inits7}
mass_dict = {'sun': Msun, 'sat': Msat, 'jup': Mjup, 'ear': Mear, 'moo': Mmoo, 'ven': Mven, 'nep': Mnep, 'iss': Miss,
             'tit': Mtit}
r_dict = {'sun': rsun, 'sat': rsat, 'jup': rjup, 'ear': rear, 'moo': rmoo, 'ven': rven, 'nep': rnep, 'iss': riss,
          'tit': rtit}
v_dict = {'sun': vsun, 'sat': vsat, 'jup': vjup, 'ear': vear, 'moo': vmoo, 'ven': vven, 'nep': vnep, 'iss': viss,
          'tit': vtit}
colour_dict = {'sun': 'yellow', 'sat': 'grey', 'jup': 'orange', 'ear': '#4d7bd1', 'moo': 'darkgrey', 'ven': '#d1996f',
               'nep': '#2a1a82', 'iss': 'purple', 'tit': 'rgb(219, 196, 61)'}
colour_marker_dict = {'sun': 'rgba(242, 238, 124, 0.5)', 'sat': 'rgba(150, 150, 131, 0.5)',
                      'jup': 'rgba(237, 180, 45, 0.5)', 'ear': 'rgba(89, 110, 247, 0.5)',
                      'moo': 'rgba(99, 103, 130, 0.5)', 'ven': 'rgba(173, 173, 101, 0.5)',
                      'nep': 'rgba(54, 48, 112, 0.5)', 'iss': 'rgba(104, 46, 140, 0.5)',
                      'tit': 'rgba(252, 229, 88, 0.5)'}
name_dict = {'sun': 'Sun', 'sat': 'Saturn', 'jup': 'Jupiter', 'ear': 'Earth', 'moo': 'Moon', 'ven': 'Venus',
             'nep': 'Neptune', 'iss': 'ISS', 'tit': 'Titan'}

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
    html.Div(className="banner", style={'background-color': '#333399', 'color': 'white'}, children=[
        dbc.Row(children=[
            dbc.Col(
                html.H2(html.A(
                    'The Three-Body Problem - Gravitational Astronomy',
                    href='https://github.com/Entonia314/ModIntPartSys',
                    style={
                        'text-decoration': 'none',
                        'color': 'inherit',
                        'background-color': '#333399',
                        'text-align': 'center'
                    }
                )), width={'size': 8, 'offset': 2}), ],
        ),
    ]),

    dbc.Container(fluid=True, children=[
        html.Section(
            className='header',
            style={"padding": "0px 0px 20px 0px", 'text-align': 'center',
                   'color': '', 'background-color': ''},
            children=[
                html.Div(
                    [
                        dbc.Button(
                            'About this Project',
                            id="collapse-button",
                            className="mb-3",
                            color="light",
                        ),
                        dbc.Collapse(
                            dbc.Card(dbc.CardBody(children=[
                                html.H4('Three-Body Problem', style={'text-align': ''}),
                                html.P(children=['''Knowing the masses, initial positions and initial velocities of three 
                                        particles, one wants to deduce their positions at any later time. Since this 
                                        system of 
                                        ordinary differential equations cannot be solved analytically, four different 
                                        methods of numerical integration 
                                        were implemented. This dashboard means to compare the methods and show some 
                                        interesting results and orbits. To learn more about the theory of the problem,
                                        see the essay in the ''',
                                                 html.A('GitHub', href='https://github.com/Entonia314/ModIntPartSys'),
                                                 ''' repository.'''],
                                       style={'font-size': '16px', 'text-align': 'justify', 'margin': '10px 50px 30px '
                                                                                                      '50px'}),

                                html.H4('Using the Dashboard', style={'text-align': ''}),
                                html.P(children=['''There are two graphs to compare results, for each of them all 
                                parameters can be chosen independently. The left card contains general options while 
                                the right one holds model-specific parameters. Two types of models are implemented: a 
                                simplified one and a simulation of the sun system with three celestial bodies at 
                                once. To switch models, tick the respective one in the left card and and go to the 
                                matching tab on the right card to choose parameters. The step size means the fineness 
                                of the discretisation for the numerical approximation. A smaller step size will lead 
                                to better results but demands higher computing capacity, so the calculation will need 
                                some time (the tab in the browser says "Updating" while still computing). For an 
                                automatic adaption of the step size, the "Adaptive Step Size" can be ticked. Here, 
                                the numerical error of the calculation is approximated and the step size is 
                                accordingly scaled down or up. In the graphs, the translucent points mark the 
                                initial positions of the particles. To see an animation of the route of the 
                                particles, press the "Play" button next to the graph. Since animations are a rather 
                                new feature of Plotly, they are a bit buggy. If one wants to change parameters while 
                                the animation is still running, please press "Pause" before changing. The particles 
                                will continue their route with new parameters if "Play" is pressed again. See the ''',
                                                 html.A('GitHub', href='https://github.com/Entonia314/ModIntPartSys'),
                                                 ''' repository for the code of the program.''',
                                                 ],
                                       style={'font-size': '16px', 'text-align': 'justify', 'margin': '10px 50px 10px '
                                                                                                      '50px'})]
                            )),
                            id="collapse",
                        ),
                    ]
                )
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    style={
                        'text-decoration': 'none',
                        'color': 'black',
                        'background-color': 'white',
                        'text-align': '',
                        'margin': '10px 10px 10px 10px'
                    },
                    children=[
                        html.H3('Graph 1', style={'text-align': 'center'}),
                        dbc.Row(
                            dcc.Loading(
                                id='loading-1',
                                type='circle',
                                children=[
                                    html.Div(
                                        id='div_graph1',
                                        style={
                                            'text-decoration': 'none',
                                            'color': 'black',
                                            'background-color': 'white',
                                            'text-align': '',
                                            'margin': '0px 0px 30px 50px'
                                        },
                                        children=[
                                            dcc.Graph(
                                                id='graph1'
                                            ),
                                            dbc.Row(id='param_infos1')],
                                    )]),
                        ),
                        dbc.Row([dbc.Col([
                            dbc.Card(
                                id='parameter_card1',
                                style={'margin': '0px 10px 10px 10px'},
                                children=[
                                    html.H3('Parameters', style={'margin': '10px 10px 10px 10px'}, ),
                                    drc.NamedDropdown(
                                        name='Method',
                                        id='method1',
                                        options=[
                                            {'label': 'Explicit Euler', 'value': 'forwardeuler'},
                                            {'label': 'Implicit Euler', 'value': 'backwardeuler'},
                                            {'label': 'Runge-Kutta', 'value': 'rungekutta'},
                                            {'label': 'Heun', 'value': 'heun'}],
                                        value='rungekutta',
                                        style={},
                                        clearable=False,
                                        searchable=False,
                                    ),
                                    drc.NamedRadioItems(
                                        name='Model',
                                        id="model1",
                                        options=[
                                            {"label": " Simplified Model", "value": 1},
                                            {"label": " Sun System", "value": 2},
                                        ],
                                        value=1,
                                        style={'margin': '10px 10px 10px 10px'},
                                    ),
                                    drc.NamedDropdown(
                                        name='Stepsize',
                                        id='h-dropdown1',
                                        options=[
                                            {
                                                'label': 'h = 0.01',
                                                'value': 0.01
                                            },
                                            {
                                                'label': 'h = 0.001',
                                                'value': 0.001
                                            },
                                            {
                                                'label': 'h = 0.0005',
                                                'value': 0.0005
                                            },
                                            {
                                                'label': 'h = 0.0001',
                                                'value': 0.0001
                                            },
                                        ],
                                        clearable=False,
                                        searchable=False,
                                        value=0.01,
                                    ),
                                    drc.NamedSlider(
                                        name='Time',
                                        id='time1',
                                        min=0,
                                        max=30,
                                        step=1,
                                        marks={0: '0', 5: '5', 10: '10',
                                               15: '15', 20: '20', 25: '25', 30: '30'},
                                        value=10
                                    ),
                                    drc.NamedRadioItems(
                                        name='Adaptive Step Size',
                                        id='ad_step1',
                                        options=[{"label": " Yes", "value": 1},
                                                 {"label": " No", "value": 0}],
                                        value=0,
                                        style={'margin': '10px 10px 10px 10px'},
                                    )
                                ])]),
                            dbc.Col([
                                dbc.Tabs([
                                    dbc.Tab(label="Simplified Model", children=[
                                        dbc.Card(
                                            id='second-card1',
                                            style={'margin': '10px 10px 10px 10px'},
                                            children=[
                                                html.H5('Simplified Model', style={'margin': '10px 10px 10px 10px'}),
                                                html.Li('Model Assumption: g = 1',
                                                        style={'margin': '5px 5px 5px 25px'}),
                                                drc.NamedDropdown(
                                                    name='Starting Values',
                                                    id='scenario-dropdown1',
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
                                                    id='simple_mass11',
                                                    min=0.01,
                                                    max=3,
                                                    step=0.00001,
                                                    marks={0.00001: '0.00001', 0.5: '0.5', 1: '1',
                                                           1.5: '1.5', 2: '2', 2.5: '2.5', 3.0: '3.0'},
                                                    value=1
                                                ),
                                                drc.NamedSlider(
                                                    name='Mass of Object 2',
                                                    id='simple_mass21',
                                                    min=0.01,
                                                    max=3,
                                                    step=0.00001,
                                                    marks={0.00001: '0.00001', 0.5: '0.5', 1: '1',
                                                           1.5: '1.5', 2: '2', 2.5: '2.5', 3.0: '3.0'},
                                                    value=1
                                                ),
                                                drc.NamedSlider(
                                                    name='Mass of Object 3',
                                                    id='simple_mass31',
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
                                                    id='object1-dropdown1',
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
                                                            'label': 'Titan',
                                                            'value': 'tit'
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
                                                    id='object2-dropdown1',
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
                                                            'label': 'Titan',
                                                            'value': 'tit'
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
                                                    id='object3-dropdown1',
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
                                                            'label': 'Titan',
                                                            'value': 'tit'
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
                                    ])])
                            ])]),
                    ]),
                dbc.Col(
                    style={
                        'text-decoration': 'none',
                        'color': 'black',
                        'background-color': 'white',
                        'text-align': '',
                        'margin': '10px 10px 10px 10px'
                    },
                    children=[
                        html.H3('Graph 2', style={'text-align': 'center'}),
                        dbc.Row(
                            dcc.Loading(
                                id='loading-2',
                                type='circle',
                                children=[
                                    html.Div(
                                        id='div_graph2',
                                        style={
                                            'text-decoration': 'none',
                                            'color': 'black',
                                            'background-color': 'white',
                                            'text-align': '',
                                            'margin': '0px 0px 30px 50px'
                                        },
                                        children=[
                                            dcc.Graph(
                                                id='graph2'
                                            ),
                                            dbc.Row(id='param_infos2')],
                                    )]),
                        ),
                        dbc.Row([dbc.Col([
                            dbc.Card(
                                id='parameter_card2',
                                style={'margin': '0px 10px 10px 10px'},
                                children=[
                                    html.H3('Parameters', style={'margin': '10px 10px 10px 10px'}, ),
                                    drc.NamedDropdown(
                                        name='Method',
                                        id='method2',
                                        options=[
                                            {'label': 'Explicit Euler', 'value': 'forwardeuler'},
                                            {'label': 'Implicit Euler', 'value': 'backwardeuler'},
                                            {'label': 'Runge-Kutta', 'value': 'rungekutta'},
                                            {'label': 'Heun', 'value': 'heun'}],
                                        value='heun',
                                        style={},
                                        clearable=False,
                                        searchable=False,
                                    ),
                                    drc.NamedRadioItems(
                                        name='Model',
                                        id="model2",
                                        options=[
                                            {"label": " Simplified Model", "value": 1},
                                            {"label": " Sun System", "value": 2},
                                        ],
                                        value=1,
                                        style={'margin': '10px 10px 10px 10px'},
                                    ),
                                    drc.NamedDropdown(
                                        name='Stepsize',
                                        id='h-dropdown2',
                                        options=[
                                            {
                                                'label': 'h = 0.01',
                                                'value': 0.01
                                            },
                                            {
                                                'label': 'h = 0.001',
                                                'value': 0.001
                                            },
                                            {
                                                'label': 'h = 0.0005',
                                                'value': 0.0005
                                            },
                                            {
                                                'label': 'h = 0.0001',
                                                'value': 0.0001
                                            },
                                        ],
                                        clearable=False,
                                        searchable=False,
                                        value=0.01,
                                    ),
                                    drc.NamedSlider(
                                        name='Time',
                                        id='time2',
                                        min=0,
                                        max=30,
                                        step=1,
                                        marks={0: '0', 5: '5', 10: '10',
                                               15: '15', 20: '20', 25: '25', 30: '30'},
                                        value=10
                                    ),
                                    drc.NamedRadioItems(
                                        name='Adaptive Step Size',
                                        id='ad_step2',
                                        options=[{"label": " Yes", "value": 1},
                                                 {"label": " No", "value": 0}],
                                        value=0,
                                        style={'margin': '10px 10px 10px 10px'},
                                    )
                                ])]),
                            dbc.Col([
                                dbc.Tabs([
                                    dbc.Tab(label="Simplified Model", children=[
                                        dbc.Card(
                                            id='second-card2',
                                            style={'margin': '10px 10px 10px 10px'},
                                            children=[
                                                html.H5('Simplified Model', style={'margin': '10px 10px 10px 10px'}),
                                                html.Li('Model Assumption: g = 1',
                                                        style={'margin': '5px 5px 5px 25px'}),
                                                drc.NamedDropdown(
                                                    name='Starting Values',
                                                    id='scenario-dropdown2',
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
                                                    id='simple_mass12',
                                                    min=0.01,
                                                    max=3,
                                                    step=0.00001,
                                                    marks={0.00001: '0.00001', 0.5: '0.5', 1: '1',
                                                           1.5: '1.5', 2: '2', 2.5: '2.5', 3.0: '3.0'},
                                                    value=1
                                                ),
                                                drc.NamedSlider(
                                                    name='Mass of Object 2',
                                                    id='simple_mass22',
                                                    min=0.01,
                                                    max=3,
                                                    step=0.00001,
                                                    marks={0.00001: '0.00001', 0.5: '0.5', 1: '1',
                                                           1.5: '1.5', 2: '2', 2.5: '2.5', 3.0: '3.0'},
                                                    value=1
                                                ),
                                                drc.NamedSlider(
                                                    name='Mass of Object 3',
                                                    id='simple_mass32',
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
                                            id='third-card2',
                                            style={'margin': '10px 10px 10px 10px'},
                                            children=[
                                                html.H5('Sun System', style={'margin': '10px 10px 10px 10px'}),
                                                drc.NamedDropdown(
                                                    name='Object 1',
                                                    id='object1-dropdown2',
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
                                                            'label': 'Titan',
                                                            'value': 'tit'
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
                                                    id='object2-dropdown2',
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
                                                            'label': 'Titan',
                                                            'value': 'tit'
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
                                                    id='object3-dropdown2',
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
                                                            'label': 'Titan',
                                                            'value': 'tit'
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
                                    ])])
                            ])]),
                    ]), ]
        ),

        html.Footer(
            html.Div('Project for the course "Modelling interacting particle systems in science".',
                     style={
                         'text-decoration': 'none',
                         'color': 'white',
                         'background-color': '#333399',
                         'text-align': 'center'
                     })
        )
    ])]
                           )


# Collapse function for the infos about the project
@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


@app.callback(
    Output(component_id='loading-1', component_property='children'),
    Output(component_id='param_infos1', component_property='children'),
    Input(component_id='method1', component_property='value'),
    Input(component_id='model1', component_property='value'),
    Input(component_id='h-dropdown1', component_property='value'),
    Input(component_id='time1', component_property='value'),
    Input(component_id='scenario-dropdown1', component_property='value'),
    Input(component_id='simple_mass11', component_property='value'),
    Input(component_id='simple_mass21', component_property='value'),
    Input(component_id='simple_mass31', component_property='value'),
    Input(component_id='object1-dropdown1', component_property='value'),
    Input(component_id='object2-dropdown1', component_property='value'),
    Input(component_id='object3-dropdown1', component_property='value'),
    Input(component_id='ad_step1', component_property='value')
)
def update_figure(method, model, h, t_end, scenario, m1, m2, m3, o1, o2, o3, ad_step):
    time.sleep(1)
    if model == 1:
        names = ['Agent 1', 'Agent 2', 'Agent 3']
        colours = ['green', 'blue', 'red']
        colours_marker = ['rgba(124, 201, 119, 0.5)', 'rgba(105, 134, 201, 0.5)', 'rgba(207, 52, 72, 0.5)']
        init_data = init_dict[scenario]
        mass = [m1, m2, m3]
        g = 1
        if method == 'forwardeuler':
            graph1, dots1 = generate_figures(forward_euler(f, init_data, t_start, t_end, h, ad_step, mass, g),
                                             'Explicit Euler Method',
                                             names,
                                             colours, colours_marker)
        elif method == 'rungekutta':
            graph1, dots1 = generate_figures(runge_kutta_4(f, init_data, t_start, t_end, h, ad_step, mass, g),
                                             'Runge-Kutta Method',
                                             names, colours, colours_marker)
        elif method == 'backwardeuler':
            try:
                graph1, dots1 = generate_figures(backward_euler(f, init_data, t_start, t_end, h, ad_step, mass, g),
                                                 'Implicit Euler Method',
                                                 names, colours, colours_marker)
            except:
                graph1, dots1 = fig_not_convergent('Implicit Euler Method')
        elif method == 'heun':

            graph1, dots1 = generate_figures(heun(f, init_data, t_start, t_end, h, ad_step, mass, g),
                                             'Heun Method',
                                             names, colours, colours_marker)
        else:
            graph1, dots1 = fig_empty()
        div_graph1 = generate_div_graph(graph1, 1)
        return div_graph1, dots1
    elif model == 2:
        g = GG
        names = [name_dict[o1], name_dict[o2], name_dict[o3]]
        colours = [colour_dict[o1], colour_dict[o2], colour_dict[o3]]
        colours_marker = [colour_marker_dict[o1], colour_marker_dict[o2], colour_marker_dict[o3]]
        mass = np.array([mass_dict[o1], mass_dict[o2], mass_dict[o3]])
        init_data = np.array([[r_dict[o1], v_dict[o1]], [r_dict[o2], v_dict[o2]], [r_dict[o3], v_dict[o3]]])
        if method == 'forwardeuler':
            graph1, dots1 = generate_figures(forward_euler(f, init_data, t_start, t_end, h, ad_step, mass, g),
                                             'Explicit Euler Method',
                                             names,
                                             colours, colours_marker)
        elif method == 'rungekutta':
            graph1, dots1 = generate_figures(runge_kutta_4(f, init_data, t_start, t_end, h, ad_step, mass, g),
                                             'Runge-Kutta Method',
                                             names, colours, colours_marker)
        elif method == 'backwardeuler':
            try:
                graph1, dots1 = generate_figures(backward_euler(f, init_data, t_start, t_end, h, ad_step, mass, g),
                                                 'Implicit Euler Method',
                                                 names, colours, colours_marker)
            except:
                graph1, dots1 = fig_not_convergent('Implicit Euler Method')
        elif method == 'heun':

            graph1, dots1 = generate_figures(heun(f, init_data, t_start, t_end, h, ad_step, mass, g),
                                             'Heun Method',
                                             names, colours, colours_marker)
        else:
            graph1, dots1 = fig_empty()
        div_graph1 = generate_div_graph(graph1, 1)
        return div_graph1, dots1


@app.callback(
    Output(component_id='loading-2', component_property='children'),
    Output(component_id='param_infos2', component_property='children'),
    Input(component_id='method2', component_property='value'),
    Input(component_id='model2', component_property='value'),
    Input(component_id='h-dropdown2', component_property='value'),
    Input(component_id='time2', component_property='value'),
    Input(component_id='scenario-dropdown2', component_property='value'),
    Input(component_id='simple_mass12', component_property='value'),
    Input(component_id='simple_mass22', component_property='value'),
    Input(component_id='simple_mass32', component_property='value'),
    Input(component_id='object1-dropdown2', component_property='value'),
    Input(component_id='object2-dropdown2', component_property='value'),
    Input(component_id='object3-dropdown2', component_property='value'),
    Input(component_id='ad_step2', component_property='value')
)
def update_figure(method, model, h, t_end, scenario, m1, m2, m3, o1, o2, o3, ad_step):
    if model == 1:
        names = ['Agent 1', 'Agent 2', 'Agent 3']
        colours = ['green', 'blue', 'red']
        colours_marker = ['rgba(124, 201, 119, 0.5)', 'rgba(105, 134, 201, 0.5)', 'rgba(207, 52, 72, 0.5)']
        init_data = init_dict[scenario]
        mass = [m1, m2, m3]
        g = 1
        if method == 'forwardeuler':
            graph2, dots2 = generate_figures(forward_euler(f, init_data, t_start, t_end, h, ad_step, mass, g),
                                             'Explicit Euler Method',
                                             names,
                                             colours, colours_marker)
        elif method == 'rungekutta':
            graph2, dots2 = generate_figures(runge_kutta_4(f, init_data, t_start, t_end, h, ad_step, mass, g),
                                             'Runge-Kutta Method',
                                             names, colours, colours_marker)
        elif method == 'backwardeuler':
            try:
                graph2, dots2 = generate_figures(backward_euler(f, init_data, t_start, t_end, h, ad_step, mass, g),
                                                 'Implicit Euler Method',
                                                 names, colours, colours_marker)
            except:
                graph2, dots2 = fig_not_convergent('Implicit Euler Method')
        elif method == 'heun':

            graph2, dots2 = generate_figures(heun(f, init_data, t_start, t_end, h, ad_step, mass, g),
                                             'Heun Method',
                                             names, colours, colours_marker)
        else:
            graph2, dots2 = fig_empty()
        div_graph2 = generate_div_graph(graph2, 2)
        return div_graph2, dots2
    elif model == 2:
        g = GG
        names = [name_dict[o1], name_dict[o2], name_dict[o3]]
        colours = [colour_dict[o1], colour_dict[o2], colour_dict[o3]]
        colours_marker = [colour_marker_dict[o1], colour_marker_dict[o2], colour_marker_dict[o3]]
        mass = np.array([mass_dict[o1], mass_dict[o2], mass_dict[o3]])
        init_data = np.array([[r_dict[o1], v_dict[o1]], [r_dict[o2], v_dict[o2]], [r_dict[o3], v_dict[o3]]])
        if method == 'forwardeuler':
            graph2, dots2 = generate_figures(forward_euler(f, init_data, t_start, t_end, h, ad_step, mass, g),
                                             'Explicit Euler Method',
                                             names,
                                             colours, colours_marker)
        elif method == 'rungekutta':
            graph2, dots2 = generate_figures(runge_kutta_4(f, init_data, t_start, t_end, h, ad_step, mass, g),
                                             'Runge-Kutta Method',
                                             names, colours, colours_marker)
        elif method == 'backwardeuler':
            try:
                graph2, dots2 = generate_figures(backward_euler(f, init_data, t_start, t_end, h, ad_step, mass, g),
                                                 'Implicit Euler Method',
                                                 names, colours, colours_marker)
            except:
                graph2, dots2 = fig_not_convergent('Implicit Euler Method')
        elif method == 'heun':

            graph2, dots2 = generate_figures(heun(f, init_data, t_start, t_end, h, ad_step, mass, g),
                                             'Heun Method',
                                             names, colours, colours_marker)
        else:
            graph2, dots2 = fig_empty()
        div_graph2 = generate_div_graph(graph2, 2)
        return div_graph2, dots2


def f(y, mass, g):
    d0 = ((-g * mass[0] * mass[1] * (y[0] - y[1]) / norm(y[0] - y[1]) ** 3) +
          (-g * mass[0] * mass[2] * (y[0] - y[2]) / norm(y[0] - y[2]) ** 3)) / mass[0]
    d1 = ((-g * mass[1] * mass[2] * (y[1] - y[2]) / norm(y[1] - y[2]) ** 3) + (
            -g * mass[1] * mass[0] * (y[1] - y[0]) / norm(y[1] - y[0]) ** 3)) / mass[1]
    d2 = ((-g * mass[2] * mass[0] * (y[2] - y[0]) / norm(y[2] - y[0]) ** 3) + (
            -g * mass[2] * mass[1] * (y[2] - y[1]) / norm(y[2] - y[1]) ** 3)) / mass[2]
    return np.array([d0, d1, d2])


def flong(y):
    d00 = ((-g * mass[0] * mass[1] * (y[0] - y[2]) / norm([y[0] - y[2], y[1] - y[3]]) ** 3) +
           (-g * mass[0] * mass[2] * (y[0] - y[4]) / norm([y[0] - y[4], y[1] - y[5]]) ** 3)) / mass[0]
    d01 = ((-g * mass[0] * mass[1] * (y[1] - y[3]) / norm([y[0] - y[2], y[1] - y[3]]) ** 3) + (
            -g * mass[0] * mass[2] * (y[1] - y[5]) / norm([y[0] - y[4], y[1] - y[5]]) ** 3)) / mass[0]
    d10 = ((-g * mass[1] * mass[2] * (y[2] - y[4]) / norm([y[2] - y[4], y[3] - y[5]]) ** 3) + (
            -g * mass[1] * mass[0] * (y[2] - y[0]) / norm([y[0] - y[2], y[1] - y[3]]) ** 3)) / mass[1]
    d11 = ((-g * mass[1] * mass[2] * (y[3] - y[5]) / norm([y[2] - y[4], y[3] - y[5]]) ** 3) + (
            -g * mass[1] * mass[0] * (y[3] - y[1]) / norm([y[0] - y[2], y[1] - y[3]]) ** 3)) / mass[1]
    d20 = ((-g * mass[2] * mass[0] * (y[4] - y[0]) / norm([y[0] - y[4], y[1] - y[5]]) ** 3) + (
            -g * mass[2] * mass[1] * (y[4] - y[2]) / norm([y[2] - y[4], y[3] - y[5]]) ** 3)) / mass[2]
    d21 = ((-g * mass[2] * mass[0] * (y[5] - y[1]) / norm([y[0] - y[4], y[1] - y[5]]) ** 3) + (
            -g * mass[2] * mass[1] * (y[5] - y[3]) / norm([y[2] - y[4], y[3] - y[5]]) ** 3)) / mass[2]
    return np.array([d00, d01, d10, d11, d20, d21])


def forward_euler(f, y0, t0, t1, h, ad_step, mass, g):
    """
    Explicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float, step-size
    :param ad_step: int, 0 if adaptive step size is deactivated and 1 if it activated
    :param mass: list of floats or ints, masses of particles
    :param g: float or int, gravitational constant
    :return: list of postitions, list of approximated errors at each step, number of steps, total time
    """
    h_min = h / 256
    h_max = h
    h_sum = 0
    k = 0
    t = t0
    v = np.zeros((len(y0), 1000000, 2))
    y = np.zeros((len(y0), 1000000, 2))
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    errlist = []
    eps = 1e-15
    while h_sum < t1 and k < 50000:

        y[:, k + 1, :] = y[:, k, :] + h * v[:, k, :]  # + h**2 * 0.5 * f(t, y[:, k, :])
        v[:, k + 1, :] = v[:, k, :] + h * f(y[:, k, :], mass, g)

        energy = norm(
            mass[0] * f(y[:, k, :], mass, g)[0] + mass[1] * f(y[:, k, :], mass, g)[1] + mass[2] *
            f(y[:, k, :], mass, g)[2])

        errlist.append(energy)

        if ad_step == 1:

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
        else:
            k += 1
            h_sum += h

    y = y[:, :k + 1, :]
    print('Forward Euler k: ', k)
    print('Maximaler Error Forward Euler: ', max(errlist))
    return y, errlist, k, h_sum


def runge_kutta(f, y0, t0, t1, h, mass, g):
    """
    Runge-Kutta method of order 3 for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float, step-size
    :param mass: list of floats or ints, masses of particles
    :param g: float or int, gravitational constant
    :return: list of postitions, list of approximated errors at each step, number of steps, total time
    """
    N = int(np.ceil((t1 - t0) / h))
    t = t0
    v = np.zeros((len(y0), N + 1, 2))
    y = np.zeros((len(y0), N + 1, 2))
    t_list = [0] * (N + 1)
    t_list[0] = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    for k in range(N):
        for i in range(len(y0)):
            k1 = f(y[:, k, :], mass, g)[i]
            k2 = f((y[:, k, :] + h * k1 / 2), mass, g)[i]
            k3 = f((y[:, k, :] + h * (-k1 + 2 * k2)), mass, g)[i]
            v[i, k + 1] = v[i, k, :] + h / 6 * (k1 + 4 * k2 + k3)
            k12 = v[i, k, :]
            k22 = v[i, k, :] + h * k1 / 2
            k32 = v[i, k, :] + h * (-k1 + 2 * k2)
            y[i, k + 1] = y[i, k, :] + h / 6 * (k12 + 4 * k22 + k32)
        t = t + h
        t_list[k + 1] = t
    return y


A = np.array([0, 2 / 9, 1 / 3, 3 / 4, 1, 5 / 6])
B = np.matrix([[0, 0, 0, 0, 0], [2 / 9, 0, 0, 0, 0], [1 / 12, 1 / 4, 0, 0, 0], [69 / 128, -243 / 128, 135 / 64, 0, 0],
               [-17 / 12, 27 / 4, -27 / 5, 16 / 15, 0], [65 / 432, -5 / 16, 13 / 16, 4 / 27, 5 / 144]])
C = np.array([1 / 9, 0, 9 / 20, 16 / 45, 1 / 12])
CH = np.array([47 / 450, 0, 12 / 25, 32 / 225, 1 / 30, 6 / 25])
CT = np.array([-1 / 150, 0, 3 / 100, -16 / 75, -1 / 20, 6 / 25])


def runge_kutta_4(f, y0, t0, t1, h, ad_step, mass, g):
    """
    Runge-Kutta-Fehlberg method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float, step-size
    :param ad_step: int, 0 if adaptive step size is deactivated and 1 if it activated
    :param mass: list of floats or ints, masses of particles
    :param g: float or int, gravitational constant
    :return: list of postitions, list of approximated errors at each step, number of steps, total time
    """
    h_min = h / 16
    h_max = h
    h_sum = 0
    eps = 1e-16
    t = t0
    v = np.zeros((len(y0), 50001, 2))
    y = np.zeros((len(y0), 50001, 2))
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    k = 0
    errlist = []
    while k < 50000 and h_sum < t1:
        i = 0

        l1_all = np.zeros((3, 2))
        l2_all = np.zeros((3, 2))
        l3_all = np.zeros((3, 2))
        l4_all = np.zeros((3, 2))
        l5_all = np.zeros((3, 2))
        l6_all = np.zeros((3, 2))

        while i < (len(y0)):
            k1 = h * f(y[:, k, :], mass, g)[i]
            k2 = h * f((y[:, k, :] + B[1, 0] * h * k1), mass, g)[i]
            k3 = h * f((y[:, k, :] + B[2, 0] * k1 + B[2, 1] * k2), mass, g)[i]
            k4 = h * f((y[:, k, :] + B[3, 0] * k1 + B[3, 1] * k2 + B[3, 2] * k3), mass, g)[i]
            k5 = h * f((y[:, k, :] + B[4, 0] * k1 + B[4, 1] * k2 + B[4, 2] * k3 + B[4, 3] * k4), mass, g)[i]
            k6 = h * f(
                (y[:, k, :] + B[5, 0] * k1 + B[5, 1] * k2 + B[5, 2] * k3 + B[5, 3] * k4 + B[5, 4] * k5), mass, g)[i]
            v[i, k + 1] = v[i, k] + CH[0] * k1 + CH[1] * k2 + CH[2] * k3 + CH[3] * k4 + CH[4] * k5 + CH[5] * k6

            l1 = v[i, k, :]
            l2 = v[i, k, :] + B[1, 0] * h * k1
            l3 = v[i, k, :] + B[2, 0] * k1 + B[2, 1] * k2
            l4 = v[i, k, :] + B[3, 0] * k1 + B[3, 1] * k2 + B[3, 2] * k3
            l5 = v[i, k, :] + B[4, 0] * k1 + B[4, 1] * k2 + B[4, 2] * k3 + B[4, 3] * k4
            l6 = v[i, k, :] + B[5, 0] * k1 + B[5, 1] * k2 + B[5, 2] * k3 + B[5, 3] * k4 + B[5, 4] * k5
            y[i, k + 1] = y[i, k] + h * CH[0] * l1 + h * CH[1] * l2 + h * CH[2] * l3 + h * CH[3] * l4 + h * CH[
                4] * l5 + h * CH[5] * l6 + h ** 2 * 0.5 * f(y[:, k, :], mass, g)[i]

            l1_all[i, :] = l1
            l2_all[i, :] = l2
            l3_all[i, :] = l3
            l4_all[i, :] = l4
            l5_all[i, :] = l5
            l6_all[i, :] = l6

            i += 1

        TE = norm(
            CT[0] * l1_all + CT[1] * l2_all + CT[2] * l3_all + CT[3] * l4_all + CT[4] * l5_all + CT[5] * l6_all)
        errlist.append(TE)

        if ad_step == 1:

            if TE < eps:
                k += 1
                h_sum = h_sum + h
                if TE < eps ** 2 and h < h_max:
                    h = h * 1.1
            elif TE > eps and h > h_min:
                h = h * 0.9 * (eps / TE) ** 0.2
            else:
                k += 1
                h_sum = h_sum + h

        else:
            k += 1
            h_sum = h_sum + h

    print('RK4 k: ', k)
    print('Maximaler Error RKF: ', max(errlist))

    y = y[:, :k + 1, :]
    return y, errlist, k, h_sum


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


def backward_euler(f, y0, t0, t1, h, ad_step, mass, g):
    """
    Implicit Euler method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float, step-size
    :param ad_step: int, 0 if adaptive step size is deactivated and 1 if it activated
    :param mass: list of floats or ints, masses of particles
    :param g: float or int, gravitational constant
    :return: list of postitions, list of approximated errors at each step, number of steps, total time
    """
    h_min = h / 16
    h_max = h
    h_sum = 0
    k = 1
    v = np.zeros((len(y0), 50001, 2))
    y = np.zeros((len(y0), 50001, 2))
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    eps = 5e-16
    t = t0
    errlist = []
    while k < 50001 and h_sum < t1:

        for i in range(len(y0)):

            def fixpoint(x):
                terms = []
                for j in range(len(x)):
                    if j != i:
                        term = (-g * mass[0] * mass[1] * (x[i, :] - x[j, :]) / norm(x[i, :] - x[j, :]) ** 3) / \
                               mass[i]
                        terms.append(term)

                return v[i, k - 1, :] + h * (terms[0] + terms[1]) - x[i, :]

            def fixpoint_deriv(x):
                terms = []
                for j in range(len(x)):
                    if j != i:
                        term = (g * mass[0] * mass[1] * (
                                2 * x[i, :] ** 2 - 3 * x[i, :] * x[j, :] + x[j, :] ** 2) / norm(
                            x[i, :] - x[j, :]) ** (5 / 2)) / mass[i]
                        terms.append(term)
                return h * (terms[0] + terms[1]) - 1

            v[i, k, :] = newton_raphson(fixpoint, fixpoint_deriv, y[:, k - 1, :], 0.0001, 5)[i, :]
            y[i, k, :] = y[i, k - 1, :] + h * v[i, k - 1, :]  # + h ** 2 * 0.5 * f(t, y[:, k - 1, :])[i]

        energy = norm(
            mass[0] * f(y[:, k, :], mass, g)[0] + mass[1] * f(y[:, k, :], mass, g)[1] + mass[2] *
            f(y[:, k, :], mass, g)[2])
        errlist.append(energy)

        if ad_step == 1:

            if abs(energy) < eps:
                k = k + 1
                h_sum = h_sum + h
                if abs(energy) < eps * 0.5 and h < h_max:
                    h = h * 2
            elif abs(energy) > eps and h > h_min:
                h = h * 0.5
            else:
                k = k + 1
                h_sum = h_sum + h
        else:
            k += 1
            h_sum += h

    y = y[:, :k, :]

    print('Backward Euler k: ', k)
    print('Maximaler Error Backward Euler: ', max(errlist))

    return y, errlist, k - 1, h_sum


def predictor_corrector(f, y0, t0, t1, h, ad_step, mass, g):
    """
    Predictor-Corrector method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float or int, step-size
    :param ad_step: int, 0 if adaptive step size is deactivated and 1 if it activated
    :param mass: list of floats or ints, masses of particles
    :param g: float or int, gravitational constant
    :return: list of postitions, list of approximated errors at each step, number of steps, total time
    """
    v = np.zeros((len(y0), 1000000, 2))
    y = np.zeros((len(y0), 1000000, 2))
    t = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    h_sum = 0
    h_min = h / 256
    h_max = h
    k = 1
    errlist = [0]
    eps = 1e-15
    while h_sum < t1 and k < 50000:

        y_pre = y[:, k - 1, :] + h * v[:, k - 1, :] + h ** 2 * 0.5 * f(y[:, k - 1, :], mass, g)

        for i in range(len(y0)):
            v[i, k, :] = v[i, k - 1, :] + h * f(y_pre, mass, g)[i]
            y[i, k, :] = y[i, k - 1, :] + h * v[i, k - 1, :]

        err = norm(
            mass[0] * f(t, y[:, k, :])[0] + mass[1] * f(y[:, k, :], mass, g)[1] + mass[2] * f(y[:, k, :], mass, g)[2])
        errlist.append(err)

        if ad_step == 1:
            if err < eps:
                k = k + 1
                h_sum = h_sum + h
                if err < eps ** 2 and h < h_max:
                    h = h * 2
            elif err > eps and h > h_min:
                h = h * 0.5
            else:
                k = k + 1
                h_sum = h_sum + h

        else:
            k = k + 1
            h_sum += h

        t = t + h
    y = y[:, 1:k, :]
    print('PK k: ', k)
    max_error = max(errlist)
    return y, errlist, k - 1, h_sum


def heun(f, y0, t0, t1, h, ad_step, mass, g):
    """
    Heun method for systems of differential equations: y' = f(t, y); with f,y,y' n-dimensional vectors.
    :param f: list of functions
    :param y0: list of floats or ints, initial values y(t0)=y0
    :param t0: float or int, start of interval for parameter t
    :param t1: float or int, end of interval for parameter t
    :param h: float, step-size
    :param ad_step: int, 0 if adaptive step size is deactivated and 1 if it activated
    :param mass: list of floats or ints, masses of particles
    :param g: float or int, gravitational constant
    :return: list of postitions, list of approximated errors at each step, number of steps, total time
    """
    v = np.zeros((len(y0), 1000000, 2))
    y = np.zeros((len(y0), 1000000, 2))
    t = t0
    v[:, 0, :] = y0[:, 1, :]
    y[:, 0, :] = y0[:, 0, :]
    h_sum = 0
    h_min = h / 256
    h_max = h
    k = 1
    errlist = [0]
    eps = 5e-16
    while h_sum < t1 and k < 50001:

        v_pre = v[:, k - 1, :] + h * f(y[:, k - 1, :], mass, g)
        y_pre = y[:, k - 1, :] + h * v[:, k - 1, :] + h ** 2 * 0.5 * f(y[:, k - 1, :], mass, g)

        v[:, k, :] = 0.5 * v[:, k - 1, :] + 0.5 * (v_pre + h * f(y_pre, mass, g))
        y[:, k, :] = 0.5 * y[:, k - 1, :] + 0.5 * (
                y_pre + h * v[:, k - 1, :] + h ** 2 * 0.5 * f(y[:, k - 1, :], mass, g))

        err = norm(
            mass[0] * f(y[:, k, :], mass, g)[0] + mass[1] * f(y[:, k, :], mass, g)[1] + mass[2] *
            f(y[:, k, :], mass, g)[2])
        errlist.append(err)

        if ad_step == 1:
            if err < eps:
                k = k + 1
                h_sum = h_sum + h
                if err < eps ** 2 and h < h_max:
                    h = h * 2
            elif err > eps and h > h_min:
                h = h * 0.5
            else:
                k = k + 1
                h_sum = h_sum + h
                # print('h zu klein')
        else:
            k = k + 1
            h_sum += h

        t = t + h
    y = y[:, :k, :]
    print('Heun k: ', k)
    max_error = max(errlist)
    print('Maximaler Error Heun: ', max_error)

    return y, errlist, k - 1, h_sum


def generate_figures(method, title, names, colours, colours_marker):
    y, errlist, k, t_stop = method
    fig = go.Figure(
        data=[go.Scatter(x=y[0, :, 0], y=y[0, :, 1],
                         mode="lines", name=names[0],
                         line=dict(width=2, color=colours[0])),
              go.Scatter(x=y[1, :, 0], y=y[1, :, 1],
                         mode="lines", name=names[1],
                         line=dict(width=2, color=colours[1])),
              go.Scatter(x=y[2, :, 0], y=y[2, :, 1],
                         mode="lines", name=names[2],
                         line=dict(width=2, color=colours[2])),
              go.Scatter(x=y[0, :, 0], y=y[0, :, 1],
                         mode="lines",
                         line=dict(width=2, color=colours[0]),
                         showlegend=False),
              go.Scatter(x=y[1, :, 0], y=y[1, :, 1],
                         mode="lines",
                         line=dict(width=2, color=colours[1]),
                         showlegend=False),
              go.Scatter(x=y[2, :, 0], y=y[2, :, 1],
                         mode="lines",
                         line=dict(width=2, color=colours[2]),
                         showlegend=False)
              ],
        layout=go.Layout(
            xaxis=dict(autorange=True, zeroline=False, range=[-2, 2]),
            yaxis=dict(autorange=True, zeroline=False, range=[-2, 2]),
            title=title, hovermode="closest",
            updatemenus=[dict(type="buttons",
                              buttons=[dict(label="Play",
                                            method="animate",
                                            args=[None, {"frame": {"duration": 30, "redraw": False},
                                                         "fromcurrent": True, "transition": {"duration": 10,
                                                                                             "easing": "quadratic-in-out"}}]),
                                       dict(label="Pause",
                                            method="animate",
                                            args=[[None], {"frame": {"duration": 0, "redraw": False},
                                                           "mode": "immediate",
                                                           "transition": {"duration": 0}}])
                                       ])]),
        frames=[go.Frame(
            data=[go.Scatter(
                x=[y[0, n, 0]],
                y=[y[0, n, 1]],
                mode="markers",
                marker=dict(color=colours[0], size=10)),
                go.Scatter(x=[y[1, n, 0]], y=[y[1, n, 1]],
                           mode="markers",
                           marker=dict(color=colours[1], size=10)),
                go.Scatter(x=[y[2, n, 0]], y=[y[2, n, 1]],
                           mode="markers",
                           marker=dict(color=colours[2], size=10))
            ])

            for n in range(k + 1)]
    )
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=[y[0, 0, 0]],
            y=[y[0, 0, 1]],
            name='Start Value',
            marker=dict(
                color=colours_marker[0],
                size=10,
                line=dict(
                    color=colours[0],
                    width=2
                )
            ),
            showlegend=False
        ),
    )
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=[y[1, 0, 0]],
            y=[y[1, 0, 1]],
            name='Start Value',
            marker=dict(
                color=colours_marker[1],
                size=10,
                line=dict(
                    color=colours[1],
                    width=2
                )
            ),
            showlegend=False
        ),
    )
    fig.add_trace(
        go.Scatter(
            mode='markers',
            x=[y[2, 0, 0]],
            y=[y[2, 0, 1]],
            name='Start Value',
            marker=dict(
                color=colours_marker[2],
                size=10,
                line=dict(
                    color=colours[2],
                    width=2
                )
            ),
            showlegend=False
        )
    )
    parameter_infos = [dbc.Col(html.Ul([
        html.Li(str('Mean Error: ' + str(np.mean(errlist)))),
        html.Li(str('Max. Error: ' + str(max(errlist))))
    ])),
        dbc.Col([html.Ul([
            html.Li(str('Total steps: ' + str(k))),
            html.Li(str('End Time: ' + str(round(t_stop, 2))))
        ])])]
    return fig, parameter_infos


def fig_not_convergent(title):
    fig = px.scatter(x=[1, 1, 1, 1, 1, 1, 1, 1], y=[3, 2.75, 2.5, 2.25, 2, 1.75, 1.5, 0.75],
                     title=str(title + ': not convergent'))
    dots = [html.Li('Not convergent!')]
    return fig, dots


def fig_empty():
    fig = px.scatter(x=[1, 1, 1, 1, 1, 1, 1, 1], y=[3, 2.75, 2.5, 2.25, 2, 1.75, 1.5, 0.75],
                     title='No method selected')
    dots = []
    return fig, dots


def generate_div_graph(graph, graph_nr):
    """
    Generate dashboard environment for graph
    :param graph: plotly figure
    :param graph_nr: int, for id of bullet points
    :return: html component
    """
    div_graph = html.Div(
        style={
            'text-decoration': 'none',
            'color': 'black',
            'background-color': 'white',
            'text-align': '',
            'margin': '0px 0px 30px 50px'
        },
        children=[
            dcc.Graph(
                figure=graph
            ),
            dbc.Row(id=str('param_infos'+str(graph_nr)))],
    )
    return div_graph


def error_data_to_csv():
    """
    Get csv of maximal errors
    :return: nothing, csv will be saved into repository 'csv_data'
    """
    g = 1
    m = [1, 1, 1]
    adStep_dict = {0: 'without AdStep', 1: 'AdStep'}
    error_dict = {}

    for i in range(1, 4):

        for j in [0, 1]:

            y, errFE1, n, t1 = forward_euler(f, init_dict[i], 0, 10, 0.01, j, m, g)
            y, errFE2, n, t1 = forward_euler(f, init_dict[i], 0, 10, 0.001, j, m, g)
            try:
                y, errBE1, n, t1 = backward_euler(f, init_dict[i], 0, 10, 0.01, j, m, g)
            except:
                errBE1 = 'not convergent'
            try:
                y, errBE2, n, t1 = backward_euler(f, init_dict[i], 0, 10, 0.001, j, m, g)
            except:
                errBE2 = 'not convergent'
            y, errRK1, n, t1 = runge_kutta_4(f, init_dict[i], 0, 10, 0.01, j, m, g)
            y, errRK2, n, t1 = runge_kutta_4(f, init_dict[i], 0, 10, 0.001, j, m, g)
            y, errH1, n, t1 = heun(f, init_dict[i], 0, 10, 0.01, j, m, g)
            y, errH2, n, t1 = heun(f, init_dict[i], 0, 10, 0.001, j, m, g)

            error_dict[str('Szenario '+str(i)+', '+adStep_dict[j]+', h=0.01')] = [errFE1, errBE1, errRK1, errH1]
            error_dict[str('Szenario '+str(i)+', '+adStep_dict[j]+', h=0.001')] = [errFE2, errBE2, errRK2, errH2]
            print(error_dict)

    error_data = pd.DataFrame(data=error_dict, index=['Explicit Euler', 'Implicit Euler', 'Runge-Kutta', 'Heun'])
    error_data.to_csv(path_or_buf='csv_data/Error_Data.csv')


def array_to_csv(y, name):
    """
    Get results as csv data
    :param y: results
    :param name:
    :return:
    """
    x0 = y[0, :, 0]
    y0 = y[0, :, 1]
    x1 = y[1, :, 0]
    y1 = y[1, :, 1]
    x2 = y[2, :, 0]
    y2 = y[2, :, 1]
    df = pd.DataFrame([x0, y0, x1, y1, x2, y2], index=['Body 1, x-Axis', 'Body 1, y-Axis', 'Body 2, x-Axis',
                                                       'Body 2, y-Axis', 'Body 3, x-Axis', 'Body 3, y-Axis'])
    df = df.transpose()
    df.to_csv(name)
    return df


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
