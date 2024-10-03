# import libraries
from dash import Dash, dcc, html, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.colors import n_colors


# dataset
df = pd.read_csv("c:/users/sahil/downloads/Call-Center-Dataset (1).csv")

# changes to dataset
df['Talk Duration(sec)'] = pd.to_timedelta(df['AvgTalkDuration']).dt.total_seconds()
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
df['Hour'] = df['Time'].dt.hour
df['Date'] = pd.to_datetime(df['Date'])

# plotly app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H3('Shree Call Center', className='title'),
                html.Img(src="/assets/call-center-logo.jpg", style={'height': '50px', 'margin-right': '15px'})
            ], className='title-div')    
        ], className='title_col')
    ]),

    dbc.Row([
        dbc.Col([
            html.H5("Q1 Analysis Dashboard", className="sub-header")
        ], className='header-col')
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Calls'),
                dbc.CardBody(id='total_calls')
            ])
        ], width={'size': 1}, className='custom_cards', id='calls'),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Calls Answered'),
                dbc.CardBody(id='calls_answered')
            ])
        ], width={'size': 2}, className='custom_cards', id='answered'),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader('Problem Resolved'),
                dbc.CardBody(id='problems_resolved')
            ])    
        ], width={'size': 2}, className='custom_cards'),
    
        dbc.Col([
            dbc.Card([
                dbc.CardHeader('AVG Call Time'),
                dbc.CardBody(id='avg_call_time')
            ])
        ], width={'size': 2}, className='custom_cards'),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader('AVG Speed of Ans'),
                dbc.CardBody(id='answer_speed')
            ])
        ], width={'size': 2}, className='custom_cards'),
    
        dbc.Col([
            dcc.Graph(id='gauge_chart')
        ], width={'size': 3}, className='gauge_col'),
    ]),    

    # inputs
    html.Div([
            dcc.Dropdown(
                id='dropdown',
                options=[{'label': i, 'value': i} for i in df['Agent'].unique()],
                value=None,
                placeholder="Select an Agent",
                className='filters',
                multi=True
            ),

            dcc.Dropdown(
                id='dropdown2',
                options=[{'label': topic, 'value': topic} for topic in df['Topic'].unique()],
                value=None,
                placeholder="select Topic",
                className='filters'
            ),

            dcc.DatePickerRange(
                id='date_picker',
                min_date_allowed=df['Date'].min(),
                max_date_allowed=df['Date'].max(),
                start_date=df['Date'].min(),
                end_date=df['Date'].max(),
                className='filters'
            ),
    ], id='container'),    

    # charts
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='bar_chart')
        ]),

        dbc.Col([
            dcc.Graph(id='scatter_chart')
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='line_chart')
        ]),

        dbc.Col([
            dcc.Graph(id='pie_chart1')
        ], width={'size': 4}),
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='line_chart2',
            )
        ]),

        dbc.Col([
            dcc.Graph(id='pie_chart2')
        ], width={'size': 4})
    ]),

    dbc.Row([
        dbc.Col([
            dcc.Graph(id='table_chart')
        ])
    ])
], className="dbc-cont")

# Callback
@app.callback(
    [Output('total_calls', 'children'), Output('calls_answered', 'children'),
     Output('problems_resolved', 'children'), Output('avg_call_time', 'children'),
     Output('answer_speed', 'children'),
     Output('gauge_chart', 'figure'),
     Output('bar_chart', 'figure'), Output('scatter_chart', 'figure'),
     Output('line_chart', 'figure'), Output('pie_chart1', 'figure'),
     Output('line_chart2', 'figure'), Output('pie_chart2', 'figure'),
     Output('table_chart', 'figure')
     ],
    [Input('dropdown', 'value'), Input('dropdown2', 'value'),
     Input('date_picker', 'start_date'), Input('date_picker', 'end_date')
     ]
)
def update_graph(agent_value, topic_value, start_date, end_date):

    dff = df

    # filter inputs
    dff = dff[(dff['Date'] >= start_date) & (dff['Date'] <= end_date)]

    if agent_value:  # agent_value will be a list when multi=True
        dff = dff[dff['Agent'].isin(agent_value)]

    if topic_value is not None:
        dff = dff[dff['Topic'] == topic_value]


    # filter cards
    total_calls = dff['Agent'].count()    

    calls_answered = dff['Answered (Y/N)'][(dff['Answered (Y/N)'] == 'Y')].count()

    problems_resolved = dff['Resolved'][(dff['Resolved'] == 'Y')].count()

    avg_call_time = f"{round(dff['Talk Duration(sec)'].mean() / 60, 2)} min"

    answer_speed = f"{round(dff['Speed of answer in seconds'].mean())} sec"

    # gauge chart
    avg_sat = dff['Satisfaction rating'].mean()
    print(f"avg_csat- {avg_sat}")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_sat,
        title={'text': "AVG CSAT", 'font': {'size': 15}},
        delta={'reference': 4, 'font': {'size': 15}},  # delta=reference-value
        gauge={'axis': {'range': [None, 5],
                        'tickvals': [0, 1, 2, 3, 4, 5],},
            'bar': {'color': "green"},
            'steps': [
                    {'range': [0, 3.5], 'color': "lightgray"},
                    {'range': [3.5, 4.5], 'color': "gray"}],
            'threshold': {'line': {'color': "red", },  'value': 4}},
        number={'font': {'size': 20}}
    )).update_layout(
            width=195,  
            height=190,  
            margin=dict(l=15, r=15, t=5, b=15),
            paper_bgcolor='rgb(250,250,250)',
        )
    
    # bar chart
    answered_count = dff[dff['Answered (Y/N)'] == 'Y'].groupby('Topic').size().reset_index(name='count')
    resolved_count = dff[dff['Resolved'] == 'Y'].groupby('Topic').size().reset_index(name='count')
    merged1 = answered_count.merge(resolved_count, on='Topic', suffixes=('_ans', '_resolve'))
    print(merged1)
    fig_bar = go.Figure(data=[
    go.Bar(name='Answered', x=merged1['Topic'], y=merged1['count_ans'], text=[str(value) for value in merged1['count_ans']],
           marker={'color': '#ffbe0b'}),
    go.Bar(name='Resolved', x=merged1['Topic'], y=merged1['count_resolve'], text=[str(value) for value in merged1['count_resolve']],
           textposition='outside', marker=dict(color='#fb5607'))
    ])
    fig_bar.update_layout(
    barmode='group',
    title=f"Calls Answered vs Problem Resolved by {agent_value if agent_value else 'All Agents'}",
    xaxis_title='Topic',
    yaxis_title='Count',
    legend=dict(x=0, y=1, orientation='h', font=dict(size=10)),
    margin=dict(l=20, r=20, t=50, b=0,),
    paper_bgcolor='rgb(250,250,250)',
    xaxis=dict(
        tickvals=['Admin Support', 'Contract related', 'Payment related', 'Streaming', 'Technical Support'],
        ticktext=[
            'Admin<br>Support',
            'Contract<br>related',
            'Payment<br>related',
            'Streaming',
            'Technical<br>Support',
        ])
    )

    # scatter chart
    df_resolved = dff[dff['Resolved'] == 'Y']
    a = (df_resolved.groupby('Agent').agg({'Resolved': 'count', 'Speed of answer in seconds': lambda x: round(x.mean(), 2)})
     .reset_index())
    fig_scatter = (px.scatter(data_frame=a, x='Resolved', y='Speed of answer in seconds', text='Agent',
                              title="Agent Performance", color='Agent').
                   update_layout(margin=dict(l=20, r=20, t=50, b=20),
                                 xaxis_title="Problem Resolved", yaxis_title="AVG Speed of Answer (sec)",
                                 showlegend=False,
                                 paper_bgcolor='rgb(250,250,250)')
                   )
    fig_scatter.update_traces(textposition='top center')

    # line chart1
    hours_grouped = dff.groupby('Hour').size().reset_index(name='count')
    fig_line = px.line(data_frame=hours_grouped, x='Hour', y='count', text=[str(value) for value in hours_grouped['count']],
                       title=f"Calls by Time {agent_value if agent_value else 'All Agents'}",
                       height=400)
    fig_line.update_traces(textposition='top center', line=dict(color='#ffbe0b'))
    fig_line.update_layout(margin=dict(l=20, r=20, t=60, b=20),
                           paper_bgcolor='rgb(250,250,250)',
                           )

    # pie chart1
    b = dff.groupby('Answered (Y/N)').size().reset_index(name='count')
    fig_pie1 = px.pie(data_frame=b, names='Answered (Y/N)', values='count',
                      title=f"Calls Answered by {agent_value if agent_value else 'All Agents'}", color='Answered (Y/N)',
                      color_discrete_map={'Y': '#ffbe0b', 'N': '#fb5607'},
                      height=380)
    fig_pie1.update_layout(paper_bgcolor='rgb(250,250,250)')
    
    # line chart2
    date_grouped = dff.groupby('Date')['Call Id'].size().reset_index(name='count')
    fig_line2 = px.line(data_frame=date_grouped, x='Date', y='count',
                        title=f"Daily Calls by {agent_value if agent_value else 'All Agents'}",
                        height=430)
    fig_line2.update_traces(line=dict(color='#fb5607'))
    fig_line2.update_layout(margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor='rgb(250,250,250)'
                            )

    # pie chart2
    c = dff.groupby('Resolved').size().reset_index(name='count')
    fig_pie2 = px.pie(data_frame=c, names='Resolved', values='count',
                      title=f"Problem Resolved by {agent_value if agent_value else 'All Agents'}", color='Resolved',
                      color_discrete_map={'Y': '#ffbe0b', 'N': '#fb5607'},
                      height=400)
    fig_pie2.update_layout(paper_bgcolor='rgb(250,250,250)')

    # filtering for table
    df_answered = dff[dff['Answered (Y/N)'] == 'Y']
    ansY_group = df_answered.groupby('Agent')['Call Id'].count().reset_index(name='Answered')
    resY_group = df_resolved.groupby('Agent')['Call Id'].count().reset_index(name='Resolved')
    ansS_group = dff.groupby('Agent')['Speed of answer in seconds'].mean().round(2).reset_index(name='AVG Ans Speed(sec)')
    avg_csat = (dff.groupby('Agent')['Satisfaction rating'].mean()).round(2).reset_index().rename(
        columns={'Satisfaction rating': 'AVG CSAT'})
    perf = (resY_group['Resolved'] / ansY_group['Answered'] * 100).round(2).reset_index(name='performance')
    result = (pd.concat([ansY_group['Agent'], perf['performance']], axis=1,).sort_values(ascending=False, by='performance'))
    
    # concat groups for table
    tab_concat = pd.concat([ansY_group, resY_group, ansS_group, avg_csat, result], axis=1)
    final_table = tab_concat.loc[:, ~tab_concat.columns.duplicated()]

    # color gradient for table cell values
    colors = n_colors('rgb(255, 200, 200)', 'rgb(200, 0, 0)', 9, colortype='rgb')

    # color mapping function for cells
    def map_values_to_color(values, color):
        min_val, max_val = values.min(), values.max()
        if max_val == min_val:
            return np.array([color[len(color) // 2]] * len(values))
        normalized = ((values - min_val) / (max_val - min_val) * (len(color) - 1)).astype(int)
        return np.array(color)[normalized]

    # Map colors for cells
    answered_colors = map_values_to_color(final_table['Answered'], colors)
    resolved_colors = map_values_to_color(final_table['Resolved'], colors)
    avg_speed_colors = map_values_to_color(final_table['AVG Ans Speed(sec)'], colors)
    avg_csat_colors = map_values_to_color(final_table['AVG CSAT'], colors)
    performance_colors = map_values_to_color(final_table['performance'], colors)

    # table chart
    fig_table = go.Figure(data=(go.Table(
    header=dict(values=list(final_table.columns), fill_color='paleturquoise', align='left', font=dict(size=13)),
    cells=dict(values=[final_table['Agent'], final_table['Answered'], final_table['Resolved'],
                       final_table['AVG Ans Speed(sec)'], final_table['AVG CSAT'], final_table['performance']],
               align='left',
               fill_color=[['lavender'] * len(final_table),  
                           answered_colors,  
                           resolved_colors,  
                           avg_speed_colors, 
                           avg_csat_colors,  
                           performance_colors],
                           font=dict(color='black', size=12))
                        )))
    fig_table.update_layout(title='Agents Overview',                          
                            paper_bgcolor='rgb(250,250,250)',
                            margin=dict(t=40, b=0),
                            height=290,
                            )

    # return charts
    return (total_calls, calls_answered, problems_resolved, avg_call_time, answer_speed, fig_gauge, fig_bar,
            fig_scatter, fig_line, fig_pie1, fig_line2, fig_pie2, fig_table
            )


if __name__ == '__main__':
    app.run_server(debug=True, port=8001)
