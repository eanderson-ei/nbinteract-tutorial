import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# data read
districts = pd.read_csv('https://raw.githubusercontent.com/eanderson-ei/sws-viz/master/data/raw/districts.csv')
pms = pd.read_csv('https://raw.githubusercontent.com/eanderson-ei/sws-viz/master/data/raw/pms.csv')
pm_data = pd.read_csv('https://raw.githubusercontent.com/eanderson-ei/sws-viz/master/data/raw/pm_data.csv')

df_temp = pd.merge(pm_data, pms, on=['district', 'pm_id'], how='left')
df = pd.merge(df_temp, districts, on='district', how='left')
df['progress'] = df['result'].diff()
df['temp_trans'] = df['pm_id'].diff()
filt = df['temp_trans'] != 0
df.loc[filt, 'progress'] = np.nan
df.drop('temp_trans', inplace=True, axis=1)

results = (df.groupby(['concept', 'district', 'pm_cat', 'year', 'quarter'])
           .mean()[['result']])
results_tidy = results.reset_index()
results_tidy['period'] = results_tidy['year'] + results_tidy['quarter']/10

# Widget options
concept_options = [
                    {'label': 'One', 'value': 1}, 
                    {'label': 'Three', 'value': 3}, 
                    {'label': 'Four', 'value': 4}
                ]

district_options = [{'label': district, 'value': district} 
                   for district in districts['district'].unique()]

# Figure definitions
# Indicator 1.1
def create_fig1_1():
    df = pd.DataFrame({'concept': [1, 2, 3, 4],
                  'fy18': [1, 0.92, 0.255, 0.524],
                  'fy19': [0.889, np.nan, 0.893, 0.38]})
    
    fig = go.Figure()

    x = ['FY18', 'FY19']
    for row in df.iterrows():
        y = (row[1][1], row[1][2])
        fig.add_trace(go.Scatter(x=x, y=y, name='Concept ' + str(int(row[1][0])),
                                hovertemplate='%{y:.0%}'))
        
        label = 'Concept ' + str(int(row[1][0]))
        fig.add_annotation(dict(x=0, y=y[0], text=label,
                            xanchor='right', yanchor='middle', xshift=-10, 
                            showarrow=False))

    # update yaxes
    fig.update_yaxes(range=[-.1,1.1], showgrid=False, zeroline=False,
                    side='right', tickformat='%') 

    # update xaxes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                    title_text='<b>Indicator 1.1</b>' +
                    '<br>Percent of coalition participants reporting ' +
                    '<br>an improvement in WASH system understanding',
                    showlegend=False)

    # add target line
    fig.add_shape(dict(
        type='line',
        x0=0, x1=1, y0=.9, y1=.9,
        line=dict(color='grey', width=1, dash='dash')))

    fig.add_annotation(dict(x=0, y=.9, text='LOP Target (90%)',
                            xanchor='left', yanchor='top', xshift=10, 
                            showarrow=False, bgcolor='white', opacity=.9))
    
    return fig

# Indicator 1.1 2020
def create_fig1_1_2020():
    df = pd.DataFrame({'concept': [1, 2, 3, 4],
                  'fy18': [1, 0.92, 0.255, 0.524],
                  'fy19': [0.889, np.nan, 0.893, 0.38],
                  'fy20': [1, .8, .9, .5]})

    fig = go.Figure()

    x = ['FY18', 'FY19', 'FY20']
    for row in df.iterrows():
        y = (row[1][1], row[1][2], row[1][3])
        fig.add_trace(go.Scatter(x=x, y=y, name='Concept ' + str(int(row[1][0])),
                                hovertemplate='%{y:.0%}'))
        
        label = 'Concept ' + str(int(row[1][0]))
        fig.add_annotation(dict(x=0, y=y[0], text=label,
                            xanchor='right', yanchor='middle', xshift=-10, 
                            showarrow=False))

    # update yaxes
    fig.update_yaxes(range=[-.1,1.1], showgrid=False, zeroline=False,
                    side='right', tickformat='%') 

    # update xaxes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                    title_text='<b>Indicator 1.1 (2020)</b>' +
                    '<br>Percent of coalition participants reporting ' +
                    '<br>an improvement in WASH system understanding',
                    showlegend=False)

    # add target line
    fig.add_shape(dict(
        type='line',
        x0=0, x1=1, y0=.9, y1=.9,
        line=dict(color='grey', width=1, dash='dash')))

    fig.add_annotation(dict(x=0, y=.9, text='LOP Target (90%)',
                            xanchor='left', yanchor='top', xshift=10, 
                            showarrow=False, bgcolor='white', opacity=.9))

    # Add target line
    fig.add_shape(dict(
        type='line',
        x0=0, x1=2, y0=.9, y1=.9,
        line=dict(color='grey', width=1, dash='dash')))

    return fig


# Indicator 1.2
def create_fig1_2():
    df2 = pd.DataFrame({'concept': [1, 2, 3, 4],
                        'fy17': [17, 0, 0, 0],
                        'fy18': [4, 2, 3, 1],
                        'fy19': [10, 0, 1, 2]})
    df2['fy18_c'] = df2['fy17']+df2['fy18']
    df2['fy19_c'] = df2['fy18_c']+df2['fy19']
    
    fig = go.Figure()

    x=['FY17', 'FY18', 'FY19']
    filt = df2['concept'] == 1
    y = df2.loc[filt, ['fy17', 'fy18_c', 'fy19_c']].to_numpy()[0]
    fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy', mode='none',
                            name='Concept One'))

    filt = df2['concept'] == 2
    y = y + df2.loc[filt, ['fy17', 'fy18_c', 'fy19_c']].to_numpy()[0]
    fig.add_trace(go.Scatter(x=x, y=y, fill='tonexty', mode='none',
                            name='Concept Two'))

    filt = df2['concept'] == 3
    y = y + df2.loc[filt, ['fy17', 'fy18_c', 'fy19_c']].to_numpy()[0]
    fig.add_trace(go.Scatter(x=x, y=y, fill='tonexty', mode='none',
                            name='Concept Three'))

    filt = df2['concept'] == 4
    y = y + df2.loc[filt, ['fy17', 'fy18_c', 'fy19_c']].to_numpy()[0]
    fig.add_trace(go.Scatter(x=x, y=y, fill='tonexty', mode='none',
                            name='Concept Four'))

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                      title_text='<b>Indicator 1.2</b><br>'+
                      'Number of analyses conducted to improve'+
                      'understanding of WASH systems')

    # update y axes
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                     range=[0,44])

    # add target line
    fig.add_shape(dict(
        type='line',
        x0=0, x1=2, y0=40, y1=40,
        line=dict(color='grey', width=1, dash='dash')))

    fig.add_annotation(dict(x=0, y=40, text='LOP Target (40)',
                            xanchor='left', yanchor='top', xshift=10, 
                            showarrow=False, bgcolor='white', opacity=.9))

    return fig


# Indicator 1.3
def create_fig1_3():
    df13 = pd.read_csv('https://raw.githubusercontent.com/eanderson-ei/sws-viz/master/data/raw/indicator-1-3.csv')
    df13_group = df13.groupby(['Coalition', 'Quarter and Fiscal Year of Measurement']).sum()
    df13_group.reset_index(inplace=True)
    
    df13_pivot = df13_group.pivot(index='Coalition', 
                                  columns='Quarter and Fiscal Year of Measurement', 
                                  values='Total # of Stakeholders Reached')
    df13_pivot['sum'] = df13_pivot.sum(axis=1)
    df13_pivot.sort_values('sum', inplace=True)
    df13_pivot.drop('sum', inplace=True, axis=1)
    df13_pivot.fillna(0, inplace=True)
    
    df13_c = df13_pivot.cumsum(axis=1)
    
    fig = go.Figure()

    y_store = np.array([0]* len(df13_c.columns))

    for row in df13_c.iterrows():
        x = df13_c.columns
        y = row[1].to_numpy()
        y_store = y_store + y
        if row[0] == df13_c.index[0]:
            fig.add_trace(go.Scatter(x=x, y=y_store, fill='tozeroy', 
                                    mode='none', name=row[0]))
        else:
            fig.add_trace(go.Scatter(x=x, y=y_store, fill='tonexty', 
                                    mode='none', name=row[0])) 

    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)',
                    title_text='<b>Indicator 1.3</b><br>'+
                    'Number of stakeholders reached with'+
                    'findings from systems analyses')

    # update y axes
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                    range=[0, 2100])

    # update x axes
    fig.update_xaxes(tickangle=-45)

    # add target line
    fig.add_shape(dict(
        type='line',
        x0=0, x1=len(df13_c.columns)-1, y0=600, y1=600,
        line=dict(color='grey', width=1, dash='dash')))

    fig.add_annotation(dict(x=0, y=600, text='FY18 Target (600)',
                            xanchor='left', yanchor='top', xshift=10, 
                            showarrow=False, bgcolor='white', opacity=.9))

    fig.add_shape(dict(
        type='line',
        x0=0, x1=len(df13_c.columns)-1, y0=2000, y1=2000,
        line=dict(color='grey', width=1, dash='dash')))

    fig.add_annotation(dict(x=0, y=2000, text='LOP Target (2000)',
                            xanchor='left', yanchor='top', xshift=10, 
                            showarrow=False, bgcolor='white', opacity=.9))
    
    return fig


# Indicator 5.1
def create_fig5_1():
    df51 = pd.read_csv('https://raw.githubusercontent.com/eanderson-ei/sws-viz/master/data/raw/indicator-5-1.csv')
    df51['Type'] = df51['Type'].fillna('Not Specified')
    df51['Fiscal Year'] = [entry[:4] for entry in 
                           df51['Quarter and Fiscal Year of Measurement']]
    df51_group = df51.groupby(['SWS Partners Involved', 'Fiscal Year', 'Type']).count()
    df51_group.reset_index(inplace=True)
    
    teams = np.sort(df51['SWS Partners Involved'].unique())
    
    fig = make_subplots(rows=1, cols=5,
                        shared_yaxes=True, 
                        shared_xaxes=True)

    # set legend color options
    legend_colors = {'Grey Literature': '#636efa', 
                    'Verbal Presentation':'#ef553b', 
                    'Published Literature':'#00cc96', 
                    'Not Specified':'#ab63fa'}

    # which group has the most entries and controls the legend? 
    # note this might break if one group does not contain all entries
    max_entries = 0
    max_team = None
    for team in teams:
        filt = df51_group['SWS Partners Involved']==team
        df51_maxteam = df51_group.loc[filt, :]
        if len(df51_maxteam['Type'].unique()) > max_entries:
            max_team = team
    max_team 

    # get types of content from max group
    filt = df51_group['SWS Partners Involved']==max_team
    learn_types = df51_group.loc[filt, 'Type'].unique()

    # plot teams
    plots = [(1,1), (1,2), (1,3), (1,4), (1,5)]
    plot_idx = 0
    ns_in_legend = False
    for team in teams:
        filt = df51_group['SWS Partners Involved']==team
        df51_team = df51_group.loc[filt, :]

        # plot bars
        filt = df51_group['SWS Partners Involved']==team
        learn_types = df51_group.loc[filt, 'Type'].unique()
        for learn_type in learn_types:
            filt = df51_team['Type']==learn_type
            df_temp = df51_team.loc[filt, ['Fiscal Year', 
                                        'Initials of Person Entering Data']]

            x = df_temp['Fiscal Year']            
            y = df_temp['Initials of Person Entering Data']
            
            
            
            if team == max_team:
                fig.add_trace(go.Bar(name=learn_type, x=x, y=y, 
                                    legendgroup=learn_type,
                                    marker_color=legend_colors[learn_type],
                                    opacity=.8), 
                            row=plots[plot_idx][0], col=plots[plot_idx][1])
            
            
            elif learn_type == 'Not Specified' and not ns_in_legend:
                fig.add_trace(go.Bar(name=learn_type, x=x, y=y, 
                                    legendgroup=learn_type,
                                    marker_color=legend_colors[learn_type],
                                    opacity=.8), 
                            row=plots[plot_idx][0], col=plots[plot_idx][1])
                ns_in_legend=True
            
            else: 
                fig.add_trace(go.Bar(name=learn_type, x=x, y=y, 
                                    legendgroup=learn_type,
                                    marker_color=legend_colors[learn_type],
                                    opacity=.8,
                                    showlegend=False), 
                            row=plots[plot_idx][0], col=plots[plot_idx][1])
            
        plot_idx +=1

    fig.update_layout(barmode='stack',
                    plot_bgcolor='white',
                    title_text='<b>Indicator 5.1</b> ' +
                    '<br>Knowledge products & presentations by SWS partners')

    # update y axes
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey',
                    range=[0,32])
    
    # ninja plot titles
    fig.update_xaxes(title_text = 'Concept 1', row=1, col=1)
    fig.update_xaxes(title_text = 'Concept 2', row=1, col=2)
    fig.update_xaxes(title_text = 'Concept 3', row=1, col=3)
    fig.update_xaxes(title_text = 'Concept 4', row=1, col=4)
    fig.update_xaxes(title_text = 'Learning Team', row=1, col=5)

    # update x axes
    fig.update_xaxes(range=[-.5,3.5])

    # fig.update_layout(height=800,
    #                 xaxis_showticklabels=True, 
    #                 xaxis2_showticklabels=True, 
    #                 xaxis3_showticklabels=True)
    
    return fig
    

# App
app.layout = html.Div(children=[
    html.H1('SWS Indicator Visualizations'),
    
    dcc.Markdown("""
                 Updated visualizations for the 2019 MEL Plan Annex.
                 
                 *All data and text are placeholder values and should be updated with 
                 accurate data and descriptions before use.*
                 """),
    
    html.Hr(),
    
    dcc.Markdown(
        """
        ## Indicator 1.1
        
        Across all concept teams, 72 percent of WASH actors surveyed or interviewed reported
        an improvement in understanding of WASH systems. The large disparities across project
        teams are likely due to different data collection methods. However, of all Concepts,
        Concept 4 is not demonstrating progression, and may need additional focus on this 
        indicator in 2020.
        """
    ),
    
    html.Div(
        className='row',
        children=[
            dcc.Graph(
                className='six columns',
                id='indicator1-1',
                config={
                    'displayModeBar': False
                },
                figure = create_fig1_1()
            ),
            dcc.Graph(
                className='six columns',
                id='indicator1-1-2020',
                config={
                    'displayModeBar': False
                },
                figure = create_fig1_1_2020()
            )
        ]
    ),
    
    dcc.Markdown(
        """
        ## Indicator 1.2
        
        Concept teams  met the Life of Project target of 40 analyses in FY19. Analyses
        conducted during the reporting period include Network Analyses (2), Financial
        Analyses (1), an a Building Block Assessment (1).
        """
    ),
    
    dcc.Graph(
        className='row',
        id='indicator1-2',
        config={
            'displayModeBar': False
        },
        figure = create_fig1_2()
    ),
    
    dcc.Markdown(
        """
        ## Indicator 1.3
        
        Concept teams reached a combined 838 stakeholders with findings from systems
        analyses this period. Outreach activities included coalition meetings, meetings
        with specified stakeholders, conferences and workshops. 
        """
    ),
    
    dcc.Graph(
        className='row',
        id='indicator1-3',
        config={
            'displayModeBar': False
        },
        figure = create_fig1_3()
    ),
    
    html.Hr(),
    
    dcc.Markdown(
        """
        ## Indicator 3.2
        
        Coalition partners measure progress towards a vision of more sustainable
        services through progress markers. Progress markers vary by partner. 
        Progress markers are catagorized as 'Expect to See', 'Like to See', and 
        'Love to See'.
        
        **Explore the progress made by Concept Teams or Districts using the 
        dropdown menus below**.
        """
    ),
    
    html.Div(
        className='row',
        children=[
            dcc.Dropdown(
                className='six columns',
                id='group-by',
                options=[
                    {'label': 'Concept', 'value': 'concept'}, 
                    {'label': 'District', 'value': 'district'}
                ],
                value='district',
                placeholder='Select a group',
                clearable=False
            ),
            dcc.Dropdown(
                className='six columns',
                id='group-select',
                options=district_options,                
                value=district_options[0]['value'],
                placeholder='Select a concept or district',
                clearable=False
            )
        ]
    ),
    
    html.Div(
        className='row',
        children=[
            dcc.Graph(
                className='six columns', 
                id='progress-steps',
                config={
                    'displayModeBar': False
                }
                ),
            dcc.Graph(
                className='six columns',
                id='progress-breakdown',
                config={
                    'displayModeBar': False
                }
                )
        ]
    ),
    
    html.Hr(),
    
    dcc.Markdown(
        """
        ## Indicator 5.1
        
        SWS produced a combined 47 knowledge products and presentations in the first half
        of FY19. Overall, the project has completed 111 knowledge products and presentations,
        significantly exceeding the LOP target of 40. 
        """
    ),
    
    dcc.Graph(
        className='row',
        id='indicator5-1',
        config={
            'displayModeBar': False
        },
        figure=create_fig5_1()
    )
])

@app.callback(
    [Output('group-select', 'options'),
     Output('group-select', 'value')],
    [Input('group-by', 'value')]
)
def update_options(group):
    # update options in group-select
    if group == 'concept':
        options = concept_options
        value = concept_options[0]['value']
    elif group == 'district':
        options = district_options
        value = district_options[0]['value']
    
    return options, value
    

@app.callback(
    Output('progress-steps', 'figure'),
    [Input('group-by', 'value')])
def update_plot_progress(group):
    # get positive data
    filt = df['progress'] > 0
    df2 = df.loc[filt, :]
    p_data = df2.groupby([group]).sum()[['progress']]

    # get negative data
    filt = df['progress'] < 0
    df3 = df.loc[filt, :]
    n_data = df3.groupby([group]).sum()[['progress']]
    
    # merge on indices
    df_merge = pd.merge(p_data, n_data, how='outer', 
                            left_index=True, right_index=True, 
                            suffixes=('_p', '_n'))
    df_merge.sort_values(by='progress_p', ascending=True, inplace=True)
    x1 = df_merge['progress_p']
    x2 = df_merge['progress_n']
    y = df_merge.index
        
    if group == 'concept':
        y = ['Concept ' + str(idx) for idx in df_merge.index]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(x=x1, y=y, orientation='h',
                         name='Positive Steps'))
    
    fig.add_trace(go.Bar(x=-x2, y=y, orientation='h',
                        base=x2, name='Lost Steps',
                        hovertext=-x2,
                        hovertemplate='<extra>Lost Steps</extra>' +
                        '%{hovertext}'))
    
    fig.update_layout(title_text = 'Count of Progress Steps Made')
    
    # Update x axes
    fig.update_xaxes(title_text='Total Progress Steps')
    
    # update y axes
    fig.update_yaxes(type='category')
    
    # Update chart elements
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode='y')
    
    fig.update_layout(barmode='stack')
    
    return fig


@app.callback(
    Output('progress-breakdown', 'figure'),
    [Input('group-by', 'value'),
     Input('group-select', 'value')])
def update_plot_group(group, selector):
    try:
        data = []
        baselines = []
        annotations = []
        for pm_cat in results_tidy['pm_cat'].unique():
            # store data
            data_df = (results_tidy
            .groupby([group, 'pm_cat', 'period'])
            .mean()[['result']]
            )

            y = data_df.loc[pd.IndexSlice[selector, pm_cat], 'result']
            x = np.arange(len(y))
            
            trace = go.Scatter(x=x, y=y,
                            hovertemplate='<b>Average Rating<b>: %{y:.1f}' +
                            '<extra></extra>')

            data.append(trace)
            
            # store baseline
            baseline = dict(
                type='line',
                x0=0, x1=len(x)-1, y0=y.iloc[0], y1=y.iloc[0],
                line=dict(color='grey', width=1, dash='dash'))
            
            baselines.append(baseline)
            
            # store annotation
            change = y.iloc[-1] - y.iloc[0]
            text = (str(['+' if change>0 else '' for change in [change]][0])
                    + str(round(change, 2)))
            annotation = dict(x=len(x)-1, y=y.iloc[-1], text=text,
                            xanchor='left', yanchor='middle', xshift=10, 
                            showarrow=False)
            
            annotations.append(annotation)
        
        x_tick_text = data_df.loc[pd.IndexSlice[selector, results_tidy['pm_cat'].unique()[0]], :].index
        
        fig = make_subplots(rows=len(data), cols=1,
                    subplot_titles=results_tidy['pm_cat'].unique(),
                    shared_xaxes=True)
        
        # Add traces
        for idx, trace in enumerate(data):
            fig.add_trace(trace, row=idx+1, col=1)
        
        # Add baselines
        for idx, baseline in enumerate(baselines):
            fig.add_shape(baseline, row=idx+1, col=1)
        
        # Add annotations
        for idx, annotation in enumerate(annotations):
            fig.add_annotation(annotation, row=idx+1, col=1)
            
        # Update y-axes properties
        fig.update_yaxes(range=[0.75,3.25], 
                        showgrid=False, 
                        tickvals=[1,2,3], 
                        ticktext=['L', 'M', 'H'])

        # Update x-axes properties
        x_tick_text = data_df.loc[pd.IndexSlice[selector, 'Expect to See'],:].index
        fig.update_xaxes(tickvals=x, ticktext=x_tick_text)
        fig.update_layout(showlegend=False)
        
        # Update chart elements
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white',
            hovermode='x',
            title_text='Progress by PM Category')
        
        return fig
    
    except:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='white',
            plot_bgcolor='white'
        )
        return go.Figure()


if __name__ == '__main__':
    app.run_server(debug=False)