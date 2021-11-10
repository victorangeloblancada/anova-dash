######################
## Import Libraries ##
######################

# Libraries for Dash front-end
import base64
import datetime
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable

import plotly.express as px

# Libraries for back-end analysis
import numpy as np
import pandas as pd

# Libraries for ANOVA
import statsmodels.api as sm
from statsmodels.formula.api import ols

# For visualizations
import plotly.graph_objects as go

##############
## Settings ##
##############

# App title
app_title = 'Analysis of Variance'

# Fix random seed
np.random.seed(0)

# Silence Pandas warnings
pd.options.mode.chained_assignment = None

######################
## Define Functions ##
######################

def parse_contents(contents, filename):
    """Parse the contents of an uploaded file to a dataframe in Python
    contents: The contents of an uploaded file
    filename: The filename of an uploaded file
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assuming that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assuming that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return pd.DataFrame()
    return df

def render_table(df, id):
    """Render a dataframe as an HTML table in Dash
    df: The source dataframe
    id: The element ID
    """
    return DataTable(id=id,
                     columns=[{'name': i, 'id': i} for i in df.columns],
                     export_format='xlsx',
                     export_headers='display',
                     data=df.to_dict('records'))

def render_bar(df, x_col, y_col):
    """Render a bar chart using Plotly
    df: The user-uploaded data
    x_col: The name of the input column for the ANOVA analysis
    y_col: The name of the response column for the ANOVA analysis
    """
    return px.bar(df, 
                  x=x_col, 
                  y=y_col)

def anova_one_way(df, response, factor):
    """Run a one-way analysis of variance and return the results table as a dataframe.
    df: The source dataframe
    response: The response column
    factor: The factor column  
    """
    model = ols(response + ' ~ C('+factor+')',
                data=df).fit()
    return sm.stats.anova_lm(model, typ=2)


def anova_two_way(df, response, factor_1, factor_2):
    """Run a two-way analysis of variance and return the results table as a dataframe.
    df: The source dataframe
    response: The response column
    factor_1: The first factor column 
    factor_2: The second factor column
    """
    model = ols(response+' ~ C('+factor_1+') + C('+factor_2+') + C('+factor_1+'):C('+factor_2+')',
                data=df).fit()
    return sm.stats.anova_lm(model, typ=2)


def overlaid_hist(df, response, factor):
    """Display overlaid histograms.
    df: The source dataframe
    response: The column to show the distribution of in the histogram
    factor: The column to split the data into separate histograms
    """
    # Create figure and add histogram traces for each value of the factor
    fig = go.Figure()

    for value in np.unique(df[factor]):
        fig.add_trace(go.Histogram(x=df.loc[df[factor] == value,
                                            response],
                                   name=str(value)))

    # Overlay both histograms
    fig.update_layout(barmode='overlay')
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)

    # Format figure
    fig.update_layout(title='Histograms of '+response+' by '+factor,
                      xaxis_title=factor+' Value',
                      yaxis_title='Count',
                      legend_title=response)

    # Display results
    return fig


def side_box_plots(df, response, factor):
    """Display box plots side by side.
    df: The source dataframe
    response: The column to show the distribution of in the box plots
    factor: The column to split the data into separate box plots
    """
    # Create figure and add box plot traces for each value of the factor
    fig = go.Figure()

    for value in np.unique(df[factor]):
        fig.add_trace(go.Box(y=df.loc[df[factor] == value,
                                      response],
                             name=str(value)))

    # Format figure
    fig.update_layout(title='Box Plots of '+response+' by '+factor,
                      xaxis_title=response,
                      yaxis_title=factor+' Value',
                      legend_title=response)

    # Display results
    return fig

################
## App Layout ##
################

# Formatting template file
external_stylesheets = ['assets/style.css']
app = dash.Dash(__name__, 
                external_stylesheets=external_stylesheets,
                meta_tags=[{"name": "viewport", 
                            "content": "width=device-width, initial-scale=1"}])

# App title (appears in browser tab title or window's top bar)
app.title = app_title

# App layout
app.layout = html.Div([
    # A hidden table where the user-uploaded data will be temporarily cached while the user is accessing the web app
    html.Div(id='input-data', style={'display': 'none'}),
    # A header
    html.H1(app_title),
    # A paragraph
    html.P('This web app performs a one-way analysis of variance on user-uploaded data.'),
    # Label for file upload component
    html.H6('File Upload'),
    # File upload component
    dcc.Upload(
        id='upload-data',
        className='upload',
        children=[
            html.P('Drag and Drop or '),
            html.A('Select Files'),
        ]
    ),
    # Label for dropdown
    html.H6('Factor column (X)'),
    # A dropdown menu whose values are populated by the columns of user-uploaded data
    dcc.Dropdown(
        id='x-dropdown',
        options=[
        ]
    ),
    # Label for dropdown
    html.H6('Response column (Y)'),
    # A dropdown menu whose values are populated by the columns of user-uploaded data
    dcc.Dropdown(
        id='y-dropdown',
        options=[
        ]
    ),
    # Print/ show output of analysis
    html.Div(id='output-section'),
    # Display a chart
    html.Div(id='chart-section',
             children=[dcc.Graph(id='output-chart-1'),
                       dcc.Graph(id='output-chart-2')],
             hidden=True)
])

###################
## App Callbacks ##
###################

@app.callback([Output('input-data', 'children'),
               Output('x-dropdown', 'options'),
               Output('y-dropdown', 'options'),
               Output('x-dropdown', 'value'),
               Output('y-dropdown', 'value')],
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def load_data(contents, filename):
    """Load data, populate dropdown menus, and temporarily cache data while user is accessing the web app 
    contents: Contents of the user-uploaded file
    filename: Name of the user-uploaded file
    """
    # Check if user has uploaded data
    if contents is not None:
        # Load data
        df = parse_contents(contents, filename)
        # Populate dropdown menus
        columns = [{'label': col, 'value': col} for col in df.columns]
        # Temporarily store input data to hidden table while user is accessing the web app
        df = df.to_json(date_format='iso', 
                        orient='split')
        return df, columns, columns, 'Factor1', 'Response'
    # Don't do anything if user has not uploaded data
    else:
        return '', [], [], '', ''

@app.callback([Output('output-section', 'children'),
               Output('output-chart-1', 'figure'),
               Output('output-chart-2', 'figure'),
               Output('chart-section', 'hidden')],
              [Input('input-data', 'children'),
               Input('x-dropdown', 'value'),
               Input('y-dropdown', 'value')])
def run_anova(df, x_col, y_col):
    """Run ANOVA and display results
    df: The user-uploaded data
    x_col: The name of the input column for the ANOVA analysis
    y_col: The name of the response column for the ANOVA analysis
    """
    # Check if user has uploaded data
    if len(df)>0:
        # Load cached data from hidden table
        df = pd.read_json(df, orient='split')
        try:
            # Run one-way ANOVA
            aov_table = anova_one_way(df, y_col, x_col)
            aov_table = aov_table

            # Show ANOVA results as plain text
            #text_output = str(aov_table)

            # Show ANOVA results as a table
            table_output = render_table(np.round(aov_table.reset_index(), 4), id='anova-table')

            # Add text conclusions
            text_output = ''
            for index, row in aov_table.iterrows():
                if row['PR(>F)']<0.05:
                    text_output += '\n'+str(index)+' has a significant effect on '+y_col+'.'

            # Show text conclusion before table
            table_output = [text_output, table_output]

            # Show the overlaid histograms
            chart_output_1 = overlaid_hist(df, y_col, x_col)
            # Show the box plots
            chart_output_2 = side_box_plots(df, y_col, x_col)

            return table_output, chart_output_1, chart_output_2, False
        
        except:
            # Tell user to select input and response columns
            text_output = 'Please select valid input and response columns.'
            # Show a blank chart
            chart_output = render_bar(pd.DataFrame({'x': [0], 'y': [0]}), x_col='x', y_col='y')
            return text_output, chart_output, chart_output, True
    
    # Tell the user to upload data first if they have not yet uploaded data
    else:
        # Tell user to select input and response columns
        text_output = 'Please upload data for analysis first.'
        # Show a blank chart
        chart_output = render_bar(pd.DataFrame({'x': [0], 'y': [0]}), x_col='x', y_col='y')
        return text_output, chart_output, chart_output, True

################
## Run Server ##
################

if __name__ == '__main__':
    app.run_server(debug=False, port=8080, host='0.0.0.0')