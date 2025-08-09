import plotly.graph_objects as go
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import numpy as np


df = pd.read_csv('US_Accidents(2016-2025)\data.csv')


numerical_cols = df.select_dtypes(include=[np.number]).columns
corr_df = df[numerical_cols].corr()


app = Dash(__name__)


heatmap_fig = px.imshow(
    corr_df,
    text_auto=True,
    aspect="auto",
    color_continuous_scale='RdBu',
    zmin=-1,
    zmax=1,
    title='Correlation Matrix of Numerical Features'
).update_layout(
    height=600
)

app.layout = html.Div(style={
    'backgroundColor': '#F8F9FA',
    'color': '#212529',
    'fontFamily': 'Roboto, sans-serif',
    'padding': '10px'
}, children=[
    html.H1('US Accidents Dashboard (2016-2025)',
            style={'textAlign': 'center',
                   'color': '#212529'}),

    html.H4('Filter by Time of Day',
            style={'textAlign': 'center', 'color': '#6C757D'}),
    dcc.Dropdown(
        id="dropdown",
        options=[{'label': time, 'value': time}
                 for time in df['Time_of_Day'].unique()],
        value=df['Time_of_Day'].unique()[0],
        clearable=False,
        style={'marginBottom': '20px'}
    ),

    html.Div([
        html.Div([
            html.Div(dcc.Graph(id="heatmap", figure=heatmap_fig),
                     style={'flex': 1, 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'}),
            html.Div(dcc.Graph(id='weather_with_accidents'),
                     style={'flex': 1, 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'}),
        ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '20px', 'marginBottom': '20px'}),

        html.Div([
            html.Div(dcc.Graph(id='cities_map_chart'),
                     style={'flex': 1, 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'}),
        ], style={'display': 'flex', 'flexDirection': 'row', 'marginBottom': '20px'}),

        html.Div([
            html.Div(dcc.Graph(id='cities_with_most_accidents'),
                     style={'flex': 1, 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'}),
            html.Div(dcc.Graph(id='accidents_over_time'),
                     style={'flex': 1, 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'}),
        ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '20px', 'marginBottom': '20px'}),

        html.Div([
            html.Div(dcc.Graph(id='severity_with_weather'),
                     style={'flex': 1, 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'}),
            html.Div(dcc.Graph(id='severity_with_time_elapsed'),
                     style={'flex': 1, 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'}),
        ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '20px', 'marginBottom': '20px'}),


        html.Div([
            html.Div(dcc.Graph(id='pressure_vs_wind_chart'),
                     style={'flex': 1, 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'}),
            html.Div(dcc.Graph(id='temp_vs_humidity_chart'),
                     style={'flex': 1, 'padding': '10px', 'backgroundColor': 'white', 'borderRadius': '5px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.05)'}),
        ], style={'display': 'flex', 'flexDirection': 'row', 'gap': '20px', 'marginBottom': '20px'}),
    ])
])


def update_map_chart(time_of_day):
    filtered = df[df["Time_of_Day"] == time_of_day]
    state_counts = filtered.groupby(
        ['State', 'Severity']).size().reset_index(name='Accident_Count')

    fig = px.choropleth(
        state_counts,
        locations='State',
        locationmode='USA-states',
        color='Accident_Count',
        scope='usa',
        color_continuous_scale="Reds",
        title=f"Accident Distribution by State — {time_of_day}",
        facet_col='Severity',
        facet_col_wrap=2
    )
    fig.update_layout(geo_scope='usa')
    return fig


def update_accidents_over_time(time_of_day):
    filtered = df[df["Time_of_Day"] == time_of_day]
    trend = filtered.groupby('Time_Elapsed').size(
    ).reset_index(name='Accident_Count')
    fig = px.line(
        trend,
        x='Time_Elapsed',
        y='Accident_Count',
        title=f"Accident Count by Time Elapsed — {time_of_day}",
        markers=True
    )
    fig.update_traces(line_color='#007BFF')
    return fig


def update_cities_chart(time_of_day):
    filtered = df[df["Time_of_Day"] == time_of_day]
    cities = filtered['City'].value_counts().head(20)
    fig = px.bar(
        x=cities.index,
        y=cities.values,
        orientation='v',
        title=f'Top 20 Cities with Most Accidents — {time_of_day}',
        labels={'x': 'Number of Accidents', 'y': 'City'},
        text=cities.values
    )
    fig.update_traces(marker_color='#28A745', textposition='outside')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig


def update_weather_chart(time_of_day):
    filtered = df[df["Time_of_Day"] == time_of_day]
    weather_counts = filtered['Weather_Condition'].value_counts()
    top_n = 9
    if len(weather_counts) > top_n:
        top_counts = weather_counts.head(top_n).copy()
        top_counts['Other'] = weather_counts.iloc[top_n:].sum()
        weather_counts = top_counts

    fig = px.pie(
        names=weather_counts.index,
        values=weather_counts.values,
        title=f"Top Weather Conditions During Accidents — {time_of_day}",
        hole=0.3
    )
    fig.update_layout(legend_title_text='Weather Condition')
    return fig


def update_severity_weather_chart(time_of_day):
    filtered = df[df["Time_of_Day"] == time_of_day]
    top_weather = filtered['Weather_Condition'].value_counts().nlargest(
        15).index
    filtered_by_weather = filtered[filtered['Weather_Condition'].isin(
        top_weather)]
    grouped = filtered_by_weather.groupby(
        ['Weather_Condition', 'Severity']).size().reset_index(name='Count')
    fig = px.bar(
        grouped,
        x='Weather_Condition',
        y='Count',
        color='Severity',
        title=f"Severity vs Top 15 Weather Conditions — {time_of_day}",
        barmode='stack'
    )
    fig.update_xaxes(tickangle=45)
    return fig


def update_severity_time_elapsed_chart(time_of_day):
    filtered = df[df["Time_of_Day"] == time_of_day]
    grouped = filtered.groupby(
        ['Time_Elapsed', 'Severity']).size().reset_index(name='Count')
    fig = px.line(
        grouped,
        x='Time_Elapsed',
        y='Count',
        color='Severity',
        title=f"Accident Count by Severity Over Time Elapsed — {time_of_day}",
        labels={'Time_Elapsed': 'Time Elapsed',
                'Count': 'Number of Accidents'},
        markers=True
    )
    return fig


def update_pressure_wind_chart(time_of_day):
    filtered = df[df["Time_of_Day"] == time_of_day]
    fig = px.density_heatmap(
        filtered,
        x='Wind_Speed(mph)',
        y='Pressure(in)',
        facet_col='Severity',
        facet_col_wrap=4,
        title=f"Pressure vs. Wind Speed Density by Severity — {time_of_day}",
        labels={'Wind_Speed(mph)': 'Wind Speed (mph)',
                'Pressure(in)': 'Pressure (in)'},
        color_continuous_scale='Blues'
    )
    return fig


def update_temp_humidity_chart(time_of_day):
    filtered = df[df["Time_of_Day"] == time_of_day]
    fig = px.density_heatmap(
        filtered,
        x='Temperature(F)',
        y='Humidity(%)',
        facet_col='Severity',
        facet_col_wrap=4,
        title=f"Temperature vs. Humidity Density by Severity — {time_of_day}",
        labels={'Temperature(F)': 'Temperature (F)',
                'Humidity(%)': 'Humidity (%)'},
        color_continuous_scale='Reds'
    )
    return fig


@app.callback(
    Output('cities_map_chart', 'figure'),
    Output('accidents_over_time', 'figure'),
    Output('cities_with_most_accidents', 'figure'),
    Output('weather_with_accidents', 'figure'),
    Output('severity_with_weather', 'figure'),
    Output('severity_with_time_elapsed', 'figure'),
    Output('pressure_vs_wind_chart', 'figure'),
    Output('temp_vs_humidity_chart', 'figure'),
    Input('dropdown', 'value')
)
def update_all_graphs(time_of_day):
    return (
        update_map_chart(time_of_day),
        update_accidents_over_time(time_of_day),
        update_cities_chart(time_of_day),
        update_weather_chart(time_of_day),
        update_severity_weather_chart(time_of_day),
        update_severity_time_elapsed_chart(time_of_day),
        update_pressure_wind_chart(time_of_day),
        update_temp_humidity_chart(time_of_day)
    )


if __name__ == '__main__':
    app.run(debug=True)
