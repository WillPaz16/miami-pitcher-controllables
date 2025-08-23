import pandas as pd
import numpy as np
import glob
from catboost import CatBoostClassifier
import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output

###########################################################################
# Import Data
###########################################################################

# 2024 reference dataset
data_2024 = pd.read_csv("2024TMDataWhole.csv")

# New game data
csv_files = glob.glob("Demo CSVs/*.csv")
df_list = [pd.read_csv(file) for file in csv_files]
final_df = pd.concat(df_list, ignore_index=True)
final_df.columns = [col.lower() for col in final_df.columns]
final_df = final_df[final_df['pitcherteam'] == 'MIA_RED']

###########################################################################
# Load xBABIP Model
###########################################################################

xBABIP_Model = CatBoostClassifier()
xBABIP_Model.load_model("xBABIP_Model.cbm")

###########################################################################
# Metric Functions
###########################################################################

def define_strike_zone(df, top=3.5, bottom=1.5, left=-0.708, right=0.708):
    df['zone'] = (
        (df['platelocheight'] >= bottom) &
        (df['platelocheight'] <= top) &
        (df['platelocside'] >= left) &
        (df['platelocside'] <= right)
    )
    return df

# STRIKE metrics
def calc_strike_metrics(df):
    df = define_strike_zone(df)
    df['strike'] = df['pitchcall'].isin(['StrikeCalled', 'StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable'])
    df['chase'] = df['strike'] & (~df['zone'])
    df['swing'] = df['pitchcall'].isin(['BallInPlay', 'StrikeSwinging', 'FoulBallFieldable', 'FoulBallNotFieldable'])
    df['called_strike'] = df['pitchcall'] == 'StrikeCalled'

    agg = df.groupby('pitcher').agg(
        strike_pct=('strike', 'mean'),
        zone_pct=('zone', 'mean'),
        chase_pct=('chase', 'mean'),
        swing_pct=('swing', 'mean'),
        called_strike_pct=('called_strike', 'mean')
    ).reset_index()
    return agg

# MISSING BATS (updated to include Out-of-Zone Miss)
def calc_miss_metrics(df):
    df['miss'] = df['pitchcall'] == 'StrikeSwinging'
    df['in_zone_miss'] = df['miss'] & df['zone']
    df['out_zone_miss'] = df['miss'] & (~df['zone'])

    agg = df.groupby('pitcher').agg(
        miss_pct=('miss', 'mean'),
        in_zone_miss_pct=('in_zone_miss', 'mean'),
        out_zone_miss_pct=('out_zone_miss', 'mean')
    ).reset_index()
    return agg

# WEAK CONTACT
def calc_weak_contact(df, model):
    hitCols = ['exitspeed', 'angle', 'hitspinrate', 'distance', 'bearing']

    df['xbabip'] = np.nan
    inplay_mask = df['pitchcall'] == 'InPlay'
    inplay_data = df.loc[inplay_mask, :]

    valid_mask = inplay_data[hitCols].notnull().all(axis=1)
    X_valid = inplay_data.loc[valid_mask, hitCols].copy()
    X_valid = X_valid.apply(pd.to_numeric, errors='coerce')

    predictions = model.predict_proba(X_valid)[:, 1]
    df.loc[inplay_mask & valid_mask, 'xbabip'] = predictions

    df['gb'] = df['taggedhittype'] == 'GroundBall'

    agg = df.groupby('pitcher').agg(
        xbabip=('xbabip', 'mean'),
        gb_pct=('gb', 'mean')
    ).reset_index()
    return agg

###########################################################################
# Aggregate Metrics
###########################################################################

strike_df = calc_strike_metrics(final_df)
miss_df = calc_miss_metrics(final_df)
weak_df = calc_weak_contact(final_df, xBABIP_Model)

pitcher_metrics = strike_df.merge(miss_df, on='pitcher').merge(weak_df, on='pitcher')

# Aggregate 2024 reference metrics
strike_2024 = calc_strike_metrics(data_2024)
miss_2024 = calc_miss_metrics(data_2024)
weak_2024 = calc_weak_contact(data_2024, xBABIP_Model)
pitcher_metrics_2024 = strike_2024.merge(miss_2024, on='pitcher').merge(weak_2024, on='pitcher')

# Percentiles relative to 2024
def add_percentiles_relative(df, reference_df, metric_cols):
    for col in metric_cols:
        ref_values = reference_df[col].dropna().values
        df[col + '_pct'] = df[col].apply(lambda x: (ref_values < x).mean() * 100)
    return df

metrics = ['strike_pct', 'zone_pct', 'chase_pct', 'swing_pct', 'called_strike_pct',
           'miss_pct', 'in_zone_miss_pct', 'out_zone_miss_pct', 'xbabip', 'gb_pct']

pitcher_metrics = add_percentiles_relative(pitcher_metrics, pitcher_metrics_2024, metrics)

# LHH / RHH metrics
df_lhh = final_df[final_df['batterside'] == 'Left'].copy()
df_rhh = final_df[final_df['batterside'] == 'Right'].copy()

pitcher_metrics_lhh = add_percentiles_relative(
    calc_strike_metrics(df_lhh)
    .merge(calc_miss_metrics(df_lhh), on='pitcher')
    .merge(calc_weak_contact(df_lhh, xBABIP_Model), on='pitcher'),
    pitcher_metrics_2024, metrics
)

pitcher_metrics_rhh = add_percentiles_relative(
    calc_strike_metrics(df_rhh)
    .merge(calc_miss_metrics(df_rhh), on='pitcher')
    .merge(calc_weak_contact(df_rhh, xBABIP_Model), on='pitcher'),
    pitcher_metrics_2024, metrics
)

###########################################################################
# Dash App
###########################################################################

import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd

# ------------------------
# Metrics grouped by category
# ------------------------
categories = {
    "Strikes": [
        'strike_pct', 'zone_pct', 'chase_pct', 'swing_pct', 'called_strike_pct'
    ],
    "Missing Bats": [
        'miss_pct', 'in_zone_miss_pct', 'out_zone_miss_pct'
    ],
    "Weak Contact": [
        'xbabip', 'gb_pct'
    ]
}

# Map raw metric names to aesthetic titles
column_name_map = {
    "pitcher": "Player",
    "strike_pct": "K",
    "zone_pct": "Zone",
    "chase_pct": "Chase",
    "swing_pct": "Swing",
    "called_strike_pct": "Called K",
    "miss_pct": "Miss",
    "in_zone_miss_pct": "IZ Miss",
    "out_zone_miss_pct": "OZ Miss",
    "xbabip": "xBABIP",
    "gb_pct": "GB"
}

metrics = [m for group in categories.values() for m in group]

# ------------------------
# Build multi-level column definitions
# ------------------------
columns = [{"name": ["", "Player"], "id": "pitcher"}]
for cat, cols in categories.items():
    for col in cols:
        col_name = column_name_map.get(col, col)
        columns.append({
            "name": [cat, col_name, "Value"], 
            "id": col
        })
        columns.append({
            "name": [cat, col_name, "Percentile"], 
            "id": f"{col}_pct"
        })

# ------------------------
# Initialize app
# ------------------------
app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Miami Baseball Pitcher Controllables Dashboard", style={
        'textAlign': 'center', 'color': '#2c3e50', 'fontFamily': 'Arial', 'marginBottom': '20px'
    }),

    html.Div([
        html.Label("Filter by Batter Handedness:", style={
            'fontWeight': 'bold', 'marginRight': '10px', 'fontSize': '16px'
        }),
        dcc.RadioItems(
            id='handed-filter',
            options=[
                {'label': 'Overall', 'value': 'Overall'},
                {'label': 'Vs LHH', 'value': 'LHH'},
                {'label': 'Vs RHH', 'value': 'RHH'}
            ],
            value='Overall',
            inline=True,
            style={'fontSize': '16px', 'marginLeft': '10px'}
        )
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '20px'}),

    dash_table.DataTable(
        id='pitcher-table',
        columns=columns,
        merge_duplicate_headers=True,
        sort_action='native',
        filter_action='native',
        style_table={'overflowX': 'auto', 'margin': '0 auto', 'width': '95%'},
        style_header={
            'backgroundColor': '#2f3640',
            'color': 'white',
            'fontWeight': 'bold',
            'textAlign': 'center',
            'fontSize': '14px',
            'border': '1px solid #2c3e50'
        },
        style_cell={
            'textAlign': 'center',
            'padding': '8px',
            'fontSize': '14px',
            'fontFamily': 'Arial'
        },
        style_data_conditional=[]
    )
])

# ------------------------
# Color mapping
# ------------------------
def color_from_percentile(v01: float) -> str:
    v01 = float(np.clip(v01, 0.0, 1.0))
    if v01 < 0.5:
        ratio = v01 / 0.5
        r, g, b = 255, int(255 * ratio), int(255 * ratio)
    else:
        ratio = (v01 - 0.5) / 0.5
        r, g, b = int(255 * (1 - ratio)), 255, int(255 * (1 - ratio))
    return f'rgb({r},{g},{b})'

def percentile_color_rules(metric_cols, step=1):
    rules = []
    for col in metric_cols:
        pct_col = f"{col}_pct"
        for lower in range(0, 100, step):
            upper = min(lower + step, 100)
            mid = lower + (upper - lower) / 2.0
            color = color_from_percentile(mid / 100.0)
            rules.append({
                'if': {
                    'filter_query': f'{{{pct_col}}} >= {lower} && {{{pct_col}}} < {upper}',
                    'column_id': pct_col
                },
                'backgroundColor': color,
                'color': 'black'
            })
        rules.append({
            'if': {'filter_query': f'{{{pct_col}}} = 100', 'column_id': pct_col},
            'backgroundColor': color_from_percentile(1.0),
            'color': 'black'
        })
    return rules

# ------------------------
# Callback
# ------------------------
@app.callback(
    Output('pitcher-table', 'data'),
    Output('pitcher-table', 'style_data_conditional'),
    Input('handed-filter', 'value')
)
def update_table(handed_filter):
    if handed_filter == 'Overall':
        df = pitcher_metrics.copy()
    elif handed_filter == 'LHH':
        df = pitcher_metrics_lhh.copy()
    else:
        df = pitcher_metrics_rhh.copy()

    # Convert all metric values to percentages (0–100) and round to 1 decimal
    for col in metrics:
        df[col] = pd.to_numeric(df[col], errors='coerce') * 100
        df[col] = df[col].round(1)
        df[col + '_pct'] = pd.to_numeric(df[col + '_pct'], errors='coerce').round(1)

    styles = percentile_color_rules(metrics, step=1)
    return df.to_dict('records'), styles

# ------------------------
# Run server
# ------------------------
import os

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))

