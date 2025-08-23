import pandas as pd
import numpy as np
import glob
from catboost import CatBoostClassifier
import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import os

# ------------------------
# Load 2024 reference metrics (for percentiles only)
# ------------------------
pitcher_metrics_ref = pd.read_csv("pitcher_metrics_overall.csv")

# ------------------------
# Load new game data (ONLY Demo CSVs)
# ------------------------
csv_files = glob.glob("Demo CSVs/*.csv")
df_list = [pd.read_csv(file) for file in csv_files]
final_df = pd.concat(df_list, ignore_index=True)
final_df.columns = [col.lower() for col in final_df.columns]
final_df = final_df[final_df['pitcherteam'] == 'mia_red']  # restrict to Miami pitchers

# ------------------------
# Load xBABIP Model
# ------------------------
xBABIP_Model = CatBoostClassifier()
xBABIP_Model.load_model("xBABIP_Model.cbm")

# ------------------------
# Metric Functions
# ------------------------
def define_strike_zone(df, top=3.5, bottom=1.5, left=-0.708, right=0.708):
    df['zone'] = (
        (df['platelocheight'] >= bottom) &
        (df['platelocheight'] <= top) &
        (df['platelocside'] >= left) &
        (df['platelocside'] <= right)
    )
    return df

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

def calc_miss_metrics(df):
    df = define_strike_zone(df)
    df['miss'] = df['pitchcall'] == 'StrikeSwinging'
    df['in_zone_miss'] = df['miss'] & df['zone']
    df['out_zone_miss'] = df['miss'] & (~df['zone'])

    agg = df.groupby('pitcher').agg(
        miss_pct=('miss', 'mean'),
        in_zone_miss_pct=('in_zone_miss', 'mean'),
        out_zone_miss_pct=('out_zone_miss', 'mean')
    ).reset_index()
    return agg

def calc_weak_contact(df, model):
    hitCols = ['exitspeed', 'angle', 'hitspinrate', 'distance', 'bearing']
    df['xbabip'] = np.nan
    inplay_mask = df['pitchcall'] == 'inplay'
    inplay_data = df.loc[inplay_mask, :]

    valid_mask = inplay_data[hitCols].notnull().all(axis=1)
    X_valid = inplay_data.loc[valid_mask, hitCols].apply(pd.to_numeric, errors='coerce')

    if len(X_valid) > 0:
        predictions = model.predict_proba(X_valid)[:, 1]
        df.loc[inplay_mask & valid_mask, 'xbabip'] = predictions

    df['gb'] = df['taggedhittype'] == 'groundball'

    agg = df.groupby('pitcher').agg(
        xbabip=('xbabip', 'mean'),
        gb_pct=('gb', 'mean')
    ).reset_index()
    return agg

def aggregate_metrics(df):
    if df.empty:
        return pd.DataFrame()
    strike_df = calc_strike_metrics(df)
    miss_df = calc_miss_metrics(df)
    weak_df = calc_weak_contact(df, xBABIP_Model)
    return strike_df.merge(miss_df, on='pitcher').merge(weak_df, on='pitcher')

def add_percentiles_relative(df, reference_df, metric_cols):
    for col in metric_cols:
        ref_values = reference_df[col].dropna().values
        df[col + '_pct'] = df[col].apply(lambda x: (ref_values < x).mean() * 100)
    return df

# ------------------------
# Dashboard Setup
# ------------------------
categories = {
    "Strikes": ['strike_pct', 'zone_pct', 'chase_pct', 'swing_pct', 'called_strike_pct'],
    "Missing Bats": ['miss_pct', 'in_zone_miss_pct', 'out_zone_miss_pct'],
    "Weak Contact": ['xbabip', 'gb_pct']
}
column_name_map = {
    "pitcher": "Player","strike_pct": "K","zone_pct": "Zone","chase_pct": "Chase","swing_pct": "Swing",
    "called_strike_pct": "Called K","miss_pct": "Miss","in_zone_miss_pct": "IZ Miss","out_zone_miss_pct": "OZ Miss",
    "xbabip": "xBABIP","gb_pct": "GB"
}
metrics = [m for group in categories.values() for m in group]

columns = [{"name": ["", "Player"], "id": "pitcher"}]
for cat, cols in categories.items():
    for col in cols:
        col_name = column_name_map.get(col, col)
        columns.append({"name": [cat, col_name, "Value"], "id": col})
        columns.append({"name": [cat, col_name, "Percentile"], "id": f"{col}_pct"})

app = dash.Dash(__name__)
server = app.server 

app.layout = html.Div([
    html.H1("Miami Baseball Pitcher Controllables Dashboard", style={'textAlign': 'center', 'color': '#2c3e50'}),

    # Filters
    html.Div([
        html.Label("Filter by Batter Handedness:"),
        dcc.RadioItems(
            id='handed-filter',
            options=[
                {'label': 'Overall', 'value': 'Overall'},
                {'label': 'Vs LHH', 'value': 'Left'},
                {'label': 'Vs RHH', 'value': 'Right'}
            ],
            value='Overall',
            inline=True
        ),
        html.Label("Filter by Pitch Type:", style={'marginLeft': '40px'}),
        dcc.Dropdown(
            id='pitch-type-filter',
            options=[{'label': pt, 'value': pt} for pt in sorted(final_df['pitchtype'].dropna().unique())],
            multi=False,
            placeholder="All Pitch Types"
        )
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'gap': '20px','marginBottom': '20px'}),

    dash_table.DataTable(
        id='pitcher-table',
        columns=columns,
        merge_duplicate_headers=True,
        sort_action='native',
        filter_action='native',
        style_table={'overflowX': 'auto', 'margin': '0 auto', 'width': '95%'},
        style_header={'backgroundColor': '#2f3640','color': 'white','fontWeight': 'bold','textAlign': 'center'},
        style_cell={'textAlign': 'center','padding': '8px'},
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
                'if': {'filter_query': f'{{{pct_col}}} >= {lower} && {{{pct_col}}} < {upper}','column_id': pct_col},
                'backgroundColor': color,'color': 'black'
            })
        rules.append({'if': {'filter_query': f'{{{pct_col}}} = 100','column_id': pct_col},
                      'backgroundColor': color_from_percentile(1.0),'color': 'black'})
    return rules

# ------------------------
# Callback
# ------------------------
@app.callback(
    Output('pitcher-table', 'data'),
    Output('pitcher-table', 'style_data_conditional'),
    Input('handed-filter', 'value'),
    Input('pitch-type-filter', 'value')
)
def update_table(handed_filter, pitch_type_filter):
    # Apply filters
    df_filtered = final_df.copy()
    if handed_filter != 'Overall':
        df_filtered = df_filtered[df_filtered['batterside'] == handed_filter]
    if pitch_type_filter:
        df_filtered = df_filtered[df_filtered['pitchtype'] == pitch_type_filter]

    # Recompute metrics on filtered data
    pitcher_metrics = aggregate_metrics(df_filtered)
    if pitcher_metrics.empty:
        return [], []  # nothing to show

    # Add percentiles relative to full 2024 reference
    pitcher_metrics = add_percentiles_relative(pitcher_metrics, pitcher_metrics_ref, metrics)

    # Format columns
    for col in metrics:
        if col != 'xbabip':
            pitcher_metrics[col] = pd.to_numeric(pitcher_metrics[col], errors='coerce') * 100
            pitcher_metrics[col] = pitcher_metrics[col].round(1)
        else:
            pitcher_metrics[col] = pd.to_numeric(pitcher_metrics[col], errors='coerce').round(3)
        pitcher_metrics[col + '_pct'] = pd.to_numeric(pitcher_metrics[col + '_pct'], errors='coerce').round(1)

    styles = percentile_color_rules(metrics, step=1)
    return pitcher_metrics.to_dict('records'), styles

# ------------------------
# Run server
# ------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
