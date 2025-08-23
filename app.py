import pandas as pd
import numpy as np
import glob
from catboost import CatBoostClassifier
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output

# ------------------------
# Load preprocessed 2024 reference metrics
# ------------------------
pitcher_metrics_ref = pd.read_csv("Processed CSVs/pitcher_metrics_overall.csv")
pitcher_metrics_ref_lhh = pd.read_csv("Processed CSVs/pitcher_metrics_lhh.csv")
pitcher_metrics_ref_rhh = pd.read_csv("Processed CSVs/pitcher_metrics_rhh.csv")

# ------------------------
# Load Demo CSVs for the app
# ------------------------
csv_files = glob.glob("Demo CSVs/*.csv")
df_list = [pd.read_csv(file) for file in csv_files]
demo_df = pd.concat(df_list, ignore_index=True)
demo_df.columns = [c.lower() for c in demo_df.columns]

# Keep only Miami Red Sox
demo_df = demo_df[demo_df['pitcherteam'] == 'MIA_RED'].copy()

# ------------------------
# Load xBABIP Model
# ------------------------
xBABIP_Model = CatBoostClassifier()
xBABIP_Model.load_model("xBABIP_Model.cbm")

# ------------------------
# Metric functions (same as preprocessing)
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

    return df.groupby('pitcher').agg(
        strike_pct=('strike', 'mean'),
        zone_pct=('zone', 'mean'),
        chase_pct=('chase', 'mean'),
        swing_pct=('swing', 'mean'),
        called_strike_pct=('called_strike', 'mean')
    ).reset_index()

def calc_miss_metrics(df):
    df['miss'] = df['pitchcall'] == 'StrikeSwinging'
    df['in_zone_miss'] = df['miss'] & df['zone']
    df['out_zone_miss'] = df['miss'] & (~df['zone'])

    return df.groupby('pitcher').agg(
        miss_pct=('miss', 'mean'),
        in_zone_miss_pct=('in_zone_miss', 'mean'),
        out_zone_miss_pct=('out_zone_miss', 'mean')
    ).reset_index()

def calc_weak_contact(df, model):
    hitCols = ['exitspeed', 'angle', 'hitspinrate', 'distance', 'bearing']
    df['xbabip'] = np.nan
    inplay_mask = df['pitchcall'] == 'InPlay'
    inplay_data = df.loc[inplay_mask, :]

    valid_mask = inplay_data[hitCols].notnull().all(axis=1)
    X_valid = inplay_data.loc[valid_mask, hitCols].apply(pd.to_numeric, errors='coerce')
    if len(X_valid) > 0:
        df.loc[inplay_mask & valid_mask, 'xbabip'] = model.predict_proba(X_valid)[:, 1]

    df['gb'] = df['taggedhittype'] == 'GroundBall'
    return df.groupby('pitcher').agg(
        xbabip=('xbabip', 'mean'),
        gb_pct=('gb', 'mean')
    ).reset_index()

def calc_all_metrics(df):
    return calc_strike_metrics(df).merge(calc_miss_metrics(df), on='pitcher').merge(calc_weak_contact(df, xBABIP_Model), on='pitcher')

def add_percentiles_relative(df, reference_df, metrics_list):
    df = df.copy()
    for col in metrics_list:
        ref_values = reference_df[col].dropna().values
        df[col + '_pct'] = df[col].apply(lambda x: (ref_values < x).mean() * 100 if pd.notna(x) else np.nan)
    return df

# ------------------------
# Metrics & columns
# ------------------------
categories = {
    "Strikes": ['strike_pct', 'zone_pct', 'chase_pct', 'swing_pct', 'called_strike_pct'],
    "Missing Bats": ['miss_pct', 'in_zone_miss_pct', 'out_zone_miss_pct'],
    "Weak Contact": ['xbabip', 'gb_pct']
}

metrics_list = [m for v in categories.values() for m in v]

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

columns = [{"name": ["", "Player"], "id": "pitcher"}]
for cat, cols in categories.items():
    for col in cols:
        col_name = column_name_map.get(col, col)
        columns.append({"name": [cat, col_name, "Value"], "id": col})
        columns.append({"name": [cat, col_name, "Percentile"], "id": f"{col}_pct"})

# ------------------------
# Dash App
# ------------------------
app = Dash(__name__)
server = app.server

# Extract unique pitch types from Demo CSVs
pitch_types_demo = demo_df['taggedpitchtype'].dropna().unique()
pitch_types_options = [{'label': 'All', 'value': 'All'}] + [{'label': pt, 'value': pt} for pt in pitch_types_demo]

app.layout = html.Div([
    html.H1("Miami Baseball Pitcher Controllables Dashboard", style={'textAlign': 'center', 'color': '#2c3e50'}),
    html.Div([
        html.Label("Filter by Batter Handedness:"),
        dcc.RadioItems(
            id='handed-filter',
            options=[
                {'label': 'Overall', 'value': 'Overall'},
                {'label': 'Vs LHH', 'value': 'LHH'},
                {'label': 'Vs RHH', 'value': 'RHH'}
            ],
            value='Overall',
            inline=True
        ),
        html.Label("Filter by Pitch Type:", style={'marginLeft': '20px'}),
        dcc.Dropdown(
            id='pitchtype-filter',
            options=pitch_types_options,
            value='All',
            clearable=False
        )
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '20px'}),
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
# Color rules
# ------------------------
# ------------------------
# Diverging color mapping (red -> white -> green)
# ------------------------
def color_from_percentile(v01: float) -> str:
    """
    Map percentile [0,1] to a diverging color: 0=red, 0.5=white, 1=green
    """
    v01 = np.clip(v01, 0.0, 1.0)
    if v01 < 0.5:
        # Red to white
        ratio = v01 / 0.5
        r = 255
        g = int(255 * ratio)
        b = int(255 * ratio)
    else:
        # White to green
        ratio = (v01 - 0.5) / 0.5
        r = int(255 * (1 - ratio))
        g = 255
        b = int(255 * (1 - ratio))
    return f'rgb({r},{g},{b})'


def percentile_color_rules(metric_cols):
    """
    Generate style_data_conditional rules for all percentile columns.
    Uses smooth mapping for all values between 0–100.
    """
    rules = []
    for col in metric_cols:
        pct_col = f"{col}_pct"
        for lower in range(0, 100, 1):
            upper = min(lower + 1, 100)
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
        # ensure 100% is green
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
    Input('handed-filter', 'value'),
    Input('pitchtype-filter', 'value')
)
def update_table(handed, pitchtype):
    df_filtered = demo_df.copy()
    
    if pitchtype != 'All':
        df_filtered = df_filtered[df_filtered['taggedpitchtype'] == pitchtype]

    # Split by handedness
    if handed == 'Overall':
        ref_df = pitcher_metrics_ref
    elif handed == 'LHH':
        ref_df = pitcher_metrics_ref_lhh
        df_filtered = df_filtered[df_filtered['batterside'] == 'Left']
    else:
        ref_df = pitcher_metrics_ref_rhh
        df_filtered = df_filtered[df_filtered['batterside'] == 'Right']

    # Compute metrics
    df_metrics = calc_all_metrics(df_filtered)
    df_metrics = add_percentiles_relative(df_metrics, ref_df, metrics_list)

    # Format
    for col in metrics_list:
        df_metrics[col] = (df_metrics[col] * 100).round(1)
        df_metrics[col + '_pct'] = df_metrics[col + '_pct'].round(1)

    styles = percentile_color_rules(metrics_list)
    return df_metrics.to_dict('records'), styles

# ------------------------
# Run server
# ------------------------
import os

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)))
