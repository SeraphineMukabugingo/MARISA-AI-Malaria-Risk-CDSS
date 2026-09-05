import os
import requests
import pandas as pd

from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# CONFIGURATION
# ============================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://backend:8000")

app = Dash(__name__)
app.title = "MARISA National Intelligence Dashboard"


# ============================================================
# COLOR SYSTEM
# ============================================================

COLORS = {
    "primary": "#0B3C5D",
    "secondary": "#1D5D7B",
    "accent": "#00A6A6",
    "success": "#198754",
    "warning": "#F4A261",
    "danger": "#D62828",
    "background": "#F4F7FB",
    "card": "#FFFFFF",
    "text": "#102A43",
    "muted": "#627D98",
    "border": "#D9E2EC",
}


# ============================================================
# API HELPERS
# ============================================================

def get_api_data(endpoint, default=None):
    """
    Safely retrieve data from the MARISA backend API.
    """

    if default is None:
        default = {}

    try:
        response = requests.get(
            f"{API_BASE_URL}{endpoint}",
            timeout=10
        )

        response.raise_for_status()

        return response.json()

    except Exception as error:
        print(f"API ERROR {endpoint}: {error}")

        return default


# ============================================================
# CARD COMPONENT
# ============================================================

def metric_card(title, value, subtitle="", icon="📊", color="#0B3C5D"):

    return html.Div(
        [
            html.Div(
                icon,
                style={
                    "fontSize": "30px",
                    "marginBottom": "10px",
                }
            ),

            html.Div(
                title,
                style={
                    "fontSize": "14px",
                    "fontWeight": "700",
                    "color": COLORS["muted"],
                }
            ),

            html.Div(
                str(value),
                style={
                    "fontSize": "30px",
                    "fontWeight": "800",
                    "color": color,
                    "marginTop": "6px",
                }
            ),

            html.Div(
                subtitle,
                style={
                    "fontSize": "12px",
                    "color": COLORS["muted"],
                    "marginTop": "6px",
                }
            ),
        ],
        style={
            "backgroundColor": COLORS["card"],
            "borderRadius": "14px",
            "padding": "20px",
            "boxShadow": "0 4px 12px rgba(0,0,0,0.08)",
            "border": f"1px solid {COLORS['border']}",
            "minWidth": "200px",
            "flex": "1",
        }
    )


# ============================================================
# HEADER
# ============================================================

def create_header():

    return html.Div(
        [

            html.Div(
                [
                    html.Div(
                        "🧬",
                        style={
                            "fontSize": "48px",
                            "marginRight": "18px",
                        }
                    ),

                    html.Div(
                        [

                            html.H1(
                                "MARISA",
                                style={
                                    "margin": "0px",
                                    "color": "white",
                                    "fontWeight": "800",
                                }
                            ),

                            html.Div(
                                "Malaria Artificial Intelligence Risk Assessment",
                                style={
                                    "color": "#D9F3FF",
                                    "fontSize": "16px",
                                    "fontWeight": "500",
                                }
                            ),

                            html.Div(
                                "National Monitoring & Clinical Decision Support Intelligence Platform",
                                style={
                                    "color": "#B9D9EA",
                                    "fontSize": "13px",
                                    "marginTop": "4px",
                                }
                            ),

                        ]
                    ),

                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                }
            ),

        ],
        style={
            "background": "linear-gradient(135deg, #06283D, #0B3C5D, #1D5D7B)",
            "padding": "28px",
            "borderRadius": "0 0 20px 20px",
            "boxShadow": "0 5px 18px rgba(0,0,0,0.18)",
            "marginBottom": "25px",
        }
    )


# ============================================================
# NATIONAL DASHBOARD
# ============================================================

def create_national_dashboard(data):

    total_predictions = data.get("total_predictions", 0)

    total_pregnant = data.get(
        "total_pregnant_assessed",
        0
    )

    total_high_risk = data.get(
        "total_high_risk",
        0
    )

    pregnant_high_risk = data.get(
        "total_pregnant_high_risk",
        0
    )

    high_risk_rate = data.get(
        "overall_high_risk_rate",
        0
    )

    followup_completed = data.get(
        "followup_completed",
        0
    )

    followup_rate = data.get(
        "followup_completion_rate",
        0
    )

    mean_risk = data.get(
        "mean_risk_score",
        0
    )

    province_data = data.get(
        "by_province",
        {}
    )


    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    cards = html.Div(
        [

            metric_card(
                "Total Predictions",
                total_predictions,
                "All MARISA assessments",
                "📊",
                COLORS["primary"]
            ),

            metric_card(
                "Pregnant Women Assessed",
                total_pregnant,
                "Pregnancy assessments",
                "🤰",
                "#7B2CBF"
            ),

            metric_card(
                "High Risk Cases",
                total_high_risk,
                "Require clinical attention",
                "⚠️",
                COLORS["danger"]
            ),

            metric_card(
                "Pregnant High Risk",
                pregnant_high_risk,
                "High-risk pregnant patients",
                "🤱",
                "#C1121F"
            ),

            metric_card(
                "Follow-up Completed",
                followup_completed,
                f"{followup_rate * 100:.1f}% completion",
                "✅",
                COLORS["success"]
            ),

            metric_card(
                "Average Risk Score",
                f"{mean_risk:.3f}",
                "Mean AI risk probability",
                "🧠",
                COLORS["accent"]
            ),

        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(210px, 1fr))",
            "gap": "18px",
            "marginBottom": "25px",
        }
    )


    # --------------------------------------------------------
    # PROVINCE DATA
    # --------------------------------------------------------

    province_rows = []

    for province, values in province_data.items():

        province_rows.append(
            {
                "Province": province,
                "Predictions": values.get(
                    "predictions",
                    0
                ),
                "Mean Risk Score": values.get(
                    "mean_risk_score",
                    0
                ),
            }
        )


    province_df = pd.DataFrame(
        province_rows
    )


    if province_df.empty:

        province_chart = go.Figure()

        province_chart.add_annotation(
            text="No province data available",
            showarrow=False
        )

    else:

        province_chart = px.bar(
            province_df,
            x="Province",
            y="Predictions",
            text="Predictions",
            title="Predictions by Province",
        )

        province_chart.update_traces(
            textposition="outside"
        )

        province_chart.update_layout(
            template="plotly_white",
            title_font_size=20,
            height=420,
            margin=dict(
                l=40,
                r=40,
                t=70,
                b=40
            ),
        )


    # --------------------------------------------------------
    # RISK SCORE BY PROVINCE
    # --------------------------------------------------------

    if province_df.empty:

        risk_chart = go.Figure()

        risk_chart.add_annotation(
            text="No risk data available",
            showarrow=False
        )

    else:

        risk_chart = px.bar(
            province_df,
            x="Province",
            y="Mean Risk Score",
            text="Mean Risk Score",
            title="Average AI Risk Score by Province",
        )

        risk_chart.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )

        risk_chart.update_layout(
            template="plotly_white",
            title_font_size=20,
            height=420,
            margin=dict(
                l=40,
                r=40,
                t=70,
                b=40
            ),
        )


    return html.Div(
        [

            html.H2(
                "🇷🇼 MARISA National Monitoring Dashboard",
                style={
                    "color": COLORS["primary"],
                    "fontWeight": "800",
                }
            ),

            html.P(
                "National overview of malaria risk assessments, high-risk patients, pregnancy assessments, and follow-up monitoring.",
                style={
                    "color": COLORS["muted"],
                    "fontSize": "15px",
                }
            ),

            cards,

            html.Div(
                [

                    html.Div(
                        dcc.Graph(
                            figure=province_chart
                        ),
                        style={
                            "backgroundColor": "white",
                            "borderRadius": "14px",
                            "padding": "10px",
                            "boxShadow": "0 3px 10px rgba(0,0,0,0.06)",
                        }
                    ),

                    html.Div(
                        dcc.Graph(
                            figure=risk_chart
                        ),
                        style={
                            "backgroundColor": "white",
                            "borderRadius": "14px",
                            "padding": "10px",
                            "boxShadow": "0 3px 10px rgba(0,0,0,0.06)",
                        }
                    ),

                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(450px, 1fr))",
                    "gap": "20px",
                }
            ),

        ]
    )


# ============================================================
# RBC DISTRICT DASHBOARD
# ============================================================

def create_rbc_dashboard(data):

    districts = data.get(
        "districts",
        []
    )

    flagged_cases = data.get(
        "live_flagged_cases",
        []
    )


    # --------------------------------------------------------
    # DISTRICT TABLE DATA
    # --------------------------------------------------------

    district_df = pd.DataFrame(
        districts
    )


    # --------------------------------------------------------
    # DISTRICT CHART
    # --------------------------------------------------------

    if district_df.empty:

        district_chart = go.Figure()

        district_chart.add_annotation(
            text="No district data available",
            showarrow=False
        )

    else:

        district_chart = px.bar(
            district_df,
            x="district",
            y="total_predictions",
            text="total_predictions",
            title="Total Predictions by District",
        )

        district_chart.update_traces(
            textposition="outside"
        )

        district_chart.update_layout(
            template="plotly_white",
            height=430,
        )


    # --------------------------------------------------------
    # HIGH RISK RATE CHART
    # --------------------------------------------------------

    if district_df.empty:

        high_risk_chart = go.Figure()

        high_risk_chart.add_annotation(
            text="No district risk data available",
            showarrow=False
        )

    else:

        high_risk_chart = px.bar(
            district_df,
            x="district",
            y="high_risk_rate",
            text="high_risk_rate",
            title="High-Risk Rate by District",
        )

        high_risk_chart.update_traces(
            texttemplate="%{text:.1%}",
            textposition="outside"
        )

        high_risk_chart.update_layout(
            template="plotly_white",
            height=430,
            yaxis_tickformat=".0%",
        )


    # --------------------------------------------------------
    # DISTRICT TABLE
    # --------------------------------------------------------

    district_table = dash_table.DataTable(

        data=districts,

        columns=[
            {
                "name": "District",
                "id": "district",
            },
            {
                "name": "Predictions",
                "id": "total_predictions",
            },
            {
                "name": "Pregnant Assessed",
                "id": "pregnant_assessed",
            },
            {
                "name": "High Risk Cases",
                "id": "high_risk_cases",
            },
            {
                "name": "High Risk Rate",
                "id": "high_risk_rate",
            },
            {
                "name": "Follow-up Completed",
                "id": "followup_completed",
            },
            {
                "name": "Follow-up Completion",
                "id": "followup_completion_rate",
            },
        ],

        style_header={
            "backgroundColor": COLORS["primary"],
            "color": "white",
            "fontWeight": "bold",
            "border": "none",
        },

        style_cell={
            "textAlign": "center",
            "padding": "12px",
            "fontFamily": "Arial",
            "color": COLORS["text"],
        },

        style_data={
            "backgroundColor": "white",
        },

        style_table={
            "overflowX": "auto",
            "borderRadius": "12px",
        },

        page_size=10,

    )


    # --------------------------------------------------------
    # LIVE FLAGGED CASES TABLE
    # --------------------------------------------------------

    flagged_table = dash_table.DataTable(

        data=flagged_cases,

        columns=[
            {
                "name": "Prediction ID",
                "id": "prediction_id",
            },
            {
                "name": "District",
                "id": "district",
            },
            {
                "name": "Province",
                "id": "province",
            },
            {
                "name": "Clinic",
                "id": "clinic_id",
            },
            {
                "name": "Age",
                "id": "age",
            },
            {
                "name": "Risk Score",
                "id": "risk_score",
            },
            {
                "name": "Risk Level",
                "id": "risk_level",
            },
            {
                "name": "Follow-up Status",
                "id": "followup_status",
            },
            {
                "name": "Tested",
                "id": "tested",
            },
            {
                "name": "Test Result",
                "id": "test_result",
            },
            {
                "name": "Referred",
                "id": "referred",
            },
            {
                "name": "Treated",
                "id": "treated",
            },
            {
                "name": "Created",
                "id": "created_at",
            },
        ],

        style_header={
            "backgroundColor": "#7A1F1F",
            "color": "white",
            "fontWeight": "bold",
        },

        style_cell={
            "textAlign": "center",
            "padding": "10px",
            "fontFamily": "Arial",
            "fontSize": "13px",
            "color": COLORS["text"],
            "whiteSpace": "normal",
            "height": "auto",
        },

        style_data={
            "backgroundColor": "white",
        },

        style_table={
            "overflowX": "auto",
        },

        page_size=10,

    )


    return html.Div(
        [

            html.H2(
                "🏥 Rwanda Biomedical Centre District Accountability Dashboard",
                style={
                    "color": COLORS["primary"],
                    "fontWeight": "800",
                }
            ),

            html.P(
                "District-level monitoring of MARISA predictions, high-risk patients, and documented follow-up actions.",
                style={
                    "color": COLORS["muted"],
                    "fontSize": "15px",
                }
            ),

            html.Div(
                [

                    metric_card(
                        "Districts Reporting",
                        len(districts),
                        "Districts with recorded predictions",
                        "🗺️",
                        COLORS["primary"]
                    ),

                    metric_card(
                        "Live Flagged Cases",
                        len(flagged_cases),
                        "High-risk cases requiring monitoring",
                        "🚨",
                        COLORS["danger"]
                    ),

                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(250px, 1fr))",
                    "gap": "18px",
                    "marginBottom": "25px",
                }
            ),

            html.Div(
                [

                    html.Div(
                        dcc.Graph(
                            figure=district_chart
                        ),
                        style={
                            "backgroundColor": "white",
                            "borderRadius": "14px",
                            "padding": "10px",
                        }
                    ),

                    html.Div(
                        dcc.Graph(
                            figure=high_risk_chart
                        ),
                        style={
                            "backgroundColor": "white",
                            "borderRadius": "14px",
                            "padding": "10px",
                        }
                    ),

                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(450px, 1fr))",
                    "gap": "20px",
                }
            ),

            html.Br(),

            html.H3(
                "📋 District Performance",
                style={
                    "color": COLORS["primary"],
                }
            ),

            district_table,

            html.Br(),
            html.Br(),

            html.H3(
                "🚨 Live High-Risk Cases",
                style={
                    "color": COLORS["danger"],
                }
            ),

            html.P(
                "These records are monitoring alerts and do not represent a clinical diagnosis.",
                style={
                    "color": COLORS["muted"],
                }
            ),

            flagged_table,

        ]
    )


# ============================================================
# DAILY TREND DASHBOARD
# ============================================================

def create_daily_trend_dashboard(data):

    daily_data = data.get(
        "daily_trend",
        []
    )


    trend_df = pd.DataFrame(
        daily_data
    )


    if not trend_df.empty:

        trend_df["date"] = pd.to_datetime(
            trend_df["date"]
        )


    # --------------------------------------------------------
    # TOTAL PREDICTIONS TREND
    # --------------------------------------------------------

    if trend_df.empty:

        prediction_trend = go.Figure()

        prediction_trend.add_annotation(
            text="No daily trend data available",
            showarrow=False
        )

        high_risk_trend = go.Figure()

        high_risk_trend.add_annotation(
            text="No high-risk trend data available",
            showarrow=False
        )

    else:

        prediction_trend = px.line(
            trend_df,
            x="date",
            y="total_predictions",
            markers=True,
            title="Daily MARISA Predictions",
        )

        prediction_trend.update_layout(
            template="plotly_white",
            height=430,
        )


        high_risk_trend = px.line(
            trend_df,
            x="date",
            y="high_risk_predictions",
            markers=True,
            title="Daily High-Risk Predictions",
        )

        high_risk_trend.update_layout(
            template="plotly_white",
            height=430,
        )


    # --------------------------------------------------------
    # DAILY COMPARISON
    # --------------------------------------------------------

    if trend_df.empty:

        comparison_chart = go.Figure()

        comparison_chart.add_annotation(
            text="No comparison data available",
            showarrow=False
        )

    else:

        comparison_chart = go.Figure()

        comparison_chart.add_trace(
            go.Bar(
                x=trend_df["date"],
                y=trend_df["total_predictions"],
                name="Total Predictions",
            )
        )

        comparison_chart.add_trace(
            go.Bar(
                x=trend_df["date"],
                y=trend_df["high_risk_predictions"],
                name="High Risk",
            )
        )

        comparison_chart.add_trace(
            go.Bar(
                x=trend_df["date"],
                y=trend_df["pregnant_predictions"],
                name="Pregnant Patients",
            )
        )

        comparison_chart.update_layout(
            barmode="group",
            template="plotly_white",
            title="Daily Prediction Comparison",
            height=450,
        )


    # --------------------------------------------------------
    # HIGH-RISK RATE
    # --------------------------------------------------------

    if trend_df.empty:

        rate_chart = go.Figure()

        rate_chart.add_annotation(
            text="No high-risk rate data available",
            showarrow=False
        )

    else:

        rate_chart = px.line(
            trend_df,
            x="date",
            y="high_risk_rate",
            markers=True,
            title="Daily High-Risk Rate",
        )

        rate_chart.update_layout(
            template="plotly_white",
            height=430,
            yaxis_tickformat=".0%",
        )


    return html.Div(
        [

            html.H2(
                "📈 Daily Prediction Trend Dashboard",
                style={
                    "color": COLORS["primary"],
                    "fontWeight": "800",
                }
            ),

            html.P(
                "Time-based monitoring of MARISA predictions, high-risk flags, and pregnancy assessments.",
                style={
                    "color": COLORS["muted"],
                    "fontSize": "15px",
                }
            ),

            html.Div(
                [

                    metric_card(
                        "Days Recorded",
                        data.get(
                            "total_days",
                            0
                        ),
                        "Days with prediction activity",
                        "📅",
                        COLORS["primary"]
                    ),

                    metric_card(
                        "Latest Day Predictions",
                        (
                            int(
                                trend_df.iloc[-1]["total_predictions"]
                            )
                            if not trend_df.empty
                            else 0
                        ),
                        "Most recent reporting day",
                        "📊",
                        COLORS["accent"]
                    ),

                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(250px, 1fr))",
                    "gap": "18px",
                    "marginBottom": "25px",
                }
            ),

            html.Div(
                [

                    html.Div(
                        dcc.Graph(
                            figure=prediction_trend
                        ),
                        style={
                            "backgroundColor": "white",
                            "borderRadius": "14px",
                        }
                    ),

                    html.Div(
                        dcc.Graph(
                            figure=high_risk_trend
                        ),
                        style={
                            "backgroundColor": "white",
                            "borderRadius": "14px",
                        }
                    ),

                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(450px, 1fr))",
                    "gap": "20px",
                }
            ),

            html.Br(),

            html.Div(
                dcc.Graph(
                    figure=comparison_chart
                ),
                style={
                    "backgroundColor": "white",
                    "borderRadius": "14px",
                    "padding": "10px",
                }
            ),

            html.Br(),

            html.Div(
                dcc.Graph(
                    figure=rate_chart
                ),
                style={
                    "backgroundColor": "white",
                    "borderRadius": "14px",
                    "padding": "10px",
                }
            ),

        ]
    )


# ============================================================
# MAIN LAYOUT
# ============================================================

app.layout = html.Div(
    [

        dcc.Interval(
            id="refresh-interval",
            interval=30 * 1000,
            n_intervals=0
        ),

        create_header(),

        html.Div(
            [

                dcc.Tabs(
                    id="dashboard-tabs",
                    value="national",
                    children=[

                        dcc.Tab(
                            label="🇷🇼 National Dashboard",
                            value="national",
                        ),

                        dcc.Tab(
                            label="🏥 RBC District Dashboard",
                            value="rbc",
                        ),

                        dcc.Tab(
                            label="📈 Daily Trends",
                            value="trend",
                        ),

                    ],
                    style={
                        "fontWeight": "700",
                    }
                ),

                html.Div(
                    id="dashboard-content",
                    style={
                        "marginTop": "25px",
                    }
                ),

            ],
            style={
                "padding": "0 25px 40px 25px",
                "maxWidth": "1600px",
                "margin": "auto",
            }
        ),

    ],
    style={
        "backgroundColor": COLORS["background"],
        "minHeight": "100vh",
        "fontFamily": "Arial, sans-serif",
    }
)


# ============================================================
# CALLBACK
# ============================================================

@app.callback(
    Output(
        "dashboard-content",
        "children"
    ),
    [
        Input(
            "dashboard-tabs",
            "value"
        ),

        Input(
            "refresh-interval",
            "n_intervals"
        ),
    ]
)
def update_dashboard(selected_tab, n_intervals):

    national_data = get_api_data(
        "/dashboard",
        {}
    )

    rbc_data = get_api_data(
        "/rbc-dashboard",
        {}
    )

    trend_data = get_api_data(
        "/daily-trend",
        {}
    )


    if selected_tab == "national":

        return create_national_dashboard(
            national_data
        )


    elif selected_tab == "rbc":

        return create_rbc_dashboard(
            rbc_data
        )


    elif selected_tab == "trend":

        return create_daily_trend_dashboard(
            trend_data
        )


    return html.Div(
        "Dashboard not available."
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8050,
        debug=False
    )