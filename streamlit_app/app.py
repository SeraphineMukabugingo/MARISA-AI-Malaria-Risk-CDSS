"""
MARISA — Clinician Risk Assessment Tool (Streamlit)

Improved frontend for the MARISA CDSS.
- Calls FastAPI /predict.
- English/Kinyarwanda support.
- Adds district capture for RBC monitoring.
- Improves accessibility, contrast, hierarchy, and risk-result presentation.
"""

import os
import requests
import streamlit as st
import plotly.graph_objects as go
import streamlit.components.v1 as components

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="MARISA Risk Assessment",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DISTRICTS_BY_PROVINCE = {
    "Kigali City": ["Gasabo", "Kicukiro", "Nyarugenge"],
    "Southern Province": ["Gisagara", "Huye", "Kamonyi", "Muhanga", "Nyamagabe", "Nyanza", "Nyaruguru", "Ruhango"],
    "Northern Province": ["Burera", "Gakenke", "Gicumbi", "Musanze", "Rulindo"],
    "Eastern Province": ["Bugesera", "Gatsibo", "Kayonza", "Kirehe", "Ngoma", "Nyagatare", "Rwamagana"],
    "Western Province": ["Karongi", "Ngororero", "Nyabihu", "Nyamasheke", "Rubavu", "Rusizi", "Rutsiro"],
}

T = {

    # ==========================================================
    # ENGLISH
    # ==========================================================

    "en": {

        # ------------------------------------------------------
        # APPLICATION HEADER
        # ------------------------------------------------------

        "title": "MARISA Malaria Risk Assessment",

        "subtitle": (
            "Explainable AI-powered clinical decision support · "
            "Malaria risk assessment for women of reproductive age"
        ),

        "tab_new": "🩺 New Assessment",
        "tab_history": "📋 Session History",
        "tab_summary": "📊 Clinic Summary",


        # ------------------------------------------------------
        # CLINICAL SAFETY NOTICE
        # ------------------------------------------------------

        "caveat": (
            "**Prototype clinical decision-support tool — not a diagnostic test. "
            "The model estimates malaria risk and does not confirm, diagnose, "
            "or rule out malaria. <b>Validation specifically among pregnant women "
            "was based on only 23 confirmed cases and showed unstable performance "
            "(average recall 11%). Always follow standard malaria testing, antenatal "
            "care, and clinical protocols regardless of the predicted score</b>.**"
        ),


        # ------------------------------------------------------
        # PATIENT INFORMATION
        # ------------------------------------------------------

        "patient_info": "Patient Information",

        "patient_info_sub": (
            "Enter information available during today's clinical visit."
        ),

        "clinic_id": "Clinic ID (optional)",

        "name": (
            "Patient name or ID "
            "(stored only in this browser session)"
        ),

        "age": "Age (years)",

        "pregnant": "Currently pregnant?",

        "yes": "Yes",
        "no": "No",
        "unknown": "Don't know",


        # ------------------------------------------------------
        # GEOGRAPHIC INFORMATION
        # ------------------------------------------------------

        "province": "Province",
        "district": "District",

        "residence": "Place of residence",

        "rural": "Rural",
        "urban": "Urban",


        # ------------------------------------------------------
        # SOCIO-DEMOGRAPHIC INFORMATION
        # ------------------------------------------------------

        "marital": "Marital status",

        "education": "Highest education level completed",

        "insurance": (
            "Does the patient have health insurance "
            "(Mutuelle or another scheme)?"
        ),


        # ------------------------------------------------------
        # HOUSEHOLD AND PREVENTION
        # ------------------------------------------------------

        "household": "Household & Malaria Prevention",

        "has_net": "Does the household have a mosquito net?",

        "itn": (
            "Is the mosquito net insecticide-treated "
            "(ITN)?"
        ),

        "sprayed": (
            "Has the dwelling been sprayed against mosquitoes "
            "within the last 12 months?"
        ),

        "water": "Main source of drinking water",

        "toilet": "Toilet facility type",

        "electricity": "Does the household have electricity?",


        # ------------------------------------------------------
        # ASSESSMENT
        # ------------------------------------------------------

        "calculate": "Calculate Malaria Risk Assessment",

        "risk_assessment": "Malaria Risk Assessment",

        "risk_assessment_sub": (
            "AI-based malaria risk estimate with explainable "
            "SHAP-based contributing factors."
        ),

        "why_score": (
            "Why this risk score? — Key contributing factors"
        ),

        "recommendations": "Recommended Clinical Actions",


        # ------------------------------------------------------
        # PREGNANCY SAFETY FLAG
        # ------------------------------------------------------

        "pregnant_flag": (
            "PREGNANT PATIENT: This model has limited validation specifically "
            "among pregnant women. The predicted score must not replace standard "
            "antenatal malaria prevention, testing, or clinical assessment. "
            "Always follow national and facility clinical protocols."
        ),


        # ------------------------------------------------------
        # BUTTONS AND ACTIONS
        # ------------------------------------------------------

        "print_btn": "🖨 Print / Save as PDF",

        "save_btn": "💾 Save to Session History",

        "saved_msg": (
            "Assessment successfully saved to this session's history."
        ),


        # ------------------------------------------------------
        # HISTORY AND SUMMARY
        # ------------------------------------------------------

        "no_history": (
            "No assessments have been saved in this session yet."
        ),

        "no_summary": (
            "No assessment data is available yet. "
            "Complete an assessment to generate a clinic summary."
        ),


        # ------------------------------------------------------
        # MODEL INFORMATION
        # ------------------------------------------------------

        "model_label": "Model version:",


        # ------------------------------------------------------
        # EMPTY RESULT STATE
        # ------------------------------------------------------

        "awaiting_title": "Assessment Ready",

        "awaiting_text": (
            "Patient risk results and contributing factors will appear "
            "here after the assessment is submitted."
        ),


        # ------------------------------------------------------
        # RISK-BASED RECOMMENDATIONS
        # ------------------------------------------------------

        "recos": {

            "Low": [

                "Continue routine malaria prevention education.",

                "Encourage consistent and correct use of insecticide-treated nets.",

                "Continue routine clinical assessment and follow standard malaria testing protocols when indicated.",

                "Do not rely on this risk score alone to determine malaria status.",
            ],


            "Moderate": [

                "Perform clinical assessment and consider malaria testing (RDT) when symptoms or clinical concerns are present.",

                "Reinforce consistent and correct use of insecticide-treated nets.",

                "Review household malaria prevention measures.",

                "Assess access to healthcare and health-insurance coverage.",
            ],


            "High": [

                "Prioritize clinical assessment and malaria testing (RDT) according to facility protocols.",

                "Reinforce consistent and correct use of insecticide-treated nets.",

                "Review household-level malaria prevention measures.",

                "Consider closer follow-up according to clinical findings and local protocols.",
            ],


            "Very High": [

                "Prioritize prompt clinical assessment and malaria testing according to facility protocols.",

                "Follow national malaria management and referral guidelines when clinically indicated.",

                "Review and reinforce household malaria prevention measures, including access to insecticide-treated nets.",

                "Document and arrange appropriate follow-up based on clinical findings and local protocols.",
            ],
        },
    },


    # ==========================================================
    # KINYARWANDA
    # ==========================================================

    "rw": {

        # ------------------------------------------------------
        # APPLICATION HEADER
        # ------------------------------------------------------

        "title": "MARISA — Isuzuma ry'Ibyago bya Malariya",

        "subtitle": (
            "Uburyo bwa AI bufasha mu gufata ibyemezo by'ubuvuzi · "
            "Isuzuma ry'ibyago bya malariya ku bagore bageze mu myaka yo kubyara"
        ),

        "tab_new": "🩺 Isuzuma Rishya",

        "tab_history": "📋 Amateka y'Isuzuma",

        "tab_summary": "📊 Incamake y'Ivuriro",


        # ------------------------------------------------------
        # CLINICAL SAFETY NOTICE
        # ------------------------------------------------------

        "caveat": (
            "**Iki ni igikoresho cy'icyitegererezo gifasha mu gufata ibyemezo "
            "by'ubuvuzi — ntabwo ari ikizamini cyangwa uburyo bwo gusuzuma malariya. "
            "Model igereranya ibyago bya malariya ariko ntishobora kwemeza, gusuzuma, "
            "cyangwa gukuraho malariya. Igenzura ryakozwe by'umwihariko ku bagore "
            "batwite ryashingiye ku bantu 23 gusa bari bafite malariya yemejwe kandi "
            "ryagaragaje ibisubizo bidahagaze neza (average recall 11%). Buri gihe "
            "kurikiza uburyo busanzwe bwo gupima malariya, gahunda z'ubuvuzi bw'umubyeyi "
            "utwite, n'amabwiriza y'ubuvuzi n'ubwo amanota y'ibyago yaba ameze ate.**"
        ),


        # ------------------------------------------------------
        # PATIENT INFORMATION
        # ------------------------------------------------------

        "patient_info": "Amakuru y'Umurwayi",

        "patient_info_sub": (
            "Andika amakuru aboneka igihe umurwayi aje kwisuzumisha uyu munsi."
        ),

        "clinic_id": "Nimero cyangwa ID y'Ivuriro (si ngombwa)",

        "name": (
            "Izina cyangwa nimero y'umurwayi "
            "(bibikwa muri iki cyiciro gusa)"
        ),

        "age": "Imyaka",

        "pregnant": "Aratwite ubu?",

        "yes": "Yego",

        "no": "Oya",

        "unknown": "Simbizi",


        # ------------------------------------------------------
        # GEOGRAPHIC INFORMATION
        # ------------------------------------------------------

        "province": "Intara",

        "district": "Akarere",

        "residence": "Aho atuye",

        "rural": "Icyaro",

        "urban": "Umujyi",


        # ------------------------------------------------------
        # SOCIO-DEMOGRAPHIC INFORMATION
        # ------------------------------------------------------

        "marital": "Uko abana n'undi muntu",

        "education": "Urwego rwo hejuru rw'amashuri yize",

        "insurance": (
            "Afite ubwishingizi bw'ubuzima "
            "(Mutuelle cyangwa ubundi bwishingizi)?"
        ),


        # ------------------------------------------------------
        # HOUSEHOLD AND PREVENTION
        # ------------------------------------------------------

        "household": "Urugo n'Uburyo bwo Kwirinda Malariya",

        "has_net": "Urugo rufite inzitiramibu?",

        "itn": (
            "Iyo nzitiramibu ifite umuti wica imibu (ITN)?"
        ),

        "sprayed": (
            "Inzu ituwemo yasizwe imiti irwanya imibu "
            "mu mezi 12 ashize?"
        ),

        "water": "Isoko nyamukuru ry'amazi yo kunywa",

        "toilet": "Ubwoko bw'ubwiherero",

        "electricity": "Urugo rufite amashanyarazi?",


        # ------------------------------------------------------
        # ASSESSMENT
        # ------------------------------------------------------

        "calculate": "Bara Ibyago bya Malariya",

        "risk_assessment": "Isuzuma ry'Ibyago bya Malariya",

        "risk_assessment_sub": (
            "Ikigereranyo cy'ibyago bya malariya gitangwa na AI "
            "hamwe n'ibisobanuro by'impamvu zabigizemo uruhare."
        ),

        "why_score": (
            "Impamvu z'aya manota — Ibintu by'ingenzi byayagizeho uruhare"
        ),

        "recommendations": "Ibikorwa Byasabwa",


        # ------------------------------------------------------
        # PREGNANCY SAFETY FLAG
        # ------------------------------------------------------

        "pregnant_flag": (
            "UMURWAYI UTWITE: Iyi model ntabwo yagenzuwe bihagije "
            "by'umwihariko ku bagore batwite. Amanota atanzwe n'iyi model "
            "ntagomba gusimbura uburyo busanzwe bwo kwirinda malariya, "
            "kuyipima cyangwa isuzuma ry'ubuvuzi. Buri gihe kurikiza "
            "amabwiriza y'ubuvuzi bw'umubyeyi utwite n'ay'ikigo nderabuzima."
        ),


        # ------------------------------------------------------
        # BUTTONS AND ACTIONS
        # ------------------------------------------------------

        "print_btn": "🖨 Chapa / Bika nka PDF",

        "save_btn": "💾 Bika mu Mateka y'Isuzuma",

        "saved_msg": (
            "Isuzuma ryabitswe neza mu mateka y'iki cyiciro."
        ),


        # ------------------------------------------------------
        # HISTORY AND SUMMARY
        # ------------------------------------------------------

        "no_history": (
            "Nta suzuma rirabikwa muri iki cyiciro."
        ),

        "no_summary": (
            "Nta makuru y'isuzuma arahari. "
            "Banza ukore isuzuma kugira ngo ubone incamake y'ivuriro."
        ),


        # ------------------------------------------------------
        # MODEL INFORMATION
        # ------------------------------------------------------

        "model_label": "Verisiyo ya Model:",


        # ------------------------------------------------------
        # EMPTY RESULT STATE
        # ------------------------------------------------------

        "awaiting_title": "Isuzuma Ryiteguye",

        "awaiting_text": (
            "Ibisubizo by'ibyago n'ibintu byabigizemo uruhare "
            "bizagaragara hano nyuma yo kohereza isuzuma."
        ),


        # ------------------------------------------------------
        # RISK-BASED RECOMMENDATIONS
        # ------------------------------------------------------

        "recos": {

            "Low": [

                "Komeza gutanga inyigisho zisanzwe zo kwirinda malariya.",

                "Shishikariza gukoresha neza kandi buri gihe inzitiramibu zifite umuti.",

                "Komeza isuzuma risanzwe ry'ubuvuzi kandi ukurikize amabwiriza yo gupima malariya igihe bikenewe.",

                "Ntushingire kuri aya manota gusa kugira ngo umenye niba umurwayi afite malariya.",
            ],


            "Moderate": [

                "Kora isuzuma ry'ubuvuzi kandi utekereze gupima malariya (RDT) niba hari ibimenyetso cyangwa impungenge z'ubuvuzi.",

                "Shishikariza gukoresha neza kandi buri gihe inzitiramibu zifite umuti.",

                "Suzuma uburyo bwo kwirinda malariya mu rugo.",

                "Suzuma uburyo umurwayi abona ubuvuzi n'ubwishingizi bw'ubuzima.",
            ],


            "High": [

                "Shyira imbere isuzuma ry'ubuvuzi no gupima malariya (RDT) ukurikije amabwiriza y'ikigo nderabuzima.",

                "Shishikariza gukoresha neza kandi buri gihe inzitiramibu zifite umuti.",

                "Suzuma uburyo bwo kwirinda malariya mu rugo.",

                "Teganya ubukurikirane bukwiye hashingiwe ku isuzuma ry'ubuvuzi n'amabwiriza akurikizwa.",
            ],


            "Very High": [

                "Shyira imbere isuzuma ryihuse ry'ubuvuzi no gupima malariya ukurikije amabwiriza y'ikigo nderabuzima.",

                "Kurikiza amabwiriza y'igihugu ajyanye no kuvura no kohereza umurwayi igihe bikenewe.",

                "Suzuma kandi ushimangire uburyo bwo kwirinda malariya mu rugo, harimo no kubona inzitiramibu zifite umuti.",

                "Andika gahunda y'ubukurikirane ikurikije ibisubizo by'isuzuma ry'ubuvuzi n'amabwiriza akurikizwa.",
            ],
        },
    },
}
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "history" not in st.session_state:
    st.session_state.history = []

t = T[st.session_state.lang]

st.markdown(
    """
<style>
:root {
    --navy:#0B1F3A; --navy2:#123456; --teal:#087E8B; --teal2:#14A6B8;
    --bg:#F4F7FA; --card:#FFFFFF; --text:#172033; --muted:#52606D;
    --border:#D7E1EA; --input:#202733; --input-border:#394455;
    --success:#2E9B62; --warning:#D97706; --danger:#D64545;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 8% 8%, rgba(20,166,184,.13), transparent 22rem),
        radial-gradient(circle at 92% 18%, rgba(11,31,58,.08), transparent 26rem),
        linear-gradient(180deg, #EAF3F8 0%, #F6FAFC 45%, #EAF2F7 100%);
    color:var(--text);
}
.block-container { padding-top:1.25rem; padding-bottom:2.5rem; max-width:1500px; }
h1,h2,h3,h4,h5,h6,[data-testid="stMarkdownContainer"] strong { color:var(--text); }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:var(--muted)!important; }
[data-testid="stWidgetLabel"] p,[data-testid="stWidgetLabel"] label,[data-testid="stRadio"] p,[data-testid="stRadio"] label,[data-testid="stRadio"] label span {
    color:var(--text)!important; opacity:1!important; font-weight:650!important;
}
div[data-testid="stForm"] {
    border:1px solid var(--border); border-radius:20px; padding:26px 28px 24px;
    background:rgba(255,255,255,.98); box-shadow:0 14px 38px rgba(11,31,58,.08);
}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input {
    background:var(--input)!important; color:#FFF!important; border-color:var(--input-border)!important; border-radius:11px!important;
}
[data-baseweb="select"]>div { background:var(--input)!important; color:#FFF!important; border-color:var(--input-border)!important; border-radius:11px!important; }
[data-baseweb="select"] span { color:#FFF!important; }
[data-baseweb="menu"] { background:#FFF!important; }
[data-baseweb="menu"] li { color:var(--text)!important; }
[data-testid="stRadio"] { margin-bottom:.35rem; }
.stButton>button,.stFormSubmitButton>button { border-radius:11px!important; font-weight:750!important; min-height:46px; transition:.15s ease; }
.stFormSubmitButton>button { color:#FFF!important; border:0!important; background:linear-gradient(135deg,var(--teal) 0%,var(--teal2) 100%)!important; box-shadow:0 9px 22px rgba(8,126,139,.25); }
.stFormSubmitButton>button:hover,.stButton>button:hover { transform:translateY(-1px); filter:brightness(1.03); }
.stTabs [data-baseweb="tab"] { font-weight:700; }
.stTabs [aria-selected="true"] { color:var(--teal)!important; }
.marisa-header { background:linear-gradient(135deg,var(--navy) 0%,#0D4E58 100%); padding:23px 30px; border-radius:18px; box-shadow:0 14px 34px rgba(11,31,58,.18); position:relative; overflow:hidden; }
.marisa-header:after { content:""; position:absolute; width:180px; height:180px; right:-55px; top:-75px; border-radius:50%; background:rgba(255,255,255,.07); }
.marisa-header-title { color:#FFF!important; margin:0; font-size:1.75rem; font-weight:800; }
.marisa-header-sub { color:#CFECEF!important; margin:5px 0 0; font-size:.98rem; }
.section-chip { display:inline-block; margin:12px 0 15px; padding:7px 12px; border-radius:999px; background:#E8F6F7; color:#075A64; font-weight:800; font-size:.86rem; }
.result-ready { min-height:355px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; background:linear-gradient(180deg,#FFF 0%,#F8FBFC 100%); border:1px dashed #B8CDD2; border-radius:20px; padding:36px; }
.medical-icon-wrap { height:84px; width:84px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:38px; background:linear-gradient(135deg,#E6F7F8,#D7F0F2); border:1px solid #B9DEE2; box-shadow:0 10px 24px rgba(8,126,139,.12); }
.result-ready h3 { margin:18px 0 7px; color:var(--navy); }
.result-ready p { max-width:420px; color:var(--muted); line-height:1.55; }
.factor-row { display:flex; align-items:center; gap:10px; padding:10px 0; border-bottom:1px solid #E8EEF2; }
.factor-label { width:52%; font-size:12.8px; font-weight:700; color:var(--text); }
.factor-track { flex:1; background:#E8EEF2; border-radius:999px; height:9px; overflow:hidden; }
.factor-fill { height:9px; border-radius:999px; }
.factor-val { width:55px; text-align:right; font-size:11.8px; font-weight:800; color:var(--muted); }
.reco-item { display:flex; gap:11px; padding:10px 0; border-bottom:1px solid #E8EEF2; font-size:13.6px; color:var(--text); }
.reco-dot { flex:0 0 23px; height:23px; width:23px; border-radius:50%; background:#E5F5F3; color:#087E8B; display:flex; align-items:center; justify-content:center; font-size:11px; font-weight:900; }
[data-testid="stMetric"] { background:#FFF; border:1px solid var(--border); padding:15px 16px; border-radius:15px; box-shadow:0 7px 20px rgba(11,31,58,.05); }
[data-testid="stMetricLabel"] p { color:var(--muted)!important; font-weight:650; }
[data-testid="stMetricValue"] { color:var(--navy)!important; }
[data-testid="stAlert"] { border-radius:13px; }
.marisa-footer { color:var(--muted); font-size:.84rem; text-align:center; line-height:1.55; padding:8px 0 4px; }
@media(max-width:900px){ .block-container{padding-left:1rem;padding-right:1rem;} div[data-testid="stForm"]{padding:20px 17px;} .factor-label{width:45%;} }
</style>
""",
    unsafe_allow_html=True,
)

col_head, col_lang = st.columns([5, 1])
with col_head:
    st.markdown(
        f'''<div class="marisa-header"><h2 class="marisa-header-title">🩺 {t['title']}</h2><p class="marisa-header-sub">{t['subtitle']}</p></div>''',
        unsafe_allow_html=True,
    )
with col_lang:
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    lang_choice = st.radio("Language", ["EN", "KIN"], horizontal=True, label_visibility="collapsed", index=0 if st.session_state.lang == "en" else 1, key="lang_radio")
    new_lang = "en" if lang_choice == "EN" else "rw"
    if new_lang != st.session_state.lang:
        st.session_state.lang = new_lang
        st.rerun()

t = T[st.session_state.lang]
st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
st.warning(t["caveat"])

tab_new, tab_history, tab_summary = st.tabs([t["tab_new"], t["tab_history"], t["tab_summary"]])

with tab_new:
    col_form, col_result = st.columns([1.08, 0.92], gap="large")

    with col_form:
        st.subheader(t["patient_info"])
        st.caption(t["patient_info_sub"])

        # Geographic selectors are intentionally OUTSIDE the form.
        # This makes Streamlit rerun immediately when Province changes,
        # so the District list always matches the selected Province.
        geo1, geo2 = st.columns(2)
        with geo1:
            province = st.selectbox(
                t["province"],
                list(DISTRICTS_BY_PROVINCE.keys()),
                key="province_selector",
            )

        district_options = DISTRICTS_BY_PROVINCE[province]

        # Reset the district whenever the province changes.
        if st.session_state.get("_district_province") != province:
            st.session_state["district_selector"] = district_options[0]
            st.session_state["_district_province"] = province

        # Safety check in case an old district is stored in session state.
        if st.session_state.get("district_selector") not in district_options:
            st.session_state["district_selector"] = district_options[0]

        with geo2:
            district = st.selectbox(
                t["district"],
                district_options,
                key="district_selector",
            )

        with st.form("assessment_form"):
            c1, c2 = st.columns(2)
            with c1:
                clinic_id = st.text_input(t["clinic_id"], value="")
                name = st.text_input(t["name"], value="")
                age = st.number_input(t["age"], min_value=15, max_value=49, value=27, step=1)
                pregnant = st.radio(t["pregnant"], [t["no"], t["yes"]], horizontal=True) == t["yes"]
            with c2:
                residence_choice = st.radio(t["residence"], [t["rural"], t["urban"]], horizontal=True)
                residence_type = "Rural" if residence_choice == t["rural"] else "Urban"

            c3, c4 = st.columns(2)
            with c3:
                marital_status = st.selectbox(t["marital"], ["never", "married", "widowed", "divorced"])
                education = st.selectbox(t["education"], ["none", "primary", "secondary", "higher"])
            with c4:
                insurance_choice = st.radio(t["insurance"], [t["yes"], t["no"], t["unknown"]], horizontal=True)
                health_insurance = {"Yes":"yes","No":"no","Don't know":"unknown","Yego":"yes","Oya":"no","Simbizi":"unknown"}.get(insurance_choice, "unknown")
                electricity = st.radio(t["electricity"], [t["no"], t["yes"]], horizontal=True) == t["yes"]

            st.markdown(f"<div class='section-chip'>🛡️ {t['household']}</div>", unsafe_allow_html=True)
            c5, c6 = st.columns(2)
            with c5:
                has_net = st.radio(t["has_net"], [t["yes"], t["no"]], horizontal=True) == t["yes"]
                itn_treated = None
                if has_net:
                    itn_treated = st.radio(t["itn"], [t["yes"], t["no"]], horizontal=True) == t["yes"]
                sprayed_choice = st.radio(t["sprayed"], [t["yes"], t["no"], t["unknown"]], horizontal=True)
                sprayed = {"Yes":"yes","No":"no","Don't know":"unknown","Yego":"yes","Oya":"no","Simbizi":"unknown"}.get(sprayed_choice, "unknown")
            with c6:
                water_source = st.selectbox(t["water"], ["improved", "unimproved"])
                toilet_facility = st.selectbox(t["toilet"], ["improved", "unimproved"])

            submitted = st.form_submit_button(t["calculate"], type="primary", use_container_width=True)

    with col_result:
        st.subheader(t["risk_assessment"])
        st.caption(t["risk_assessment_sub"])

        if submitted:
            payload = {
                "clinic_id": clinic_id or None,
                "district": district,
                "age": age,
                "pregnant": pregnant,
                "province": province,
                "residence_type": residence_type,
                "marital_status": marital_status,
                "education": education,
                "health_insurance": health_insurance,
                "has_net": has_net,
                "itn_treated": itn_treated,
                "sprayed_last_12mo": sprayed,
                "water_source": water_source,
                "toilet_facility": toilet_facility,
                "electricity": electricity,
            }
            try:
                with st.spinner("Running MARISA risk assessment..."):
                    resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=20)
                resp.raise_for_status()
                result = resp.json()

                score = float(result["risk_score"])
                level = result["risk_level"]
                color_map = {"Low":"#2E9B62","Moderate":"#D97706","High":"#E76624","Very High":"#D64545"}
                color = color_map.get(level, "#52606D")

                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=score*100,
                    number={"suffix":"%", "font":{"size":38, "color":"#0B1F3A"}},
                    gauge={
                        "axis":{"range":[0,100], "tickcolor":"#52606D"},
                        "bar":{"color":color}, "bgcolor":"#FFFFFF", "borderwidth":0,
                        "steps":[
                            {"range":[0,10],"color":"#EAF7F0"},
                            {"range":[10,20],"color":"#FFF4DE"},
                            {"range":[20,35],"color":"#FFF0E7"},
                            {"range":[35,100],"color":"#FCE8E8"},
                        ],
                        "threshold":{"line":{"color":color,"width":3},"thickness":0.8,"value":score*100},
                    },
                    title={"text":level.upper() + (" RISK" if st.session_state.lang=="en" else " IBYAGO"), "font":{"size":18,"color":"#172033"}},
                ))
                fig.update_layout(height=290, margin=dict(t=55,b=15,l=30,r=30), paper_bgcolor="rgba(0,0,0,0)", font={"color":"#172033"})
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

                if result.get("validation_note"):
                    st.error(t["pregnant_flag"])
                if result.get("calibration_note"):
                    st.caption(result["calibration_note"])

                st.markdown(f"### {t['why_score']}")
                for factor in result.get("top_factors", []):
                    contribution = float(factor.get("contribution", 0))
                    bar_color = "#D64545" if factor.get("direction") == "increases_risk" else "#2E9B62"
                    pct = min(abs(contribution)*400, 100)
                    st.markdown(
                        f"<div class='factor-row'><div class='factor-label'>{factor.get('label','Feature')}</div><div class='factor-track'><div class='factor-fill' style='width:{pct}%;background:{bar_color};'></div></div><div class='factor-val'>{contribution:+.3f}</div></div>",
                        unsafe_allow_html=True,
                    )

                st.markdown(f"### {t['recommendations']}")
                recos = t["recos"].get(level, []).copy()
                if pregnant:
                    recos.insert(0, t["pregnant_flag"])
                for i, recommendation in enumerate(recos, 1):
                    st.markdown(f"<div class='reco-item'><div class='reco-dot'>{i}</div><div>{recommendation}</div></div>", unsafe_allow_html=True)

                st.caption(f"{t['model_label']} {result.get('model_version','unknown')}")
                col_print, col_save = st.columns(2)
                with col_print:
                    if st.button(t["print_btn"], use_container_width=True):
                        components.html("<script>window.parent.print();</script>", height=0)
                with col_save:
                    if st.button(t["save_btn"], use_container_width=True):
                        st.session_state.history.append({
                            "name": name or "Unnamed",
                            "age": age,
                            "pregnant": pregnant,
                            "province": province,
                            "district": district,
                            "score": score,
                            "level": level,
                        })
                        st.success(t["saved_msg"])

            except requests.exceptions.HTTPError as exc:
                response_text = exc.response.text[:1000] if exc.response is not None else ""
                st.error(f"The MARISA backend returned an error.\n\nAPI: {API_BASE_URL}\n\n{response_text or exc}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Could not reach the MARISA prediction API.\n\nAPI: {API_BASE_URL}\n\n{exc}")
            except (KeyError, TypeError, ValueError) as exc:
                st.error(f"The prediction API returned an unexpected response format.\n\n{exc}")
        else:
            st.markdown(
                f'''<div class="result-ready"><div class="medical-icon-wrap">🩺</div><h3>{t['awaiting_title']}</h3><p>{t['awaiting_text']}</p></div>''',
                unsafe_allow_html=True,
            )

with tab_history:
    if not st.session_state.history:
        st.info(t["no_history"])
    else:
        color_map = {"Low":"#2E9B62","Moderate":"#D97706","High":"#E76624","Very High":"#D64545"}
        for record in reversed(st.session_state.history):
            c = color_map.get(record["level"], "#52606D")
            preg_tag = " · 🤰" if record["pregnant"] else ""
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;align-items:center;padding:14px 16px;margin-bottom:8px;background:#FFF;border:1px solid #D7E1EA;border-radius:14px;'><div><b style='color:#172033'>{record['name']}</b>{preg_tag}<br><span style='color:#52606D;font-size:12px;'>{record['age']} yrs · {record['district']} · {record['province']}</span></div><div style='padding:5px 12px;border-radius:999px;background:{c}18;color:{c};font-weight:800;font-size:12px;'>{record['level']} ({record['score']*100:.1f}%)</div></div>",
                unsafe_allow_html=True,
            )

with tab_summary:
    try:
        resp = requests.get(f"{API_BASE_URL}/drift-summary", timeout=10)
        resp.raise_for_status()
        d = resp.json()
        if d["total_predictions"] == 0:
            st.info(t["no_summary"])
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total predictions", d["total_predictions"])
            c2.metric("Last 7 days", d["predictions_last_7_days"])
            c3.metric("Mean risk", f"{(d['mean_risk_score_overall'] or 0):.3f}")
            c4.metric("High-risk rate", f"{(d['high_risk_rate_overall'] or 0)*100:.1f}%")
            st.markdown("### Mean risk score by province")
            for province_name, stats in d["by_province"].items():
                st.markdown(f"**{province_name}** · mean risk **{stats['mean_risk_score']}** · {stats['count']} predictions")
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not reach the backend summary endpoint.\n\n{exc}")

st.markdown("---")
st.markdown(
    """<div class="marisa-footer"><b>MARISA — Explainable AI-Powered Clinical Decision Support System</b><br>Research prototype using Rwanda DHS-derived patterns. MARISA supports clinical decision-making and does not replace laboratory malaria testing.</div>""",
    unsafe_allow_html=True,
)
