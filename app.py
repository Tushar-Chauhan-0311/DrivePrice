import streamlit as st
import pandas as pd
import joblib
import html
from pathlib import Path

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="DrivePrice | Smart Car Valuation",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "car_price_model.pkl"


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Place car_price_model.pkl "
        "in the same folder as app.py."
    )
    st.stop()
except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()


# =========================================================
# CUSTOM CSS
# =========================================================

st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080b10;
    --panel: #101722;
    --panel-light: #141e2b;
    --border: #283648;
    --text: #f5f7fb;
    --muted: #9baabd;
    --accent: #ff4057;
}

.stApp {
    background:
        radial-gradient(ellipse at 85% 0%, rgba(47, 73, 110, .13), transparent 35%),
        var(--bg);
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1380px;
    padding: 1.6rem 2.6rem 3rem;
}

h1, h2, h3 {
    color: var(--text) !important;
    font-family: 'Outfit', sans-serif !important;
}

/* BRAND */

.brand {
    font-family: 'Outfit', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    letter-spacing: -1.1px;
    margin: 0 0 28px;
    color: var(--text);
}

.brand span {
    color: var(--accent);
}

/* HERO */

.hero {
    min-height: 385px;
    display: flex;
    align-items: center;
    padding: 45px;
    margin-bottom: 45px;
    border: 1px solid var(--border);
    border-radius: 24px;
    overflow: hidden;
    position: relative;
    isolation: isolate;
    background:
        linear-gradient(
            90deg,
            rgba(5, 9, 15, .98) 0%,
            rgba(5, 9, 15, .90) 40%,
            rgba(5, 9, 15, .25) 100%
        ),
        url('https://images.unsplash.com/photo-1492144534655-ae79c964c9d7?auto=format&fit=crop&w=2000&q=85');
    background-size: cover;
    background-position: center 57%;
    animation: riseIn .65s ease-out both;
}

.hero-content {
    max-width: 590px;
    position: relative;
    z-index: 1;
}

.hero-tag {
    display: inline-block;
    padding: 9px 14px;
    margin-bottom: 22px;
    border: 1px solid rgba(255, 64, 87, .3);
    border-radius: 30px;
    background: rgba(255, 64, 87, .09);
    color: #ff8b99;
    font-size: .73rem;
    font-weight: 700;
    letter-spacing: 1.5px;
}

.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: clamp(2.5rem, 4.5vw, 4.2rem);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -1.8px;
    color: #fff;
    margin-bottom: 20px;
}

.hero-title span {
    color: var(--accent);
}

.hero-description {
    max-width: 465px;
    color: #c3cedd;
    font-size: 1rem;
    line-height: 1.9;
}

/* SECTION HEADINGS */

.section-title {
    color: var(--text);
    font-family: 'Outfit', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: -.5px;
    margin-bottom: 8px;
}

.section-description {
    color: var(--muted);
    font-size: .91rem;
    line-height: 1.8;
    margin-bottom: 24px;
}

/* FORM CARD */

div[data-testid="stForm"] {
    background: linear-gradient(145deg, #121b27, #0d131c);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 26px;
}

.form-heading {
    color: #f2f5f9;
    font-family: 'Outfit', sans-serif;
    font-size: 1.08rem;
    font-weight: 700;
    margin: 9px 0 6px;
}

.form-caption {
    color: var(--muted);
    font-size: .83rem;
    line-height: 1.7;
    margin-bottom: 20px;
}

div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label {
    color: #cbd5e1 !important;
    font-size: .88rem !important;
    font-weight: 600 !important;
}

div[data-testid="stNumberInput"] input {
    background: #111b27 !important;
    color: #f5f7fb !important;
    border: 1px solid #304052 !important;
    border-radius: 10px !important;
    min-height: 44px;
}

div[data-baseweb="select"] > div {
    background: #111b27 !important;
    color: #f5f7fb !important;
    border: 1px solid #304052 !important;
    border-radius: 10px !important;
    min-height: 44px;
}

div[data-testid="stNumberInput"] input:focus,
div[data-baseweb="select"] > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

/* SUBMIT BUTTON */

div[data-testid="stFormSubmitButton"] button {
    width: 100%;
    min-height: 52px;
    margin-top: 15px;
    border: 1px solid #ff667b;
    border-radius: 11px;
    background: linear-gradient(110deg, #ff4057, #d91f43);
    color: white;
    font-family: 'Outfit', sans-serif;
    font-size: .98rem;
    font-weight: 700;
    box-shadow: 0 8px 24px rgba(255, 64, 87, .13);
    transition: transform .2s ease, box-shadow .2s ease;
}

div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(255, 64, 87, .24);
    color: white;
}

/* RESULT CARD */

.result-card {
    box-sizing: border-box;
    width: 100%;
    min-height: 360px;
    padding: 30px;
    border: 1px solid #2a394b;
    border-radius: 20px;
    background:
        radial-gradient(ellipse at 100% 0%, rgba(255, 64, 87, .12), transparent 48%),
        linear-gradient(145deg, #131d29, #0c121b);
    box-shadow: 0 16px 40px rgba(0, 0, 0, .12);
    overflow-wrap: anywhere;
    animation: riseIn .45s ease-out both;
}

.result-label {
    color: #a8b6c7;
    font-size: .73rem;
    font-weight: 700;
    letter-spacing: 1.7px;
}

.result-heading {
    color: #f5f7fb;
    font-family: 'Outfit', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    line-height: 1.5;
    margin-top: 13px;
}

.price {
    color: var(--accent);
    font-family: 'Outfit', sans-serif;
    font-size: clamp(2.1rem, 3vw, 3.3rem);
    font-weight: 800;
    line-height: 1.2;
    letter-spacing: -1.2px;
    margin: 30px 0 12px;
    overflow-wrap: anywhere;
}

.price-caption {
    color: #a5b3c4;
    font-size: .88rem;
    line-height: 1.8;
}

.result-status {
    display: inline-block;
    margin-top: 20px;
    padding: 8px 12px;
    border: 1px solid rgba(255, 64, 87, .28);
    border-radius: 30px;
    background: rgba(255, 64, 87, .09);
    color: #ff8b99;
    font-size: .8rem;
}

.result-divider {
    height: 1px;
    margin: 26px 0;
    background: linear-gradient(90deg, var(--accent), #344252, transparent);
}

.result-note {
    color: #a5b3c4;
    font-size: .87rem;
    line-height: 2;
}

.result-note strong {
    color: #f5f7fb;
}

/* EMPTY STATE */

.empty-icon {
    font-size: 2.3rem;
    margin: 30px 0 15px;
}

.empty-title {
    color: #e8edf5;
    font-family: 'Outfit', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 9px;
}

.empty-description {
    color: var(--muted);
    font-size: .88rem;
    line-height: 1.9;
    max-width: 370px;
}

/* DISCLAIMER AND FOOTER */

.disclaimer {
    margin-top: 18px;
    padding: 16px 18px;
    border: 1px solid #273444;
    border-radius: 13px;
    background: #0e151e;
    color: var(--muted);
    font-size: .81rem;
    line-height: 1.8;
}

.disclaimer strong {
    color: #e7edf5;
}

.footer {
    margin-top: 48px;
    padding-top: 22px;
    border-top: 1px solid #202b38;
    color: #7e8da0;
    text-align: center;
    font-size: .81rem;
    line-height: 1.9;
}

/* ANIMATION */

@keyframes riseIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
    }
}

/* RESPONSIVE */

@media (max-width: 900px) {
    .block-container {
        padding: 1.2rem 1.2rem 2rem;
    }

    .hero {
        min-height: 325px;
        padding: 28px;
    }

    .hero-title {
        font-size: 2.7rem;
    }

    div[data-testid="stForm"] {
        padding: 21px;
    }

    .result-card {
        min-height: auto;
        padding: 23px;
    }
}

@media (max-width: 600px) {
    .hero {
        min-height: 290px;
        padding: 21px;
    }

    .hero-title {
        font-size: 2.15rem;
        letter-spacing: -.8px;
    }

    .section-title {
        font-size: 1.5rem;
    }

    .result-card {
        padding: 20px;
    }
}
</style>
""")


# =========================================================
# BRAND AND HERO
# =========================================================

st.html("""
<div class="brand">Drive<span>Price</span></div>

<section class="hero">
    <div class="hero-content">
        <div class="hero-tag">✦ &nbsp; INTELLIGENT CAR VALUATION</div>
        <div class="hero-title">
            Every car has a story.<br>
            Know its <span>worth.</span>
        </div>
        <div class="hero-description">
            Discover your car's estimated resale value.
            Enter a few details, get an estimate, and make
            a more informed buying or selling decision.
        </div>
    </div>
</section>
""")


# =========================================================
# MAIN CONTENT
# =========================================================

form_col, result_col = st.columns([1.05, 0.95], gap="large")


# =========================================================
# INPUT FORM
# =========================================================

with form_col:

    st.html("""
    <div class="section-title">Your vehicle</div>
    <div class="section-description">
        Tell us about the car to calculate its estimated resale value.
    </div>
    """)

    with st.form("car_details_form"):

        st.html("""
        <div class="form-heading">01 &nbsp; Basic details</div>
        <div class="form-caption">
            Vehicle age, original price and usage
        </div>
        """)

        manufacturing_year = st.number_input(
    "Manufacturing year",
    min_value=2003,
    max_value=2018,
    value=2018,
    step=1,
)

        present_price = st.number_input(
            "Original showroom price (₹ lakh)",
            min_value=0.1,
            max_value=200.0,
            value=8.0,
            step=0.1,
        )

        kms_driven = st.number_input(
            "Kilometres driven",
            min_value=0,
            max_value=1_000_000,
            value=30000,
            step=1000,
        )

        owner = st.selectbox(
            "Previous owners",
            options=[0, 1, 2, 3],
            format_func=lambda x: (
                "First owner" if x == 0
                else f"{x} previous owner{'s' if x != 1 else ''}"
            ),
        )

        st.html("""
        <div class="form-heading">02 &nbsp; Specifications</div>
        <div class="form-caption">
            Select the car's fuel type, transmission and seller category.
        </div>
        """)

        fuel_type = st.selectbox(
            "Fuel type",
            ["Petrol", "Diesel", "CNG"],
        )

        transmission = st.selectbox(
            "Transmission",
            ["Manual", "Automatic"],
        )

        seller_type = st.selectbox(
            "Seller type",
            ["Dealer", "Individual"],
        )

        submitted = st.form_submit_button(
            "✦   Calculate estimated price",
            use_container_width=True,
        )


# =========================================================
# PREDICTION AND RESULT
# =========================================================

with result_col:

    st.html("""
    <div class="section-title">Price estimate</div>
    <div class="section-description">
        An estimated resale value based on the details you provide.
    </div>
    """)

    if submitted:

        input_data = pd.DataFrame([{
            "Car_Age": 2018 - manufacturing_year,
            "Present_Price": present_price,
            "Kms_Driven": kms_driven,
            "Fuel_Type": fuel_type,
            "Seller_Type": seller_type,
            "Transmission": transmission,
            "Owner": owner,
        }])

        try:
            prediction = float(model.predict(input_data)[0])

            if pd.isna(prediction):
                raise ValueError("The model returned an invalid value.")

            prediction = max(0.0, prediction)

            result_html = f"""
            <div class="result-card">
                <div class="result-label">DRIVEPRICE VALUATION</div>
                <div class="result-heading">Estimated selling price</div>

                <div class="price">₹ {prediction:.2f} lakh</div>

                <div class="price-caption">
                    Estimated resale value in Indian rupees.
                </div>

                <div class="result-status">
                    ✓ &nbsp; Estimate generated
                </div>

                <div class="result-divider"></div>

                <div class="result-note">
                    <strong>Your vehicle summary</strong><br>
                    Manufacturing year: {manufacturing_year}<br>
                    Original price: ₹ {present_price:.2f} lakh<br>
                    Distance driven: {kms_driven:,} km<br>
                    Previous owners: {html.escape(
                        "First owner" if owner == 0
                        else f"{owner} previous owner{'s' if owner != 1 else ''}"
                    )}<br>
                    Fuel type: {html.escape(fuel_type)}<br>
                    Transmission: {html.escape(transmission)}<br>
                    Seller type: {html.escape(seller_type)}
                </div>
            </div>
            """

            st.html(result_html)

        except Exception as exc:
            st.error(
                "The prediction could not be generated. "
                "Check that the model was trained with the expected feature names."
            )
            st.exception(exc)

    else:

        st.html("""
        <div class="result-card">
            <div class="result-label">DRIVEPRICE VALUATION</div>
            <div class="result-heading">Your estimated resale price</div>

            <div class="price">₹ — —</div>

            <div class="price-caption">
                Your valuation will appear here after you submit your vehicle details.
            </div>

            <div class="result-divider"></div>

            <div class="empty-icon">🚘</div>
            <div class="empty-title">Ready to discover your car's value?</div>
            <div class="empty-description">
                Fill in the form on the left and calculate your estimated
                resale price in just one step.
            </div>
        </div>
        """)

    st.html("""
    <div class="disclaimer">
        <strong>Important note</strong><br>
        This estimate is based on historical used-car data.
        Actual selling prices may vary depending on vehicle condition,
        location, mileage, demand and current market conditions.
    </div>
    """)


# =========================================================
# FOOTER
# =========================================================

st.html("""
<div class="footer">
    <strong style="color:#e4eaf2;">DrivePrice</strong><br>
    Smart car valuation powered by machine learning.<br>
    Built with Python, Scikit-learn and Streamlit.
</div>
""")