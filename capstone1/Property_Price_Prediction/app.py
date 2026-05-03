import os

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# Set page configuration for layout options
st.set_page_config(
    page_title="Property Price Prediction", layout="wide", page_icon="🏠"
)

# Define model files as a module-level constant
MODEL_FILES = {
    "CatBoost (Tuned)": "catboost_model.pkl",
    "XGBoost": "xgboost_model.pkl",
    "LightGBM": "lightgbm_model.pkl",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_data():
    # Attempt to load the data if it exists
    try:
        df = pd.read_csv(os.path.join(SCRIPT_DIR, "House Price.csv"))
        return df
    except FileNotFoundError:
        return None


@st.cache_resource
def load_preprocessors():
    # Load the saved scaler and locality map
    try:
        scaler = joblib.load(os.path.join(SCRIPT_DIR, "scaler.pkl"))
        locality_map = joblib.load(os.path.join(SCRIPT_DIR, "locality_freq_map.pkl"))
        # Also load the list of columns the scaler was trained on
        scaler_cols = joblib.load(os.path.join(SCRIPT_DIR, "scaler_cols.pkl"))
        # Load the final list of model features
        model_features = joblib.load(os.path.join(SCRIPT_DIR, "model_features.pkl"))
        # Load the categories for categorical features
        posted_by_categories = joblib.load(
            os.path.join(SCRIPT_DIR, "posted_by_categories.pkl")
        )
        bhk_or_rk_categories = joblib.load(
            os.path.join(SCRIPT_DIR, "bhk_or_rk_categories.pkl")
        )
        city_categories = joblib.load(os.path.join(SCRIPT_DIR, "city_categories.pkl"))
        return (
            scaler,
            locality_map,
            scaler_cols,
            model_features,
            posted_by_categories,
            bhk_or_rk_categories,
            city_categories,
        )
    except FileNotFoundError:
        return None, None, None, None, None, None, None


@st.cache_resource
def load_models():
    # Attempt to load the trained models
    models = {}
    for model_name, filename in MODEL_FILES.items():
        try:
            models[model_name] = joblib.load(os.path.join(SCRIPT_DIR, filename))
        except (FileNotFoundError, ModuleNotFoundError, ImportError):
            pass  # Skip if file not found or module (like catboost) is missing

    return models


def main():
    st.title("🏠 Property Price Prediction App")
    st.markdown(
        "This application analyzes real estate data and predicts property prices."
    )

    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose a section:", ["Data Exploratory (EDA)", "Price Prediction", "About"]
    )

    df = load_data()
    (
        scaler,
        locality_map,
        scaler_cols,
        model_features,
        posted_by_categories,
        bhk_or_rk_categories,
        city_categories,
    ) = load_preprocessors()

    if app_mode == "Data Exploratory (EDA)":
        st.header("📊 Data Exploration & Layout Options for Graphics")

        if df is None:
            st.error(
                "Dataset 'House Price.csv' not found. Please ensure it is in the same directory."
            )
            return

        # Display dataset
        with st.expander("View Raw Dataset"):
            st.dataframe(df.head(100))

        st.markdown(
            "### Layout Option 1: Streamlit Tabs (Best for categorizing different chart types)"
        )
        tab1, tab2, tab3 = st.tabs(["Distributions", "Correlations", "Map View"])

        with tab1:
            st.subheader("Price Distribution")
            fig_price = px.histogram(
                df,
                x="TARGET(PRICE_IN_LACS)",
                nbins=50,
                title="Distribution of Property Prices (in Lacs)",
                marginal="box",
            )
            st.plotly_chart(fig_price, width="stretch")

        with tab2:
            st.subheader("Features vs Price")
            # Layout Option 2: Streamlit Columns (Best for side-by-side charts)
            col1, col2 = st.columns(2)
            with col1:
                fig_sqft = px.scatter(
                    df,
                    x="SQUARE_FT",
                    y="TARGET(PRICE_IN_LACS)",
                    title="Square Footage vs Price",
                    log_x=True,
                    log_y=True,
                )
                st.plotly_chart(fig_sqft, width="stretch")
            with col2:
                fig_box = px.box(
                    df,
                    x="BHK_NO.",
                    y="TARGET(PRICE_IN_LACS)",
                    title="Price by BHK Number",
                    log_y=True,
                )
                st.plotly_chart(fig_box, width="stretch")

        with tab3:
            st.subheader("Geographical Distribution")
            # Map layout using standard container width
            st.markdown("Property locations based on Latitude and Longitude.")
            # Drop missing coords for map
            map_data = df.dropna(subset=["LATITUDE", "LONGITUDE"])
            # The dataset has LATITUDE and LONGITUDE swapped. Rename them correctly first.
            map_data = map_data.rename(
                columns={"LATITUDE": "longitude", "LONGITUDE": "latitude"}
            )
            # Filter outliers to keep the map focused on India using the corrected columns
            map_data = map_data[
                (map_data["latitude"] >= 6.0)
                & (map_data["latitude"] <= 38.0)
                & (map_data["longitude"] >= 68.0)
                & (map_data["longitude"] <= 98.0)
            ]
            st.map(map_data[["latitude", "longitude"]])

        st.markdown(
            "### Layout Option 3: Streamlit Containers (Best for distinct vertical sections)"
        )
        with st.container():
            st.write("---")
            st.subheader("Categorical Feature Analysis")
            col_a, col_b = st.columns(2)
            with col_a:
                val_counts = df["POSTED_BY"].value_counts().reset_index()
                val_counts.columns = ["Posted By", "Count"]
                fig_pie = px.pie(
                    val_counts,
                    names="Posted By",
                    values="Count",
                    title="Properties Posted By",
                )
                st.plotly_chart(fig_pie, width="stretch")
            with col_b:
                val_counts_status = (
                    df["UNDER_CONSTRUCTION"].value_counts().reset_index()
                )
                val_counts_status.columns = ["Under Construction", "Count"]
                val_counts_status["Under Construction"] = val_counts_status[
                    "Under Construction"
                ].map({0: "No", 1: "Yes"})
                fig_bar = px.bar(
                    val_counts_status,
                    x="Under Construction",
                    y="Count",
                    title="Construction Status",
                )
                st.plotly_chart(fig_bar, width="stretch")

    elif app_mode == "Price Prediction":
        st.header("📈 Price Prediction & Model Performance")
        loaded_models = load_models()

        if not loaded_models:
            st.warning(
                "⚠️ No models found! Please save your trained models in the Jupyter notebook first."
            )
        if any(
            x is None
            for x in [
                scaler,
                locality_map,
                scaler_cols,
                model_features,
                posted_by_categories,
                bhk_or_rk_categories,
                city_categories,
            ]
        ):
            st.warning(
                "⚠️ Preprocessors not found! Please save `scaler.pkl`, `locality_freq_map.pkl`, `scaler_cols.pkl`, `model_features.pkl`, `posted_by_categories.pkl`, `bhk_or_rk_categories.pkl`, and `city_categories.pkl` from your notebook."
            )

        if not loaded_models:
            st.code(
                "import joblib\njoblib.dump(catboost_best, 'catboost_model.pkl')\njoblib.dump(models['XGBoost'], 'xgboost_model.pkl')",
                language="python",
            )

        tab1, tab2 = st.tabs(
            ["🏠 Single Property Prediction", "📊 Performance Comparison"]
        )

        with tab2:
            st.subheader("Model Performance Comparison")
            st.markdown(
                "Compare the evaluation metrics of the trained models based on the test dataset."
            )

            # Placeholder metrics - update these from your notebook results!
            performance_data = pd.DataFrame(
                {
                    "Model": [
                        "CatBoost (Tuned)",
                        "XGBoost",
                        "LightGBM",
                    ],
                    "R2 Score": [
                        0.85,
                        0.82,
                        0.81,
                    ],  # Change to your notebook's r2_score values
                    "MAE (Lacs)": [
                        12.5,
                        14.2,
                        15.0,
                    ],  # Change to your notebook's MAE values
                }
            )

            # Filter out models that might not have loaded
            available_model_names = list(loaded_models.keys()) if loaded_models else []
            if available_model_names:
                performance_data = performance_data[
                    performance_data["Model"].isin(available_model_names)
                ]

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                fig_r2 = px.bar(
                    performance_data,
                    x="Model",
                    y="R2 Score",
                    title="R² Score Comparison (Higher is better)",
                    color="Model",
                )
                fig_r2.update_layout(showlegend=False)
                st.plotly_chart(fig_r2, width="stretch")

            with col_m2:
                fig_mae = px.bar(
                    performance_data,
                    x="Model",
                    y="MAE (Lacs)",
                    title="Mean Absolute Error (Lower is better)",
                    color="Model",
                )
                fig_mae.update_layout(showlegend=False)
                st.plotly_chart(fig_mae, width="stretch")

            st.info(
                "💡 **Note:** The values currently shown are placeholders. Please update the `performance_data` DataFrame in `app.py` with the actual metrics evaluated on your test set in your Jupyter Notebook."
            )

        with tab1:
            st.markdown(
                "Enter property details below to get an estimated price from our ML models."
            )

            # Use columns for form inputs to make it look cleaner
            col1, col2 = st.columns(2)

            with col1:
                posted_by = st.selectbox("Posted By", ["Owner", "Dealer", "Builder"])
                under_construction = st.radio(
                    "Under Construction?",
                    [0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                )
                rera = st.radio(
                    "RERA Approved?",
                    [0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                )
                bhk_no = st.number_input(
                    "Number of BHK / Rooms", min_value=1, max_value=10, value=2
                )

            with col2:
                bhk_or_rk = st.selectbox("Property Type", ["BHK", "RK"])
                square_ft = st.number_input(
                    "Square Footage", min_value=100.0, max_value=50000.0, value=1000.0
                )
                ready_to_move = st.radio(
                    "Ready to Move?",
                    [0, 1],
                    format_func=lambda x: "Yes" if x == 1 else "No",
                )
                resale = st.radio(
                    "Resale?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No"
                )

            st.markdown("### Location")

            # Precompute cities and localities for the dropdowns from the raw dataset
            if df is not None:
                if "city" not in df.columns:
                    df[["locality", "city"]] = df["ADDRESS"].str.rsplit(
                        ",", n=1, expand=True
                    )
                    df["city"] = df["city"].str.strip().str.title()
                    df["locality"] = df["locality"].str.strip().str.title()

                # Filter cities to only those in training data (excluding 'Other')
                if city_categories is not None:
                    known_cities = [c for c in city_categories if c != "Other"]
                    cities = sorted(known_cities)
                else:
                    cities = sorted(df["city"].dropna().unique())
                default_city_idx = (
                    cities.index("Bangalore") if "Bangalore" in cities else 0
                )
            else:
                cities = ["Bangalore"]
                default_city_idx = 0

            col3, col4 = st.columns(2)
            with col3:
                selected_city = st.selectbox("City", cities, index=default_city_idx)
            with col4:
                if df is not None:
                    all_localities = (
                        df[df["city"] == selected_city]["locality"].dropna().unique()
                    )
                    # Filter to known localities (those in frequency map with freq > 0)
                    if locality_map is not None:
                        known_localities = [
                            l
                            for l in all_localities
                            if l in locality_map.index and locality_map[l] > 0
                        ]
                        localities_in_city = (
                            sorted(known_localities) if known_localities else ["Other"]
                        )
                    else:
                        localities_in_city = sorted(all_localities)
                else:
                    localities_in_city = ["Other"]
                selected_locality = st.selectbox("Locality", localities_in_city)

            # Check if city is known and warn if not
            # Removed: Now only known cities are shown in dropdown

            # Compute approximate latitude and longitude behind the scenes based on selection
            if df is not None:
                location_data = df[
                    (df["city"] == selected_city)
                    & (df["locality"] == selected_locality)
                ]
                if not location_data.empty:
                    latitude = float(location_data["LATITUDE"].median())
                    longitude = float(location_data["LONGITUDE"].median())
                else:
                    # Fallback to city median if locality coordinates are missing
                    latitude = float(
                        df[df["city"] == selected_city]["LATITUDE"].median()
                    )
                    longitude = float(
                        df[df["city"] == selected_city]["LONGITUDE"].median()
                    )
            else:
                latitude = 12.9716
                longitude = 77.5946

            # Prediction button
            if st.button("Predict Price", type="primary"):
                if (
                    not loaded_models
                    or df is None
                    or any(
                        x is None
                        for x in [
                            scaler,
                            locality_map,
                            model_features,
                            posted_by_categories,
                            bhk_or_rk_categories,
                            city_categories,
                        ]
                    )
                ):
                    st.error(
                        "Models, data, or preprocessors are not loaded. Please check your files and notebook."
                    )
                else:
                    # 1. Create a DataFrame from the user input
                    # This needs to match the structure BEFORE one-hot encoding
                    input_data = pd.DataFrame(
                        {
                            "POSTED_BY": [posted_by],
                            "UNDER_CONSTRUCTION": [under_construction],
                            "RERA": [rera],
                            "BHK_NO.": [bhk_no],
                            "BHK_OR_RK": [bhk_or_rk],
                            "SQUARE_FT": [square_ft],
                            "READY_TO_MOVE": [ready_to_move],
                            "RESALE": [resale],
                            "LONGITUDE": [longitude],
                            "LATITUDE": [latitude],
                            "city": [selected_city],
                            "locality": [selected_locality],
                        }
                    )

                    try:
                        # 2. Preprocess the input data using the loaded artifacts
                        # Clean column names to match notebook
                        input_data.columns = [
                            c.lower().replace(" ", "_").replace(".", "")
                            for c in input_data.columns
                        ]

                        # --- Apply saved preprocessing steps ---

                        # 2a. Frequency encode locality
                        input_data["locality_freq"] = (
                            input_data["locality"].map(locality_map).fillna(0)
                        )

                        # 2b. Scale numerical features
                        # Ensure the columns are in the same order as when the scaler was fit
                        input_data[scaler_cols] = scaler.transform(
                            input_data[scaler_cols]
                        )

                        # 2c. Set categories for categorical features to match training
                        input_data["posted_by"] = pd.Categorical(
                            input_data["posted_by"], categories=posted_by_categories
                        )
                        input_data["bhk_or_rk"] = pd.Categorical(
                            input_data["bhk_or_rk"], categories=bhk_or_rk_categories
                        )
                        input_data["city"] = pd.Categorical(
                            input_data["city"], categories=city_categories
                        )

                        # 2e. One-Hot Encode categorical features
                        input_data = pd.get_dummies(
                            input_data,
                            columns=["posted_by", "bhk_or_rk", "city"],
                            drop_first=True,
                        )

                        # Re-align the input data to match the model's expected features.
                        # This adds missing one-hot encoded columns, fills them with 0,
                        # and ensures the column order is correct, preventing dtype issues.
                        final_input = input_data.reindex(
                            columns=model_features, fill_value=0
                        )

                        # 3. Make the Predictions for all loaded models
                        st.write("---")
                        st.subheader("Model Predictions")

                        # Disclaimer removed: Now only known cities are selectable

                        # Display metrics in columns
                        cols = st.columns(len(loaded_models))
                        for idx, (model_name, model) in enumerate(
                            loaded_models.items()
                        ):
                            pred = model.predict(final_input)[0]
                            with cols[idx]:
                                st.metric(label=model_name, value=f"₹ {pred:.2f} L")

                    except Exception as e:
                        st.error(f"Error making prediction: {e}")
                        st.info(
                            "Make sure the categorical inputs ('Posted By', etc.) were preprocessed exactly the same way they were during model training in your notebook."
                        )

    elif app_mode == "About":
        st.header("About This App")
        st.write(
            "This application was built using Streamlit as the final front-end for the Property Price Prediction Capstone Project."
        )
        st.write("It demonstrates how to leverage:")
        st.markdown(
            "- **Streamlit caching** for loading datasets and machine learning models."
        )
        st.markdown("- **Plotly Express** for rich, interactive data visualizations.")
        st.markdown(
            "- **Advanced Layouts** using `st.columns`, `st.tabs`, and `st.expander`."
        )


if __name__ == "__main__":
    main()
