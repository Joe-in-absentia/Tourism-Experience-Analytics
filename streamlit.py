import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sqlalchemy import create_engine


st.set_page_config(page_title="Tourism Recommendation System",layout="wide")
st.title("🌍 Tourism Experience Analytics")               # Main title.
st.markdown("---")

st.sidebar.title("🧭 Dashboard Controls")                 # Sidebar title.
st.sidebar.markdown("---")    

# Database connection.

engine = create_engine("postgresql://postgres:123456@localhost:5432/tourism_data")
df = pd.read_sql("SELECT * FROM cleaned_tourism_data",engine)
                            

# Load the trained model.
regression_model = joblib.load("regression_model.pkl")
encoder = joblib.load("rating_encoder.pkl")
classification_model = joblib.load("classification_model.pkl")
visitmode_encoder = joblib.load("visitmode_encoder.pkl")
feature_encoder = joblib.load("feature_encoder.pkl")
feature_columns = joblib.load("feature_columns.pkl")
kmeans = joblib.load("clustering_model.pkl")
scaler = joblib.load("scaler.pkl")
x_encoder = joblib.load("cluster_encoder.pkl")

page = st.sidebar.radio("Select Page",["Home",":red[Rating Prediction]",":red[Visit Mode Prediction]",
                                            ":blue[Recommendations]"])


def home_page():

    st.subheader("🚶 Visit Mode & Rating Distribution")
    col1, col2 = st.columns(2)
    
    vc = df["VisitMode"].value_counts().reset_index()
    vc.columns = ["VisitMode", "Count"]
    
    rd = df["Rating"].value_counts().reset_index()
    rd.columns = ["Rating","Count"]
 
    with col1:
       fig = px.bar(vc, x="VisitMode", y="Count",
             title="Visit Mode Distribution",color_discrete_sequence=["#174CDD"])
       st.plotly_chart(fig, width='stretch')
    with col2:      
        fig = px.bar(rd, x='Rating',y='Count', title="⭐ Rating Distribution",
                     color_discrete_sequence=["#E29F1A"])
        st.plotly_chart(fig, width='stretch')
     
    st.markdown("---")

    st.subheader("🏝️ Top Attractions")

    ta = df["Attraction"].value_counts().reset_index()
    ta.columns = ["Attraction", "Count"]
       
    fig = px.bar(ta.head(25), x='Attraction',y='Count',color_discrete_sequence=["#1E693D"])
    st.plotly_chart(fig, width='stretch')

    st.markdown("---")

def predict_rating():

    st.header("⭐ Rating Prediction")

    visit_year = st.number_input("Visit Year",value=2025)
    visit_month = st.selectbox("Visit Month",range(1,13))
    visit_mode = st.selectbox("Visit Mode",df["VisitMode"].unique())
    attraction = st.selectbox("Attraction",df["Attraction"].unique())
    continent = st.selectbox("Continent",df["Continent"].unique())
    region = st.selectbox("Region",df["Region"].unique())
    country = st.selectbox("Country",df["Country"].unique())
    season = st.selectbox("Season",df["Season"].unique())
    attraction_type = st.selectbox("Attraction Type",df["AttractionType"].unique())
    user_visit_count = st.number_input("User Visit Count",min_value=0,value=5)
    unique_attractions = st.number_input("Unique Attractions Visited",min_value=0, value=3)
    attraction_avg_rating = st.slider("Attraction Average Rating",1.0,5.0,4.0)
    user_avg_rating = st.slider("User Average Rating",1.0,5.0,4.0)
    visit_mode_avg_rating = st.slider("Visit Mode Average Rating",1.0,5.0,4.0)

    if st.button(":blue[🚀 Predict Rating]"):
      input_df = pd.DataFrame({
      "VisitYear":[visit_year],
      "VisitMonth":[visit_month],
      "VisitMode":[visit_mode],
      "Attraction":[attraction],
      "Continent":[continent],
      "Region":[region],
      "Country":[country],
      "Season":[season],
      "UserVisitCount":[user_visit_count],
      "AttractionAvgRating":[attraction_avg_rating],
      "UserAvgRating":[user_avg_rating],
      "VisitModeAvgRating":[visit_mode_avg_rating],
      "UniqueAttractionsVisited":[unique_attractions],
      "AttractionType":[attraction_type]})

      input_encoded = encoder.transform(input_df)  
      prediction = regression_model.predict(input_encoded)
      st.success(
          f"Predicted Rating: {prediction[0]:.2f}")
   

def predict_visitmode():

    st.header("🚶 Visit Mode Prediction")

    attraction_address = st.selectbox("Attraction Address",df["AttractionAddress"].unique())
    attraction_type_id = st.selectbox("Attraction Type ID",df["AttractionTypeId"].unique())
    attraction_type = st.selectbox("Attraction Type",df["AttractionType"].unique())
    season = st.selectbox("Season",df["Season"].unique())
    country = st.selectbox("Country",df["Country"].unique())
    city = st.selectbox("City Name",df["CityName"].unique())
    rating = st.slider("Rating",1.0,5.0,4.0)
    user_id = st.number_input("User ID",value=1)
    user_avg_rating = st.slider("User Average Rating",1.0,5.0,4.0)

    if st.button(":blue[🚀 Predict Visit Mode]"):

        input_df = pd.DataFrame({
            "TransactionId": [df["TransactionId"].mode()[0]],
            "AttractionAddress": [attraction_address],
            "AttractionTypeId": [attraction_type_id],
            "AttractionType": [attraction_type],
            "Season": [season],
            "Country": [country],
            "CityName": [city],
            "Rating": [rating],
            "UserId": [user_id],
            "UserAvgRating": [user_avg_rating]})

        input_df = input_df[feature_columns]
        input_encoded = feature_encoder.transform(input_df)
        prediction = classification_model.predict(input_encoded)
        visit_mode = visitmode_encoder.inverse_transform(prediction)

        st.success(f"Predicted Visit Mode: {visit_mode[0]}")

def recommendation():

    st.header("📌 Cluster-Based Recommendation System")

    user_visit_count = st.number_input("User Visit Count",min_value=0,value=5)
    rating = st.slider("Rating",1.0,5.0,4.0)
    unique_attractions = st.number_input("Unique Attractions Visited",min_value=0,value=3)
    user_avg_rating = st.slider("User Average Rating",1.0,5.0,4.0)
    visit_mode_avg_rating = st.slider("Visit Mode Average Rating",1.0,5.0,4.0)

    if st.button("🚀 Get Recommendations",type="primary"):

        user_input = pd.DataFrame({
            "UserVisitCount": [user_visit_count],
            "UniqueAttractionsVisited": [unique_attractions],
            "UserAvgRating": [user_avg_rating],
            "Rating": [rating],
            "VisitModeAvgRating": [visit_mode_avg_rating]})

        user_encoded = x_encoder.transform(user_input)
        user_scaled = scaler.transform(user_encoded)
        cluster = kmeans.predict(user_scaled)[0]

        st.success(f" Traveler Cluster: {cluster + 1}")

        cluster_data = df[[
                "UserVisitCount",
                "UniqueAttractionsVisited",
                "UserAvgRating",
                "Rating",
                "VisitModeAvgRating"]]

        # Encode database data
        cluster_encoded = x_encoder.transform(cluster_data)
        # Scale database data
        cluster_scaled = scaler.transform(cluster_encoded)
        # Predict clusters for all users
        df["Cluster"] = kmeans.predict(cluster_scaled)
        cluster_users = df[df["Cluster"] == cluster]

        if len(cluster_users) == 0:
            st.warning("No similar travelers found.")
            return

        recommendations = (cluster_users.groupby("Attraction").agg(
            Visits=("TransactionId","count"),AvgRating=("Rating","mean")).reset_index()
                 .sort_values(["AvgRating","Visits"],ascending=False).head(15))

        st.subheader("🔥 Recommended Attractions")

        st.dataframe(recommendations,use_container_width=True)

        fig = px.bar(recommendations,x="Attraction",y="Visits",
                 title="Popular Attractions in Your Cluster")

        st.plotly_chart(fig,use_container_width=True)


if page == "Home":
    home_page()
elif page == ":red[Rating Prediction]":
    predict_rating()
elif page == ":red[Visit Mode Prediction]":
    predict_visitmode()
elif page == ":blue[Recommendations]":
    recommendation()