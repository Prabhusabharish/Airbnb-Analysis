import re
import ast
import json
import nltk
import folium
import pymongo
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import streamlit as st
import geopandas as gpd
from jinja2 import Template
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# - - - - - - - - - - - - - - -set st addbar page - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

icon = Image.open("C:/Users/prabh/Downloads/Datascience/Project/Airbnb/1.png")
st.set_page_config(page_title= "Airbnb",
                   page_icon= icon,
                   layout= "wide",
                   initial_sidebar_state= "expanded",
                   menu_items={'About': """# This Airbnb page is created by *Prabakaran!"""})
st.markdown("<h1 style='text-align: center; color: white;'>Airbnb Data Analysis</h1>", unsafe_allow_html=True)

# - - - - - - - - - - - - - - -set bg image - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
st.markdown("<h1 style='text-align: center; color: white;'></h1>", unsafe_allow_html=True)

def setting_bg():
    st.markdown(f""" <style>.stApp {{
                        background: url("https://cutewallpaper.org/27/catherine-logo-wallpaper/review-updates-sustainability-and-more-from-catherine-resource-center--airbnb.png");
                        background-size: cover}}
                     </style>""",unsafe_allow_html=True) 
setting_bg()

SELECT = option_menu(
    menu_title=None,
    options=["Home", "Explore Data", "Contact"],
    icons=["house", "bar-chart", "at"],
    default_index=0,
    orientation="horizontal",
    styles={"container": {"padding": "0!important", "background-color": "white", "size": "cover", "width": "100"},
            "icon": {"color": "black", "font-size": "20px"},

            "nav-link": {"font-size": "20px", "text-align": "center", "margin": "-2px", "--hover-color": "#6F36AD"},
            "nav-link-selected": {"background-color": "#6F36AD"}})

# ------------------------------ ----------------------Home--------------------------------------------------#
# Home page creation

if SELECT == "Home":
    st.title("Home Page")

    col1,col2 = st.columns(2)

    with col1 :

        with col1:
            st.markdown("""
                Technologies Used:
                - Python
                - Pandas
                - Streamlit
                - Pandas
                - Pymongo
             """)

    with col2 :
        image_path = "home.png"
        st.image(image_path, use_column_width=True)

# ------------------------------ ----------------------Contact--------------------------------------------------#
# Contact page creation
elif SELECT == "Contact":
    Name = (f'{"Name :"}  {"Prabakaran T"}')
    mail = (f'{"Mail :"}  {"prabhusabharish78@gmail.com"}')
    description = "An Aspiring DATA-SCIENTIST..!"
    social_media = {
        "GITHUB": "https://github.com/Prabhusabharish",
        "LINKEDIN": "https://www.linkedin.com/feed/"}

    col1, col2 = st.columns(2)
    col1.image(Image.open("C:/Users/prabh/Downloads/Datascience/Project/Airbnb/ds.png"), width=300)

    with col2:
        st.header('Airbnb Analysis')
        st.subheader(
            "This project aims to analyze Airbnb data using MongoDB Atlas, perform data cleaning and preparation, develop interactive geospatial visualizations, and create dynamic plots to gain insights into pricing variations, availability patterns, and location-based trends.")
        st.write("---")
        st.subheader(Name)
        st.subheader(mail)

### ----------------------------------------------------- connect with mongoDB dataframe --------------------------------------------------------------------------------------------- 
elif SELECT == "Explore Data":
    st.title("~PrabhuSabharish~")

# my local file connection creation 
df=pd.read_csv("C:/Users/prabh/Downloads/Datascience/Project/Airbnb/output.csv")
df
st.dataframe(df)
    
### ----------------------------------------------------- Data Extract and Cleaning --------------------------------------------------------------------------------------------- 

# Missing values handling
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)
exclude_columns = ['reviews', 'amenities']  # Add other columns as needed
df_stringified = df.drop(columns=exclude_columns).astype(str)
duplicate_rows = df_stringified.duplicated().sum()
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Exploratory Data Analysis (EDA)
# Basic Statistics
basic_stats = df.describe()

# Univariate Analysis
plt.figure(figsize=(12, 8))
sns.histplot(df['price'], bins=30, kde=True)
plt.title('Distribution of Prices')
plt.show()

# Bivariate Analysis
plt.figure(figsize=(12, 8))
sns.scatterplot(x='bedrooms', y='price', data=df)
plt.title('Relationship between Bedrooms and Price')
plt.show()

# Price Analysis and Visualization
# Variations Across Locations
plt.figure(figsize=(14, 10))
sns.boxplot(x='neighborhood_overview', y='price', data=df)
plt.title('Price Variations Across Neighborhoods')
plt.xticks(rotation=45)
plt.show()

# Step 7: Availability Analysis
plt.figure(figsize=(12, 8))
sns.lineplot(x='last_scraped', y='availability', data=df)
plt.title('Availability Over Time')
plt.show()

# Availability columns claen and creation 
df["availability"] = df["availability"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df["availability_30"] = df["availability"].apply(lambda x: x.get("availability_30") if isinstance(x, dict) else None)
df["availability_60"] = df["availability"].apply(lambda x: x.get("availability_60") if isinstance(x, dict) else None)
df["availability_90"] = df["availability"].apply(lambda x: x.get("availability_90") if isinstance(x, dict) else None)
df["availability_365"] = df["availability"].apply(lambda x: x.get("availability_365") if isinstance(x, dict) else None)


# Distribution of review scores
plt.figure(figsize=(10, 6))
sns.histplot(df['review_scores'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Review Scores')
plt.xlabel('Review Scores')
plt.ylabel('Frequency')
plt.show()

# Extract review scores into separate columns
review_score_categories = [
    'review_scores_accuracy',
    'review_scores_cleanliness',
    'review_scores_checkin',
    'review_scores_communication',
    'review_scores_location',
    'review_scores_value',
    'review_scores_rating'
]

for category in review_score_categories:
    df[category] = df['reviews'].apply(lambda x: x.get(category) if isinstance(x, dict) else None)

# Select relevant columns for correlation analysis
selected_columns = review_score_categories + ['accommodates', 'bedrooms', 'bathrooms', 'price']

# Correlation matrix
correlation_matrix = df[selected_columns].corr()

# Heatmap for correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix')
plt.show()

# Assuming 'last_scraped' column is a datetime type
df['last_scraped'] = pd.to_datetime(df['last_scraped'])

# Extract 'month' and 'year'
df['month'] = df['last_scraped'].dt.month
df['year'] = df['last_scraped'].dt.year

# Group by 'year' and 'month' and count the number of listings
time_series_data = df.groupby(['year', 'month']).size().reset_index(name='count')

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(time_series_data['year'].astype(str) + '-' + time_series_data['month'].astype(str), time_series_data['count'], marker='o')
plt.title('Number of Listings Over Time')
plt.xlabel('Year-Month')
plt.ylabel('Number of Listings')
plt.xticks(rotation=45)
plt.show()

# Drop rows where 'price' is missing
df = df.dropna(subset=['price'])

# Select relevant features for the model
features = ['accommodates', 'bedrooms', 'bathrooms', 'guests_included', 'minimum_nights', 'maximum_nights', 'number_of_reviews']

# Select target variable
target = 'price'

# Create X and y
X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
rf_model = RandomForestRegressor()

# Fit the model
rf_model.fit(X_train, y_train)

# Get feature importances
feature_importances = rf_model.feature_importances_

# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot Feature Importances
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, hue='Feature', palette='viridis', legend=False)
plt.title('Feature Importances')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()


# Plot the Time Series for 'price'
plt.figure(figsize=(12, 6))
df['price'].plot()
plt.title('Time Series Analysis for Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Example Streamlit components for visualization
st.title("Price Distribution")
# Plotting a histogram
plt.figure(figsize=(12, 6))
sns.histplot(df['price'], kde=True, bins=30, color='blue')
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Plotting a box plot
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['price'], color='green')
plt.title('Box Plot of Prices')
plt.xlabel('Price')
plt.show()

# Plotting a kernel density plot
plt.figure(figsize=(12, 6))
sns.kdeplot(df['price'], color='orange')
plt.title('Kernel Density Plot of Prices')
plt.xlabel('Price')
plt.ylabel('Density')
plt.show()

# Select numerical features and target variable
numerical_features = ['accommodates', 'bedrooms', 'bathrooms', 'guests_included', 'minimum_nights', 'maximum_nights', 'number_of_reviews', 'price']

# Create a correlation matrix
correlation_matrix = df[numerical_features].corr()

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Set the style of seaborn
sns.set(style="whitegrid")

# Create a box plot for median prices across different months
plt.figure(figsize=(12, 8))
sns.boxplot(x='price', y='month', data=df, hue='month', palette='viridis', width=0.6, showfliers=False, legend=False)
plt.title('Seasonal Trends: Median Prices Across Different Months')
plt.xlabel('Price')
plt.ylabel('Month')
plt.show()

# Download the VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Assuming 'reviews' is the column containing review data
reviews = df['reviews']

# Create a SentimentIntensityAnalyzer object
sia = SentimentIntensityAnalyzer()

# Apply sentiment analysis to each review and create a new column 'sentiment_score'
df['sentiment_score'] = reviews.apply(lambda x: sia.polarity_scores(x)['compound'])

# Assuming 'price' is the target variable, you can analyze the relationship
# between sentiment scores and prices
sentiment_price_analysis = df[['sentiment_score', 'price']]

# Visualize the relationship (scatter plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sentiment_score', y='price', data=sentiment_price_analysis)
plt.title('Sentiment Analysis vs. Price')
plt.xlabel('Sentiment Score')
plt.ylabel('Price')
plt.show()

# Cluster analysis

cluster_features = ['accommodates', 'bedrooms', 'bathrooms', 'guests_included', 'minimum_nights', 'maximum_nights', 'number_of_reviews']

# Create a subset of the data for clustering
cluster_data = df[cluster_features]

# Standardize the data
scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(cluster_data_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Method graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

# Based on the Elbow Method, choose the optimal number of clusters (K)
optimal_k = 3

# Apply K-Means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(cluster_data_scaled)

# Visualize the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='accommodates', y='price', hue='cluster', data=df, palette='viridis')
plt.title('K-Means Clustering of Listings')
plt.xlabel('Accommodates')
plt.ylabel('Price')
plt.show()



# Model Evaluation
y_pred = rf_model.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae:.2f}')

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (MSE): {mse:.2f}')

# Calculate R-squared
r_squared = r2_score(y_test, y_pred)
print(f'R-squared: {r_squared:.2f}')


# Outlier Detection
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['price'])
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.show()

# Calculate the Interquartile Range (IQR)
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]

# Print the identified outliers
print("Identified Outliers:")
print(outliers[['price']])

# Visualize the outliers
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df.index, y=df['price'], color='blue', label='Inliers')
sns.scatterplot(x=outliers.index, y=outliers['price'], color='red', label='Outliers')
plt.title('Outlier Detection')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.show()


# Text Data analysis
nltk.download('vader_lexicon')


columns_to_analyze = ['description', 'summary', 'neighborhood_overview']

# Initialize the Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Create new columns for sentiment scores
for column in columns_to_analyze:
    df[f'{column}_sentiment'] = df[column].apply(lambda x: sia.polarity_scores(str(x))['compound'])


# Calculate mean sentiment scores
mean_description_sentiment = df['description_sentiment'].mean()
mean_summary_sentiment = df['summary_sentiment'].mean()
mean_neighborhood_sentiment = df['neighborhood_overview_sentiment'].mean()


# Box plot for sentiment scores
plt.figure(figsize=(12, 8))
sns.boxplot(data=df[['description_sentiment', 'summary_sentiment', 'neighborhood_overview_sentiment']])
plt.title('Sentiment Scores Comparison')
plt.xlabel('Text Columns')
plt.ylabel('Sentiment Score')
plt.show()

# Extract latitude and longtitude columns
df["latitude"] = df["address"].apply(lambda x: ast.literal_eval(x)['location']['coordinates'][1])
df["longitude"] = df["address"].apply(lambda x: ast.literal_eval(x)['location']['coordinates'][0])


# Geospatial Visualization
st.title("Geospatial Visualization")
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
m = folium.Map(location=[df.latitude.mean(), df.longitude.mean()], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)
for idx, row in gdf.iterrows():
    folium.Marker([row['latitude'], row['longitude']], popup=f"Price: ${row['price']:.2f}").add_to(marker_cluster)
folium_static(m)




