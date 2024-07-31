# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import ast
from pymongo import MongoClient
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from datetime import datetime
import certifi
from bson.decimal128 import Decimal128
import geopy
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
import plotly.express as px

################### FUNCTIONS ###################

def get_data():
    client = MongoClient('mongodb+srv://sai27lalitha:sailalitha@sample-airbnb-data.zv08wpn.mongodb.net/', tlsCAFile=certifi.where())
    db = client['sample_airbnb']
    listings = db['listingsAndReviews']
    data = pd.DataFrame(list(listings.find()))
    return data

def convert_decimal128_to_float(x):
    if isinstance(x, Decimal128):
        return float(x.to_decimal())
    return x

def clean_data(df):
    unwanted_cols = ['listing_url', 'name', 'summary', 'space', 'description', 'notes', 'first_review',
                     'last_review', 'transit', 'access', 'interaction', 'house_rules', 'images']
    df = df.drop(unwanted_cols, axis=1)

    columns_to_convert = ['weekly_price', 'monthly_price', 'reviews_per_month', 'security_deposit', 'cleaning_fee']
    for column in columns_to_convert:
        df[column] = df[column].apply(convert_decimal128_to_float)

    columns_to_convert = ['minimum_nights', 'maximum_nights', 'bathrooms', 'price', 'extra_people', 'guests_included']
    df[columns_to_convert] = df[columns_to_convert].astype(str).astype(float)

    df['weekly_price'] = df['weekly_price'].fillna(df['weekly_price'].mean())
    df['monthly_price'] = df['monthly_price'].fillna(df['monthly_price'].mean())
    df['cleaning_fee'] = df['cleaning_fee'].fillna(df['cleaning_fee'].mean())
    df['security_deposit'] = df['security_deposit'].fillna(df['security_deposit'].mean())
    df['reviews_per_month'] = df['reviews_per_month'].fillna(df['reviews_per_month'].median())

    df = df.dropna(subset=['beds', 'bedrooms', 'bathrooms'])

    availability_expanded = df['availability'].apply(pd.Series)
    availability_expanded.columns = ['availability_30', 'availability_60', 'availability_90', 'availability_120']
    df = pd.concat([df.drop(columns=['availability']), availability_expanded], axis=1)

    review_scores_expanded = df['review_scores'].apply(pd.Series)
    review_scores_expanded.columns = ['review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'review_scores_rating']
    df = pd.concat([df.drop(columns=['review_scores']), review_scores_expanded], axis=1)

    address_expanded = df['address'].apply(pd.Series)
    address_expanded.columns = ['street', 'suburb', 'government_area', 'market', 'country', 'country_code', 'location']
    df = pd.concat([df.drop(columns=['address']), address_expanded], axis=1)

    host_expanded = df['host'].apply(pd.Series)
    host_expanded.columns = ['host_id', 'host_url', 'host_name', 'host_location', 'host_about', 'host_response_time', 'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood', 'host_response_rate', 'host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'host_listings_count', 'host_total_listings_count', 'host_verifications']
    df = pd.concat([df.drop(columns=['host']), host_expanded], axis=1)

    df = df.drop(['host_id', 'host_url', 'host_name', 'host_about', 'host_thumbnail_url', 'host_picture_url', 'host_neighbourhood', 'host_response_rate', 'host_is_superhost', 'host_has_profile_pic', 'host_listings_count', 'host_verifications', '_id', 'neighborhood_overview'], axis=1)

    review_scores_columns = ['review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value', 'review_scores_rating']
    for column in review_scores_columns:
        df[column] = df[column].fillna(df[column].median())

    df['host_response_time'] = df['host_response_time'].fillna(df['host_response_time'].mode()[0])

    df['coordinates'] = df['location'].apply(lambda x: x['coordinates'] if isinstance(x, dict) else None)
    df[['longitude', 'latitude']] = pd.DataFrame(df['coordinates'].tolist(), index=df.index)
    df.drop(columns=['coordinates', 'location'], inplace=True)

    df['month_scraped'] = pd.to_datetime(df['calendar_last_scraped']).dt.month

    return df

def plot_histogram_boxplot(df, columns):
    for col in columns:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(df[col])
        plt.title(f'{col} Distribution')

        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f'{col} Box Plot')

        plt.tight_layout()
        plt.savefig(f'{col}_distribution.png')
        plt.close()

def plot_correlation_matrix(df, columns):
    correlation_matrix = df[columns].corr()
    plt.figure(figsize=(12, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Price Columns')
    plt.savefig('correlation_matrix.png')
    plt.close()

def plot_price_over_time(df):
    monthly_avg_price = df.groupby(df['last_scraped'].dt.to_period('M'))['price'].mean()
    monthly_avg_price.plot()
    plt.xlabel('Month')
    plt.ylabel('Average Price')
    plt.title('Average Price Over Time')
    plt.savefig('average_price_over_time.png')
    plt.close()

def plot_geospatial(df):
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
    plt.figure(figsize=(12, 6))
    gdf.plot(column='price', cmap='OrRd', legend=True, markersize=5)
    plt.title('Price by Location')
    plt.savefig('price_by_location.png')
    plt.close()

def plot_price_by_property_room_type(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='property_type', y='price', data=df)
    plt.title('Price by Property Type')
    plt.xticks(rotation=45)
    plt.savefig('price_by_property_type.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.barplot(x='room_type', y='price', data=df)
    plt.title('Price by Room Type')
    plt.xticks(rotation=45)
    plt.savefig('price_by_room_type.png')
    plt.close()

def plot_price_by_country(df):
    plt.figure(figsize=(12, 6))
    sns.barplot(x='country', y='price', data=df)
    plt.xticks(rotation=90)
    plt.savefig('price_by_country.png')
    plt.close()

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

def plot_availability(df):
    df['season'] = df['month_scraped'].apply(get_season)
    df['total_availability'] = df[['availability_30', 'availability_60', 'availability_90', 'availability_120']].sum(axis=1)

    seasonal_availability = df.groupby('season')['total_availability'].sum()
    seasonal_availability.plot(kind='bar', color=['blue', 'green', 'red', 'orange'])
    plt.xlabel('Season')
    plt.ylabel('Total Number of Available Days')
    plt.title('Airbnb Availability by Season')
    plt.savefig('availability_by_season.png')
    plt.close()

    periodic_availability = df.groupby('season')[['availability_30', 'availability_60', 'availability_90', 'availability_120']].mean()
    periodic_availability.plot(kind='bar', figsize=(12, 6))
    plt.xlabel('Season')
    plt.ylabel('Average Number of Available Days')
    plt.title('Average Airbnb Availability by Season and Period')
    plt.legend(title='Period')
    plt.savefig('availability_by_season_period.png')
    plt.close()

def plot_location_insights(df):
    coords = df[['latitude', 'longitude']].values
    db = DBSCAN(eps=0.01, min_samples=5, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    df['cluster'] = db.labels_

    sns.boxplot(x='cluster', y='price', data=df)
    plt.xlabel('Cluster')
    plt.ylabel('Price')
    plt.title('Price Distribution by Cluster')
    plt.savefig('price_by_cluster.png')
    plt.close()

    sns.boxplot(x='cluster', y='review_scores_rating', data=df)
    plt.xlabel('Cluster')
    plt.ylabel('Review Scores Rating')
    plt.title('Review Scores Rating by Cluster')
    plt.savefig('review_scores_by_cluster.png')
    plt.close()

def visualize_heatmap(df):
    map_center = df[['latitude', 'longitude']].mean().values.tolist()
    m = folium.Map(location=map_center, zoom_start=12)
    heat_data = df[['latitude', 'longitude', 'price']].values.tolist()
    HeatMap(heat_data).add_to(m)
    return m

################### STREAMLIT APP ###################

# Set up the page configuration
st.set_page_config(page_title="Airbnb Data Analysis", layout="wide")

# Title of the app
st.title('Airbnb Data Analysis')

# Load data
@st.cache_data
def load_data():
    return get_data()

df = load_data()
cleaned_data = clean_data(df)

# Define a function to display images with titles
def display_images(image_titles):
    for title, image_path in image_titles:
        st.write(f"### {title}")
        st.image(image_path)

# Sidebar for navigation
with st.sidebar:
    st.header("Navigation")
    options = ["Home", "Price Analysis", "Geospatial Analysis", "Availability Analysis"]
    choice = st.selectbox("Select Analysis Type", options)

# Display Home Page
if choice == "Home":
    st.write("### Cleaned Data Sample")
    st.write(cleaned_data.head())
    
    st.write("### Key Takeaways")
    st.write("1. Insights from price analysis.")
    st.write("2. Observations from geospatial analysis.")
    st.write("3. Findings on availability trends.")
    st.write("4. Conclusions from location insights.")

# Price Analysis
elif choice == "Price Analysis":
    st.write("## Price Analysis")
    display_images([
        ("Price Distribution", 'price_distribution.png'),
        ("Weekly Price Distribution", 'weekly_price_distribution.png'),
        ("Monthly Price Distribution", 'monthly_price_distribution.png'),
        ("Security Deposit Distribution", 'security_deposit_distribution.png'),
        ("Cleaning Fee Distribution", 'cleaning_fee_distribution.png')
    ])
    plot_columns = ['price', 'weekly_price', 'monthly_price', 'security_deposit', 'cleaning_fee']
    plot_correlation_matrix(cleaned_data, plot_columns)
    st.image('correlation_matrix.png')
    plot_price_over_time(cleaned_data)
    st.image('average_price_over_time.png')

# Geospatial Analysis
elif choice == "Geospatial Analysis":
    st.write("## Geospatial Analysis")
    display_images([
        ("Price by Location", 'price_by_location.png')
    ])
    st.write("### Geospatial Heatmap")
    heatmap = visualize_heatmap(cleaned_data)
    st_folium(heatmap, width=700)

# Availability Analysis
elif choice == "Availability Analysis":
    st.write("## Availability Analysis")
    display_images([
        ("Availability by Season", 'availability_by_season.png'),
        ("Availability by Season Period", 'availability_by_season_period.png')
    ])
