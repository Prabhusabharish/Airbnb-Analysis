import ast
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
# - - - - - - - - - - - - - - -set st addbar page - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
icon = Image.open("C:/Users/prabh/Downloads/Datascience/Project/Airbnb/1.png")
st.set_page_config(page_title= "Airbnb",
                   page_icon= icon,
                   layout= "wide",
                   initial_sidebar_state= "expanded",
                   menu_items={'About': """# This Airbnb page is created by *Prabakaran!"""})
st.markdown("<h1 style='text-align: center; color: white;'>AIRBNB DATA ANALYSIS</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: white;'></h1>", unsafe_allow_html=True)
# - - - - - - - - - - - - - - -set bg image - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def setting_bg():
    st.markdown(f""" <style>.stApp {{
                        background: url("https://cutewallpaper.org/27/catherine-logo-wallpaper/review-updates-sustainability-and-more-from-catherine-resource-center--airbnb.png");
                        background-size: cover}}
                     </style>""",unsafe_allow_html=True) 
setting_bg()
SELECT = option_menu(
    menu_title=None,
    options=["Home", "Data Exploration", "About"],
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
    with col1:
        st.write(
            '''
            Discover the enchantment of Airbnb, where every stay is a journey and every host crafts a unique experience. 
            From cozy hideaways to chic apartments, Airbnb connects wanderers with distinctive accommodations. 
            Delve into our interactive analysis to unveil trends and unravel the secrets behind the world's beloved hospitality platform. 
            Your adventure into the heart of hospitality starts right here!
            '''
        )
    with col2:
        image_path = "home.png"
        st.image(image_path, use_column_width=True)
# ------------------------------ ----------------------Home--------------------------------------------------#
# Contact page creation
elif SELECT == "About":
    Name = (f'{"Name :"}  {"Prabakaran T"}')
    mail = (f'{"Mail :"}  {"prabhusabharish78@gmail.com"}')
    github = (f'{"Github :"}  {"https://github.com/Prabhusabharish"}')
    linkedin = (f'{"LinkedIn :"}  {"https://www.linkedin.com/feed/"}')
    description = "An Aspiring DATA-SCIENTIST..!"
    st.header('Airbnb Analysis')
    st.subheader(
        "This project aims to analyze Airbnb data using MongoDB Atlas, perform data cleaning and preparation, develop interactive geospatial visualizations, and create dynamic plots to gain insights into pricing variations, availability patterns, and location-based trends.")
    st.write("---")
    st.subheader(Name)
    st.subheader(mail)

    st.markdown(github, unsafe_allow_html=True)
    st.markdown(linkedin, unsafe_allow_html=True)
### -----------------------------------------------------  --------------------------------------------------------------------------------------------- 
elif SELECT == "Data Exploration":  
    def datadir():
        df = pd.read_csv(f"C:/Users/prabh/Downloads/Datascience/Project/Airbnb/calendata.csv")
        return df  
    df = datadir()  
    country = df["country"].unique()[0]  
    room_type = df["room_type"].unique()[0]
    property_type = df["property_type"].unique()[0]
    host_response_time = df["host_response_time"].unique()[0]
    selected_countries = st.sidebar.multiselect("Select the Countries", sorted(df["country"].unique()), default=[df["country"].unique()[0]])
    selected_room_types = st.sidebar.multiselect("Select the Room Types", sorted(df["room_type"].unique()), default=[df["room_type"].unique()[0]])
    selected_property_types = st.sidebar.multiselect("Select the Property Types", sorted(df["property_type"].unique()), default=[df["property_type"].unique()[0]])
    host_response_time_tab1 = st.sidebar.selectbox("Select the Host Response Time", df["host_response_time"].unique(), key="host_response_time_tab1")
    host_response_time_tab2 = st.sidebar.selectbox("Select the Host Response Time", df["host_response_time"].unique(), key="host_response_time_tab2")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["PRICE ANALYSIS", "AVAILABILITY ANALYSIS", "LOCATION ANALYSIS", "GEOSPATIAL VISUALIZATION", "TOP CHARTS"])
    with tab1:
        st.title("**PRICE DIFFERENCE**")
        col1, col2 = st.columns(2)
        df_filtered_host_response_time = df[df["host_response_time"] == host_response_time]
        with col1:
            df_filtered = df[df["country"] == country]
            df_filtered.reset_index(drop=True, inplace=True)
            df_filtered_room_type = df_filtered[df_filtered["room_type"] == room_type]
            df_filtered_room_type.reset_index(drop=True, inplace=True)
            df_bar = pd.DataFrame(df_filtered_room_type.groupby("property_type")[["price", "review_scores_rating", "number_of_reviews"]].sum())
            df_bar.reset_index(inplace=True)
            custom_colors = {'Entire home/apt': '#1f78b4', 'Private room': '#33a02c', 'Shared room': '#e31a1c'}
            fig_bar = px.bar(df_bar, x='property_type', y="price", title="PRICE FOR PROPERTY_TYPES",
                            hover_data=["number_of_reviews", "review_scores_rating"],
                            color='property_type', color_discrete_map=custom_colors, width=650, height=500)
            st.plotly_chart(fig_bar)
        with col2:
            filtered_data = df[df["property_type"].isin(selected_property_types)]
            df_filtered_property_type = df[df["property_type"] == property_type]
            df_filtered_property_type.reset_index(drop=True, inplace=True)
            df_filtered_host_response_time = df_filtered_property_type[df_filtered_property_type["host_response_time"] == host_response_time]
            df_filtered_host_response_time.reset_index(drop=True, inplace=True)
            df_pie = pd.DataFrame(df_filtered_host_response_time.groupby("host_response_time")[["price", "bedrooms"]].sum())
            df_pie.reset_index(inplace=True)
            fig_pie = px.pie(df_pie, values="price", names="host_response_time",
                            hover_data=["bedrooms"],
                            color_discrete_sequence=px.colors.sequential.BuPu_r,
                            title="PRICE DIFFERENCE BASED ON HOST RESPONSE TIME",
                            width=600, height=500)
            st.plotly_chart(fig_pie)
        with col1:
            df_do_bar = pd.DataFrame(df_filtered_host_response_time.groupby("bed_type")[["minimum_nights", "maximum_nights", "price"]].sum())
            df_do_bar.reset_index(inplace=True)
            fig_do_bar = px.bar(df_do_bar, x='bed_type', y=['minimum_nights', 'maximum_nights'],
                                title='MINIMUM NIGHTS AND MAXIMUM NIGHTS', hover_data="price",
                                barmode='group', color_discrete_sequence=px.colors.sequential.Rainbow, width=600, height=500)

            st.plotly_chart(fig_do_bar)
        with col2:
            df_do_bar_2 = pd.DataFrame(df_filtered_host_response_time.groupby("bed_type")[["bedrooms", "beds", "accommodates", "price"]].sum())
            df_do_bar_2.reset_index(inplace=True)
            fig_do_bar_2 = px.bar(df_do_bar_2, x='bed_type', y=['bedrooms', 'beds', 'accommodates'],
                                title='BEDROOMS AND BEDS ACCOMMODATES', hover_data="price",
                                barmode='group', color_discrete_sequence=px.colors.sequential.Rainbow_r, width=600, height=500)

            st.plotly_chart(fig_do_bar_2)
# ### ----------------------------------------------------- tab2 --------------------------------------------------------------------------------------------- 
    with tab2:
        st.title("***AVAILABILITY ANALYSIS***")
        df_tab2 = df[df["country"] == country]
        df_tab2.reset_index(drop=True, inplace=True)
        df_tab2 = df_tab2[df_tab2["property_type"] == property_type]
        df_tab2.reset_index(drop=True, inplace=True)
        df_sunb_30 = px.sunburst(df_tab2, path=["room_type", "bed_type", "is_location_exact"], values="availability_30",
                                width=600, height=500, title="Availability_30",
                                color_discrete_sequence=px.colors.sequential.Peach_r)
        st.plotly_chart(df_sunb_30)
        df_sunb_60 = px.sunburst(df_tab2, path=["room_type", "bed_type", "is_location_exact"], values="availability_60",
                                width=600, height=500, title="Availability_60",
                                color_discrete_sequence=px.colors.sequential.Blues_r)
        st.plotly_chart(df_sunb_60)
        df_sunb_90 = px.sunburst(df_tab2, path=["room_type", "bed_type", "is_location_exact"], values="availability_90",
                                width=600, height=500, title="Availability_90",
                                color_discrete_sequence=px.colors.sequential.Aggrnyl_r)
        st.plotly_chart(df_sunb_90)
        df_sunb_365 = px.sunburst(df_tab2, path=["room_type", "bed_type", "is_location_exact"], values="availability_365",
                                width=600, height=500, title="Availability_365",
                                color_discrete_sequence=px.colors.sequential.Greens_r)
        st.plotly_chart(df_sunb_365)
        df_tab2 = df_tab2[df_tab2["room_type"] == room_type]
        df_mul_bar = pd.DataFrame(
            df_tab2.groupby("host_response_time")[["availability_30", "availability_60", "availability_90", "availability_365", "price"]].sum())
        df_mul_bar.reset_index(inplace=True)
        fig_df_mul_bar = px.bar(df_mul_bar, x='host_response_time', y=['availability_30', 'availability_60', 'availability_90', "availability_365"],
                                title='AVAILABILITY BASED ON HOST RESPONSE TIME', hover_data="price",
                                barmode='group', color_discrete_sequence=px.colors.sequential.Rainbow_r, width=1000)

        st.plotly_chart(fig_df_mul_bar)
# ### ----------------------------------------------------- tab3 --------------------------------------------------------------------------------------------- 
    # tab3 creation
    def convert_address(value):
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
    def select_the_df(sel_val, df_selected, differ_max_min):
        if sel_val == f"{df_selected['price'].min()} to {differ_max_min * 0.30 + df_selected['price'].min()} (30% of the Value)":
            df_val_30 = df_selected[df_selected["price"] <= differ_max_min * 0.30 + df_selected['price'].min()]
            df_val_30.reset_index(drop=True, inplace=True)
            return df_val_30
        elif sel_val == f"{differ_max_min * 0.30 + df_selected['price'].min()} to {differ_max_min * 0.60 + df_selected['price'].min()} (30% to 60% of the Value)":
            df_val_60 = df_selected[df_selected["price"] >= differ_max_min * 0.30 + df_selected['price'].min()]
            df_val_60_1 = df_val_60[df_val_60["price"] <= differ_max_min * 0.60 + df_selected['price'].min()]
            df_val_60_1.reset_index(drop=True, inplace=True)
            return df_val_60_1
        elif sel_val == f"{differ_max_min * 0.60 + df_selected['price'].min()} to {df_selected['price'].max()} (60% to 100% of the Value)":
            df_val_100 = df_selected[df_selected["price"] >= differ_max_min * 0.60 + df_selected['price'].min()]
            df_val_100.reset_index(drop=True, inplace=True)
            return df_val_100
        else:
            print("No")
            return df_selected  
    with tab3:
        st.title("LOCATION ANALYSIS")
        st.write("")
        country_location = df["country"].unique()[0]
        property_type_location = df["property_type"].unique()[0]
        room_type_location = df["room_type"].unique()[0]
        df_location_country = df[df["country"] == country_location]
        df_location_country.reset_index(drop=True, inplace=True)
        df_location_property_type = df_location_country[df_location_country["property_type"] == property_type_location]
        df_location_property_type.reset_index(drop=True, inplace=True)
        differ_max_min = df_location_property_type['price'].max() - df_location_property_type['price'].min()
        min_price = round(df_location_property_type['price'].min())
        max_price_30 = round(differ_max_min * 0.30 + df_location_property_type['price'].min())
        max_price_60 = round(differ_max_min * 0.60 + df_location_property_type['price'].min())
        max_price_100 = round(df_location_property_type['price'].max())
        val_sel_location = st.sidebar.radio("Select the Price Range",
                                            [f"{min_price} to {max_price_30} (30% of the Value)",
                                            f"{max_price_30} to {max_price_60} (30% to 60% of the Value)",
                                            f"{max_price_60} to {max_price_100} (60% to 100% of the Value)"])
        df_val_sel_location = select_the_df(val_sel_location, df_location_property_type, differ_max_min)
        df_val_sel_location = df_val_sel_location.drop(columns=["amenities"])
        st.dataframe(df_val_sel_location)
        numeric_columns = df_val_sel_location.select_dtypes(include=[np.number]).columns
        df_val_sel_corr_location = df_val_sel_location[numeric_columns].corr()
        st.dataframe(df_val_sel_corr_location)
        non_numeric_columns = df_val_sel_location.select_dtypes(exclude=[np.number]).columns
        df_val_sel_corr_location_non_numeric = df_val_sel_location.drop(columns=non_numeric_columns).corr()
        st.dataframe(df_val_sel_corr_location_non_numeric)
        st.dataframe(df_val_sel_location)
        df_val_sel_gr_location = pd.DataFrame(
            df_val_sel_location.groupby("accommodates")[["cleaning_fee", "bedrooms", "beds", "extra_people"]].sum()
        )
        fig_1_location = px.bar(df_val_sel_gr_location, x=df_val_sel_gr_location.index, y=['cleaning_fee', 'bedrooms', 'beds'],
                                title="ACCOMMODATES",
                                hover_data=["extra_people"], barmode='group', color_discrete_sequence=px.colors.sequential.Rainbow_r,
                                width=1000)
        df_val_sel_rt_location = df_val_sel_location[df_val_sel_location["room_type"] == room_type_location]
        st.dataframe(df_val_sel_location)
        st.dataframe(df_val_sel_rt_location)
# ### ----------------------------------------------------- tab4 --------------------------------------------------------------------------------------------- 
    with tab4:
        st.title("GEOSPATIAL VISUALIZATION")
        st.write("")
        fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='price', size='accommodates',
                        color_continuous_scale= "rainbow",hover_name='name',range_color=(0,49000), mapbox_style="carto-positron",
                        zoom=1)
        fig.update_layout(width=1150,height=800,title='Geospatial Distribution of Listings')
        st.plotly_chart(fig)
# ### ----------------------------------------------------- tab4 --------------------------------------------------------------------------------------------- 
    with tab5:
        st.title("PRICING ANALYSIS")
        country_t = df["country"].unique()[0]
        df_country_t = df[df["country"] == country_t]
        df_country_t_sorted = df_country_t.sort_values(by="price")
        df_price_neighbourhood = pd.DataFrame(df_country_t_sorted.groupby("host_neighbourhood")["price"].agg(["sum", "mean"])).reset_index()
        df_price_neighbourhood.columns = ["host_neighbourhood", "Total_price", "Average_price"]
        col1, col2 = st.columns(2)
        with col1:
            fig_price_neighbourhood_total = px.bar(df_price_neighbourhood, x="Total_price", y="host_neighbourhood", orientation='h',
                                            title="PRICE BASED ON HOST_NEIGHBOURHOOD", width=600, height=800)
            st.plotly_chart(fig_price_neighbourhood_total)
        with col2:
            fig_price_neighbourhood_avg = px.bar(df_price_neighbourhood, x="Average_price", y="host_neighbourhood", orientation='h',
                                            title="AVERAGE PRICE BASED ON HOST_NEIGHBOURHOOD", width=600, height=800)
            st.plotly_chart(fig_price_neighbourhood_avg)
        df_price_location = pd.DataFrame(df_country_t_sorted.groupby("host_location")["price"].agg(["sum", "mean"])).reset_index()
        df_price_location.columns = ["host_location", "Total_price", "Average_price"]
        col1, col2 = st.columns(2)
        with col1:
            fig_price_location_total = px.bar(df_price_location, x="Total_price", y="host_location", orientation='h',
                                            width=600, height=800, color_discrete_sequence=px.colors.sequential.Bluered_r,
                                            title="PRICE BASED ON HOST_LOCATION")
            st.plotly_chart(fig_price_location_total)
        with col2:
            fig_price_location_avg = px.bar(df_price_location, x="Average_price", y="host_location", orientation='h',
                                            width=600, height=800, color_discrete_sequence=px.colors.sequential.Bluered_r,
                                            title="AVERAGE PRICE BASED ON HOST_LOCATION")
            st.plotly_chart(fig_price_location_avg)
        room_type_t = df["room_type"].unique()[0]
        df_top_100_price = df_country_t_sorted[df_country_t_sorted["room_type"] == room_type_t].head(100)
        fig_top_100_price_1 = px.bar(df_top_100_price, x="name", y="price", color="price",
                                    color_continuous_scale="rainbow",
                                    range_color=(0, df_top_100_price["price"].max()),
                                    title="MINIMUM_NIGHTS MAXIMUM_NIGHTS AND ACCOMMODATES",
                                    width=1200, height=800,
                                    hover_data=["minimum_nights", "maximum_nights", "accommodates"])
        st.plotly_chart(fig_top_100_price_1)
        fig_top_100_price_2 = px.bar(df_top_100_price, x="name", y="price", color="price",
                                    color_continuous_scale="greens",
                                    title="BEDROOMS, BEDS, ACCOMMODATES AND BED_TYPE",
                                    range_color=(0, df_top_100_price["price"].max()),
                                    width=1200, height=800,
                                    hover_data=["accommodates", "bedrooms", "beds", "bed_type"])
        st.plotly_chart(fig_top_100_price_2)
