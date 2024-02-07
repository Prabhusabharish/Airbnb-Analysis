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

    with col1 :

        with col1:
            st.write('''***Airbnb is an online marketplace that connects people who want to rent out
              their property with people who are looking for accommodations,
              typically for short stays. Airbnb offers hosts a relatively easy way to
              earn some income from their property.Guests often find that Airbnb rentals
              are cheaper and homier than hotels.***''')
            st.write('''***Airbnb Inc (Airbnb) operates an online platform for hospitality services.
                  The company provides a mobile application (app) that enables users to list,
                  discover, and book unique accommodations across the world.
                  The app allows hosts to list their properties for lease,
                  and enables guests to rent or lease on a short-term basis,
                  which includes vacation rentals, apartment rentals, homestays, castles,
                  tree houses and hotel rooms. The company has presence in China, India, Japan,
                  Australia, Canada, Austria, Germany, Switzerland, Belgium, Denmark, France, Italy,
                  Norway, Portugal, Russia, Spain, Sweden, the UK, and others.
                  Airbnb is headquartered in San Francisco, California, the US.***''')

    with col2 :
        image_path = "home.png"
        st.image(image_path, use_column_width=True)

### ----------------------------------------------------- connect with mongoDB dataframe --------------------------------------------------------------------------------------------- 
elif SELECT == "Data Exploration":
       
    def datadir():
        df = pd.read_csv(f"C:/Users/prabh/Downloads/Datascience/Project/Airbnb/calendata.csv")
        return df  # Return the DataFrame

    df = datadir()  # Call the function to load the data
  
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["PRICE ANALYSIS","AVAILABILITY ANALYSIS","LOCATION ANALYSIS", "GEOSPATIAL VISUALIZATION", "TOP CHARTS"])

### ----------------------------------------------------- start pages and EDA --------------------------------------------------------------------------------------------- 

    with tab1:
        st.title("**PRICE DIFFERENCE**")
        col1,col2= st.columns(2)

        with col1:
            country = st.selectbox("Select the country", df["country"].unique(), key="country_tab1")

            df_filtered = df[df["country"] == country]
            df_filtered.reset_index(drop=True, inplace=True)

            room_type = st.selectbox("Select the Room Type", df_filtered["room_type"].unique(), key="room_type_tab1")

            df_filtered_room_type = df_filtered[df_filtered["room_type"] == room_type]
            df_filtered_room_type.reset_index(drop=True, inplace=True)

            df_bar = pd.DataFrame(df_filtered_room_type.groupby("property_type")[["price", "review_scores_rating", "number_of_reviews"]].sum())
            df_bar.reset_index(inplace=True)

            custom_colors = {'Entire home/apt': '#1f78b4', 'Private room': '#33a02c', 'Shared room': '#e31a1c'}

            # Bar chart
            fig_bar = px.bar(df_bar, x='property_type', y="price", title="PRICE FOR PROPERTY_TYPES",
                            hover_data=["number_of_reviews", "review_scores_rating"],
                            color='property_type', color_discrete_map=custom_colors, width=650, height=500)
            st.plotly_chart(fig_bar)

        with col2:
            property_type = st.selectbox("Select the Property_type", df["property_type"].unique(), key="property_type_tab1")

            df_filtered_property_type = df[df["property_type"] == property_type]
            df_filtered_property_type.reset_index(drop=True, inplace=True)

            host_response_time = st.selectbox("Select the Host Response Time", df_filtered_property_type["host_response_time"].unique(), key="host_response_time_tab1")

            # Ensure case sensitivity in the column name, use 'host_response_time' instead of 'Host_response_time'
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
            host_response_time = st.selectbox("Select the host_response_time", df["host_response_time"].unique(), key="host_response_time_tab2")

            df_filtered_host_response_time = df[df["host_response_time"] == host_response_time]

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



### ----------------------------------------------------- tab2 --------------------------------------------------------------------------------------------- 

    with tab2:
        st.title("***AVAILABILITY ANALYSIS***")
        col1, col2 = st.columns(2)

        with col1:
            country = st.selectbox("Select the Country", df["country"].unique(), key="country_tab2")

            df = df[df["country"] == country]
            df.reset_index(drop=True, inplace=True)

            property_ty = st.selectbox("Select the Property Type", df["property_type"].unique(), key="property_type_tab2")

            df = df[df["property_type"] == property_ty]
            df.reset_index(drop=True, inplace=True)

            df_sunb_30 = px.sunburst(df, path=["room_type", "bed_type", "is_location_exact"], values="availability_30",
                                    width=600, height=500, title="Availability_30",
                                    color_discrete_sequence=px.colors.sequential.Peach_r)
            st.plotly_chart(df_sunb_30)

        with col2:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")

            df_sunb_60 = px.sunburst(df, path=["room_type", "bed_type", "is_location_exact"], values="availability_60",
                                    width=600, height=500, title="Availability_60",
                                    color_discrete_sequence=px.colors.sequential.Blues_r)
            st.plotly_chart(df_sunb_60)

        col1, col2 = st.columns(2)

        with col1:
            df_sunb_90 = px.sunburst(df, path=["room_type", "bed_type", "is_location_exact"], values="availability_90",
                                    width=600, height=500, title="Availability_90",
                                    color_discrete_sequence=px.colors.sequential.Aggrnyl_r)
            st.plotly_chart(df_sunb_90)

        with col2:
            df_sunb_365 = px.sunburst(df, path=["room_type", "bed_type", "is_location_exact"], values="availability_365",
                                        width=600, height=500, title="Availability_365",
                                        color_discrete_sequence=px.colors.sequential.Greens_r)
            st.plotly_chart(df_sunb_365)

        roomtype = st.selectbox("Select the Room Type", df["room_type"].unique(), key="room_type_tab2")

        df = df[df["room_type"] == roomtype]

        df_mul_bar = pd.DataFrame(
            df.groupby("host_response_time")[["availability_30", "availability_60", "availability_90", "availability_365", "price"]].sum())
        df_mul_bar.reset_index(inplace=True)

        fig_df_mul_bar = px.bar(df_mul_bar, x='host_response_time', y=['availability_30', 'availability_60', 'availability_90', "availability_365"],
                                title='AVAILABILITY BASED ON HOST RESPONSE TIME', hover_data="price",
                                barmode='group', color_discrete_sequence=px.colors.sequential.Rainbow_r, width=1000)

        st.plotly_chart(fig_df_mul_bar)



### ----------------------------------------------------- tab3 --------------------------------------------------------------------------------------------- 

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

    with tab3:
        st.title("LOCATION ANALYSIS")
        st.write("")

        country_location = st.selectbox("Select the Country", df["country"].unique())
        df_location_country = df[df["country"] == country_location]
        df_location_country.reset_index(drop=True, inplace=True)

        property_type_location = st.selectbox("Select the Property Type", df_location_country["property_type"].unique())
        df_location_property_type = df_location_country[df_location_country["property_type"] == property_type_location]
        df_location_property_type.reset_index(drop=True, inplace=True)

        differ_max_min = df_location_property_type['price'].max() - df_location_property_type['price'].min()
        val_sel_location = st.radio("Select the Price Range",
                                    [f"{df_location_property_type['price'].min()} to {differ_max_min * 0.30 + df_location_property_type['price'].min()} (30% of the Value)",
                                    f"{differ_max_min * 0.30 + df_location_property_type['price'].min()} to {differ_max_min * 0.60 + df_location_property_type['price'].min()} (30% to 60% of the Value)",
                                    f"{differ_max_min * 0.60 + df_location_property_type['price'].min()} to {df_location_property_type['price'].max()} (60% to 100% of the Value)"])

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

        room_type_location = st.selectbox("Select the Room_Type", df_val_sel_location["room_type"].unique())
        df_val_sel_rt_location = df_val_sel_location[df_val_sel_location["room_type"] == room_type_location]

        if 'host_neighbourhood' in df_val_sel_location.columns:
            df_grouped = df_val_sel_location.groupby(["street", "host_location", "host_neighbourhood"]).agg({'market': 'sum'}).reset_index()

            st.dataframe(df_grouped)

            fig_2_location = px.bar(df_grouped, x="host_neighbourhood", y="market",
                                    title="MARKET",
                                    hover_data=["street", "host_location", "host_neighbourhood", "market"],
                                    barmode='group', orientation='h',
                                    color_discrete_sequence=px.colors.sequential.Rainbow_r, width=1000)

            st.plotly_chart(fig_2_location)
        else:
            st.warning("")
        
        st.dataframe(df_val_sel_location)

        st.dataframe(df_val_sel_rt_location)



### ----------------------------------------------------- tab4 --------------------------------------------------------------------------------------------- 

    with tab4:
        st.title("GEOSPATIAL VISUALIZATION")

        st.write("")

        fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='price', size='accommodates',
                        color_continuous_scale= "rainbow",hover_name='name',range_color=(0,49000), mapbox_style="carto-positron",
                        zoom=1)
        fig.update_layout(width=1150,height=800,title='Geospatial Distribution of Listings')
        st.plotly_chart(fig)


### ----------------------------------------------------- tab4 --------------------------------------------------------------------------------------------- 

    with tab5:
        country_t = st.selectbox("Select the Country_t", df["country"].unique())
        df1_t = df[df["country"] == country_t]

        property_ty_t = st.selectbox("Select the Property_type_t", df1_t["property_type"].unique())
        df2_t_sorted = df1_t[df1_t["property_type"] == property_ty_t].sort_values(by="price")

        df_price = pd.DataFrame(df2_t_sorted.groupby("host_neighbourhood")["price"].agg(["sum", "mean"])).reset_index()
        df_price.columns = ["host_neighbourhood", "Total_price", "Avarage_price"]

        col1, col2 = st.columns(2)

        with col1:
            fig_price = px.bar(df_price, x="Total_price", y="host_neighbourhood", orientation='h',
                            title="PRICE BASED ON HOST_NEIGHBOURHOOD", width=600, height=800)
            st.plotly_chart(fig_price)

        with col2:
            fig_price_2 = px.bar(df_price, x="Avarage_price", y="host_neighbourhood", orientation='h',
                                title="AVERAGE PRICE BASED ON HOST_NEIGHBOURHOOD", width=600, height=800)
            st.plotly_chart(fig_price_2)

        df_price_1 = pd.DataFrame(df2_t_sorted.groupby("host_location")["price"].agg(["sum", "mean"])).reset_index()
        df_price_1.columns = ["host_location", "Total_price", "Avarage_price"]

        col1, col2 = st.columns(2)

        with col1:
            fig_price_3 = px.bar(df_price_1, x="Total_price", y="host_location", orientation='h',
                                width=600, height=800, color_discrete_sequence=px.colors.sequential.Bluered_r,
                                title="PRICE BASED ON HOST_LOCATION")
            st.plotly_chart(fig_price_3)

        with col2:
            fig_price_4 = px.bar(df_price_1, x="Avarage_price", y="host_location", orientation='h',
                                width=600, height=800, color_discrete_sequence=px.colors.sequential.Bluered_r,
                                title="AVERAGE PRICE BASED ON HOST_LOCATION")
            st.plotly_chart(fig_price_4)

        room_type_t = st.selectbox("Select the Room_Type_t", df2_t_sorted["room_type"].unique())
        df3_t_sorted_price = df2_t_sorted[df2_t_sorted["room_type"] == room_type_t].sort_values(by="price")

        df3_top_50_price = df3_t_sorted_price.head(100)

        fig_top_50_price_1 = px.bar(df3_top_50_price, x="name", y="price", color="price",
                                    color_continuous_scale="rainbow",
                                    range_color=(0, df3_top_50_price["price"].max()),
                                    title="MINIMUM_NIGHTS MAXIMUM_NIGHTS AND ACCOMMODATES",
                                    width=1200, height=800,
                                    hover_data=["minimum_nights", "maximum_nights", "accommodates"])

        st.plotly_chart(fig_top_50_price_1)

        fig_top_50_price_2 = px.bar(df3_top_50_price, x="name", y="price", color="price",
                                    color_continuous_scale="greens",
                                    title="BEDROOMS, BEDS, ACCOMMODATES AND BED_TYPE",
                                    range_color=(0, df3_top_50_price["price"].max()),
                                    width=1200, height=800,
                                    hover_data=["accommodates", "bedrooms", "beds", "bed_type"])

        st.plotly_chart(fig_top_50_price_2)





# ------------------------------ ----------------------about--------------------------------------------------#
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
