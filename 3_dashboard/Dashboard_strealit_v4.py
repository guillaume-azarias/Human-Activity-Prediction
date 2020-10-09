# This is a script to show a dashboard during the final project presentation
# 
# Since it will not be further used by Caru, it is not interactive in term of 
# data plotting, but only shows example traces from our analysis results.
# 
# To run the script, go to the folder where it is located in the Terminal, then run:
# streamlit run Dashboard_strealit_v1.py

# Questions: guillaume.azarias@hotmail.com

import streamlit as st


################################################
# Top of the dashboard: Logo, user and resident
# Logo
logo_pic = "Header.png"
st.image(logo_pic, use_column_width=True)

# User section in the sidebar
st.sidebar.image('User_section.png', width=300)

# Select Resident
# See https://www.fakeaddressgenerator.com/World_more/Switzerland_address_generator
pics = {
    "Yvonne Pfeifer": "Yvonne.png",
    "Daniel Maurer": "Daniel.png"
}
resident = st.sidebar.selectbox("Select Resident", list(pics.keys()), key=11)
st.image(pics[resident], use_column_width=True)


################################################
# Middle of the dashboard: Parameters
# Sidebar
# Parameter picture
st.sidebar.image('Parameters.png', width=300)

# Summary report
summary_normal_pic = "Summary of activity_normal.png"
summary_pic_abnormal = "Summary of activity_abnormal.png"
if resident == "Yvonne Pfeifer":
    st.image(summary_normal_pic, use_column_width=True)
elif resident == "Daniel Maurer":
    st.image(summary_pic_abnormal, use_column_width=True)


# Analysis sidebar
analysis = [
    "Anomaly detection",
    "Between-days clustering",
    "Within-day clustering"
]
analysis_type = st.sidebar.selectbox("Analysis", analysis, key=12)

# Date
date_graph = {
    "Last day": "https://storage.needpix.com/rsynced_images/science-fiction-2971848_1280.jpg",
    "Last week": "https://cdn.pixabay.com/photo/2016/09/24/22/20/cat-1692702_960_720.jpg"
}
date_type = st.sidebar.selectbox("Date", list(date_graph.keys()), key=14)

# Parameters
parameter = {
    "Presence (CO2)": "https://storage.needpix.com/rsynced_images/science-fiction-2971848_1280.jpg",
    "Light": "https://cdn.pixabay.com/photo/2016/09/24/22/20/cat-1692702_960_720.jpg",
    "Temperature": "https://cdn.pixabay.com/photo/2019/03/15/19/19/puppy-4057786_960_720.jpg",
}
parameter_type = st.sidebar.selectbox(
    "Parameter", list(parameter.keys()), key=15)



# Analysis results to be loaded
forecasting_usual = "Normal_2803.png"
forecasting_usual_week = "Normal_2803_1w.png"
forecasting_unusual = "Abnormal_2803.png"

betweendays = "between-days clustering.png"

withinday_1 = "within-day clustering_1.png"
withinday_2 = "within-day clustering_2.png"
withinday_3 = "within-day clustering_3.png"


# Loading of analysis results
if resident == "Yvonne Pfeifer":
    if analysis_type == "Anomaly detection":
        if date_type == 'Last day':
            st.image(forecasting_usual, use_column_width=True)
        elif date_type == 'Last week':
            st.image(forecasting_usual_week, use_column_width=True)
    else:
        st.markdown("No data for Yvonne")
elif resident == "Daniel Maurer":
    if analysis_type == "Anomaly detection":
        st.image(forecasting_unusual, use_column_width=True)
    elif analysis_type == "Between-days clustering":
        st.image(betweendays, use_column_width=True)
    elif analysis_type == "Within-day clustering":
        st.image(withinday_1, use_column_width=True)
        st.image(withinday_2, use_column_width=True)
        st.image(withinday_3, use_column_width=True)
    else:
        st.markdown("Sorry, no data...")
else:
    st.markdown("No data...")


# Contact section

"""
\n
## Contact details
"""

contact_pic = "Contact_section.png"
st.image(contact_pic, use_column_width=True)
