from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt    
import joblib
import pandas as pd
import plotly.express as px
import util
import os
from sklearn.metrics import classification_report,ConfusionMatrixDisplay, confusion_matrix
from pathlib import Path

def readCSV(filepath , index = "Unnamed: 0"):
    df = pd.read_csv(filepath)
    df.index = df[index].values
    df.drop(labels= index,axis = 1,inplace=True)
    return df

PATH = Path(__file__).resolve().parent.parent
DATA_PATH = PATH/"Datas"
MODEL_PATH = PATH/"Model"

df = readCSV(DATA_PATH/"data.csv")
author_df = readCSV(DATA_PATH/"author_stats.csv")
publisher_df = readCSV(DATA_PATH/"publisher_stats.csv")
subject_area_df = readCSV(DATA_PATH/"subject_area_stats.csv")
country_df = readCSV(DATA_PATH/"mapData.csv")

lineChartDataCitations = readCSV(DATA_PATH/"line_chart_datesXcitations.csv", "date").fillna(0)
lineChartDataPapers = readCSV(DATA_PATH/"line_chart_datesXpapers.csv", "date").fillna(0)
networkData = readCSV(DATA_PATH/"authors_network.csv")
countriesNetworkData = readCSV(DATA_PATH/"countries_network.csv")

df.sort_values(by="Publishing Date", ascending=True,inplace=True)

columns = df.columns
unique_authors = author_df["Author"].values
unique_publisher = publisher_df["Publisher"].values
unique_subject = subject_area_df["Subject Area"].values
unique_country = country_df["Country"].values

def OverviewPage():
    MIN_DATE , MAX_DATE = dt.datetime.strptime(df["Publishing Date"].min(), "%Y-%m-%d"), dt.datetime.strptime(df["Publishing Date"].max(), "%Y-%m-%d")
    ROW_HEIGHT = 90
    st.title("Scopus Data")

    period = st.slider("Select Published Date range", 
                        MIN_DATE, MAX_DATE,
                        value = (MIN_DATE, MAX_DATE),
                        step = dt.timedelta(days=1),
                        format="YYYY-MM-DD")
    
    checkBox,col1, col2, col3 = st.columns([3,7,7,7])

    with checkBox:
        st.text("Ascending")
        ascending = st.checkbox("", value=True)
    with col1:
        container = st.container(height=ROW_HEIGHT,vertical_alignment="top")
        with container:
            sortBy = st.multiselect(label="Sort by",options = columns, key="scopus sort key")
    with col2:
        container2 = st.container(height=ROW_HEIGHT,vertical_alignment="top")
        with container2:
            names_set = set(st.multiselect(label="Choose Author" , options=unique_authors, key="select scopus author"))
    with col3:
        container3 = st.container(height=ROW_HEIGHT,vertical_alignment="top")
        with container3:
            publisher_set = set(st.multiselect(label="Choose Publisher" , options=unique_publisher , key="select scopus publisher"))

    start, end = str(period[0].date()) , str(period[1].date())
    dataForTable = df[(df["Publishing Date"] >= start) & (df["Publishing Date"] <= end)]
    dataForTable = dataForTable.sort_values(sortBy,ascending=ascending)
    lineCitationsData = lineChartDataCitations.copy().loc[start:end]
    linePapersData = lineChartDataPapers.copy().loc[start:end]

    if (len(names_set) != 0) :
        mask = dataForTable["Author"].str.split(", ").apply(
            lambda authors: bool(set(authors) & names_set)
        )
        dataForTable = dataForTable[mask]
    if (len(publisher_set) != 0):
        mask = dataForTable["Publisher"].str.split(", ").apply(
            lambda publisher: bool(set(publisher) & publisher_set)
        )
        dataForTable = dataForTable[mask]

    st.dataframe(dataForTable[["Publishing Date" , *[col for col in columns if col != "Publishing Date"]]],width="stretch", height=600, hide_index=True)

    st.subheader("Citations on Subject Areas")
    subjectAreaForLineChart = st.multiselect(label="Subject Area for Line Chart",options=lineCitationsData.columns ,key="citations line chart multiselect")
    st.line_chart(x_label="Date",y_label="Citations on Subject Areas", data=lineCitationsData[subjectAreaForLineChart])
    
    st.subheader("Papers published on Subject Areas")
    subjectAreaForLineChart = st.multiselect(label="Subject Area for Line Chart",options=linePapersData.columns , key = "papers line chart multiselect")
    st.line_chart(x_label="Date",y_label="Papers published on Subject Areas", data=linePapersData[subjectAreaForLineChart])


def AuthorsPage():
    st.title("Authors")

    container = st.container(height=200,vertical_alignment="top")
    
    with container:

        col1 , col2 = st.columns([7,3],vertical_alignment="center")

        with col1:
            authorSortedBy = st.multiselect(label="Sort by" , options=author_df.columns , key = "author table sort key")
            author_names_set = st.multiselect(label="Choose Author" , options=unique_authors , key = "author table selection")
        with col2:
            authorAscending = st.checkbox("Ascending", value=False , key="author ascending check box") 

    authorTableData = author_df
    if (len(author_names_set) != 0) :
        authorTableData = author_df[author_df["Author"].isin(author_names_set)]
    if (len(authorSortedBy) == 0) :
        authorSortedBy = ["Total Citations"]
        
    st.dataframe(authorTableData.sort_values(by=authorSortedBy , ascending = authorAscending), width="stretch", height=600, hide_index=True)

    st.subheader("Bar Chart")
    st.text("Please choose authors you want to visualize above.")
    y_axis = st.selectbox(label="Y-axis" ,options=author_df.columns[1:-1], key="select box bar chart authors")
    st.bar_chart(data=author_df[author_df["Author"].isin(author_names_set)],x="Author" , y=y_axis,horizontal=False , sort=f"-{y_axis}" , width=800)


def PublishersPage():
    st.title("Publisher")

    
    container = st.container(height=200,vertical_alignment="top")
    
    with container:
        col1 , col2 = st.columns([7,3],vertical_alignment="center")
        with col1:
            publisherSortedBy = st.multiselect(label="Sort by" , options=publisher_df.columns , key = "publisher table sort key")
            publisher_names_set = st.multiselect(label="Choose Publisher" , options=unique_publisher , key = "publisher table selection")
        with col2:
            publisherAscending = st.checkbox("Ascending", value=False , key="publisher ascending check box") 

    publisherTableData = publisher_df
    if (len(publisher_names_set) != 0) :
        publisherTableData = publisher_df[publisher_df["Publisher"].isin(publisher_names_set)]
    if (len(publisherSortedBy) == 0) :
        publisherSortedBy = ["Total Citations"]
    st.dataframe(publisherTableData.sort_values(by=publisherSortedBy , ascending = publisherAscending), width="stretch", height=600, hide_index=True)
    
    st.subheader("Bar Chart")
    st.text("Please choose publishers you want to visualize above.")
    y_axis = st.selectbox(label="Y-axis" ,options=publisher_df.columns[1:-1], key="select box bar chart publishers")
    st.bar_chart(data=publisher_df[publisher_df["Publisher"].isin(publisher_names_set)],x="Publisher" , y=y_axis,horizontal=False , sort=f"-{y_axis}" , width=800)



def SubjectAreaPage():
    st.title("Subject-Area")
    container = st.container(height=200,vertical_alignment="top")
    
    with container:
        col1 , col2 = st.columns([7,3],vertical_alignment="center")

        with col1:
            subjectSortedBy = st.multiselect(label="Sort by" , options=subject_area_df.columns , key = "subject table sort key")
            subject_names_set = st.multiselect(label="Choose Subject Area" , options=unique_subject , key = "subject table selection")
        with col2:
            subjectAscending = st.checkbox("Ascending", value=False , key="subject ascending check box") 

    subjectTableData = subject_area_df
    if (len(subject_names_set) != 0) :
        subjectTableData = subject_area_df[subject_area_df["Subject Area"].isin(subject_names_set)]
    if (len(subjectSortedBy) == 0) :
        subjectSortedBy = ["Total Citations"]
    st.text(subjectTableData.shape)
    st.dataframe(subjectTableData.sort_values(by=subjectSortedBy , ascending = subjectAscending), width="stretch", height=600, hide_index=True)


def CountryPage():
    st.title("Country")
    container = st.container(height=200,vertical_alignment="top")
    
    with container:
        col1 , col2 = st.columns([7,3],vertical_alignment="center")

        with col1:
            countrySortedBy = st.multiselect(label="Sort by" , options=country_df.columns , key = "country table sort key")
            country_names_set = st.multiselect(label="Choose Country" , options=unique_country , key = "country table selection")
        with col2:
            countryAscending = st.checkbox("Ascending", value=False , key="country ascending check box") 

    countryTableData = country_df
    if (len(country_names_set) != 0) :
        countryTableData = country_df[country_df["Country"].isin(country_names_set)]
    if (len(countrySortedBy) == 0) :
        countrySortedBy = ["Total Citations"]
    st.text(countryTableData.shape)
    st.dataframe(countryTableData.sort_values(by=countrySortedBy , ascending = countryAscending), width="stretch", height=600, hide_index=True)
    
    st.subheader("Bar Chart")
    st.text("Please choose countries you want to visualize above.")
    y_axis = st.selectbox(label="Y-axis" ,options=country_df.columns[1:4], key="select box bar chart country")
    st.bar_chart(data=country_df[country_df["Country"].isin(country_names_set)],x="Country" , y=y_axis,horizontal=False , sort=f"-{y_axis}" , width=800)

    
    st.subheader("Map")
    size_dict = {"Published" : 50 , "Total Citations": 0.5 , "Average Citations": 1000}
    show = st.selectbox(label="Show" ,options=country_df.columns[1:4], key="select box map country")
    size = st.slider("Dot Size", 0.5, 10.0 ,1.0,step=0.5 ,key="size slider map")
    mapData = country_df[["Country", "Latitude" , "Longitude", show]].copy()
    mapData.dropna(axis=0 ,how="any",inplace=True)
    mapData["size"] = mapData[show] * size *size_dict[show]
    st.map(mapData, latitude="Latitude",longitude="Longitude", size = "size")


def CollaborationPage():
    newDf = df[["Affiliation Names","Affiliated Cities","Affiliated Countries"]].copy()
    interMask = (
    newDf["Affiliated Countries"]
      .fillna("")                       
      .str.split(", ")                       
      .apply(lambda lst: len(set([c for c in lst if c])) >= 2)
    )

    nationalMask = (newDf["Affiliated Cities"]
      .fillna("")                         
      .str.split(", ")                     
      .apply(lambda lst: len(set([c for c in lst if c])) >= 2)
    ) & ~interMask

    singleMask = (newDf["Affiliation Names"]
      .fillna("")                         
      .str.split(", ")                     
      .apply(lambda lst: len(set([c for c in lst if c])) < 2)
    ) & ~nationalMask 
    
    internationalCollab = interMask.sum()
    nationalCollab = nationalMask.sum()
    singleAuthorship = singleMask.sum()


    temp = pd.DataFrame([{"International Collaboration" : internationalCollab , "Only National Collaboration" : nationalCollab , "Single Authorship" : singleAuthorship}])
    
    temp = temp.melt(
        var_name="Collaboration Type",
        value_name="Count"
    )
    
    fig = px.pie(
        temp,
        names="Collaboration Type",
        values="Count",
        title="Authorship / Collaboration Distribution"
    )

    st.plotly_chart(fig)
    st.subheader("Co-Author Network graph")
    util.createNetwork_weighted(networkData, graph_title="Co-Authors Network graph", idx="Authors")
    st.subheader("Collaboration between Countries Network graph")
    util.createNetwork_weighted(countriesNetworkData, graph_title="Collaboration between Countries Network graph", idx="Countries")


def AiOverviewTab():
    y_test = np.load(DATA_PATH/"y_test.npy")
    y_train = np.load(DATA_PATH/"y_train.npy")
    y_pred_test = np.load(DATA_PATH/"y_pred_test.npy")
    y_pred_train = np.load(DATA_PATH/"y_pred_train.npy")

    train_report_dict = classification_report(y_train, y_pred_train, output_dict=True)
    test_report_dict = classification_report(y_test, y_pred_test, output_dict=True)

    st.header("Classification Reports")

    st.subheader("Train Set Classification Report")
    util.get_classification_report(train_report_dict)

    
    
    st.subheader("Test Set Classification Report")
    util.get_classification_report(test_report_dict)
    
    cm = confusion_matrix(y_train,y_pred_train)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Unsuccessful" , "Successful"])

    fig, ax = plt.subplots()
    disp.plot(ax=ax)
 
    st.space("large")
    st.header("Confusion Matrix")
    st.subheader("Train set Confusion Matrix")
    
    st.pyplot(fig)

    cm2 = confusion_matrix(y_test, y_pred_test)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=["Unsuccessful" , "Successful"])

    fig2, ax2 = plt.subplots()
    disp2.plot(ax=ax2)
 
    st.subheader("Test set Confusion Matrix")
    
    st.pyplot(fig2)


def AiPage():

    src_dict = {'Journal': {'j'},
    'Conference Proceeding': {'p'},
    'Book Series': {'k'},
    'Book': {'b'}}

    df = pd.DataFrame()

    title = st.text_area(label="Title",key="title", height=50)
    abstract = st.text_area(label="Abstract or Discription",key="abstract")
    keyword = st.text_area(label="Keywords",key="auth-keywords", height=50)
    area = st.text_area(label="Subject Area",key="subject-area", height=50)
    sourceTitle = st.text_area(label="Source Title",key="sourcetitle", height=50)
    author = st.text_area(label="Author",key="author", height=50)
    publisher = st.text_area(label="Publisher",key="publisher", height=50)
    language = st.text_area(label="Language",key="language", height=50)

    col1,col2,col3 = st.columns(3)
    with col1:
        sourceType = src_dict[st.selectbox("Source Type" , options=["Journal", "Conference Proceeding","Book Series","Book"], key="src type")]
    with col2:
        openAccess = st.selectbox(label="Open Access" , options=[0,1,2])
    with col3:
        year = st.selectbox(label="year" , options=range(2017,2025))

    d= {}
    d["title_abstract"] = f"{title} [SEP] {abstract}"
    d["authkeywords"] = f"{keyword}"
    d["subject-area"] = f"{area}"
    d["sourcetitle"] = f"{sourceTitle}"
    d["srctype"] =f"{sourceType}"
    d["openaccess"] = openAccess
    d["year"] = year
    d["author"] = author
    d["publisher"] = publisher
    d["language"] = language

    #"author" , "publisher", "language"
    df = pd.DataFrame([d])
    result = ["unsuccessful" , "successful"]

    model = joblib.load(MODEL_PATH/"logreg_pipeline.joblib")
    st.subheader("AI Predictions:")
    st.write(f" Success Rate : {round(model.predict_proba(df)[0][1]*100 , 2)}%")
    st.write(f" The Paper is predicted to be {result[model.predict(df)[0]]}.")
