import pandas as pd
import numpy as np
import streamlit as st
import datetime as dt
import pages as pages

st.set_page_config(layout="wide")

st.markdown("<p style=\"font-size: 70px; font-weight: bold;\">DSDE Project</p>",unsafe_allow_html=True)

edaTab, overviewTab, authorsTab , publishersTab , subjectAreaTab,countryTab,collaborationTab,aiOverviewTab, aiTab,researchQuestionTab= st.tabs(["EDA","Overview" , "Authors" , "Publishers","Subject Areas", "Countries", "Collaboration","AI Overview","AI", "Research Question"])

with edaTab:
    pages.EDAPage()
with overviewTab:
    pages.OverviewPage()
with authorsTab:
    pages.AuthorsPage()
with publishersTab:
    pages.PublishersPage()
with subjectAreaTab:
    pages.SubjectAreaPage()
with countryTab:
    pages.CountryPage()
with collaborationTab:
    pages.CollaborationPage()
with aiOverviewTab:
    pages.AiOverviewTab()
with aiTab:
    pages.AiPage()
with researchQuestionTab:
    pages.ResearchQuestionPage()
