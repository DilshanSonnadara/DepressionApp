#!/usr/bin/env python
# coding: utf-8

# # Import the model and libraries

# In[1]:


import numpy as np
import pandas as pd 
import pickle
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier


# In[2]:


model = pickle.load(open("Final_Year_Research_Model.pkl", 'rb'))


# # Title and description

# In[3]:


st.write("""
# Depression Prediction 


""")


# In[ ]:


st.subheader("This app will use machine learning technology to let you know if you are at risk of developing depressive symptoms based on your background factors. Please fill in the form given on the left hand side of the app to assess yourself.")


# # Sidebar

# In[4]:


st.sidebar.header("User Input Parameters")


# In[5]:


Life_Satisfaction=st.sidebar.slider("How satisfied are you with life? (1-Not at all satisfied, 5-Very Satisfied)",1,5)
Level_of_stress_with_respect_to_academic=st.sidebar.slider("How stressful is the work related to academics? (1-Not at all stressful, 5-Very stressful)",0,5)
Enjoyment_of_university_life=st.sidebar.slider("To what extent do you enjoy your university life?? (1-I do not enjoy it at all, 5-I enjoy it very much)",0,5)
BMI=st.sidebar.slider("What is your BMI?",7.0,110.0)
Meal_Categorization=st.sidebar.selectbox('Please select one of the following?',('I have a nutritionally balanced diet', 'On a diet', 'Unable to eat 3 meals a day','Other'))
Frequency_of_socializing=st.sidebar.selectbox('How often do you socialize with your friends?',('Less than once a month', 'Once a month', '2 to 3 times a month','Once a week','More than once a week'))
Interaction_with_family=st.sidebar.slider("How well do you interact with your family? (1 - Negative Interaction, 5 - Positive Interaction)",1,5)
Preferance_of_lectures=st.sidebar.selectbox('Please select one of the following regarding lectures',('Like online and happens online','Like online but happens onsite','Like onsite and happens onsite','Like onsite but happens online'))
Frequency_of_Physical_Exercise=st.sidebar.slider("How many days of the week do you work out/ get physical exercise?",0,7)
Satisfaction_with_academic_achievements=st.sidebar.selectbox('Are you satisfied with your academic achievements?',('Yes', 'No','Neither satisfied nor dissatisfied','No GPA as of yet'))
University_Enter_Attempt=st.sidebar.selectbox('Which attempt did you enter university from?',('1st attempt','2nd attempt','3rd attempt'))
Employment_Status=st.sidebar.selectbox('Please select one of the following',('I do a part time job', 'I do a full time job', 'I am not employed'))
Never_Love=st.sidebar.selectbox('Have you ever been in a love affair?',('Yes', 'No'))
Life_Threatening_Events=st.sidebar.selectbox('How many serious life-threatening events have you been exposed to?',('None', 'One', 'Two', 'Three or greater'))
Physical_Appearence_Satisfaction=st.sidebar.slider("How satisfied are you with your physical appearance? (1-Not at all satisfied, 5-Very Satisfied)",1,5)
Way_of_living=st.sidebar.slider("To what extent has the economic problems in the country affected your way of living? (1-Not at all Affected, 5-Very significantly affected)",1,5)
Awareness_of_professional_help=st.sidebar.selectbox('Are you aware of the professional help that is available in the university to students who are depressed?',('Yes', 'No'))
Year_in_university_Year_3=st.sidebar.selectbox('Are you currently in your third year?',('Yes', 'No'))
Having_Siblings=st.sidebar.selectbox('Do you have any siblings?',('Yes', 'No'))
Frequency_seeing_family=st.sidebar.selectbox('How frequently do you see your family?',('Almost never/ Never', 'Rarely', 'Occasionally','Quite Frequently','Almost everyday/ Everyday'))
Satisfied_Relationship=st.sidebar.selectbox('Are you in a satisfactory relationship?',('Yes', 'No'))
Loss_of_job=st.sidebar.selectbox('Did anyone in your family (Including yourself) lose their job/business in the past three years?',('Yes', 'No'))
Financially_supporting_family=st.sidebar.selectbox('Do you have to financially support your family?',('Yes', 'No'))
Income=st.sidebar.selectbox('What is your total monthly family income in rupees?',('Less than 30,000', '30,000-100,000','100,00-250,000','250,000-500,000','Greater than 500,000'))
Restriction_to_home=st.sidebar.selectbox('Do you feel as though you have been restricted to your home (most of the time) due to various situations such as covid, curfew and current situation in the country?',('Yes','No'))

Sleeping_Hours=st.sidebar.slider("How many hours do you sleep at night (on average)?",0,24)
Sports_in_university=st.sidebar.selectbox('Do you take part in sports in university?',('Yes', 'No'))
Financial_support_from_family=st.sidebar.selectbox('Do you have adequate financial support from your family?',('Yes', 'No'))
Accomodation_University_Satisfaction=st.sidebar.slider("How satisfied are you with your accommodation?",1,5)
Recently_Broke_Up=st.sidebar.selectbox('Have you gone through a breakup recently?',('Yes', 'No'))
Accomadation_University=st.sidebar.selectbox('Where do you stay when attending university?',('Rented Place', 'Other'))
Societies_in_university=st.sidebar.selectbox('Do you take part in societies in university?',('Yes', 'No'))
Loss_of_loved_one=st.sidebar.selectbox('Has someone very close to you passed away in the last three years?',('Yes', 'No'))
Mode_of_transport_to_university=st.sidebar.selectbox('How do you travel to university?',('Walking', 'Other'))
Difficulty_in_English=st.sidebar.selectbox('Do you have any difficulty following academic activities because of the English language?',('Yes', 'No'))
Parents_Employment=st.sidebar.slider("How many of your parents are employed?",0,2)


# # Encoding the variables

# In[6]:


Meal_Categorization_I_have_a_nutritionally_balanced_diet= [1 if Meal_Categorization=="I have a nutritionally balanced diet" else 0 ][0]
Meal_Categorization_On_a_diet= [1 if Meal_Categorization=="On a diet" else 0][0]
Meal_Categorization_Meal_Categorization_Unable_to_eat_3_meals_a_day= [1 if Meal_Categorization=="Unable to eat 3 meals a day" else 0][0]
Never_Love_No=[1 if Never_Love=="No" else 0][0]
Never_Love_Yes=[1 if Never_Love=="Yes" else 0][0]
Having_Siblings_Yes=[1 if Having_Siblings=="Yes" else 0][0]
Having_Siblings_No=[1 if Having_Siblings=="No" else 0][0]

Awareness_of_professional_help_Yes=[1 if Awareness_of_professional_help=="Yes" else 0][0]
Awareness_of_professional_help_No=[1 if Awareness_of_professional_help=="No" else 0][0]
Loss_of_job_Yes=[1 if Loss_of_job=="Yes" else 0][0]
Loss_of_job_No=[1 if Loss_of_job=="No" else 0][0]

Satisfaction_with_academic_achievements_Yes=[1 if Satisfaction_with_academic_achievements=="Yes" else 0  ][0]
Satisfaction_with_academic_achievements_No=[1 if Satisfaction_with_academic_achievements=="No" else 0][0]
Satisfaction_with_academic_achievements_Neither_satisfied_nor_dissatisfied=[1 if Satisfaction_with_academic_achievements=="Neither satisfied nor dissatisfied" else 0][0]
Satisfaction_with_academic_achievements_No_GPA_as_of_yet=[1 if Satisfaction_with_academic_achievements=="No GPA as of yet" else 0][0]

Sports_in_university_Yes=[1 if Sports_in_university=="Yes" else 0 ][0]
Sports_in_university_No=[1 if Sports_in_university=="No" else 0 ][0]
Satisfied_Relationship_Yes=[1 if Satisfied_Relationship=="Yes" else 0][0]
Satisfied_Relationship_No=[1 if Satisfied_Relationship=="No" else 0][0]

Loss_of_loved_one_No=[1 if Loss_of_loved_one=="No" else 0][0]
Societies_in_university_Yes=[1 if Societies_in_university=="Yes" else 0][0]

Financial_support_from_family_No=[1 if Financial_support_from_family=="No" else 0][0]
Financially_supporting_family_No=[1 if Financially_supporting_family=="No" else 0][0]

Recently_Broke_Up_Yes=[1 if Recently_Broke_Up=="Yes" else 0][0]
Recently_Broke_Up_No=[1 if Recently_Broke_Up=="No" else 0][0]
Restriction_to_home_Yes=[1 if Restriction_to_home=="Yes" else 0][0]
Restriction_to_home_No=[1 if Restriction_to_home=="No" else 0][0]

Mode_of_transport_to_university_Walking=[1 if Mode_of_transport_to_university=="Walking" else 0][0]

Accomadation_University_Rented_Place=[1 if Accomadation_University=='Rented Place' else 0][0]
Preferance_of_lectures_I_like_online_lectures_but_most_of_my_lectures_are_onsite=[1 if Preferance_of_lectures=="Like online and happens online" else 0][0]
Preferance_of_lectures_I_like_online_lectures_and_most_of_my_lectures_are_online=[1 if Preferance_of_lectures=="Like online but happens onsite" else 0][0]
Preferance_of_lectures_I_like_onsite_lectures_and_most_of_my_lectures_are_onsite=[1 if Preferance_of_lectures=="Like onsite and happens onsite" else 0][0]
Preferance_of_lectures_I_like_onsite_lectures_but_most_of_my_lectures_are_online=[1 if Preferance_of_lectures=="Like onsite but happens online" else 0][0]
Financial_support_from_family_No=[1 if Financial_support_from_family=="No" else 0 ][0]
Year_in_university_Year_3=[1 if Year_in_university_Year_3=="Yes" else 0][0]
Difficulty_in_English_No=[1 if Difficulty_in_English=="No" else 0][0]


# In[62]:


Frequency_of_socializing=[1 if Frequency_of_socializing=="Less than once a month" else 2 if Frequency_of_socializing=="Once a month" else 3 if Frequency_of_socializing=="2 to 3 times a month" else 4 if Frequency_of_socializing=="Once a week" else 5 if Frequency_of_socializing=="More than once a week" else np.NAN][0]
University_Enter_Attempt=[1 if University_Enter_Attempt=="1st attempt" else 2 if University_Enter_Attempt=="2nd attempt" else 3 if University_Enter_Attempt=="3rd attempt" else np.NAN][0]
Employment_Status=[1 if Employment_Status=='I am not employed' else 2 if Employment_Status=='I do a part time job' else 3 if Employment_Status=='I do a full time job' else np.NAN][0]
Life_Threatening_Events = [0 if Life_Threatening_Events=="None" else 1 if Life_Threatening_Events=="One" else 2 if Life_Threatening_Events=="Two" else 3 if Life_Threatening_Events=="Three or greater" else np.NAN][0]
Frequency_seeing_family=[1 if Frequency_seeing_family=="Almost never/ Never" else 2 if Frequency_seeing_family=='Rarely' else 3 if Frequency_seeing_family=='Occasionally' else 4 if Frequency_seeing_family=='Quite Frequently' else 5 if Frequency_seeing_family=='Almost everyday/ Everyday' else np.NAN][0]
Income=[1 if Income=='Less than 30,000' else 2 if Income== '30,000-100,000' else 3 if Income=='100,00-250,000' else 4 if Income=='250,000-500,000' else 5 if Income=='Greater than 500,000' else np.NAN][0]


# # Standardizing the variables

# In[63]:


Life_Satisfaction = (Life_Satisfaction-3.54)/1.04
Level_of_stress_with_respect_to_academic=((Level_of_stress_with_respect_to_academic-3.54)/1.06)
Enjoyment_of_university_life=(Enjoyment_of_university_life-3.32)/1.01
BMI=(BMI-22.36)/3.99
Frequency_of_socializing=(Frequency_of_socializing-2.73)/1.50
Interaction_with_family=(Interaction_with_family-4.13)/0.92
Frequency_of_Physical_Exercise=(Frequency_of_Physical_Exercise-1.51)/1.8
University_Enter_Attempt=(University_Enter_Attempt-1.725)/0.71
Way_of_living=(Way_of_living-3.89)/1.05
Employment_Status=(Employment_Status-1.25)/0.59
Life_Threatening_Events=(Life_Threatening_Events-0.61)/0.95
Physical_Appearence_Satisfaction=(Physical_Appearence_Satisfaction-3.59)/1.01
Income =  (Income-2.45)/0.98
Accomodation_University_Satisfaction=(Accomodation_University_Satisfaction-3.96)/1.06
Parents_Employment=(Parents_Employment-1.22)/0.57
Frequency_seeing_family=(Frequency_seeing_family-4.32)/0.95
Sleeping_Hours=(Sleeping_Hours-6.17)/1.09
Parents_Employment=(Parents_Employment-1.22)/0.57


# # Making the dataframe for the prediction

# In[50]:


data=pd.DataFrame(np.array([Sleeping_Hours, BMI, Parents_Employment,
       Frequency_of_Physical_Exercise, Employment_Status, Income,
       Accomodation_University_Satisfaction, Frequency_of_socializing,
       Life_Threatening_Events, Life_Satisfaction,
       Physical_Appearence_Satisfaction, University_Enter_Attempt,
       Interaction_with_family, Level_of_stress_with_respect_to_academic,
       Enjoyment_of_university_life, Frequency_seeing_family,
       Way_of_living, Accomadation_University_Rented_Place,
       Meal_Categorization_I_have_a_nutritionally_balanced_diet,
       Meal_Categorization_On_a_diet,
       Meal_Categorization_Meal_Categorization_Unable_to_eat_3_meals_a_day,
       Year_in_university_Year_3, Mode_of_transport_to_university_Walking,
       Sports_in_university_No, Sports_in_university_Yes,
       Societies_in_university_Yes,
       Preferance_of_lectures_I_like_online_lectures_and_most_of_my_lectures_are_online,
       Preferance_of_lectures_I_like_online_lectures_but_most_of_my_lectures_are_onsite,
       Preferance_of_lectures_I_like_onsite_lectures_and_most_of_my_lectures_are_onsite,
       Preferance_of_lectures_I_like_onsite_lectures_but_most_of_my_lectures_are_online,
       Satisfaction_with_academic_achievements_Neither_satisfied_nor_dissatisfied,
       Satisfaction_with_academic_achievements_No,
       Difficulty_in_English_No, Awareness_of_professional_help_No,
       Awareness_of_professional_help_Yes, Having_Siblings_Yes,
       Financial_support_from_family_No, Financially_supporting_family_No,
       Loss_of_loved_one_No, Restriction_to_home_No,
       Restriction_to_home_Yes, Loss_of_job_No,
       Satisfied_Relationship_Yes, Never_Love_No, Never_Love_Yes,
       Recently_Broke_Up_Yes]).reshape(1,46))


# In[51]:


data.columns=['Sleeping_Hours', 'BMI', 'Parents_Employment',
       'Frequency_of_Physical_Exercise', 'Employment_Status', 'Income',
       'Accomodation_University_Satisfaction', 'Frequency_of_socializing',
       'Life_Threatening_Events', 'Life_Satisfaction',
       'Physical_Appearence_Satisfaction', 'University_Enter_Attempt',
       'Interaction_with_family', 'Level_of_stress_with_respect_to_academic',
       'Enjoyment_of_university_life', 'Frequency_seeing_family',
       'Way-of_living', 'Accomadation_University_Rented Place',
       'Meal_Categorization_Nutritionally Balanced Diet',
       'Meal_Categorization_On a diet',
       'Meal_Categorization_Unable to eat 3 meals a day',
       'Year_in_university_Year 3', 'Mode_of_transport_to_university_Walking',
       'Sports_in_university_No', 'Sports_in_university_Yes',
       'Societies_in_university_Yes',
       'Preferance_of_lectures_I like online lectures and most of my lectures are online',
       'Preferance_of_lectures_I like online lectures but most of my lectures are onsite',
       'Preferance_of_lectures_I like onsite lectures and most of my lectures are onsite',
       'Preferance_of_lectures_I like onsite lectures but most of my lectures are online',
       'Satisfaction_with_academic_achievements_Neither satisfied nor dissatisfied',
       'Satisfaction_with_academic_achievements_No',
       'Difficulty_in_English_No', 'Awareness_of_professional_help_No',
       'Awareness_of_professional_help_Yes', 'Having_Siblings_Yes',
       'Financial_support_from_family_No', 'Financially_supporting_family_No',
       'Loss_of_loved_one_No', 'Restriction_to_home_No',
       'Restriction_to_home_Yes', 'Loss_of_job_No',
       'Satisfied_Relationship_Yes', 'Never_Love_No', 'Never_Love_Yes',
       'Recently_Broke_Up_Yes']


# # Making the prediction

# In[53]:


st.title("The predicted value")
B=model.predict(data)
Value="Not Depressed" if B==0 else "Depressed"
if Value=="Depressed":
    st.subheader("The current input parameters indicate a risk of developing depression")
elif Value=="Not Depressed":
    st.subheader("The current input parameters do not indicate a risk of developing depression")


# # Making the plot

# In[105]:



# # Adding a background image

# In[54]:


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2016/03/27/07/32/clouds-1282314_960_720.jpg");
             background-attachment: fixed;
             background-size: cover;
             
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 


# # Changing the colour of the side bar

# In[55]:


# Set sidebar background color
st.markdown(
    """
    <style>
    .css-1aumxhk {
        background-color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# # Adding a song

# In[13]:


st.header("Like to listen to some music while filling the form?")
st.subheader("Please click the play button")

from streamlit_player import st_player

# Embed a youtube video

st_player("https://www.youtube.com/watch?v=HJrKVYJdABc")


# In[ ]:




