import pandas as pd
import streamlit as st
import numpy as np
import pickle as pk
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,classification_report 
#Importing the dependencies
import pandas as pd
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import svm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import streamlit as st
import base64
import pickle as pk
import base64
from sklearn import svm
#import seaborn as sns
import altair as alt



st.set_page_config(page_title='Cervical Cancer detection',layout='centered')



with st.sidebar:
    selection=option_menu(menu_title="Main Menu",options=["Single Prediction","Multi Prediction"],icons=["cast","book"],menu_icon="house",default_index=0)

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download your Predictions</a>'
    return href

train_data_header_names=['Age', 'Num of pregnancies', 'Hormonal Contraceptives (years)', 'IUD', 'STDs', 'STDs (number)', 'STDs:condylomatosis', 'STDs:vulvo-perineal condylomatosis', 'STDs:genital herpes', 'STDs:HIV', 'STDs: Number of diagnosis', 'Dx:Cancer', 'Dx:CIN', 'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology']

def eligibility_status(givendata):
    
    loaded_model=pk.load(open("The_Cervical_Cancer_Model.sav", "rb"))
    input_data_as_numpy_array = np.asarray(givendata)# changing the input_data to numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) # reshape the array as we are predicting for one instance
    new_input_data=pd.DataFrame(input_data_reshaped,columns=train_data_header_names)
    new_input_data=new_input_data.values
    # std=StandardScaler()
    # std_input_data_reshaped=std.fit_transform(input_data_reshaped)
    prediction = loaded_model.predict(new_input_data)

    
    if prediction[0]==0:
      return "No cervical cancer detected"
    else:
      return "Cervical cancer is present"



def main():
    st.header("cervical cancer detection system")
    theReasons=[]
    #getting user input
    
    age = st.slider('Patient age', 0, 200, key="ageslide")
    st.write("Patient is", age, 'years old')

    # sex_partners = st.slider('Number of sexual partners', 0, 80, key="sex_partners")

    # First_Intercourse = st.slider('First sexual intercourse', 0, 80, key="first_sex")

    Num_of_pregnancies = st.slider('Num of pregnancies', 0, 80, key="num_pregnancies")


    try:
        Years_of_hormonal_contraceptives = st.text_input('How long has the patient been on hormonal_contraceptives',"0",key="yearsofhormonal_Contracept")
        Years_of_hormonal_contraceptives=float(Years_of_hormonal_contraceptives)
        
    except ValueError:
        if Years_of_hormonal_contraceptives=="":
            st.error("Insert the years of Hormonal Contraceptives")
        else:
            st.error("This field can't be empty")

    st.write("\n")
    
    option2 = st.selectbox("Presence of Intrauterine Device (IUD) in patients",("",'Yes', 'No'),key="IUD")
    if (option2=='Yes'):
        iud=1        
    else:
        iud=0

    # Years_of_IUD = st.number_input("How long has the patient been on IUD ")


    option3 = st.selectbox("Any STD",("",'Yes', 'No'),key="patient_std")
    if (option3=='Yes'):
        std=1
        
    else:
        std=0
    
    # std_number = st.number_input("number of STDs")
    
    try:
        std_number = st.text_input('number of STDs',"0", key="std_number")
        std_number=float(std_number)
    except ValueError:
        st.error("This field can't be empty")

    st.write("\n")



    option4 = st.selectbox("Is condylomatosis present ?",("",'Yes', 'No'),key="the_std_condylomatosis")
    if (option4=='Yes'):
        std_condylomatosis=1    
    else:
        std_condylomatosis=0
    
    
    


    option6= st.selectbox("Any vulvo-perineal condylomatosis ?",("","Yes","No"),key="the_vulvo_perineal_condylomatosis")
    if (option6=='Yes'):
        vulvo_perineal_condylomatosis=1    
    else:
        vulvo_perineal_condylomatosis=0


    
    option9= st.selectbox("genital herpes ?",("","Yes","No"),key="the_genital herpes")
    if (option9=='Yes'):
        genital_herpes=1    
    else:
        genital_herpes=0

 

    option11= st.selectbox("HIV ?",("","Yes","No"),key="the_hiv")
    if (option11=='Yes'):
        HIV=1    
    else:
        HIV=0  
    

    try:
        Number_of_diagnosis = st.text_input('number of STD diagnosis',"0", key="diagnosis_times")
        Number_of_diagnosis=float(Number_of_diagnosis)
    except ValueError:
        st.error("This field can't be empty")

    st.write("\n")


        
    option14= st.selectbox("diagnosed with cancer ?",("","Yes","No"),key="Cancer_diagnosis")
    if (option14=='Yes'):
        Cancer_diagnosis=1    
    else:
        Cancer_diagnosis=0

    option15= st.selectbox("diagnosed with Cervical Intraepithelial Neoplasia?",("","Yes","No"),key="the_Cin_diagnosis")
    if (option15=='Yes'):
        Cin_diagnosis=1    
    else:
        Cin_diagnosis=0

    option16= st.selectbox("diagnosed with hpv ?",("","Yes","No"),key="the_diagnosed_HPV")
    if (option16=='Yes'):
        diagnosed_HPV=1    
    else:
        diagnosed_HPV=0

    option17= st.selectbox("Other diagnosis ?",("","Yes","No"),key="the_Other_diagnosis")
    if (option17=='Yes'):
        Other_diagnosis=1    
    else:
        Other_diagnosis=0

    option18= st.selectbox("Hinselmann present?",("","Yes","No"),key="the_Hinselmann")
    if (option18=='Yes'):
        Hinselmann=1    
    else:
        Hinselmann=0

    option19= st.selectbox("Schiller present ?",("","Yes","No"),key="the_Schiller")
    if (option19=='Yes'):
        Schiller=1    
    else:
        Schiller=0

    option20= st.selectbox("Citology present ?",("","Yes","No"),key="the_Citology")
    if (option20=='Yes'):
        Citology=1    
    else:
        Citology=0
      
    
    Eligible = '' #for displaying result
    Reason=""
    
    # creating a button for Prediction
    if option2!=""  and option3!=""  and option4!=""   and option6!="" and option9!=""  and option11!=""   and option14!="" and option15!="" and option16!=""  and option17!=""  and option18!=""  and option19!="" and option20!="" and st.button('Predict'):
        Eligible = eligibility_status([age,Num_of_pregnancies,Years_of_hormonal_contraceptives,iud,std,std_number,std_condylomatosis,vulvo_perineal_condylomatosis,genital_herpes,HIV,Number_of_diagnosis,Cancer_diagnosis,Cin_diagnosis,diagnosed_HPV,Other_diagnosis,Hinselmann,Schiller,Citology])
        st.success(Eligible)
    


def multi(input_data):
    loaded_model=pk.load(open("The_Cervical_Cancer_Model.sav", "rb"))
    dfinput = pd.read_csv(input_data)
    
    st.header('Dataset')
    st.markdown('Preview of the uploaded dataset')
    st.dataframe(dfinput)
    #st.markdown(dfinput.shape)
    st.write("\n")
    st.write("\n")
    #dfinput=dfinput.iloc[1:].reset_index(drop=True)
    

    std_scaler=pk.load(open("my_saved_CervicalCancer_std_scaler.pkl", "rb"))
    dfinput=std_scaler.transform(dfinput)
    
    st.write()
    st.write()
    with st.sidebar:
        predictButton=st.button("Click to Predict")
        #selectionList=["","confusion Matrix","Reality data vs Test result"]
        #selectionw=option_menu(menu_title=None,options=["Predict your result","Visualization","confusion Matrix"],icons=["cast","book","cast"],default_index=1, orientation="horizontal")
        st.write("\n")
        st.write("\n")

       

    if predictButton:
        prediction = loaded_model.predict(dfinput)
        interchange=[]
        for i in prediction:
            if i==0:
                newi="No Cervical Cancer detected"
                interchange.append(newi)
            else:
                newip="Cervical Cancer is present"
                interchange.append(newip)
            
        st.subheader('**Predicted output**')
        prediction_output = pd.Series(prediction, name='Biopsy')
        prediction_id = pd.Series(np.arange(0, len(prediction_output)))

        dfresult = pd.concat([prediction_id, prediction_output], axis=1)
        st.dataframe(prediction_output)
        st.markdown(filedownload(dfresult), unsafe_allow_html=True)
        





if selection=="Single Prediction":
    main()

if selection=="Multi Prediction":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title("Cervical Cancer")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    #--------------Visualization-------------------#
    # Main panel
    
    # Displays the dataset
    if uploaded_file is not None:
        #load_data = pd.read_table(uploaded_file)
        multi(uploaded_file)
    else:
        st.info('waiting for CSV file to be uploaded.')








