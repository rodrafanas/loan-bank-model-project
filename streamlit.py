import streamlit as st
import pandas as pd
import base64
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import shap
import streamlit.components.v1 as components
import warnings 
warnings.filterwarnings('ignore')

#Load artifacts of saved model

metadata = pd.read_csv('artifacts/metadata.csv')
pol = pd.read_csv("artifacts/prop_pol.csv").drop(columns=['Unnamed: 0'])

mean_vars_num = joblib.load('artifacts/mean_vars_num.joblib')
loaded_scaler = joblib.load('artifacts/scaler.joblib')
map_addr_state = joblib.load('artifacts/map_addr_state.joblib')
features_used_to_train = joblib.load('artifacts/features_used_to_train.joblib')
threshold_medium_risk_decision = joblib.load('artifacts/threshold_medium_risk_decision.joblib')
threshold_high_risk_decision = joblib.load('artifacts/threshold_high_risk_decision.joblib')


model = joblib.load("artifacts/best_model.joblib")

st.set_page_config(
    page_title="Loan Bank App",
    page_icon="images/yes_bank_logo.png"
)

st.set_option('deprecation.showPyplotGlobalUse', False)

######################
#main page layout
######################

st.title("Loan Default Prediction")
st.subheader("Are you sure your loan applicant is surely going to pay the loan back?💸 "
                 "This machine learning app will help you to make a prediction to help you with your decision!")

col1, col2 = st.columns([1, 1])

with col1:
    st.image("images/yes_bank_logo.png")

with col2:
    st.write("""To borrow money, credit analysis is performed. Credit analysis involves the measure to investigate
the probability of the applicant to pay back the loan on time and predict its default/ failure to pay back.

These challenges get more complicated as the count of applications increases that are reviewed by loan officers.
Human approval requires extensive hour effort to review each application, however, the company will always seek
cost optimization and improve human productivity. This sometimes causes human error and bias, as it’s not practical
to digest a large number of applicants considering all the factors involved.""")

st.subheader("To predict default/ failure to pay back status, you need to follow the steps below:")
st.markdown("""
1. Enter/choose the parameters that best describe your applicant on the left side bar;
2. Press the "Predict" button and wait for the result.

""")

st.subheader("Below you could find prediction result: ")

######################
#sidebar layout
######################

st.sidebar.title("Loan Applicant Info")
st.sidebar.image("images/loan_bank.png", width=100)
st.sidebar.write("Please choose parameters that describe the applicant")

# input features
# term
term = st.sidebar.radio("Select Loan term: ", (' 36 months', ' 60 months'))
# loan_term
loan_amnt =st.sidebar.slider("Please choose Loan amount you would like to apply:",min_value=1000, max_value=40000,step=500)
# emp_length
emp_length = st.sidebar.selectbox('Please choose your employment length', ("< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
                                                                           "6 years", "7 years","8 years","9 years","10+ years") )
# anual_inc
annual_inc =st.sidebar.slider("Please choose your annual income:", min_value=10000, max_value=200000,step=1000)
# addr_state 
addr_state =st.sidebar.selectbox('Please choose your state: ',('IA', 'ND', 'ID', 'SD', 'WY', 'VT', 'ME', 'AK', 'DC', 'MT',
                                                        'DE', 'NE', 'WV', 'RI', 'HI', 'NH', 'NM', 'MS', 'UT', 'AR', 'KS', 
                                                        'OK', 'KY', 'LA', 'OR', 'AL', 'SC', 'WI', 'NV', 'TN', 'CT', 'MO', 
                                                        'IN', 'MN', 'WA', 'CO', 'MA', 'AZ', 'MD', 'MI', 'NC', 'VA', 'GA', 
                                                        'OH', 'PA', 'NJ', 'IL', 'FL', 'TX', 'NY', 'CA'))
# dti
dti=st.sidebar.slider("Please choose DTI:",min_value=0.1, max_value=100.1,step=0.1)
# mths_since_recent_ing
mths_since_recent_inq=st.sidebar.slider("Please choose your mths_since_recent_inq:",min_value=1, max_value=25,step=1)
# bc_open_to_buy
bc_open_to_buy =st.sidebar.slider("Please choose your bc_open_to_buy:", min_value=0, max_value=700000,step=100)
# num_op_rev_tl
num_op_rev_tl=st.sidebar.slider("Please choose num_op_rev_tl:",min_value=1, max_value=50,step=1)
# home_ownership 
home_ownership = st.sidebar.radio("Select Home Ownership: ", ('MORTGAGE','OWN','RENT','OTHERS'))


## processing data user
def preprocess(loan_amnt, term, home_ownership, emp_length, annual_inc, dti, mths_since_recent_inq, addr_state, bc_open_to_buy, num_op_rev_tl):
    # Pre-processing user input
    user_input_dict={'loan_amnt':[loan_amnt], 'term':[term],
                    #   'sub_grade':[sub_grade], 
                      'home_ownership':[home_ownership], 
                      'emp_length':[emp_length], 'annual_inc':[annual_inc], 'dti':[dti],
                'mths_since_recent_inq':[mths_since_recent_inq], 
                # 'revol_util':[revol_util], 
                'addr_state':[addr_state],
                'bc_open_to_buy':[bc_open_to_buy],
                'num_op_rev_tl':[num_op_rev_tl]}     
    user_input=pd.DataFrame(data=user_input_dict)
    #user_input=np.array(user_input)
    #user_input=user_input.reshape(1,-1)
    cleaner_type = {"term": {" 36 months": 1.0, " 60 months": 2.0},
    "addr_state": map_addr_state,
    "emp_length": {"< 1 year": 0.0, '1 year': 1.0, '2 years': 2.0, '3 years': 3.0, '4 years': 4.0,
    '5 years': 5.0, '6 years': 6.0, '7 years': 7.0, '8 years': 8.0, '9 years': 9.0,
    '10+ years': 10.0 }
    }
    user_input = user_input.replace(cleaner_type)
    user_input.home_ownership.replace({"ANY":"OTHERS","OTHER":"OTHERS","NONE":"OTHERS"},inplace=True)
    user_input = pd.get_dummies(user_input, 
                      columns=['home_ownership'],
                      drop_first=False, 
                      prefix ='home_ownership',
                      prefix_sep='_',
                      dtype=int)
    # Indicate all columns used to train 
    l1 = user_input.columns.tolist()
    l2 = features_used_to_train

    columns_to_add = list(set(l2)-set(l1))
    # Adicionando as novas colunas com valores padrão NaN
    for col in columns_to_add:
        user_input[col] = 0

    user_input = user_input[l2]
    # Normalization 
    # Selecting Numeric Features
    lista_vars_numericas = list(
      metadata[((metadata.Level  == 'ordinal')|(metadata.Level == 'interval')) & (metadata.Role == 'input')]
      ['Features'])
    user_input[lista_vars_numericas] = (user_input[lista_vars_numericas].fillna(mean_vars_num)).astype(float)
    ## Standarlization
    # Use the loaded scaler to transform the new data
    scaled_features_new = loaded_scaler.transform(user_input[lista_vars_numericas])
    user_input[lista_vars_numericas] = scaled_features_new
    return user_input

#user_input=preprocess
user_input=preprocess(loan_amnt, term, home_ownership, emp_length, annual_inc, dti, mths_since_recent_inq,addr_state, bc_open_to_buy, num_op_rev_tl)



#predict button

btn_predict = st.sidebar.button("Predict")

if btn_predict:
    pred = model.predict_proba(user_input)[:, 1]

    if threshold_medium_risk_decision <= pred[0] < threshold_high_risk_decision:
        st.error(f'Warning! The applicant has has score = {np.round(pred[0]*1000,2)} It represeents medium risk to not pay the loan back!')
    if pred[0] >= threshold_high_risk_decision:
        st.error(f'Warning! The applicant has has score = {np.round(pred[0]*1000,2)} It represeents high risk to not pay the loan back!')
    if pred[0] < threshold_medium_risk_decision:
        st.success(f'It is green! The aplicant has score = {np.round(pred[0]*1000,2)} It represeents a high probability to pay the loan back!')


    st.subheader('Credit Risk Policy for Loan.')

    def highlight_rows(s):
        if s['Risk Bands'] == 'High':
            return ['background-color: red'] * len(s)
        elif s['Risk Bands'] == 'Medium High':
            return ['background-color: yellow'] * len(s)
        elif s['Risk Bands'] == 'Low':
            return ['background-color: green'] * len(s)
        elif s['Risk Bands'] == 'Very Low':
            return ['background-color: blue'] * len(s)
        else:
            return ['background-color: white'] * len(s)


    st.dataframe(pol.style.apply(highlight_rows, axis=1))            

    
    #prepare test set for shap explainability
    loans = pd.read_csv("input/Train.csv")
    loans.set_index('id',inplace=True)
    
    X = loans.drop(columns=['target'])
    y = loans[['target']]
    y_ravel = y.values.ravel()

    st.subheader('Result Interpretability - Applicant Level')
    shap.initjs()
    explainer = shap.Explainer(model, X)
    shap_values = explainer(user_input)
    fig = shap.plots.bar(shap_values[0])
    st.pyplot(fig)

    st.subheader('Model Interpretability - Overall')

    # Exibe a figura salva no Streamlit
    st.image('images/shap_plot.png', caption='SHAP Plot', use_column_width=True)
    
    # st.write(""" In this chart blue and red mean the feature value, e.g. annual income blue is a smaller value e.g. 40K USD,
    # and red is a higher value e.g. 100K USD. The width of the bars represents the number of observations on a certain feature value,
    # for example with the annual_inc feature we can see that most of the applicants are within the lower-income or blue area. And on axis x negative SHAP
    # values represent applicants that are likely to churn and the positive values on the right side represent applicants that are likely to pay the loan back.
    # What we are learning from this chart is that features such as annual_inc and sub_grade are the most impactful features driving the outcome prediction.
    # The higher the salary is, or the lower the subgrade is, the more likely the applicant to pay the loan back and vice versa, which makes total sense in our case.
    # """)

