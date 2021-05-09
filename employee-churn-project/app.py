# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib

app = Flask(__name__)

#model = joblib.load("flight_rf.pkl")
model = joblib.load("Emp_attrition_prediction.pkl")



df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    global df
    
#   input_features = [int(x) for x in request.form.values()]
    #input_features = [x for x in request.form.values()]
    #print(request.form.values())
    #capturing features
    EMPLOYEE_ID = request.form['EMPLOYEE_ID']
    
    #region continuous variables
    Age = float(request.form['AGE'])
    
    DistanceFromHome = float(request.form['DISTANCE_FROM_HOME'])
    MonthlyIncome = float(request.form['MONTHLY_INCOME'])
    TotalWorkingYears = float(request.form['TOTAL_WORKING_YEARS'])
    YearsAtCompany = float(request.form['YEARS_AT_COMPANY']) 
    
#    scaler = pickle.load(open('scaler.pkl','rb'))
 #   scaler.fit_transform(data[['MonthlyIncome']])
    #end region continuous variables 
    
    
    print(Age)
    
    #region continuous variables
    BUSINESSTRAVEL_TRAVEL = request.form['BUSINESSTRAVEL_TRAVEL']
    BusinessTravel_Travel_Rarely=0
    BusinessTravel_Travel_Frequently=0
    
    DEPARTMENT = request.form['DEPARTMENT']    
    Department_ResearchDevelopment=0
    Department_Sales=0
    
    
    EDUCATION_FIELD = request.form['EDUCATION_FIELD']
    EducationField_LifeSciences=0
    EducationField_Marketing=0
    EducationField_Medical=0
    EducationField_Other=0
    EducationField_TechnicalDegree=0
    
    GENDER = request.form['GENDER']
    Gender_Male=0
    Gender_Female=0		
    
    
    JOBROLE = request.form['JOBROLE']
    JobRole_HumanResources=0
    JobRole_LaboratoryTechnician=0
    JobRole_Manager=0
    JobRole_ManufacturingDirector=0
    JobRole_ResearchDirector=0
    JobRole_ResearchScientist=0
    JobRole_SalesExecutive=0
    JobRole_SalesRepresentative=0
    
    MARITALSTATUS = request.form['MARITALSTATUS']
    MaritalStatus_Married=0
    MaritalStatus_Single=0
    
    OVERTIME = request.form['OVERTIME']
    OverTime_Yes=0
    
    EDUCATION = request.form['EDUCATION']
    Education_2=0
    Education_3=0
    Education_4=0
    Education_5=0
    
    
    JOB_LEVEL = request.form['JOB_LEVEL']
    JobLevel_2=0
    JobLevel_3=0
    JobLevel_4=0
    JobLevel_5=0
    
    JOB_SATISFACTION = request.form['JOB_SATISFACTION']
    JobSatisfaction_2=0
    JobSatisfaction_3=0
    JobSatisfaction_4=0
    
    COMPANIES_WROKED = request.form['COMPANIES_WROKED']
    NumCompaniesWorked_1=0
    NumCompaniesWorked_2=0
    NumCompaniesWorked_3=0
    NumCompaniesWorked_4=0
    NumCompaniesWorked_5=0
    NumCompaniesWorked_6=0
    NumCompaniesWorked_7=0
    NumCompaniesWorked_8=0
    NumCompaniesWorked_9=0
    
    PERFORMANCE_RATE = request.form['PERFORMANCE_RATE']   
    PerformanceRating_4=0
    
    WORK_LIFE_BALANCE = request.form['WORK_LIFE_BALANCE'] 
    WorkLifeBalance_2=0
    WorkLifeBalance_3=0
    WorkLifeBalance_4=0
    
    YEARS_IN_CURRENT_ROLE = request.form['YEARS_IN_CURRENT_ROLE']
    YearsInCurrentRole_1=0
    YearsInCurrentRole_2=0
    YearsInCurrentRole_3=0
    YearsInCurrentRole_4=0
    YearsInCurrentRole_5=0
    YearsInCurrentRole_6=0
    YearsInCurrentRole_7=0
    YearsInCurrentRole_8=0
    YearsInCurrentRole_9=0
    YearsInCurrentRole_10=0
    YearsInCurrentRole_11=0
    YearsInCurrentRole_12=0
    YearsInCurrentRole_13=0
    YearsInCurrentRole_14=0
    YearsInCurrentRole_15=0
    YearsInCurrentRole_16=0
    YearsInCurrentRole_17=0
    YearsInCurrentRole_18=0
    
    YEARS_SINCE_LAST_PROMOTION = request.form['YEARS_SINCE_LAST_PROMOTION'] 
    YearsSinceLastPromotion_1=0
    YearsSinceLastPromotion_2=0
    YearsSinceLastPromotion_3=0
    YearsSinceLastPromotion_4=0
    YearsSinceLastPromotion_5=0
    YearsSinceLastPromotion_6=0
    YearsSinceLastPromotion_7=0
    YearsSinceLastPromotion_8=0
    YearsSinceLastPromotion_9=0
    YearsSinceLastPromotion_10=0
    YearsSinceLastPromotion_11=0
    YearsSinceLastPromotion_12=0
    YearsSinceLastPromotion_13=0
    YearsSinceLastPromotion_14=0
    YearsSinceLastPromotion_15=0
    #end region continuous variables
    
    
    
    
    
    
   
    
    
    
    
   
    
       
    
    
    
        
    
    
    
    #features_value = np.array(input_features)
    
    #validate input hours
    #if input_features[0] <0 or input_features[0] >24:
     #   return render_template('home.html', prediction_text='Please enter valid hours between 1 to 24 if you live on the Earth')
        

    #print("Here",features_value)
    #scaler = pickle.load(open("scale.pickle","rb"))
    #scaled = scaler.fit_transform(data[['MonthlyIncome','HourlyRate']])
    features_value=[Age,DistanceFromHome,MonthlyIncome,TotalWorkingYears,YearsAtCompany,BusinessTravel_Travel_Frequently,
                    BusinessTravel_Travel_Rarely,Department_ResearchDevelopment,Department_Sales,EducationField_LifeSciences,
                    EducationField_Marketing,EducationField_Medical,EducationField_Other,EducationField_TechnicalDegree,
                    Gender_Male,JobRole_HumanResources,JobRole_LaboratoryTechnician,JobRole_Manager,JobRole_ManufacturingDirector,
                    JobRole_ResearchDirector,JobRole_ResearchScientist,JobRole_SalesExecutive,JobRole_SalesRepresentative,
                    MaritalStatus_Married,MaritalStatus_Single,OverTime_Yes,Education_2,Education_3,Education_4,Education_5,
                    JobLevel_2,JobLevel_3,JobLevel_4,JobLevel_5,JobSatisfaction_2,JobSatisfaction_3,JobSatisfaction_4,
                    NumCompaniesWorked_1,NumCompaniesWorked_2,NumCompaniesWorked_3,NumCompaniesWorked_4,NumCompaniesWorked_5,
                    NumCompaniesWorked_6,NumCompaniesWorked_7,NumCompaniesWorked_8,NumCompaniesWorked_9,PerformanceRating_4,WorkLifeBalance_2,
                    WorkLifeBalance_3,WorkLifeBalance_4,YearsInCurrentRole_1,YearsInCurrentRole_2,YearsInCurrentRole_3,YearsInCurrentRole_4,
                    YearsInCurrentRole_5,YearsInCurrentRole_6,YearsInCurrentRole_7,YearsInCurrentRole_8,YearsInCurrentRole_9,YearsInCurrentRole_10,
                    YearsInCurrentRole_11,YearsInCurrentRole_12,YearsInCurrentRole_13,YearsInCurrentRole_14,YearsInCurrentRole_15,
                    YearsInCurrentRole_16,YearsInCurrentRole_17,YearsInCurrentRole_18,YearsSinceLastPromotion_1,YearsSinceLastPromotion_2,
                    YearsSinceLastPromotion_3,YearsSinceLastPromotion_4,YearsSinceLastPromotion_5,YearsSinceLastPromotion_6,
                    YearsSinceLastPromotion_7,YearsSinceLastPromotion_8,YearsSinceLastPromotion_9,YearsSinceLastPromotion_10,
                    YearsSinceLastPromotion_11,YearsSinceLastPromotion_12,YearsSinceLastPromotion_13,YearsSinceLastPromotion_14,
                    YearsSinceLastPromotion_15
                    ]
    #features_value = features_value.astype(np.float64)                    
    print(features_value)
    #output = model.predict([features_value])[0][0].round(2)
    output = model.predict([features_value])
    print(output)

    # input and predicted value store in df then save in csv file
    #df= pd.concat([df,pd.DataFrame({'Employee data':features_value,'Predicted Output':[output]})],ignore_index=True)
    #print(df)   
    #df.to_csv('smp_data_from_app.csv')

    pred=""
    if output[0]==0:
        pred ='Employee may look for job change.'
    else:
        pred ='Employee may stay for longer association.'    
    
    return render_template('home.html', prediction_text='{}'.format(pred))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8082)
    