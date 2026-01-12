import pytest
from backend.Services.my_model import predict


@pytest.fixture
def Data():
    Fake_data = {
        'Age': 50,
        'BusinessTravel': 'Travel_Rarely',
        'Department': 'Research & Development',
        'Education': 1,
        'EducationField': 'Life Sciences',
        'EnvironmentSatisfaction': 2,
        'JobInvolvement': 2,
        'JobLevel': 2,
        'JobRole': 'Research Scientist',
        'JobSatisfaction': 2,
        'MaritalStatus': 'Single',
        'MonthlyIncome': 5994,
        'OverTime': 'No',
        'PerformanceRating': 3,
        'TotalWorkingYears': 3,
        'TrainingTimesLastYear': 0,
        'WorkLifeBalance': 2,
        'YearsAtCompany': 3,
        'YearsInCurrentRole': 3,
        'YearsWithCurrManager': 3
    }
    return Fake_data

def test_predict(Data):
    prediction = predict(Data)
    assert prediction == 'yes'

    
