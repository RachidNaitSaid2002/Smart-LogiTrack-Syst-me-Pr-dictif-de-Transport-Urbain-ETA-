import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

@pytest.fixture
def Data():
    FakeData = {
        "trip_distance": 0,
        "pickup_hour": 0,
        "pickup_month": 0,
        "Airport_fee": 0,
        "pickup_day_week": 0,
        "RateCodeID": 0,
        "fare_amount": 0
    }
    return FakeData

def test_login():

    response = client.post(
        "/login",
        json={
            "email": "string",
            "hashed_password": "string"
        }
    )
    
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_predict(Data):

    login_response = client.post(
        "/login",
        json={
            "email": "string",
            "hashed_password": "string"
        }
    )

    access_token = login_response.json().get("access_token")
    headers = {"Authorization": f"Bearer {access_token}"}

    response = client.post(
        "/predictions",
        json=Data,
        headers=headers
    )
    assert response.status_code == 200
    assert "predicted_duration" in response.json()