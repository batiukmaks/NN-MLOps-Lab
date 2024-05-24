import pytest
from main import app
from model.dnn import DNN

@pytest.fixture
def client():
    return app.test_client()

def test_correct_input(client):
    response = client.post('/predict', json={'data': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    assert response.status_code == 200  # success

def test_incorrect_input(client):
    response = client.post('/predict', json={'data': [1, 2, 3, 4, 5]})
    assert response.status_code == 400  # if your app returns a 400 error for bad input