from unittest.mock import patch, MagicMock
import json
from backend.Services.gemini import gemini_func

@patch.dict("os.environ", {"HF_TOKEN": "fake-token"})
@patch("backend.Services.gemini.InferenceClient")
def test_gemini_func_simple(mock_client_class):
    sample_data = {"Age": 30, "Education": 2}

    mock_client = MagicMock()
    mock_client_class.return_value = mock_client

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(
            content='{"risk_assessment": "Test this function"}'
        ))
    ]
    mock_client.chat.completions.create.return_value = mock_response

    result = gemini_func(sample_data)

    data = json.loads(result)
    assert data["risk_assessment"] == "Test this function"
