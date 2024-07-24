from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_convert_file():
    # Test file upload
    with open("test.pdf", "rb") as file:
        response = client.post(
            "/convert",
            files={"file": ("test.pdf", file, "application/pdf")}
        )
    assert response.status_code == 200
    assert response.json() == {"success": True, "message": "File uploaded successfully"}

def test_convert_file_unsupported_format():
    # Test file upload with unsupported format
    with open("test.txt", "rb") as file:
        response = client.post(
            "/convert",
            files={"file": ("test.txt", file, "text/plain")}
        )
    assert response.status_code == 200
    assert response.json() == {"success": False, "message": "Unsupported file format"}

def test_ask_pdf():
    # First upload the test PDF
    with open("test.pdf", "rb") as file:
        client.post(
            "/convert",
            files={"file": ("test.pdf", file, "application/pdf")}
        )
    
    # Test asking a question
    response = client.post(
        "/ask_pdf",
        data={"question": "What is this document about?"}
    )
    assert response.status_code == 200
    assert "response" in response.json()

def test_ask_pdf_no_file_uploaded():
    # Test asking a question when no file is uploaded
    response = client.post(
        "/ask_pdf",
        data={"question": "What is this document about?"}
    )
    assert response.status_code == 500  # or appropriate error status code
    assert "response" not in response.json()
