import requests
import os

BASE_URL = "http://127.0.0.1:5000/api"

def test_flow():
    # 1. Signup
    print("Testing Signup...")
    signup_data = {"username": "testuser_v2", "password": "password123"}
    try:
        response = requests.post(f"{BASE_URL}/auth/signup", json=signup_data)
        print(f"Signup: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Signup failed: {e}")

    # 2. Login
    print("\nTesting Login...")
    login_data = {"username": "testuser_v2", "password": "password123"}
    token = None
    try:
        response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
        print(f"Login: {response.status_code} - {response.json()}")
        if response.status_code == 200:
            token = response.json().get('access_token')
    except Exception as e:
        print(f"Login failed: {e}")

    if not token:
        print("Stopping tests due to login failure.")
        return

    headers = {"Authorization": f"Bearer {token}"}

    # 3. Predict (Need a dummy image)
    print("\nTesting Predict...")
    # Create a dummy image
    with open("test_image.jpg", "wb") as f:
        f.write(os.urandom(1024)) # Random bytes

    files = {'image': open('test_image.jpg', 'rb')}
    try:
        response = requests.post(f"{BASE_URL}/predict", headers=headers, files=files)
        print(f"Predict: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"Predict failed: {e}")
    finally:
        files['image'].close()
        os.remove("test_image.jpg")

    # 4. History
    print("\nTesting History...")
    try:
        response = requests.get(f"{BASE_URL}/history", headers=headers)
        print(f"History: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"History failed: {e}")

if __name__ == "__main__":
    test_flow()
