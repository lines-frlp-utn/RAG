import requests
from app.config import conf

class UserExistsDTOResponse:
    def __init__(self, exists: bool):
        self.exists = exists

def user_exists(userIdentifier: str):
    print(f"user: {userIdentifier} ¿exists?")
    response = requests.get(
        f"{conf.USERS_API_URL}:{conf.USERS_API_URL}users/exists/{userIdentifier}",
    )
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
    else:
        print(f"Request successful: {response.status_code} - {response.text}")
        response.text
        return response.text

def create_user(userIdentifier, role):
    print(f"userIdentifier: {userIdentifier}")
    print(f"role: {role}")
    response = requests.post(
        f"{conf.USERS_API_URL}:{conf.USERS_API_URL}/users",
        json={
            "identifier": userIdentifier,
            "role": role,
        }
    )
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
    else:
        print(f"Request successful - user created successfully {response.status_code} - {response.text}")

def get_user_role(userIdentifier):
    print(f"userIdentifier: {userIdentifier}")
    response = requests.get(
        f"{conf.USERS_API_URL}:{conf.USERS_API_URL}/users/role/{userIdentifier}",
    )
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
    else:
        print(f"Request successful {response.status_code} - {response.text}")
    return response.text