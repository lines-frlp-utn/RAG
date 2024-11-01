import requests
from app.config import conf
from pydantic import BaseModel
from enum import Enum

class Role(str, Enum):
    ADMIN = "ADMIN"
    CLIENTE = "CLIENTE"
class UserExistsDTOResponse(BaseModel):
    exists: bool
    role_name: Role | None

def user_exists(userIdentifier: str):
    print(f"user: {userIdentifier} ¿exists?")
    response = requests.get(
        f"{conf.USERS_API_URL}:{conf.USERS_API_PORT}/users/exists/{userIdentifier}",
    )
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
    else:
        data = response.json()
        user = UserExistsDTOResponse(**data)
        return user

def create_user(userIdentifier, role):
    print(f"userIdentifier: {userIdentifier}")
    print(f"role: {role}")
    response = requests.post(
        f"{conf.USERS_API_URL}:{conf.USERS_API_PORT}/users/create",
        json={
            "identifier": userIdentifier,
            "role_name": role,
        }
    )
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
    else:
        print(f"Request successful - user created successfully {response.status_code} - {response.text}")