from enum import Enum

import requests
from app.config import conf
from pydantic import BaseModel


class Role(str, Enum):
    ADMIN = "ADMIN"
    CLIENTE = "CLIENTE"


class UserExistsDTOResponse(BaseModel):
    exists: bool
    role_name: Role | None


def user_exists(userIdentifier: str, password: str):
    print(f"user: {userIdentifier} ¿exists?")
    try:
        response = requests.get(
            f"{conf.USERS_API_URL}:{conf.USERS_API_PORT}/users/login/{userIdentifier}?password={password}",
        )
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            return UserExistsDTOResponse(exists=False, role_name=None)
        else:
            data = response.json()
            user = UserExistsDTOResponse(**data)
            return user
    except Exception as e:
        print(f"Exception in user_exists: {e}")
        return UserExistsDTOResponse(exists=False, role_name=None)


def create_user(userIdentifier, role, password, providerId="", email="", picture="", name=""):
    print(f"userIdentifier: {userIdentifier}")
    print(f"role: {role}")
    response = requests.post(
        f"{conf.USERS_API_URL}:{conf.USERS_API_PORT}/users/create",
        json={
            "name": name,
            "identifier": userIdentifier,
            "role_name": role,
            "password": password or "",
            "auth_provider": providerId,
            "email": email or "",
            "picture": picture,
        },
    )
    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    else:
        print(
            f"Request successful - user created successfully {response.status_code} - {response.text}"
        )
        return response.text
