import os
import json
import asyncpg
import chainlit as cl
from enum import Enum
from typing import Optional, Dict, Any
from app.config import conf

class Role(str, Enum):
    ADMIN = "ADMIN"
    CLIENT = "CLIENT"


async def select_user_row(identifier: str) -> Optional[Dict[str, Any]]:
    connection = await asyncpg.connect(conf.DATABASE_URL)
    await connection.set_type_codec(
        'jsonb', 
        encoder=json.dumps,
        decoder=json.loads,
        schema='pg_catalog'
    )

    try:
        row = await connection.fetchrow(
            'SELECT "id","identifier","metadata" FROM "User" WHERE "identifier"=$1;', 
            identifier
        )
        if row:
            return dict(row)
        
        return None
        
    finally:
        await connection.close()


async def get_user(identifier: str) -> Optional[Dict[str, Any]]:
    row = await select_user_row(identifier)
    if row:
        metadata = row.get("metadata")
        return metadata if isinstance(metadata, dict) else {}
    return None

async def update_user(identifier: str, new_metadata: Dict[str, Any]) -> bool:
    connection = await asyncpg.connect(conf.DATABASE_URL)
    try:
        result = await connection.execute(
            'UPDATE "User" SET metadata = $1 WHERE identifier = $2',
            json.dumps(new_metadata), identifier
        )
        return result == "UPDATE 1"
    
    except Exception:
        return False
    
    finally:
        await connection.close()

async def add_password(identifier: str, password: str) -> bool:
    metadata = await get_user(identifier)

    if not metadata:
        return False
    
    if "password" in metadata:
        return True
    
    metadata["password"] = password
    success = await update_user(identifier, metadata)
    return success


async def authenticate(username: str, password: str) -> cl.User | None:
    if not username or not password:
        return None

    try:
        metadata = await get_user(username)

        if metadata:
            stored_password = metadata.get("password")
            
            if stored_password is None:
                success = await add_password(username, password)
                if success:
                    metadata["password"] = password  
                    stored_password = password
                else:
                    return None

            if stored_password != password:
                return None

            return cl.User(
                identifier=username,
                metadata={
                    **metadata,
                    "provider": metadata.get("provider", "credentials"),
                    "display_name": metadata.get("display_name", username),
                    "role": metadata.get("role", Role.CLIENT.value),
                },
            )

        new_metadata = {
            "provider": "credentials",
            "display_name": username,
            "role": Role.CLIENT.value,
            "password": password,  
            "auth_provider": "credentials"
        }
        
        return cl.User(
            identifier=username, 
            metadata=new_metadata
        )

    except Exception:
        return None
