from typing import Any, Dict, List, Optional
import httpx


class UsersApiError(Exception):
    pass


class UsersApi:
    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def create_thread(self, identifier: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(f"{self.base_url}/threads/create/{identifier}")
            r.raise_for_status()
            return r.json()

    async def link_chainlit_thread(self, thread_id: int, chainlit_id: str) -> None:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.patch(f"{self.base_url}/threads/{thread_id}/chainlit/{chainlit_id}")
            r.raise_for_status()

    async def get_thread_by_chainlit(self, chainlit_id: str) -> Optional[int]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(f"{self.base_url}/threads/by_chainlit/{chainlit_id}")
            if r.status_code == 200:
                return r.json()["thread_id"]
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return None

    async def get_latest_thread(self, identifier: str) -> Optional[int]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(f"{self.base_url}/threads/{identifier}/latest")
            if r.status_code == 200:
                return r.json()["thread_id"]
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return None

    async def list_threads(self, identifier: str) -> List[Dict[str, Any]]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(f"{self.base_url}/threads/{identifier}")
            r.raise_for_status()
            return r.json()

    async def get_messages(self, thread_id: int) -> List[Dict[str, Any]]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(f"{self.base_url}/threads/messages/{thread_id}")
            r.raise_for_status()
            return r.json()

    async def create_message(self, thread_id: int, sender: str, message: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.post(
                f"{self.base_url}/threads/message/{thread_id}",
                params={"sender": sender},
                json={"message": message},
            )
            r.raise_for_status()
            return r.json()
