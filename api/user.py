from fastapi import APIRouter

from api.exceptions import NotFoundError
from api.schemas.userSchema import Item

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/{user_id}")
def read_user(user_id: int, q: str | None = None):
    if user_id < 1:
        raise NotFoundError("User not found")
    return {"user_id": user_id, "q": q}


@router.post("/create")
def create_user(user: Item):
    return user
