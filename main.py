from fastapi import FastAPI
from pydantic import BaseModel

from api.embedding import router as embedding_router
from api.exception_handlers import register_exception_handlers
from api.user import router as user_router

app = FastAPI()
register_exception_handlers(app)


class Item(BaseModel):
    name: str
    description: str = None
    price: float
    tax: float = 10.5


app.include_router(user_router)
app.include_router(embedding_router)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


@app.post("/items")
def create_item(item: Item):
    print(item)
    return item


@app.get("/get_item/{item_id}")
def get_item(item_id: int):
    return {"item_id": item_id}
