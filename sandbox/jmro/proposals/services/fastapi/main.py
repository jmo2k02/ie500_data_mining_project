from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(
    title="Data Mining API", description="API for data mining project", version="1.0.0"
)


class Item(BaseModel):
    id: Optional[int] = None
    name: str
    description: Optional[str] = None


items_db = []


@app.get("/")
async def root():
    return {"message": "Welcome to the Data Mining API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/items", response_model=List[Item])
async def get_items():
    return items_db


@app.get("/items/{item_id}", response_model=Item)
async def get_item(item_id: int):
    for item in items_db:
        if item.id == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found")


@app.post("/items", response_model=Item)
async def create_item(item: Item):
    item.id = len(items_db) + 1
    items_db.append(item)
    return item


@app.put("/items/{item_id}", response_model=Item)
async def update_item(item_id: int, updated_item: Item):
    for idx, item in enumerate(items_db):
        if item.id == item_id:
            updated_item.id = item_id
            items_db[idx] = updated_item
            return updated_item
    raise HTTPException(status_code=404, detail="Item not found")


@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    for idx, item in enumerate(items_db):
        if item.id == item_id:
            items_db.pop(idx)
            return {"message": "Item deleted successfully"}
    raise HTTPException(status_code=404, detail="Item not found")
