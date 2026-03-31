# FastAPI Service

This is a FastAPI application for the data mining project.

## Installation

Make sure FastAPI and uvicorn are installed:

```bash
uv add fastapi uvicorn
```

## Running the API

Start the development server:

```bash
uv run uvicorn services.fastapi.main:app --reload
```

Or from the services/fastapi directory:

```bash
cd services/fastapi
uv run uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check
- `GET /items` - Get all items
- `GET /items/{item_id}` - Get specific item
- `POST /items` - Create new item
- `PUT /items/{item_id}` - Update item
- `DELETE /items/{item_id}` - Delete item
