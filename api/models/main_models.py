from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    respuesta: str

class HealthResponse(BaseModel):
    status: str
    available_models: list
    message: str
