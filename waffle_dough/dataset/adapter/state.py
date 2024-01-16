from pydantic import BaseModel


class AdapterState(BaseModel):
    status: str
