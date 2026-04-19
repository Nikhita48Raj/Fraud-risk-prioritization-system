from pydantic import BaseModel
from typing import Optional


class TransactionInput(BaseModel):
    TransactionID: int
    TransactionDT: float
    TransactionAmt: float

    card1: Optional[float] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card5: Optional[float] = None

    addr1: Optional[float] = None
    dist1: Optional[float] = None

    P_emaildomain: Optional[str] = None