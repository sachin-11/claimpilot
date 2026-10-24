from datetime import date
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel

from app.schemas.base import ReadModel


class CustomerCreate(BaseModel):
    full_name: str
    phone: str | None = None
    email: str | None = None
    dob: date | None = None
    address: str | None = None


class CustomerRead(CustomerCreate, ReadModel):
    pass


class CustomerPolicyCreate(BaseModel):
    customer_id: UUID
    policy_id: UUID
    policy_number: str
    sum_insured: Decimal
    start_date: date
    end_date: date | None = None
    status: str = "active"


class CustomerPolicyRead(CustomerPolicyCreate, ReadModel):
    pass
