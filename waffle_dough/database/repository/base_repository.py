from enum import Enum
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel
from sqlalchemy import orm
from sqlalchemy.orm import Session

from waffle_dough.database.model import Base
from waffle_dough.exception.database_exception import *

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class Query:
    class OrderBy(str, Enum):
        ASC = "ASCENDING"
        DESC = "DESCENDING"


class CRUDBase(Generic[ModelType, CreateSchemaType, UpdateSchemaType]):
    def __init__(self, model: Type[ModelType]):
        """
        CRUD object with default methods to
        Create, Read, Update, Delete (CRUD).

        **Parameters**

        * `model`: A SQLAlchemy model class
        * `schema`: A Pydantic model (schema) class
        """
        self.model = model

    def commit(self, db: Session):
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            raise e

    def get(self, db: Session, id: Any) -> Optional[ModelType]:
        return db.query(self.model).filter(self.model.id == str(id)).first()

    def get_multi(
        self,
        db: Session,
        filter_by: dict[str, Any] = {},
        filter_in: dict[str, Optional[list]] = {},
        filter_like: list[Tuple[str, str]] = [],
        order_by: List[Tuple[str, str]] = [],
        skip: int = None,
        limit: int = None,
    ) -> List[ModelType]:
        query = self._get_query(
            db=db,
            filter_by=filter_by,
            filter_in=filter_in,
            filter_like=filter_like,
            order_by=order_by,
            skip=skip,
            limit=limit,
        )
        return query.all()

    def get_count(
        self,
        db: Session,
        filter_by: dict[str, Any] = {},
        filter_in: dict[str, Optional[list]] = {},
        filter_like: list[Tuple[str, str]] = [],
        order_by: List[Tuple[str, str]] = [],
        skip: int = None,
        limit: int = None,
    ) -> int:
        query = self._get_query(
            db=db,
            filter_by=filter_by,
            filter_in=filter_in,
            filter_like=filter_like,
            order_by=order_by,
            skip=skip,
            limit=limit,
        )
        return query.count()

    def create(self, db: Session, obj_in: CreateSchemaType) -> ModelType:
        obj_in_data = obj_in.model_dump()
        db_obj = self.model(**obj_in_data)  # type: ignore
        db.add(db_obj)
        self.commit(db)
        db.refresh(db_obj)
        return db_obj

    def update(
        self,
        db: Session,
        id: str,
        obj_in: Union[UpdateSchemaType, Dict[str, Any]],
    ) -> ModelType:
        # jsonable_encoder sometimes return empty dict
        # even the model is not empty
        # obj_in_data = jsonable_encoder(db_obj)
        obj = db.get(self.model, id)
        if obj is None:
            raise DatabaseNotFoundError(f"Object with id {id} not found")
        obj_in_data = {col.name: str(getattr(obj, col.name)) for col in obj.__table__.columns}
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.model_dump(exclude_unset=True)
        for field in obj_in_data:
            if field in update_data and update_data[field] is not None:
                setattr(obj, field, update_data[field])
        db.add(obj)
        self.commit(db)
        db.refresh(obj)
        return obj

    def remove(self, db: Session, id: Any) -> ModelType:
        obj = db.get(self.model, id)
        if obj is None:
            raise DatabaseNotFoundError(f"[{self.__class__.__name__}] Object with id {id} not found")
        db.delete(obj)
        self.commit(db)
        return obj

    def remove_multi(self, db: Session) -> List[ModelType]:
        objs = db.query(self.model).all()
        db.query(self.model).delete()
        self.commit(db)
        return objs

    def remove_multi_by_query(
        self,
        db: Session,
        filter_by: dict[str, Any] = {},
        filter_in: dict[str, Optional[list]] = {},
        filter_like: list[Tuple[str, str]] = [],
        order_by: List[Tuple[str, str]] = [],
        skip: int = None,
        limit: int = None,
    ) -> list:
        query = self._get_query(
            db=db,
            filter_by=filter_by,
            filter_in=filter_in,
            filter_like=filter_like,
            order_by=order_by,
            skip=skip,
            limit=limit,
        )
        objs = [obj.id for obj in query.all()]
        query.delete()
        self.commit(db)
        return objs

    def _get_query(
        self,
        db: Session,
        filter_by: dict[str, Any] = {},
        filter_in: dict[str, Optional[list]] = {},
        filter_like: list[Tuple[str, str]] = [],
        order_by: List[Tuple[str, str]] = [],
        skip: int = None,
        limit: int = None,
    ) -> orm.Query:
        query = db.query(self.model)
        query = query.filter_by(**filter_by)
        for k, v in filter_in.items():
            if v is None:
                continue
            query = query.filter(getattr(self.model, k).in_(v))
        for k, v in filter_like:
            query = query.filter(getattr(self.model, k).like("%" + v + "%"))
        for col, order in order_by:
            if order == Query.OrderBy.ASC:
                query = query.order_by(getattr(self.model, col).asc())
            elif order == Query.OrderBy.DESC:
                query = query.order_by(getattr(self.model, col).desc())
        if skip:
            query = query.offset(skip)
        if limit:
            query = query.limit(limit)
        return query
