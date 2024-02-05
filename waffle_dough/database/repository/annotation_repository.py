from typing import Any, Union

from sqlalchemy.orm import Session

from waffle_dough.database.model import Annotation
from waffle_dough.database.repository.base_repository import CRUDBase
from waffle_dough.database.repository.category_repository import category_repository
from waffle_dough.database.repository.image_repository import image_repository
from waffle_dough.exception.database_exception import *
from waffle_dough.field import AnnotationInfo, UpdateAnnotationInfo


class AnnotationRepository(CRUDBase[Annotation, AnnotationInfo, UpdateAnnotationInfo]):
    def create(self, db: Session, obj_in: Union[AnnotationInfo, list[AnnotationInfo]]) -> Annotation:
        obj_ins = obj_in if isinstance(obj_in, list) else [obj_in]
        for obj_in in obj_ins:
            if obj_in.category_id is not None:
                category = category_repository.get(db, obj_in.category_id)
                if category is None:
                    raise DatabaseNotFoundError(f"Category with id {obj_in.category_id} not found")
                if category.task != obj_in.task:
                    raise DatabaseConstraintError(
                        f"Category task {category.task} and annotation task {obj_in.task} mismatch"
                    )

            if obj_in.image_id is not None:
                image = image_repository.get(db, obj_in.image_id)
                if image is None:
                    raise DatabaseNotFoundError(f"Image with id {obj_in.image_id} not found")

        return super().create(db, obj_ins)

    def update(
        self,
        db: Session,
        id: Union[str, list[str]],
        obj_in: Union[
            UpdateAnnotationInfo, dict[str, Any], list[Union[UpdateAnnotationInfo, dict[str, Any]]]
        ],
    ) -> list[Annotation]:
        ids = id if isinstance(id, list) else [id]
        obj_ins = obj_in if isinstance(obj_in, list) else [obj_in]

        db_objs = []
        for id, obj_in in zip(ids, obj_ins):
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

            if obj_in.category_id is not None:
                category = category_repository.get(db, obj_in.category_id)
                if category is None:
                    raise DatabaseNotFoundError(f"Category with id {obj_in.category_id} not found")

            db.add(obj)
            db_objs.append(obj)
        self.commit(db)
        return db_objs


annotation_repository = AnnotationRepository(Annotation)
