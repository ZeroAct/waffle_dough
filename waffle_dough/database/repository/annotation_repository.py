from waffle_dough.database.model import Annotation
from waffle_dough.database.repository.base_repository import CRUDBase
from waffle_dough.field import AnnotationInfo, UpdateAnnotationInfo


class AnnotationRepository(CRUDBase[Annotation, AnnotationInfo, UpdateAnnotationInfo]):
    pass


annotation_repository = AnnotationRepository(Annotation)
