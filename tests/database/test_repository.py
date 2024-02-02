import pytest

from waffle_dough.database.repository import (
    annotation_repository,
    category_repository,
    image_repository,
)
from waffle_dough.exception.database_exception import *
from waffle_dough.field import (
    AnnotationInfo,
    CategoryInfo,
    ImageInfo,
    UpdateAnnotationInfo,
    UpdateCategoryInfo,
    UpdateImageInfo,
)


@pytest.fixture
def db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    from waffle_dough.database.model import Base

    engine = create_engine(
        f"sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return Session()


def test_crud(db):
    # Create
    ## create image
    image_info = ImageInfo.agnostic(
        original_file_name="test.jpg",
        width=100,
        height=100,
    )
    image = image_repository.create(db, image_info)[0]
    assert len(image_repository.get_multi(db)) == 1
    assert image_repository.get(db, image.id) == image

    ## create category
    category_info = CategoryInfo.classification(
        name="test",
    )
    category = category_repository.create(db, category_info)[0]
    assert len(category_repository.get_multi(db)) == 1

    ## create annotation
    annotation_info = AnnotationInfo.classification(
        image_id=image.id,
        category_id=category.id,
    )
    annotation = annotation_repository.create(db, annotation_info)[0]
    assert len(annotation_repository.get_multi(db)) == 1

    annotation_info = AnnotationInfo.classification(
        image_id="some_random_non_existing_id",
        category_id="some_random_non_existing_id",
    )
    with pytest.raises(DatabaseNotFoundError):
        annotation_repository.create(db, annotation_info)

    annotation_info = AnnotationInfo.object_detection(
        image_id="some_random_non_existing_id",
        category_id="some_random_non_existing_id",
        bbox=[0, 0, 1, 1],
    )
    with pytest.raises(DatabaseNotFoundError):
        annotation_repository.create(db, annotation_info)

    annotation_info = AnnotationInfo.object_detection(
        image_id=image.id,
        category_id=category.id,
        bbox=[0, 0, 1, 1],
    )
    with pytest.raises(DatabaseConstraintError):
        annotation_repository.create(db, annotation_info)

    # Cascade delete
    category_repository.remove(db, category.id)
    assert len(annotation_repository.get_multi(db)) == 0

    category_info = CategoryInfo.classification(
        name="test",
    )
    category = category_repository.create(db, category_info)[0]
    annotation_info = AnnotationInfo.classification(
        image_id=image.id,
        category_id=category.id,
    )
    annotation = annotation_repository.create(db, annotation_info)[0]
    image_repository.remove(db, image.id)
    assert len(annotation_repository.get_multi(db)) == 0

    # Update
    ## update image
    image = image_repository.create(db, image_info)[0]
    assert image.split == "unset"

    update_image_info = UpdateImageInfo(split="train")
    image = image_repository.update(db, image.id, update_image_info)[0]
    assert image.split == "train"

    ## update category
    update_category_info = UpdateCategoryInfo(name="test2")
    category = category_repository.update(db, category.id, update_category_info)[0]
    assert category.name == "test2"

    ## update annotation
    annotation = annotation_repository.create(db, annotation_info)[0]
    update_annotation_info = UpdateAnnotationInfo.classification(category_id=category.id)
    annotation = annotation_repository.update(db, annotation.id, update_annotation_info)[0]
    assert annotation.category_id == category.id
