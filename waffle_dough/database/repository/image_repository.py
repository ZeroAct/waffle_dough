from waffle_dough.database.model import Image
from waffle_dough.database.repository.base_repository import CRUDBase
from waffle_dough.field import ImageInfo, UpdateImageInfo


class ImageRepository(CRUDBase[Image, ImageInfo, UpdateImageInfo]):
    pass


image_repository = ImageRepository(Image)
