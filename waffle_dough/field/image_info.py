from waffle_utils.logger import datetime_now
from waffle_utils.validator import setter_type_validator

from .base_field import BaseField


class ImageInfo(BaseField):
    def __init__(
        self,
        image_id: int,
        file_name: str,
        width: int,
        height: int,
        original_file_name: str = None,
        date_captured: str = None,
    ):
        self.image_id = image_id
        self.file_name = file_name
        self.width = width
        self.height = height
        self.original_file_name = original_file_name
        self.date_captured = date_captured

    # properties
    @property
    def image_id(self) -> int:
        return self.__image_id

    @image_id.setter
    @setter_type_validator(int)
    def image_id(self, v):
        if v and v < 1:
            raise ValueError("id should be greater than 0.")
        self.__image_id = v

    @property
    def file_name(self) -> str:
        return self.__file_name

    @file_name.setter
    @setter_type_validator(str, strict=False)
    def file_name(self, v):
        self.__file_name = v

    @property
    def width(self) -> int:
        return self.__width

    @width.setter
    @setter_type_validator(int)
    def width(self, v):
        self.__width = v

    @property
    def height(self) -> int:
        return self.__height

    @height.setter
    @setter_type_validator(int)
    def height(self, v):
        self.__height = v

    @property
    def original_file_name(self) -> str:
        return self.__original_file_name

    @original_file_name.setter
    @setter_type_validator(str, strict=False)
    def original_file_name(self, v):
        self.__original_file_name = v or self.file_name

    @property
    def date_captured(self) -> str:
        return self.__date_captured

    @date_captured.setter
    @setter_type_validator(str)
    def date_captured(self, v):
        if v is None:
            self.__date_captured = datetime_now()
        else:
            self.__date_captured = v

    def __eq__(self, other) -> bool:
        if isinstance(other, ImageInfo):
            return self.file_name == other.file_name
        else:
            return False

    @classmethod
    def new(
        cls,
        image_id: int,
        file_name: str,
        width: int,
        height: int,
        original_file_name: str = None,
        date_captured: str = None,
    ) -> "ImageInfo":
        """Image Format

        Args:
            image_id (int): image id. natural number.
            file_name (str): file name. relative file path.
            width (int): image width.
            height (int): image height.
            original_file_name (str): original file name. relative file path.
            date_captured (str): date_captured string. "%Y-%m-%d %H:%M:%S"

        Returns:
            Image: image class
        """
        return cls(image_id, file_name, width, height, original_file_name, date_captured)

    def to_dict(self) -> dict:
        """Get Dictionary of Category

        Returns:
            dict: annotation dictionary.
        """

        cat = {
            "image_id": self.image_id,
            "file_name": self.file_name,
            "width": self.width,
            "height": self.height,
            "original_file_name": self.original_file_name,
            "date_captured": self.date_captured,
        }

        return cat
