from sqlalchemy import JSON, Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .base import Base


class Image(Base):
    __tablename__ = "images"

    id = Column(String, primary_key=True)
    task = Column(String)
    ext = Column(String)
    width = Column(Integer)
    height = Column(Integer)
    original_file_name = Column(String)
    date_captured = Column(String)
    split = Column(String)

    annotations = relationship("Annotation", back_populates="image", cascade="all, delete-orphan")
