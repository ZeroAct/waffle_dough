from sqlalchemy import JSON, Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .base import Base


class Category(Base):
    __tablename__ = "categories"

    id = Column(String, primary_key=True)
    task = Column(String)
    name = Column(String)
    supercategory = Column(String)
    keypoints = Column(JSON)
    skeleton = Column(JSON)

    annotations = relationship("Annotation", back_populates="category", cascade="all, delete-orphan")
