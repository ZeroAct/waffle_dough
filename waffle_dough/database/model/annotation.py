from sqlalchemy import JSON, Boolean, Column, Double, ForeignKey, Integer, String
from sqlalchemy.orm import relationship

from .base import Base


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(String, primary_key=True)
    task = Column(String)
    image_id = Column(String, ForeignKey("images.id"))
    category_id = Column(String, ForeignKey("categories.id"))
    bbox = Column(JSON)
    segmentation = Column(JSON)
    area = Column(Double)
    keypoints = Column(JSON)
    num_keypoints = Column(Integer)
    caption = Column(String)
    value = Column(Double)
    iscrowd = Column(Integer)
    score = Column(Double)

    image = relationship("Image", back_populates="annotations")
    category = relationship("Category", back_populates="annotations")
