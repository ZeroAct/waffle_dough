from pathlib import Path
from typing import Union

from pycocotools.coco import COCO

from waffle_dough.exception import *
from waffle_dough.field import AnnotationInfo, CategoryInfo, ImageInfo
from waffle_dough.math.segmentation import convert_segmentation
from waffle_dough.type import SegmentationType, TaskType

from .base_adapter import BaseAdapter


class CocoAdapter(BaseAdapter):
    def __init__(
        self,
        images: dict[str, ImageInfo] = None,
        annotations: dict[str, AnnotationInfo] = None,
        categories: dict[str, CategoryInfo] = None,
        task: Union[str, TaskType] = TaskType.OBJECT_DETECTION,
        callbacks: list[callable] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, callbacks=callbacks, **kwargs)

        self.images: dict[str, ImageInfo] = images or {}
        self.annotations: dict[str, AnnotationInfo] = annotations or {}
        self.categories: dict[str, CategoryInfo] = categories or {}

        if task not in [
            TaskType.OBJECT_DETECTION,
            TaskType.INSTANCE_SEGMENTATION,
            TaskType.SEMANTIC_SEGMENTATION,
        ]:
            raise DatasetAdapterTaskError(f"Task {task} is not supported by COCO format.")
        self.task: Union[str, TaskType] = task.lower()

    @classmethod
    def from_target(
        cls,
        coco_dataset: Union[str, dict],
        task: Union[str, TaskType] = TaskType.OBJECT_DETECTION,
    ) -> "CocoAdapter":
        adapter = cls(task=task)

        if isinstance(coco_dataset, (str, Path)):
            coco = COCO(coco_dataset)
        elif isinstance(coco_dataset, dict):
            coco = COCO()
            coco.dataset = coco_dataset
            coco.createIndex()

        categories = {}
        coco_cat_id_to_new_cat_id = {}
        for cat_id, cat in coco.cats.items():
            cat_id = cat.pop("id", None) or cat_id
            cat = CategoryInfo.from_dict(
                task=task,
                d=cat,
            )
            coco_cat_id_to_new_cat_id[cat_id] = cat.id
            categories[cat.id] = cat

        images = {}
        coco_img_id_to_new_img_id = {}
        for img_id, img in coco.imgs.items():
            img = ImageInfo(
                ext=Path(img["file_name"]).suffix,
                width=img["width"],
                height=img["height"],
                original_file_name=img["file_name"],
                date_captured=img.get("date_captured", None),
                task=TaskType.AGNOSTIC,
            )
            coco_img_id_to_new_img_id[img_id] = img.id
            images[img.id] = img

        annotations = {}
        for ann in coco.anns.values():
            img = images[coco_img_id_to_new_img_id[ann["image_id"]]]
            cat = coco_cat_id_to_new_cat_id[ann["category_id"]]

            W = img.width
            H = img.height

            if task == TaskType.OBJECT_DETECTION:
                x1, y1, w, h = ann["bbox"]
                ann = AnnotationInfo.object_detection(
                    image_id=img.id,
                    category_id=cat,
                    bbox=[x1 / W, y1 / H, w / W, h / H],
                    iscrowd=getattr(ann, "iscrowd", None),
                    score=getattr(ann, "score", None),
                )
            elif task in [TaskType.INSTANCE_SEGMENTATION, TaskType.SEMANTIC_SEGMENTATION]:
                segmentation = ann["segmentation"]
                if isinstance(segmentation, dict):
                    segmentation = convert_segmentation(
                        segmentation, SegmentationType.RLE, SegmentationType.POLYGON
                    )

                for i, segmentation_ in enumerate(segmentation):
                    for j, point in enumerate(segmentation_):
                        segmentation[i][j] = point / W if j % 2 == 0 else point / H

                ann = AnnotationInfo.instance_segmentation(
                    image_id=img.id,
                    category_id=cat,
                    segmentation=segmentation,
                    bbox=getattr(ann, "bbox", None),
                    iscrowd=getattr(ann, "iscrowd", None),
                    score=getattr(ann, "score", None),
                )

            annotations[ann.id] = ann

        adapter.images = images
        adapter.annotations = annotations
        adapter.categories = categories

        return adapter

    def to_target(self, image_ids: list[str] = None, category_ids: list[str] = None) -> dict:
        coco = {}

        coco["images"] = []
        coco["categories"] = []
        coco["annotations"] = []

        target_img_id_to_coco_img_id = {}
        for img_id in image_ids or self.images.keys():
            img = self.images[img_id]
            coco_img_id = len(coco["images"]) + 1
            target_img_id_to_coco_img_id[img_id] = coco_img_id

            coco["images"].append(
                {
                    "id": coco_img_id,
                    "file_name": img.original_file_name,
                    "width": img.width,
                    "height": img.height,
                    "date_captured": img.date_captured,
                }
            )

        target_cat_id_to_coco_cat_id = {}
        for cat_id in category_ids or self.categories.keys():
            cat = self.categories[cat_id]
            coco_cat_id = len(coco["categories"]) + 1
            target_cat_id_to_coco_cat_id[cat_id] = coco_cat_id

            coco["categories"].append(
                {
                    "id": coco_cat_id,
                    "name": cat.name,
                    "supercategory": cat.supercategory,
                }
            )

        for ann_id, ann in self.annotations.items():
            img = self.images[ann.image_id]

            W = img.width
            H = img.height

            if self.task == TaskType.OBJECT_DETECTION:
                W = img.width
                H = img.height

                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": target_img_id_to_coco_img_id[ann.image_id],
                        "category_id": target_cat_id_to_coco_cat_id[ann.category_id],
                        "bbox": [ann.bbox[0] * W, ann.bbox[1] * H, ann.bbox[2] * W, ann.bbox[3] * H],
                        "iscrowd": ann.iscrowd,
                        "score": ann.score,
                    }
                )
            elif self.task in [TaskType.INSTANCE_SEGMENTATION, TaskType.SEMANTIC_SEGMENTATION]:
                segmentation = ann.segmentation

                for i, segmentation_ in enumerate(segmentation):
                    for j, point in enumerate(segmentation_):
                        segmentation[i][j] = point * W if j % 2 == 0 else point * H

                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": target_img_id_to_coco_img_id[ann.image_id],
                        "category_id": target_cat_id_to_coco_cat_id[ann.category_id],
                        "bbox": [ann.bbox[0] * W, ann.bbox[1] * H, ann.bbox[2] * W, ann.bbox[3] * H],
                        "segmentation": segmentation,
                        "iscrowd": ann.iscrowd,
                        "score": ann.score,
                    }
                )

        return coco
