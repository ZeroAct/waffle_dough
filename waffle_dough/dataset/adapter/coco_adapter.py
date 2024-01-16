import os
from pathlib import Path, PurePath
from typing import Union

from pycocotools.coco import COCO
from waffle_utils.file import io

from waffle_dough.exception import *
from waffle_dough.field import AnnotationInfo, CategoryInfo, ImageInfo
from waffle_dough.math.segmentation import convert_segmentation
from waffle_dough.type import SegmentationType, SplitType, TaskType

from .base_adapter import BaseAdapter


class COCOAdapter(BaseAdapter):
    def __init__(
        self,
        image_dict: dict[str, ImageInfo] = None,
        annotation_dict: dict[str, AnnotationInfo] = None,
        category_dict: dict[str, CategoryInfo] = None,
        task: Union[str, TaskType] = TaskType.OBJECT_DETECTION,
        callbacks: list[callable] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            image_dict=image_dict,
            annotation_dict=annotation_dict,
            category_dict=category_dict,
            task=task,
            callbacks=[] + (callbacks if callbacks else []),
            *args,
            **kwargs,
        )

        if self.task not in [
            TaskType.OBJECT_DETECTION,
            TaskType.INSTANCE_SEGMENTATION,
            TaskType.SEMANTIC_SEGMENTATION,
        ]:
            raise DatasetAdapterTaskError(f"Task {self.task} is not supported by COCO format.")

    def import_target(
        self,
        coco_dataset: Union[str, dict],
    ) -> "COCOAdapter":
        try:
            if isinstance(coco_dataset, dict):
                coco = COCO()
                coco.dataset = coco_dataset
                coco.createIndex()
            else:
                coco = COCO(str(coco_dataset))
        except Exception as e:
            raise DatasetAdapterImportError(f"Failed to import COCO dataset") from e

        self.run_callback_hooks("on_loop_start", len(coco.cats) + len(coco.imgs) + len(coco.anns))

        category_dict = {}
        coco_cat_id_to_new_cat_id = {}
        for cat_id, cat in coco.cats.items():
            cat_id = cat.pop("id", None) or cat_id
            cat = CategoryInfo.from_dict(
                task=self.task,
                d=cat,
            )
            coco_cat_id_to_new_cat_id[cat_id] = cat.id
            category_dict[cat.id] = cat
            self.run_callback_hooks("on_step_end")

        image_dict = {}
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
            image_dict[img.id] = img
            self.run_callback_hooks("on_step_end")

        annotation_dict = {}
        for ann in coco.anns.values():
            img = image_dict[coco_img_id_to_new_img_id[ann["image_id"]]]
            cat = coco_cat_id_to_new_cat_id[ann["category_id"]]

            W = img.width
            H = img.height

            if self.task == TaskType.OBJECT_DETECTION:
                x1, y1, w, h = ann["bbox"]
                ann = AnnotationInfo.object_detection(
                    image_id=img.id,
                    category_id=cat,
                    bbox=[x1 / W, y1 / H, w / W, h / H],
                    iscrowd=getattr(ann, "iscrowd", None),
                    score=getattr(ann, "score", None),
                )
            elif self.task in [TaskType.INSTANCE_SEGMENTATION, TaskType.SEMANTIC_SEGMENTATION]:
                segmentation = ann["segmentation"]
                if isinstance(segmentation, dict):
                    segmentation = convert_segmentation(
                        segmentation, SegmentationType.RLE, SegmentationType.POLYGON
                    )

                for i, segmentation_ in enumerate(segmentation):
                    for j, point in enumerate(segmentation_):
                        segmentation[i][j] = point / W if j % 2 == 0 else point / H

                ann = getattr(AnnotationInfo, self.task.lower())(
                    image_id=img.id,
                    category_id=cat,
                    segmentation=segmentation,
                    bbox=getattr(ann, "bbox", None),
                    iscrowd=getattr(ann, "iscrowd", None),
                    score=getattr(ann, "score", None),
                )

            annotation_dict[ann.id] = ann
            self.run_callback_hooks("on_step_end")
        self.run_callback_hooks("on_loop_end")

        self.image_dict = image_dict
        self.annotation_dict = annotation_dict
        self.category_dict = category_dict

        return self

    def export_target(self, result_dir: Union[str, Path], image_dir: Union[str, Path]) -> str:

        self.run_callback_hooks(
            "on_loop_start",
            len(self.image_dict) + len(self.category_dict) + len(self.annotation_dict),
        )

        categories = []
        target_cat_id_to_coco_cat_id = {}
        for category_id, category in self.category_dict.items():
            coco_cat_id = len(categories) + 1
            target_cat_id_to_coco_cat_id[category_id] = coco_cat_id

            categories.append(
                {
                    "id": coco_cat_id,
                    "name": category.name,
                    "supercategory": category.supercategory,
                }
            )
            self.run_callback_hooks("on_step_end")

        split_coco = {}
        for split in SplitType:
            split_coco[split] = {
                "categories": categories,
                "images": [],
                "annotations": [],
            }

        target_img_id_to_coco_img_id = {}
        for image_id, image in self.image_dict.items():
            split = image.split
            coco = split_coco[split]

            coco_img_id = len(coco["images"]) + 1
            target_img_id_to_coco_img_id[image_id] = coco_img_id

            coco["images"].append(
                {
                    "id": coco_img_id,
                    "file_name": image.id + image.ext,  # TODO: save to original file name
                    "width": image.width,
                    "height": image.height,
                    "date_captured": image.date_captured,
                }
            )
            self.run_callback_hooks("on_step_end")

        for annotation_id, annotation in self.annotation_dict.items():
            split = self.image_dict[annotation.image_id].split
            coco = split_coco[split]

            image = self.image_dict[annotation.image_id]

            W = image.width
            H = image.height

            if self.task == TaskType.OBJECT_DETECTION:
                W = image.width
                H = image.height

                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": target_img_id_to_coco_img_id[annotation.image_id],
                        "category_id": target_cat_id_to_coco_cat_id[annotation.category_id],
                        "bbox": [
                            annotation.bbox[0] * W,
                            annotation.bbox[1] * H,
                            annotation.bbox[2] * W,
                            annotation.bbox[3] * H,
                        ],
                        "iscrowd": annotation.iscrowd,
                        "score": annotation.score,
                    }
                )
            elif self.task in [TaskType.INSTANCE_SEGMENTATION, TaskType.SEMANTIC_SEGMENTATION]:
                segmentation = annotation.segmentation

                for i, segmentation_ in enumerate(segmentation):
                    for j, point in enumerate(segmentation_):
                        segmentation[i][j] = point * W if j % 2 == 0 else point * H

                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": target_img_id_to_coco_img_id[annotation.image_id],
                        "category_id": target_cat_id_to_coco_cat_id[annotation.category_id],
                        "bbox": [
                            annotation.bbox[0] * W,
                            annotation.bbox[1] * H,
                            annotation.bbox[2] * W,
                            annotation.bbox[3] * H,
                        ],
                        "segmentation": segmentation,
                        "iscrowd": annotation.iscrowd,
                        "score": annotation.score,
                    }
                )
            self.run_callback_hooks("on_step_end")

        # save
        result_dir = Path(result_dir)
        for split, coco in split_coco.items():
            io.save_json(coco, result_dir / f"{split.lower()}.json")

        io.copy_files_to_directory(image_dir, result_dir / "images", create_directory=True)

        self.run_callback_hooks("on_loop_end")

        return str(result_dir)
