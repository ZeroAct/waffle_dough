from waffle_dough.exception import FieldMissingError, FieldValidationError


def validate_keypoints(v: list[str]) -> list[str]:
    if v is not None:
        for keypoint in v:
            if not isinstance(keypoint, str):
                raise FieldValidationError(f"each keypoint must be string: {keypoint}")
    return v


def validate_skeleton(v: list[list[int]], keypoints: list[str]) -> list[list[int]]:
    if v is not None:
        if keypoints is None:
            raise FieldMissingError("keypoints must be given when skeleton is given")
        for skeleton in v:
            if not isinstance(skeleton, list):
                raise FieldValidationError(f"each skeleton must be list: {skeleton}")
            if len(skeleton) != 2:
                raise FieldValidationError(f"each skeleton must have 2 elements: {skeleton}")
            for skeleton_element in skeleton:
                if not isinstance(skeleton_element, int):
                    raise FieldValidationError(
                        f"each skeleton element must be int: {skeleton_element}"
                    )
                if skeleton_element < 0 or skeleton_element >= len(keypoints):
                    raise FieldValidationError(
                        f"each skeleton element must be in range of keypoints: {skeleton_element}"
                    )
    return v
