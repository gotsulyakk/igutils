from typing import List, Dict


class LabelMap:
    """Class that represents label map
    in Dict[int, str] format
    """

    def __init__(self, labelmap: Dict[int, str]) -> None:
        self.__valid_labelmap(labelmap)
        self.labelmap = labelmap

    def __valid_labelmap(self, labelmap: Dict) -> None:
        if not isinstance(labelmap, dict):
            raise TypeError(
                f"Classes dict must be dictionary [int, str], current {type(labelmap)}"
            )
        if not all([isinstance(label, str) for label in labelmap.values()]):
            raise TypeError(f"All labels must be string")
        if not all([isinstance(label_id, int) for label_id in labelmap.keys()]):
            raise TypeError(f"All label index must be int")
        if not min(labelmap.keys()) == 0:
            raise ValueError(f"Label index must start from 0")

    @property
    def num_labels(self) -> int:
        return len(self.labelmap)

    @property
    def inversed(self):
        return {value: key for key, value in self.labelmap.items()}

    @property
    def labels_list(self) -> List[str]:
        return list(self.labelmap.values())

    @property
    def classes_list_of_dict_coco(self) -> List[dict]:
        tmp_list = []
        for label_id, label in self.labelmap.items():
            tmp_list.append({"id": label_id + 1, "name": label})
        return tmp_list
