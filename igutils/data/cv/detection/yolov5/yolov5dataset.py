from pathlib import Path

import cv2
import pandas as pd
import yaml


class YOLOv5Dataset:
    def __init__(self, config, root):
        self.config = self.load_config(config)
        self.images = Path(root) / "images"
        self.labels = Path(root) / "labels"

    @property
    def labelmap(self):
        return dict([i for i in enumerate(self.config["names"])])

    @property
    def wide(self):
        df = self.load_to_csv()
        df["label"] = df["label_id"].replace(self.labelmap)
        df["image_name"] = df["image_id"].replace(self.get_image_names())
        return df[
            [
                "image_name",
                "label",
                "x_center",
                "y_center",
                "width",
                "height",
                "label_id",
            ]
        ]

    @staticmethod
    def load_config(config_path):
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)
        return config

    @staticmethod
    def to_coco_bbox(df: pd.DataFrame) -> pd.DataFrame:
        """Converts yolo bboxes to coco bboxes

        width_coco = width_yolo * image_width
        height_coco = height_yolo * image_height
        x_coco = x_yolo * image_width - (width_coco / 2)
        y_coco = y_yolo * image_height - (height_coco / 2)

        Args:
            df ([type]): [description]

        Returns:
            [type]: [description]
        """
        df["width_coco"] = df["width"] * df["image_width"]
        df["height_coco"] = df["height"] * df["image_height"]
        df["x_coco"] = df["x_center"] * df["image_width"] - (df["width_coco"] / 2)
        df["y_coco"] = df["y_center"] * df["image_height"] - (df["height_coco"] / 2)

        return df

    @staticmethod
    def to_pascal_voc_bbox(df: pd.DataFrame) -> pd.DataFrame:
        df["bbox_width"] = df["width"] * df["image_width"]
        df["bbox_height"] = df["height"] * df["image_height"]
        df["bbox_x_center"] = df["x_center"] * df["image_width"]
        df["bbox_y_center"] = df["y_center"] * df["image_height"]

        df["xmin"] = df["bbox_x_center"] - (df["bbox_width"] / 2)
        df["ymin"] = df["bbox_y_center"] - (df["bbox_height"] / 2)
        df["xmax"] = df["bbox_x_center"] + (df["bbox_width"] / 2)
        df["ymax"] = df["bbox_y_center"] + (df["bbox_height"] / 2)

        return df[
            [
                "image_name",
                "image_width",
                "image_height",
                "label",
                "label_id",
                "xmin",
                "ymin",
                "xmax",
                "ymax",
            ]
        ]

    def get_image_names(self):
        image_names = {}
        for filepath in self.images.iterdir():
            image_name = str(filepath).split("/")[-1]
            image_names[image_name.split(".")[0]] = image_name

        return image_names

    def get_image_sizes(self):
        image_sizes = {}
        for filepath in self.images.iterdir():
            image_name = str(filepath).split("/")[-1]
            image = cv2.imread(str(filepath))
            height, width, _ = image.shape
            image_sizes[image_name] = (height, width)

        return image_sizes

    def load_to_csv(self):
        annotations = []
        for filepath in sorted(self.labels.iterdir()):
            with open(filepath, "r") as f:
                for line in f.readlines():
                    line = line.split(" ")
                    image_id = str(filepath).split("/")[-1].split(".")[0]
                    label = int(line[0])
                    x, y, width, height = (
                        float(line[1]),
                        float(line[2]),
                        float(line[3]),
                        float(line[4]),
                    )
                    annotations.append((image_id, label, x, y, width, height))

        return pd.DataFrame.from_records(
            annotations,
            columns=["image_id", "label_id", "x_center", "y_center", "width", "height"],
        )
