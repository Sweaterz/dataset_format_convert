import argparse
import json
import os
from PIL import Image
from tqdm import tqdm


def coco2labelme(
        coco_annotation_path,
        output_dir="labelmeFormat",
        name=''
                ):

    # 加载 COCO 标注文件
    # coco_annotation_path = "coco_style_fourclass/annotations/instances_train2017.json" # "path/to/coco/annotations.json"
    with open(coco_annotation_path, "r") as f:
        coco_data = json.load(f)

    # 自定义名称
    if name == "":
        name = coco_annotation_path.split('/')[-1].replace("instances_", "").split('.')[0]; # 这里name是train2017

    # 创建输出目录
    # output_dir = "labelmeFormat2" # 确保文件夹还未生成
    if os.path.exists(output_dir):
        print(f"目录output dir:{output_dir}已存在。")
        exit()
    make_dir1 = os.path.join(output_dir, f"labels/{name}")
    make_dir2 = os.path.join(output_dir, f"images/{name}")
    os.makedirs(make_dir1, exist_ok=True)
    os.makedirs(make_dir2, exist_ok=True) # 这里生成图片的空文件夹，自行将图片放入

    # 需要设置图片相对标注文件的位置信息(或者图片的绝对路径)
    imagesPath = f"../../images/{name}/"

    # 遍历 COCO 数据
    for image_info in tqdm(coco_data["images"], desc="转换进度"): # coco_data["images"]:
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        image_width = image_info["width"]
        image_height = image_info["height"]

        # 创建 LabelMe 格式的 JSON 文件
        labelme_data = {
            "version": "5.5.0",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.join(imagesPath, file_name),
            "imageData": None,
            "imageHeight": image_height,
            "imageWidth": image_width,
        }

        # 添加标注信息
        for annotation in coco_data["annotations"]:
            if annotation["image_id"] == image_id:
                category_id = annotation["category_id"]
                category_name = next(
                    (cat["name"] for cat in coco_data["categories"] if cat["id"] == category_id),
                    "unknown",
                )

                # 添加多边形标注
                segmentation = annotation["segmentation"]
                for polygon in segmentation:
                    shape = {
                        "label": category_name,
                        "points": [[polygon[i], polygon[i + 1]] for i in range(0, len(polygon), 2)],
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {},
                    }
                    labelme_data["shapes"].append(shape)

        # 保存 LabelMe 格式的 JSON 文件
        output_path = os.path.join(make_dir1, os.path.splitext(file_name)[0] + ".json")
        with open(output_path, "w") as f:
            json.dump(labelme_data, f, indent=2)
    print(f"输出到{output_dir}")
    print("转换完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='json convert to txt params')
    parser.add_argument('--coco-path', type=str, default='="coco_style_fourclass/annotations/instances_train2017.json"', help='json path dir')
    parser.add_argument('--output-dir', type=str, default='', help='output dir')
    parser.add_argument('--name', type=str, default='', help='data name')
    args = parser.parse_args()
    coco_annotation_path = args.coco_path
    output_dir = args.output_dir
    name = args.name

    coco2labelme(coco_annotation_path, output_dir, name)
