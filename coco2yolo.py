from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="coco_style_fourclass/annotations/", use_segments=True)	# labels_dir="path/to/coco/annotations/"
