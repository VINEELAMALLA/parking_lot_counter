from parse_annotations import parse_cvat_polygon_annotations

data = parse_cvat_polygon_annotations('dataset/annotations.xml')
print(data[0])  # Print the first parsed item to check
