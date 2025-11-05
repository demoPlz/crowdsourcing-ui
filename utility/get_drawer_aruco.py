import cv2

# Choose dictionary; 5x5_100 is good (IDs 0–99)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

# Pick two distinct IDs, e.g. 17 and 42
for marker_id in [17, 42]:
    img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, 400)  # 400 px → good print quality
    cv2.imwrite(f"aruco_{marker_id}.png", img)
    print(f"Saved aruco_{marker_id}.png")