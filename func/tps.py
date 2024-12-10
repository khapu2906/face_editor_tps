import cv2
import numpy as np
import mediapipe as mp
import json
from scipy.interpolate import RBFInterpolator
import math

with open("mediapipe_facezone.json", 'r') as file:
    full_face = json.load(file)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5
)


def get_landmarks(image):
    """Use Mediapipe to detect landmarks."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    landmarks = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                landmarks.append((x, y))

    return landmarks


def get_landmarks_point_in_zone(full_face):
    points = []
    for region in full_face:
        if isinstance(full_face[region], list):
            points.extend(full_face[region])
        elif isinstance(full_face[region], dict):
            points.extend(get_landmarks_point_in_zone(full_face[region]))

    return points


def apply_tps_with_mask(img, source_points, target_points, face_rect):
    min_x, min_y, max_x, max_y = face_rect
    h, w = max_y - min_y, max_x - min_x

    print("size face", h, w)

    for element_face, element_points in source_points:
        print(element_face, element_points)
    rbf_x = RBFInterpolator(
        source_points, target_points[:, 0],
        kernel='thin_plate_spline', epsilon=0.5
    )
    rbf_y = RBFInterpolator(
        source_points, target_points[:, 1],
        kernel='thin_plate_spline', epsilon=0.5
    )

    grid_x, grid_y = np.meshgrid(
        np.linspace(min_x, max_x, w), np.linspace(min_y, max_y, h)
    )

    # Nội suy các điểm lưới dựa trên RBFInterpolator
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    map_x = rbf_x(grid_points).reshape(h, w)
    map_y = rbf_y(grid_points).reshape(h, w)

    map_x = np.clip(map_x, 0, img.shape[1] - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, img.shape[0] - 1).astype(np.float32)

    # Tạo mặt nạ dựa trên các điểm landmark của khuôn mặt
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    hull = cv2.convexHull(np.array(source_points, dtype=np.int32))
    cv2.fillConvexPoly(mask, hull, 1)

    map_x = np.where(mask[min_y:max_y, min_x:max_x],
                     map_x, grid_x).astype(np.float32)
    map_y = np.where(mask[min_y:max_y, min_x:max_x],
                     map_y, grid_y).astype(np.float32)

    transformed_points = np.column_stack((map_x.ravel(), map_y.ravel()))

    transformed_face = cv2.remap(
        img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
    )

    img_copy = img.copy()
    img_copy[min_y:max_y, min_x:max_x] = transformed_face

    return img_copy



def transform_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Cannot read the image file.")
        return

    original_landmarks = get_landmarks(img)
    if not original_landmarks:
        print("No face found.")
        return

    # points = get_landmarks_point_in_zone(full_face)
    source_points = np.array(original_landmarks, dtype='float32')
    target_points = source_points.copy()

    # Make adjustments to target_points as in your original code
    # Eyes
    target_points[full_face["eyes"]["left"]["inter_corner"]] += [+2, -1]

    target_points[full_face["eyes"]["left"]["outer_corner"]] += [+5, 0]
    target_points[full_face["eyes"]["right"]["inter_corner"]] += [-2, -1]
    target_points[full_face["eyes"]["right"]["outer_corner"]] += [-5, 0]
    target_points[full_face["eyes"]["left"]["upper"]] += [+3, 0]
    target_points[full_face["eyes"]["left"]["tail"]] += [+6, 0]
    target_points[full_face["eyes"]["left"]["lower"]] += [+3, -2]
    target_points[full_face["eyes"]["right"]["upper"]] += [-3, 0]
    target_points[full_face["eyes"]["right"]["tail"]] += [-6, 0]
    target_points[full_face["eyes"]["right"]["lower"]] += [-3, -2]

    # nose
    target_points[full_face["nose"]["bridges"]["lower"]["w"]["left"]] += [-3, 0]
    target_points[full_face["nose"]["bridges"]["lower"]["w"]["right"]] += [-1, 0]
    target_points[full_face["nose"]["wings"]["left"]] += [-2, +1]
    target_points[full_face["nose"]["wings"]["right"]] += [-3, +2]
    target_points[full_face["nose"]["hole"]["left"]] += [-2, +2]
    target_points[full_face["nose"]["hole"]["right"]] += [0, +1]
    # Cheek
    target_points[full_face["cheek"]["left"]["tip"]] += [+2, -2]
    target_points[full_face["cheek"]["right"]["tip"]] += [-2, -2]
    ##Mouth
    target_points[full_face["mouth"]["corners"]["left"]] += [-3, 0]
    target_points[full_face["mouth"]["corners"]["right"]] += [+3, 0]
    target_points[full_face["mouth"]["lips"]["upper"]["outer"]["left"]] += [-3, -3]
    target_points[full_face["mouth"]["lips"]["upper"]["outer"]["center"]] += [0, -3]
    target_points[full_face["mouth"]["lips"]["upper"]["outer"]["right"]] += [+3, -3]
    target_points[full_face["mouth"]["lips"]["upper"]["inter"]["left"]] += [0, +1]
    target_points[full_face["mouth"]["lips"]["upper"]["inter"]["center"]] += [0, -2]
    target_points[full_face["mouth"]["lips"]["upper"]["inter"]["right"]] += [0, +1]
    target_points[full_face["mouth"]["lips"]["lower"]["outer"]["left"]] += [-2, +2]
    target_points[full_face["mouth"]["lips"]["lower"]["outer"]["center"]] += [0, +1]
    target_points[full_face["mouth"]["lips"]["lower"]["outer"]["right"]] += [+2, +2]
    target_points[full_face["mouth"]["lips"]["lower"]["inter"]["left"]] += [0, -1]
    target_points[full_face["mouth"]["lips"]["lower"]["inter"]["center"]] += [0, -1]
    target_points[full_face["mouth"]["lips"]["lower"]["inter"]["right"]] += [0, -1]
    target_points[full_face["chin"]["left"]] += [-3, +10]
    target_points[full_face["chin"]["center"]] += [-7, +8]
    target_points[full_face["chin"]["right"]] += [0, +10]
    target_points[full_face["jaw"]["left"]] += [-3, +2]
    target_points[full_face["jaw"]["right"]] += [+3, +2]

    margin_x = int((max(point[0] for point in original_landmarks) -
                    min(point[0] for point in original_landmarks)) * 0.17)

    margin_y = int((max(point[1] for point in original_landmarks) -
                min(point[1] for point in original_landmarks)) * 0.17)

    min_x = math.floor(min(point[0]for point in original_landmarks) - margin_x)
    min_y = math.floor(min(point[1] for point in original_landmarks) - margin_y)
    max_x = math.ceil(max(point[0] for point in original_landmarks) + margin_x)
    max_y = math.ceil(max(point[1] for point in original_landmarks) + margin_y)
    face_rect = (min_x, min_y, max_x, max_y)

    edited_img = img.copy()
    modified_source_points = []
    modified_target_points = []

    for sp, tp in zip(source_points, target_points):
        if not np.array_equal(sp, tp):
            modified_source_points.append(sp)
            modified_target_points.append(tp)

    def is_duplicate(point, points, threshold=1e-5):
        return any(np.linalg.norm(np.array(point) - np.array(existing_point)) < threshold for existing_point in points)

    for p in full_face["support"]:
        if not is_duplicate(target_points[p], modified_target_points):
            modified_target_points.append(target_points[p])
            modified_source_points.append(source_points[p])

    modified_source_points = np.array(modified_source_points, dtype='float32')
    modified_target_points = np.array(modified_target_points, dtype='float32')

    rect_corners = np.array([
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y] 
    ], dtype='float32')

    modified_source_points = np.vstack((modified_source_points, rect_corners))
    modified_target_points = np.vstack((modified_target_points, rect_corners))

    x_min, y_min, x_max, y_max = map(int, face_rect)
    cv2.rectangle(
        img,
        (x_min, y_min),
        (x_max, y_max),
        (0, 255, 0),
        2 
    )

    transformed_img = apply_tps_with_mask(
        edited_img, modified_source_points, modified_target_points, face_rect)
    transformed_landmarks = get_landmarks(transformed_img)


    cv2.imshow("Original Image", img)
    cv2.imshow("Transformed Image", transformed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


transform_image('../inputs/front_side.png')
