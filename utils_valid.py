import numpy as np
import cv2
import matplotlib
from scipy.spatial.distance import directed_hausdorff,cdist
matplotlib.use("Agg")
# matplotlib.use('TkAgg')
import os
import torch.nn.functional as F
import pdb
import torch
from monai.visualize.utils import blend_images
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from skimage import measure
from scipy.ndimage import distance_transform_edt
from skimage.feature import canny
import time

def pratt_fom(segmented_image, ground_truth_image):
    # Ensure the images are binary (0 and 1)
    segmented_image = (segmented_image > 0).astype(np.float32)
    ground_truth_image = (ground_truth_image > 0).astype(np.float32)

    # Detect edges using Canny
    segmented_edges = canny(segmented_image)
    ground_truth_edges = canny(ground_truth_image)

    # Create a distance transform from the ground truth edges
    ground_truth_distance_map = distance_transform_edt(ground_truth_edges == 0)

    # Pratt's FOM calculation
    ground_truth_points = np.argwhere(ground_truth_edges)
    N_g = len(ground_truth_points)
    if N_g == 0:
        return 0.0  # Avoid division by zero if the ground truth has no boundary

    total_sum = 0
    for point in ground_truth_points:
        y, x = int(point[0]), int(point[1])
        distance = ground_truth_distance_map[y, x]
        total_sum += 1 / (1 + distance ** 2)

    segmented_points = np.argwhere(segmented_edges)
    fom = total_sum / max(len(segmented_points), N_g)
    return fom


def region_growing(img, seed, threshold=1):
    """
    Perform region growing algorithm starting from the seed point.

    Parameters:
    img (numpy.ndarray): Grayscale image.
    seed (tuple): The starting point (x, y) for region growing.
    threshold (int): Threshold for intensity difference to include a pixel in the region.

    Returns:
    numpy.ndarray: Binary mask of the grown region.
    """
    # Initialize the region mask and the list of seed points
    region = np.zeros_like(img)
    region[seed[1], seed[0]] = 255
    seeds = [seed]

    # Get image dimensions
    h, w = img.shape

    while seeds:
        x, y = seeds.pop(0)

        # Check 4-neighbors (up, down, left, right)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy

            if 0 <= nx < w and 0 <= ny < h and region[ny, nx] == 0:
                if abs(int(img[ny, nx]) - int(img[y, x])) <= threshold:
                    region[ny, nx] = 255
                    seeds.append((nx, ny))

    return region

def calculate_hausdorff_distance(pred_boundary, target_boundary):
    """
    Calculate the Hausdorff distance between predicted and target boundaries.
    
    Args:
        pred_boundary (torch.Tensor): Boundary coordinates of the predicted segmentation.
        target_boundary (torch.Tensor): Boundary coordinates of the ground truth segmentation.
    
    Returns:
        hausdorff_distance (float): Hausdorff distance between the two boundaries.
    """
    # Convert the tensor boundaries to CPU-based NumPy arrays for scipy function
    pred_boundary_np = pred_boundary.cpu().numpy()
    target_boundary_np = target_boundary.cpu().numpy()
    
    return max(directed_hausdorff(pred_boundary_np, target_boundary_np)[0],
               directed_hausdorff(target_boundary_np, pred_boundary_np)[0])

def approximate_hausdorff_distance(pred_boundary, target_boundary, sample_size=100):
    """
    Calculate an approximate Hausdorff distance by sampling a subset of boundary points.
    
    Args:
        pred_boundary (torch.Tensor): Boundary coordinates of the predicted segmentation.
        target_boundary (torch.Tensor): Boundary coordinates of the ground truth segmentation.
        sample_size (int): Number of points to sample from each boundary.

    Returns:
        hausdorff_distance (float): Approximate Hausdorff distance between the two boundaries.
    """
    # Convert to CPU-based NumPy arrays
    pred_boundary_np = pred_boundary.cpu().numpy()
    target_boundary_np = target_boundary.cpu().numpy()

    # Randomly sample points if the boundary has more than sample_size points
    if len(pred_boundary_np) > sample_size:
        pred_boundary_np = pred_boundary_np[np.random.choice(len(pred_boundary_np), sample_size, replace=False)]
    if len(target_boundary_np) > sample_size:
        target_boundary_np = target_boundary_np[np.random.choice(len(target_boundary_np), sample_size, replace=False)]
    
    # Compute pairwise distances and get the maximum of minimum distances
    distances = cdist(pred_boundary_np, target_boundary_np, metric='euclidean')
    forward_hausdorff = distances.min(axis=1).max()
    reverse_hausdorff = distances.min(axis=0).max()

    return max(forward_hausdorff, reverse_hausdorff)

def calculate_metrics(pred, target, _class=None, pred_class=None, target_class=None):
    """
    Calculate mean sensitivity, overall accuracy, F1 score (Dice coefficient), Jaccard index (IoU),
    F2 score, precision, and Hausdorff distance based on the overlap of predicted and ground truth areas.

    Args:
        pred (torch.Tensor): Predicted tensor of shape [C, H, W, D].
        target (torch.Tensor): Ground truth tensor of shape [C, H, W, D].
        _class (int, optional): Specific class to calculate metrics for.
                                If None, calculate for all classes combined.

    Returns:
        mean_sensitivity (float): Mean sensitivity across all classes.
        overall_accuracy (float): Overall pixel accuracy.
        f1_score (float): Mean Dice coefficient across classes.
        jaccard_index (float): Mean IoU across classes.
        f2_score (float): F2 score across classes.
        precision (float): Mean precision across classes.
        hausdorff_distance (float): Mean Hausdorff distance across classes.
    """
    if pred_class is None or target_class is None:
        pred_class = torch.argmax(pred, dim=0)
        target_class = torch.argmax(target, dim=0)

    if _class is not None:
        # Calculate metrics for the specified class
        intersection = torch.sum((pred_class == _class) & (target_class == _class)).item()
        pred_area = torch.sum(pred_class == _class).item()
        label_area = torch.sum(target_class == _class).item()
        union = pred_area + label_area - intersection

        mean_sensitivity = intersection / label_area if label_area > 0 else 0.0
        class_accuracy = intersection / (pred_area + label_area - intersection) if (pred_area + label_area - intersection) > 0 else 0.0
        # pdb.set_trace()
        f1_score = (2 * intersection) / (pred_area + label_area) if (pred_area + label_area) > 0 else 0.0
        jaccard_index = intersection / union if union > 0 else 0.0
        f2_score = (5 * intersection) / ((4 * label_area) + pred_area) if (4 * label_area + pred_area) > 0 else 0.0
        precision = intersection / pred_area if pred_area > 0 else 0.0
        
        # Calculate Hausdorff distance (convert to CPU for numpy compatibility)
        # pred_boundary = (pred_class == _class).nonzero()
        # target_boundary = (target_class == _class).nonzero()
        # hausdorff_distance = calculate_hausdorff_distance(pred_boundary, target_boundary)
        # hausdorff_distance = approximate_hausdorff_distance(pred_boundary, target_boundary)
        
    else:
        # Calculate metrics for all classes
        sensitivities = []
        dice_scores = []
        iou_scores = []
        f2_scores = []
        precisions = []
        hausdorff_distances = []
        correct_pixels = torch.sum(pred_class == target_class).item()
        
        for i in range(1, pred.shape[0]):  # Skip background class if it exists
            intersection = torch.sum((pred_class == i) & (target_class == i)).item()
            pred_area = torch.sum(pred_class == i).item()
            label_area = torch.sum(target_class == i).item()
            union = pred_area + label_area - intersection

            # Mean Sensitivity (Recall)
            if label_area > 0:
                sensitivities.append(intersection / label_area)
            
            # F1 Score (Dice Coefficient)
            if pred_area + label_area > 0:
                dice_scores.append((2 * intersection) / (pred_area + label_area))
            
            # Jaccard Index (IoU)
            if union > 0:
                iou_scores.append(intersection / union)
            
            # F2 Score
            if (4 * label_area + pred_area) > 0:
                f2_scores.append((5 * intersection) / (4 * label_area + pred_area))
            
            # Precision
            if pred_area > 0:
                precisions.append(intersection / pred_area)
            
            # Hausdorff Distance (requires boundary extraction)
            pred_boundary = (pred_class == i).nonzero()
            target_boundary = (target_class == i).nonzero()
            if len(pred_boundary) > 0 and len(target_boundary) > 0:
                hausdorff_distances.append(calculate_hausdorff_distance(pred_boundary, target_boundary))

        # Calculate mean of each metric across classes
        mean_sensitivity = sum(sensitivities) / len(sensitivities) if sensitivities else 0.0
        overall_accuracy = correct_pixels / target_class.numel()
        f1_score = sum(dice_scores) / len(dice_scores) if dice_scores else 0.0
        jaccard_index = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
        f2_score = sum(f2_scores) / len(f2_scores) if f2_scores else 0.0
        precision = sum(precisions) / len(precisions) if precisions else 0.0
        hausdorff_distance = sum(hausdorff_distances) / len(hausdorff_distances) if hausdorff_distances else 0.0

    return mean_sensitivity, class_accuracy, f1_score, jaccard_index, f2_score, precision

def calculate_accuracy(pred, target, _class=None):
    """
    Calculate accuracy and Dice loss based on the overlap of predicted and ground truth areas.

    Args:
        pred (torch.Tensor): Predicted tensor of shape [C, H, W, D].
        target (torch.Tensor): Ground truth tensor of shape [C, H, W, D].
        _class (int, optional): Specific class to calculate the overlap accuracy and Dice loss for.
                                If None, calculate for all classes combined.

    Returns:
        accuracy (float): Overlap accuracy as a fraction of the intersected area over the target area.
        dice_loss (float): Dice loss for the specified class or all classes combined.
    """
    pred_class = torch.argmax(pred, dim=0)
    target_class = torch.argmax(target, dim=0)

    if _class is not None:
        # Calculate accuracy and Dice loss for the specified class
        intersection = torch.sum(
            (pred_class == _class) & (target_class == _class)
        ).item()
        pred_area = torch.sum(pred_class == _class).item()
        label_area = torch.sum(target_class == _class).item()

        accuracy = intersection / label_area

        dice = (2 * intersection) / (pred_area + label_area)
        dice_loss = 1 - dice
    else:
        # Calculate accuracy and Dice loss for all classes
        accuracy_list = []
        dice_list = []
        for i in range(1, pred.shape[0]):  # Skip background class if it exists
            intersection = torch.sum((pred_class == i) & (target_class == i)).item()
            pred_area = torch.sum(pred_class == i).item()
            label_area = torch.sum(target_class == i).item()

            if label_area > 0:
                accuracy_list.append(intersection / label_area)

            if pred_area + label_area > 0:
                dice = (2 * intersection) / (pred_area + label_area)
                dice_list.append(dice)

        accuracy = sum(accuracy_list) / len(accuracy_list) if accuracy_list else 0.0
        dice_loss = 1 - (sum(dice_list) / len(dice_list)) if dice_list else 1.0

    return accuracy, dice_loss


def tensor_to_numpy(tensor):
    if tensor.is_cuda:
        # If the tensor is on CUDA, detach it from the graph and move it to CPU
        tensor = tensor.detach().cpu()
    else:
        # If the tensor is not on CUDA, just detach it from the graph
        tensor = tensor.detach()

    # Convert the tensor to a NumPy array
    numpy_array = tensor.numpy()
    return numpy_array

def label2mask(labels, num_classes):
    b, h, w, d = labels.shape
    new_mask = torch.zeros(
        (
            labels.shape[0],
            num_classes,
            labels.shape[1],
            labels.shape[2],
            labels.shape[3],
        )
    )

    for i in range(num_classes):
        new_mask[:, i, :, :, :] = labels == i
    return new_mask

def output2mask(output, one_hot=False):
    probabilities = F.softmax(output, dim=1)
    class_indices = torch.argmax(probabilities, dim=1)
    if one_hot:
        num_classes = probabilities.shape[1]
        one_hot_mask = F.one_hot(class_indices, num_classes=num_classes)
        one_hot_mask = one_hot_mask.permute(0, 4, 1, 2, 3)
        return one_hot_mask
    return class_indices


def mask2onehot(pred_mask_original, num_classes):
    b, c, h, w, d = pred_mask_original.shape
    new_mask = torch.zeros(
        (
            pred_mask_original.shape[0],
            num_classes,
            pred_mask_original.shape[2],
            pred_mask_original.shape[3],
            pred_mask_original.shape[4],
        )
    )

    for i in range(num_classes):
        new_mask[:, i, :, :, :] = pred_mask_original == i
    return new_mask


def validation_visualization(
    image, output, target, epoch, image_path, iteration=0, train=False, save_dir=None
):
    # if save_dir is not None:

    probabilities = F.softmax(output, dim=1)
    class_indices = torch.argmax(probabilities, dim=1)
    num_classes = probabilities.shape[1]

    one_hot_mask = F.one_hot(class_indices, num_classes=num_classes)
    one_hot_mask = one_hot_mask.permute(0, 3, 1, 2)

    train_path = os.path.join(save_dir, "training")
    os.makedirs(train_path, exist_ok=True)
    val_path = os.path.join(save_dir, "validation")
    os.makedirs(val_path, exist_ok=True)
    image = tensor_to_numpy(image)
    output = tensor_to_numpy(one_hot_mask)
    target = tensor_to_numpy(target)
    # pdb.set_trace()
    # one_hot_mask = tensor_to_numpy(one_hot_mask)
    b, c, h, w = target.shape
    if output.shape[1] != 4:
        diff = 4 - output.shape[1]
        for i in range(diff):
            output = np.concatenate((output, np.zeros((b, 1, h, w))), axis=1)
            target = np.concatenate((target, np.zeros((b, 1, h, w))), axis=1)
    # img_slice =np.expand_dims(image[0, 0, slice_idx], axis=0)
    # img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
    colors = ["black", "blue", "yellow", "red"]  # Colors for each class
    from matplotlib.colors import ListedColormap

    custom_cmap = ListedColormap(colors)
    # Colors for the masks
    for i in range(b):
        dataset = image_path[i].split("/")[-2]

        index = image_path[i].split("/")[-1].split(".")[0]

        # img_norm=  np.expand_dims(cv2.cvtColor(np.transpose(image[i],(1,2,0)), cv2.COLOR_RGB2GRAY), axis=0)
        img_norm = image[i]  # np.expand_dims(np.transpose(,(1,2,0)), axis=0)
        # img_norm=image[i]

        first_cls_output = output[i, 1, :]
        second_cls_output = output[i, 2, :]
        third_cls_output = output[i, 3, :]

        outputs_combined = np.maximum(
            np.maximum(first_cls_output * 1, second_cls_output * 2),
            third_cls_output * 3,
        )
        outputs_combined = np.expand_dims(outputs_combined, axis=0)

        first_cls_label = target[i, 1, :]
        second_cls_label = target[i, 2, :]
        third_cls_label = target[i, 3, :]

        labels_combined = np.maximum(
            np.maximum(first_cls_label * 1, second_cls_label * 2), third_cls_label * 3
        )
        labels_combined = np.expand_dims(labels_combined, axis=0)

        fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 5 subplots in a row
        img1 = blend_images(img_norm, labels_combined, alpha=0.3, cmap=custom_cmap)
        subplot1 = np.transpose(img1, (1, 2, 0))

        img2 = blend_images(img_norm, outputs_combined, alpha=0.3, cmap=custom_cmap)
        subplot2 = np.transpose(img2, (1, 2, 0))

        img3 = blend_images(
            img_norm, np.expand_dims(first_cls_output, axis=0), alpha=1, cmap="Blues"
        )
        subplot3 = np.transpose(img3, (1, 2, 0))

        img4 = blend_images(
            img_norm, np.expand_dims(second_cls_output, axis=0), alpha=1, cmap="inferno"
        )
        subplot4 = np.transpose(img4, (1, 2, 0))

        img5 = blend_images(
            img_norm, np.expand_dims(third_cls_output, axis=0), alpha=1, cmap="Reds"
        )
        subplot5 = np.transpose(img5, (1, 2, 0))

        axs[0, 0].imshow(img_norm[0], cmap="gray")
        axs[0, 0].set_title("Raw Image")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(subplot2)
        axs[0, 1].set_title("Predict Combined")
        axs[0, 1].axis("off")

        axs[0, 2].imshow(subplot1)
        axs[0, 2].set_title("Ground Truth Combined")
        axs[0, 2].axis("off")

        axs[1, 0].imshow(subplot3)
        axs[1, 0].set_title("Predict on Liquor")
        axs[1, 0].axis("off")

        axs[1, 1].imshow(subplot4)
        axs[1, 1].set_title("Predict on Loop")
        axs[1, 1].axis("off")

        axs[1, 2].imshow(subplot5)
        axs[1, 2].set_title("Predict on Crystal")
        axs[1, 2].axis("off")

        plt.tight_layout()

        # if i==8:
        #     pdb.set_trace()
        # pdb.set_trace()
        iteration = str(iteration).zfill(4)
        if train:
            plt.savefig(
                os.path.join(
                    train_path, f"{epoch}_train_{dataset}_{index}_{iteration}.png"
                )
            )
        else:
            plt.savefig(
                os.path.join(
                    val_path, f"{epoch}_validation_{dataset}_{index}_{iteration}.png"
                )
            )
        plt.close(fig)


def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
        )
    ax.imshow(mask_image)


def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            import cv2

            contours, _ = cv2.findContours(
                m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Try to smooth contours
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)

    ax.imshow(img)


# Calculate the bounding box for each group
def calculate_bbox(points):
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    return (x_min, y_min, x_max, y_max)


def scale_bbox(bbox, scale_factor=1.1):
    """
    Enlarges a bounding box by a specified factor.

    Parameters:
    - bbox: A tuple (x_min, y_min, x_max, y_max) representing the original bounding box.
    - enlargement_factor: A float indicating how much to enlarge the bounding box (default is 1.1 for 10% increase).

    Returns:
    - A tuple (x_min_new, y_min_new, x_max_new, y_max_new) representing the enlarged bounding box.
    """
    x_min, y_min, x_max, y_max = bbox
    width = x_max - x_min
    height = y_max - y_min

    # Increase by the enlargement factor
    new_width = width * scale_factor
    new_height = height * scale_factor

    # Adjust the bounding box
    x_min_new = int(x_min - (new_width - width) / 2)
    y_min_new = int(y_min - (new_height - height) / 2)
    x_max_new = int(x_max + (new_width - width) / 2)
    y_max_new = int(y_max + (new_height - height) / 2)

    return (x_min_new, y_min_new, x_max_new, y_max_new)


def seperate_loops(loop_points):
    distances = pdist(loop_points, metric="euclidean")
    dist_matrix = squareform(distances)

    # Create an adjacency matrix where an edge exists if the distance is less than a threshold (e.g., 1.5 units)
    adjacency_matrix = dist_matrix <= 1.5

    # Convert the adjacency matrix to a sparse matrix
    sparse_matrix = csr_matrix(adjacency_matrix)

    # Use connected components to determine if the points are in a single connected component or multiple
    n_components, labels = connected_components(
        csgraph=sparse_matrix, directed=False, return_labels=True
    )
    if n_components > 1:
        # print(f"Detected {n_components} separate loops.")
        group_1_points = loop_points[labels == 0]
        group_2_points = loop_points[labels == 1]

        bbox_group_1 = calculate_bbox(group_1_points)
        bbox_group_2 = calculate_bbox(group_2_points)
    else:
        # print("Detected a single loop.")
        group_1_points = loop_points
        group_2_points = None
        bbox_group_1 = calculate_bbox(group_1_points)
        bbox_group_2 = None
    return group_1_points, group_2_points, bbox_group_1, bbox_group_2


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_masks(
    image,
    masks,
    scores,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show()


def systematic_sampling(points, num, bbox=None):

    if bbox is not None:

        t1 = time.time()
        x_min, y_min, x_max, y_max = bbox
        points = np.array(
            [
                p
                for p in points
                if not (x_min <= p[0] <= x_max and y_min <= p[1] <= y_max)
            ]
        )
        t2 = time.time()
        # print(f"Excluding points in bounding box took {t2 - t1:.2f} seconds")
    # Ensure there are enough points left after exclusion
    if len(points) < num:
        return np.array([[]])
        raise ValueError(
            f"Not enough points left after exclusion to sample {num} points."
        )
    coord_list_even = []

    coordinate_list = np.linspace(0, len(points), num=num, endpoint=False, dtype=int)
    # print(f"{len(coordinate_list)} voxels in even sampling are calculated")
    for i in coordinate_list:
        coord_list_even.append(points[i])

    coord_list = np.array(coord_list_even)
    return coord_list


def get_central_coordinates(input_coord, num_points, max_distance=50):
    # Convert input to numpy array
    coords = np.array(input_coord)

    # Calculate the centroid (mean point)
    centroid = np.mean(coords, axis=0)

    # Calculate distances from each point to the centroid
    distances = np.linalg.norm(coords - centroid, axis=1)

    mask = distances == max_distance
    # Get indices of the closest points
    # closest_indices = np.argsort(distances)[:num_points]
    mask = mask[:num_points]

    # Return the closest points
    return coords[mask]


def get_points(label_img, value, Erosion=False, **kwargs):

    points_array = np.where(label_img == value)
    if len(points_array[0]) == 0:
        return np.array([]), None

    points_array = np.column_stack((points_array[1], points_array[0]))
    x_min, y_min = np.min(points_array, axis=0)
    x_max, y_max = np.max(points_array, axis=0)
    bbox = [x_min, y_min, x_max, y_max]

    if Erosion:
        del points_array
        mask = np.zeros_like(label_img)
        mask[label_img == value] = 1
        mask = mask.astype(np.uint8)
        kernel_size = kwargs.get("kernel_size", (10, 10))
        kernel = kwargs.get("kernel", np.ones(kernel_size, np.uint8))
        mask = cv2.erode(mask, kernel, iterations=1)
        cr_points = np.where(mask == 1)
        points_array = np.column_stack((cr_points[1], cr_points[0]))
    Closing = kwargs.get("Closing", False)
    Opening = kwargs.get("Opening", False)
    if Closing:
        mask = np.zeros_like(label_img)
        mask[label_img == value] = 1
        mask = mask.astype(np.uint8)
        kernel_size = kwargs.get("kernel_size", (10, 10))
        kernel = kwargs.get("kernel", np.ones(kernel_size, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cr_points = np.where(mask == 1)
        points_array = np.column_stack((cr_points[1], cr_points[0]))
    if Opening:
        mask = np.zeros_like(label_img)
        mask[label_img == value] = 1
        mask = mask.astype(np.uint8)
        kernel_size = kwargs.get("kernel_size", (10, 10))
        kernel = kwargs.get("kernel", np.ones(kernel_size, np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cr_points = np.where(mask == 1)
        points_array = np.column_stack((cr_points[1], cr_points[0]))
    return points_array, bbox


def points_outside_bbox(points, bbox):
    """
    Find points that are outside the given bounding box.

    Args:
        points (np.ndarray): Array of points with shape (N, 2) where each point is [x, y].
        bbox (list or tuple): Bounding box specified as [x_min, y_min, x_max, y_max].

    Returns:
        np.ndarray: Array of points that are outside the bounding box.
    """
    x_min, y_min, x_max, y_max = bbox

    # Check for points outside the bounding box
    outside_mask = (
        (points[:, 0] < x_min)
        | (points[:, 0] > x_max)
        | (points[:, 1] < y_min)
        | (points[:, 1] > y_max)
    )

    # Return points that are outside the bbox
    return points[outside_mask]


def find_contour(points):
    """
    Finds the overall contour that encloses the given points.

    Args:
        points (np.ndarray): Array of points with shape (N, 2) where each point is [x, y].

    Returns:
        contour (np.ndarray): Contour points representing the outer boundary.
    """
    # Ensure the points are in the correct shape for contour finding
    points = points.reshape((-1, 1, 2)).astype(np.int32)

    # Find the convex hull of the points to get the overall contour
    contour = cv2.convexHull(points)

    return contour


def kmeans_clustering(image, k=3):
    """
    Perform K-means clustering on the image to separate it into k clusters.

    Args:
        image (np.ndarray): Input image in grayscale or single-channel format.
        k (int): Number of clusters/classes to separate.

    Returns:
        clustered_image (np.ndarray): Image with pixel values corresponding to their cluster label.
        centers (np.ndarray): The intensity values of the cluster centers.
    """
    # Reshape the image into a 2D array of pixels
    pixel_values = image.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)

    # Define criteria and apply K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Convert centers to uint8
    centers = np.uint8(centers)

    # Map the labels back to the original image dimensions
    clustered_image = centers[labels.flatten()]
    clustered_image = clustered_image.reshape(image.shape)

    return clustered_image, centers


def postprocess_refine_edges(
    masks,
    close_kernel=np.ones((20, 20), np.uint8),
    open_kernel=np.ones((15, 15), np.uint8),
):
    # Apply morphological closing to refine the edges of the masks
    refined_masks = []
    for mask in masks:
        assert mask.ndim == 2, "Mask must be a 2D array"

        closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
        refined_masks.append(closed_mask)
    open_masks = []
    for mask in refined_masks:
        open_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
        open_masks.append(open_mask)
    return np.array(open_masks)

def postprocess_sam_cr(
    predictor,
    raw_img,
    group_1_points,
    liquor_points,
    loop_points,
    bbox_group_1,
    in_number=10,
    out_number=10,
    bb_scale_factor=1.2,
    plot=False,
    crystal=False,
):
    # detecting loop regions
    if bbox_group_1 is None:
        input_box = None
    else:
        input_box = scale_bbox(np.array(bbox_group_1), scale_factor=bb_scale_factor)
    group_1_points = sample_top_50_percent_near_center(group_1_points, in_number)
    if crystal:
        liquor_points = systematic_sampling(liquor_points, out_number, input_box)
    else:
        liquor_points = systematic_sampling(liquor_points, out_number)


    input_labels_1 = np.ones(group_1_points.shape[0])
    input_labels_0 = np.zeros(liquor_points.shape[0])
    input_labels = np.concatenate((input_labels_1, input_labels_0), axis=0)
    input_points = np.concatenate((group_1_points, liquor_points), axis=0)

    # https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/image_predictor_example.ipynb
    lo_masks_1, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        box=input_box,
        multimask_output=False,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = lo_masks_1[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    # if crystal:
    #     pdb.set_trace()
    masks = masks.astype(np.uint8)
    masks = postprocess_refine_edges(masks)

    # Apply the bounding box to the masks, outside of the box is set to 0
    if input_box is not None:
        x_min, y_min, x_max, y_max = input_box
        masks[0][:y_min, :] = 0
        masks[0][y_max:, :] = 0
        masks[0][:, :x_min] = 0
        masks[0][:, x_max:] = 0
    if plot:
        matplotlib.use('TkAgg')
        show_masks(raw_img,masks,scores,input_labels=input_labels,box_coords=input_box,borders=True,point_coords=input_points,)
    return masks, scores, logits

def sample_top_50_percent_near_center(points, num_samples=10):
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)
    
    # Calculate distances from each point to the centroid
    distances = np.linalg.norm(points - centroid, axis=1)
    
    # Sort points based on distance to the centroid and get top 50% closest points
    sorted_indices = np.argsort(distances)
    top_50_percent_count = len(points) // 4
    top_50_percent_indices = sorted_indices[:top_50_percent_count]
    top_50_percent_points = points[top_50_percent_indices]
    
    # Uniformly sample the specified number of points from the top 50%
    sampled_points = top_50_percent_points[np.random.choice(len(top_50_percent_points), num_samples, replace=False)]
    
    return sampled_points

def postprocess_sam(
    predictor,
    raw_img,
    group_1_points,
    liquor_points,
    bbox_group_1,
    in_number=10,
    out_number=10,
    bb_scale_factor=1.2,
    plot=False,
    crystal=False,
):
    # detecting loop regions
    if bbox_group_1 is None:
        input_box = None
    else:
        input_box = scale_bbox(np.array(bbox_group_1), scale_factor=bb_scale_factor)
    group_1_points = systematic_sampling(group_1_points, in_number)
    if crystal:
        liquor_points = systematic_sampling(liquor_points, out_number, input_box)
    else:
        liquor_points = systematic_sampling(liquor_points, out_number)
    # try:
    #     input_labels_1 = np.ones(group_1_points.shape[0])
    #     input_labels_0 = np.zeros(liquor_points.shape[0])
    #     input_labels = np.concatenate((input_labels_1, input_labels_0), axis=0)
    #     input_points = np.concatenate((group_1_points, liquor_points), axis=0)
    # except:
    #     pdb.set_trace()
    # if crystal:
    #     pdb.set_trace()
    # pdb.set_trace()
    input_labels_1 = np.ones(group_1_points.shape[0])
    input_labels_0 = np.zeros(liquor_points.shape[0])
    input_labels = np.concatenate((input_labels_1, input_labels_0), axis=0)
    input_points = np.concatenate((group_1_points, liquor_points), axis=0)

    # https://github.com/facebookresearch/segment-anything-2/blob/main/notebooks/image_predictor_example.ipynb
    lo_masks_1, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        box=input_box,
        multimask_output=False,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = lo_masks_1[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]
    # if crystal:
    #     pdb.set_trace()
    masks = masks.astype(np.uint8)
    masks = postprocess_refine_edges(masks)

    # Apply the bounding box to the masks, outside of the box is set to 0
    if input_box is not None:
        x_min, y_min, x_max, y_max = input_box
        masks[0][:y_min, :] = 0
        masks[0][y_max:, :] = 0
        masks[0][:, :x_min] = 0
        masks[0][:, x_max:] = 0
    if plot:
        matplotlib.use('TkAgg')
        show_masks(raw_img,masks,scores,input_labels=input_labels,box_coords=input_box,borders=True,point_coords=input_points,)
    return masks, scores, logits


if __name__ == "__main__":
    profile_code()
