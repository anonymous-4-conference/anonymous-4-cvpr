import numpy as np
import cv2
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from skimage import measure
import numpy as np
from scipy import ndimage
import pdb
def refine_masks_with_sam2(predictor, raw_img, pred_slice, num_classes=5, 
                          confidence_threshold=0.85, min_area=100):
    """
    Refine predicted masks using SAM2 for multiple classes
    
    Args:
        predictor: SAM2 predictor instance
        raw_img: Original image
        pred_slice: Original prediction mask
        num_classes: Number of classes to segment
        confidence_threshold: Confidence threshold for mask acceptance
        min_area: Minimum area for connected components
    """
    height, width = raw_img.shape[:2]
    refined_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Process each class separately
    for class_idx in range(0, num_classes ):
        # Extract binary mask for current class
        class_mask = (pred_slice == class_idx).astype(np.uint8)
        
        # Skip if no pixels for this class
        if not np.any(class_mask) or np.sum(class_mask) < min_area:
            continue
            
        # Find connected components
        labeled_mask, num_features = measure.label(class_mask, return_num=True)
        
        # Process each connected component
        for label_idx in range(1, num_features + 1):
            component = (labeled_mask == label_idx)
            
            # Skip small components
            if np.sum(component) < min_area:
                continue
                
            # Get component properties
            props = measure.regionprops(component.astype(int))[0]
            
            # Generate prompts from component
            points = generate_prompts_from_component(props, component)
            
            # Get SAM2 prediction
            masks, scores, logits = predictor.predict(
                point_coords=points,
                point_labels=np.ones(len(points)),
                multimask_output=True,
                return_logits=True
            )
            
            # Select best mask based on IoU with original component
            best_mask = None
            best_score = -1
            
            for mask, score in zip(masks, scores):
                if score < confidence_threshold:
                    continue
                mask_bool = mask.astype(bool)
                iou = calculate_iou(component, mask)
                iou = calculate_iou(component, mask_bool)
                
                if iou > best_score:
                    best_score = iou
                    best_mask = mask_bool
            
            # Update refined mask with best prediction
            if best_mask is not None:
                refined_mask[best_mask] = class_idx
            pdb.set_trace()
    pdb.set_trace()
    return refined_mask

def plot_plot(img, mask, refined_mask):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(mask)
    axes[1].set_title('Original Mask')
    axes[1].axis('off')
    axes[2].imshow(refined_mask)
    axes[2].set_title('Refined Mask')
    axes[2].axis('off')
    plt.tight_layout()
    plt.show()
def generate_prompts_from_component(props, component):
    """
    Generate prompt points from component properties
    """
    points = []
    
    # Add centroid
    y0, x0 = props.centroid
    points.append([x0, y0])
    
    # Add boundary points
    boundary = measure.find_contours(component, 0.5)[0]
    num_boundary_points = min(5, len(boundary))
    step = len(boundary) // num_boundary_points
    
    for i in range(0, len(boundary), step):
        y, x = boundary[i]
        points.append([x, y])
        
    return np.array(points)

def calculate_iou(mask1, mask2):
    """
    Calculate Intersection over Union between two masks
    """
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)

def visualize_refinement(original_img, original_mask, refined_mask, save_path=None):
    """
    Visualize original and refined segmentation results
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Original mask
    axes[1].imshow(original_mask)
    axes[1].set_title('Original Mask')
    axes[1].axis('off')
    
    # Refined mask
    axes[2].imshow(refined_mask)
    axes[2].set_title('Refined Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()