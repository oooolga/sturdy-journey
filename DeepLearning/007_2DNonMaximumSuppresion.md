# The Coding Challenge: 2D Non-Maximum Suppression (NMS)
**The Task:** Implement a function that performs NMS on a set of bounding boxes based on their confidence scores and their Intersection over Union (IoU).
**Requirements:**
1. Input:
   - boxes: A tensor of shape $(N, 4)$ representing $(x1, y1, x2, y2)$ coordinates.
   - scores: A tensor of shape $(N)$ representing the confidence score for each box.
   - iou_threshold: A float (e.g., 0.5). If two boxes overlap more than this threshold, the one with the lower score is discarded.
2. Output: A list of indices of the boxes that should be kept.
3. Efficiency: While a pure PyTorch vectorized version is complex, try to minimize unnecessary computations (e.g., sort the scores first).

## Starter Code Template
```Python
import torch

def compute_iou(box1, boxes):
    """
    Args:
        box1: (4) tensor [x1, y1, x2, y2]
        boxes: (K, 4) tensor of other boxes
    Returns:
        iou: (K) tensor of IoU values
    """
    # Logic to compute IoU between one box and a set of boxes
    pass

def nms(boxes, scores, iou_threshold):
    """
    Args:
        boxes: (N, 4) [x1, y1, x2, y2]
        scores: (N)
        iou_threshold: float
    Returns:
        keep: list of indices
    """
    # 1. Sort scores in descending order
    # 2. Iteratively select the top box and remove overlaps
    pass
```

## Attempt \#1
```Python
import torch

def compute_iou(box1, boxes):
    """
    Args:
        box1: (4) tensor [x1, y1, x2, y2]
        boxes: (K, 4) tensor of other boxes
    Returns:
        iou: (K) tensor of IoU values
    """
    # Logic to compute IoU between one box and a set of boxes
    # Get intersection coordinates
    intersect_x1 = torch.max(box1[0], boxes[:, 0]) # K, 1
    intersect_x2 = torch.min(box1[2], boxes[:, 2])
    intersect_y1 = torch.max(box1[1], boxes[:, 1])
    intersect_y2 = torch.min(box1[3], boxes[:, 3])

    # Compute area of intersection
    area_intersect = torch.clamp((intersect_x2 - intersect_x1), min=0) * torch.clamp((intersect_y2 - intersect_y1), min=0)
    area_box1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area_boxes = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])

    iou = area_intersect / (area_box1+area_boxes-area_intersect+1e-6)
    return iou


def nms(boxes, scores, iou_threshold):
    """
    Args:
        boxes: (N, 4) [x1, y1, x2, y2]
        scores: (N)
        iou_threshold: float
    Returns:
        keep: list of indices
    """
    # 1. Sort scores in descending order
    order = scores.argsort(descending=True)
    
    # 2. Iteratively select the top box and remove overlaps
    pass
```