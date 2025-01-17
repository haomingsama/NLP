# GRADED FUNCTION: yolo_filter_boxes
import tensorflow as tf


box_confidence = tf.random.normal([19,19,5,1])
box_class_probs = tf.random.normal([19,19,5,80])
boxes = tf.random.uniform([19,19,5,4],minval = 0, maxval=1)

box_score = box_confidence * box_class_probs

box_class = tf.math.argmax(box_score, axis = -1)
box_class_score = tf.math.reduce_max(box_score, axis =-1)

filtering_mask = box_class_score >= 0.6
print(filtering_mask)

scores = tf.boolean_mask(box_class_score, filtering_mask)
print(scores)

boxes = tf.boolean_mask(boxes, filtering_mask)
print(boxes)

classes = tf.boolean_mask(box_class, filtering_mask)

print(box_class)

print(classes)


indice = tf.image.non_max_suppression(boxes, scores, max_output_size = 5, iou_threshold = 0.5)
print(indice)
boxes = tf.gather(boxes,indice)
scores = tf.gather(scores, indice)
classes = tf.gather(classes,indice)
print(boxes)
print(scores)
print(classes)
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    """Filters YOLO boxes by thresholding on object and class confidence.

    Arguments:
    box_confidence -- tensor of shape (19, 19, 5, 1)
    boxes -- tensor of shape (19, 19, 5, 4)
    box_class_probs -- tensor of shape (19, 19, 5, 80)
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    Note: "None" is here because you don't know the exact number of selected boxes, as it depends on the threshold.
    For example, the actual output size of scores would be (10,) if there are 10 boxes.
    """

    # Step 1: Compute box scores
    ### START CODE HERE ### (≈ 1 line)
    box_scores = box_confidence * box_class_probs
    ### END CODE HERE ###

    # Step 2: Find the box_classes using the max box_scores, keep track of the corresponding score
    ### START CODE HERE ### (≈ 2 lines)
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1, keepdims=False)
    ### END CODE HERE ###

    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    ### START CODE HERE ### (≈ 1 line)
    filtering_mask = box_class_scores >= threshold
    ### END CODE HERE ###

    # Step 4: Apply the mask to box_class_scores, boxes and box_classes
    ### START CODE HERE ### (≈ 3 lines)
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    ### END CODE HERE ###

    return scores, boxes, classes