def load_image(image_path):
    import cv2
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

def save_results(results, output_path):
    with open(output_path, 'w') as f:
        for result in results:
            f.write(f"{result}\n")

def log_activity(activity_message):
    import logging
    logging.basicConfig(filename='activity.log', level=logging.INFO)
    logging.info(activity_message)