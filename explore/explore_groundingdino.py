import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


def load_image(image_path):
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path)
    print(f"  Image size: {image.size}")
    return image


def detect_objects(
    image, processor, model, device, text_labels, threshold=0.3, text_threshold=0.25
):
    print(f"Detecting: {text_labels}")

    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]],
    )

    result = results[0]
    print(f"  Found {len(result['boxes'])} objects")

    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        box_list = [round(x, 2) for x in box.tolist()]
        print(f"    {label} | confidence: {score.item():.3f} | box: {box_list}")

    return result


def draw_boxes(image, result, prompt_name):
    image_draw = image.copy()
    draw = ImageDraw.Draw(image_draw)

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    colors = ["red", "green", "blue", "yellow", "magenta", "cyan", "orange", "purple"]

    for i, (box, score, label) in enumerate(
        zip(result["boxes"], result["scores"], result["labels"])
    ):
        box = box.tolist()
        x1, y1, x2, y2 = box
        color = colors[i % len(colors)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        text = f"{label}: {score.item():.2f}"
        draw.text((x1, y1 - 25), text, fill=color, font=font)

    return image_draw


def visualize_results(
    image, results_dict, save_path="../viz/groundingdino_results.png"
):
    num_results = len(results_dict)
    fig, axes = plt.subplots(1, num_results + 1, figsize=(6 * (num_results + 1), 6))

    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    for i, (prompt_name, image_with_boxes) in enumerate(results_dict.items(), 1):
        axes[i].imshow(image_with_boxes)
        axes[i].set_title(f"Prompt: {prompt_name}", fontsize=12)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading Grounding DINO model...")
    model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
    print("  Model loaded successfully")

    image = load_image("test_data/test_image.jpg")

    prompts = {
        "Vehicle": [["a car"]],
        "Details": [["a wheel", "a door", "a window"]],
        "Scene": [["a vehicle", "a building", "a wall"]],
        "Fine-grained": [["a vintage car", "wooden door", "yellow wall"]],
    }

    print("\nRunning object detection with different prompts...")
    results_dict = {}

    for prompt_name, text_labels in prompts.items():
        print(f"\n--- {prompt_name} ---")
        result = detect_objects(
            image,
            processor,
            model,
            device,
            text_labels,
            threshold=0.3,
            text_threshold=0.25,
        )
        image_with_boxes = draw_boxes(image, result, prompt_name)
        results_dict[prompt_name] = image_with_boxes

    print("\nGenerating visualization...")
    visualize_results(image, results_dict)

    print("\nDone!")


if __name__ == "__main__":
    main()
