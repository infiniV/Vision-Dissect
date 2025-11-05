import matplotlib.pyplot as plt
from ultralytics import YOLO


def visualize_outputs(image_path="test_image.jpg"):
    models = {
        "Detection": "yolo11n.pt",
        "Segmentation": "yolo11n-seg.pt",
        "Pose": "yolo11n-pose.pt",
    }

    fig, axes = plt.subplots(1, len(models), figsize=(18, 6))

    for idx, (name, model_path) in enumerate(models.items()):
        print(f"Processing {name}...")
        model = YOLO(model_path)

        results = model(image_path, verbose=False)[0]

        plotted = results.plot()

        ax = axes[idx]
        ax.imshow(plotted[..., ::-1])
        ax.set_title(name, fontsize=14)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("viz/model_outputs.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: model_outputs.png")


if __name__ == "__main__":
    visualize_outputs(image_path="test_image.jpg")
