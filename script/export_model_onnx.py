import argparse
import torch
from model import (
    TomatoNet,
    TomatoViT,
    TomatoComboVitTomatoPCD,
    TomatoComboVitPointNet,
    TomatoPCD,
    PointNet,
)


def load_model(model_type, num_traits):
    if model_type == "resnet50":
        model = TomatoNet(num_traits=num_traits)
    elif model_type == "vit":
        model = TomatoViT(num_traits=num_traits)
    elif model_type == "combo_vit_tomatoPCD":
        model = TomatoComboVitTomatoPCD(num_traits=num_traits)
    elif model_type == "combo_vit_pointNet":
        model = TomatoComboVitPointNet(num_traits=num_traits)
    elif model_type == "tomatoPCD":
        model = TomatoPCD()
    elif model_type == "pointNet":
        model = PointNet()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Export model architecture to ONNX format."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=[
            "resnet50",
            "vit",
            "combo_vit_tomatoPCD",
            "combo_vit_pointNet",
            "tomatoPCD",
            "pointNet",
        ],
        help="Type of model to load.",
    )
    parser.add_argument(
        "--num_traits",
        type=int,
        default=4,
        help="Number of traits the model predicts (not used for TomatoPCD and PointNet).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the ONNX model.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(args.model_type, args.num_traits).to(device)
    model.eval()

    # Dummy input for exporting
    if args.model_type in ["resnet50", "vit"]:
        dummy_input = torch.randn(1, 4, 224, 224).to(device)
    elif args.model_type in ["combo_vit_tomatoPCD", "combo_vit_pointNet"]:
        dummy_input = (
            torch.randn(1, 4, 224, 224).to(device),
            torch.randn(1, 6, 2048).to(device),
        )
    elif args.model_type == "tomatoPCD":
        dummy_input = torch.randn(1, 6, 2048).to(device)
    elif args.model_type == "pointNet":
        dummy_input = torch.randn(1, 6, 2048).to(device)

    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        args.output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
    )

    print(f"Model saved to {args.output_path}")


if __name__ == "__main__":
    main()
