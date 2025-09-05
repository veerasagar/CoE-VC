
#!/usr/bin/env python3
"""
run.py - Main entry point for AI Fashion Design System

Usage:
    python run.py --mode generate --text "summer dress"
    python run.py --mode train --model mamba
    python run.py --mode demo
"""

import argparse
import yaml
import torch
import os
from pathlib import Path

def load_config(config_path="configs/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_device():
    """Setup computing device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    return device

def generate_patterns(args, config):
    """Generate fashion patterns"""
    print(f"🎨 Generating patterns with prompt: '{args.text}'")

    # Import models
    from models.garment_diffusion_mamba import GarmentDiffusionMamba, GarmentDiffusionConfig

    # Setup
    device = setup_device()
    model_config = GarmentDiffusionConfig(**config['model']['mamba_diffusion'])
    model = GarmentDiffusionMamba(model_config).to(device).eval()

    # Generate
    conditioning = {'text': args.text}

    with torch.no_grad():
        outputs = model.generate(
            conditioning=conditioning,
            num_inference_steps=config['generation']['num_inference_steps'],
            guidance_scale=config['generation']['guidance_scale']
        )

    print(f"✅ Generated pattern shape: {outputs['edge_parameters'].shape}")

    # Save outputs
    output_dir = Path(config['generation']['output_dir'])
    output_dir.mkdir(exist_ok=True)

    torch.save(outputs, output_dir / f"generated_pattern_{args.text.replace(' ', '_')}.pt")
    print(f"💾 Saved to: {output_dir}")

def generate_3d_garment(args, config):
    """Generate 3D garments with NeRF"""
    print(f"🎭 Generating 3D garment: '{args.text}'")

    # Import NeRF model
    from models.nerf_fashion import RevolutionaryFashionNeRF, NeRFFashionConfig

    # Setup
    device = setup_device()
    model_config = NeRFFashionConfig(**config['model']['nerf_fashion'])
    model = RevolutionaryFashionNeRF(model_config).to(device).eval()

    # Generate 3D garment
    with torch.no_grad():
        garment_3d = model.generate_garment(
            text_description=args.text,
            style_image=None
        )

    print(f"✅ Generated 3D garment:")
    print(f"   RGB shape: {garment_3d['rendered_rgb'].shape}")
    print(f"   Cloth simulation: {garment_3d['cloth_simulation']['positions'].shape}")

    # Save 3D outputs
    output_dir = Path(config['generation']['output_dir']) / "3d_garments"
    output_dir.mkdir(exist_ok=True, parents=True)

    torch.save(garment_3d, output_dir / f"3d_garment_{args.text.replace(' ', '_')}.pt")
    print(f"💾 3D garment saved to: {output_dir}")

def train_model(args, config):
    """Train fashion models"""
    print(f"🏋️ Training {args.model} model...")

    # Import trainer
    from src.trainer import FashionTrainer
    from src.data_loader import create_fashion_loaders

    # Setup
    device = setup_device()

    # Create data loaders
    data_loaders = create_fashion_loaders(config['data'])
    train_loader = data_loaders['train']
    val_loader = data_loaders.get('val', data_loaders.get('test'))

    # Initialize trainer
    trainer = FashionTrainer(
        model_type=args.model,
        config=config,
        device=device
    )

    # Start training
    trainer.train(train_loader, val_loader)
    print("✅ Training completed!")

def run_demo(config):
    """Run interactive demo"""
    print("🎪 Starting interactive demo...")

    try:
        # Try to start Jupyter notebook
        os.system("jupyter notebook notebooks/tutorial.ipynb")
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Could not start demo: {e}")
        print("💡 Try running: jupyter notebook notebooks/tutorial.ipynb")

def main():
    parser = argparse.ArgumentParser(description="AI Fashion Design System")
    parser.add_argument("--mode", choices=["generate", "generate-3d", "train", "demo"], 
                       required=True, help="Operation mode")
    parser.add_argument("--model", choices=["mamba", "nerf", "vae", "gan"], 
                       default="mamba", help="Model to use")
    parser.add_argument("--text", type=str, default="elegant summer dress", 
                       help="Text prompt for generation")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Configuration file path")

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"❌ Config file not found: {args.config}")
        print("💡 Make sure config.yaml exists in configs/ directory")
        return

    print("🚀 AI Fashion Design System")
    print("=" * 40)

    # Route to appropriate function
    if args.mode == "generate":
        generate_patterns(args, config)
    elif args.mode == "generate-3d":
        generate_3d_garment(args, config)
    elif args.mode == "train":
        train_model(args, config)
    elif args.mode == "demo":
        run_demo(config)

    print("\n🎉 Operation completed!")

if __name__ == "__main__":
    main()
