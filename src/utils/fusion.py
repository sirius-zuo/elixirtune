"""LoRA adapter fusion utilities."""

import os
import subprocess
from pathlib import Path

class AdapterFusion:
    """Handle LoRA adapter fusion operations."""
    
    def __init__(self):
        """Initialize adapter fusion utility."""
        pass
    
    def fuse_adapters(self, base_model_path: str, adapter_path: str, 
                     output_path: str, verbose: bool = True) -> str:
        """Fuse LoRA adapters into base model.
        
        Args:
            base_model_path: Path to base model
            adapter_path: Path to adapter directory
            output_path: Path for fused model output
            verbose: Whether to show detailed output
            
        Returns:
            Path to fused model
        """
        # Disable tokenizer parallelism to prevent warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Prepare the fusion command
        cmd = [
            "mlx_lm.fuse",
            "--model", base_model_path,
            "--adapter-path", adapter_path,
            "--save-path", output_path
        ]
        
        if verbose:
            print("="*60)
            print("FUSING LORA ADAPTERS")
            print("="*60)
            print(f"Base model: {base_model_path}")
            print(f"Adapters: {adapter_path}")
            print(f"Output: {output_path}")
            print(f"Command: {' '.join(cmd)}")
            print("-" * 60)
        
        try:
            # Run the fusion command
            if verbose:
                result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            else:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            if verbose:
                print("-" * 60)
                print("✅ Fusion completed successfully!")
                print(f"Fused model saved to: {output_path}")
                print("="*60)
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Fusion failed with exit code {e.returncode}"
            if e.stderr:
                error_msg += f"\nError: {e.stderr}"
            if e.stdout:
                error_msg += f"\nOutput: {e.stdout}"
            
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
        
        except FileNotFoundError:
            error_msg = ("mlx_lm.fuse command not found. Make sure MLX LM is properly installed.\n"
                        "Try: pip install mlx-lm")
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    def validate_fusion_inputs(self, base_model_path: str, adapter_path: str) -> bool:
        """Validate inputs before fusion.
        
        Args:
            base_model_path: Path to base model
            adapter_path: Path to adapter directory
            
        Returns:
            True if inputs are valid
        """
        errors = []
        
        # Check base model path (allow HuggingFace model IDs)
        if not (Path(base_model_path).exists() or self._is_huggingface_model_id(base_model_path)):
            errors.append(f"Base model path does not exist and is not a valid HuggingFace model ID: {base_model_path}")
        
        # Check adapter path
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            errors.append(f"Adapter path does not exist: {adapter_path}")
        elif adapter_dir.is_dir():
            # Check for required adapter files
            adapter_config = adapter_dir / "adapter_config.json"
            adapter_weights = adapter_dir / "adapters.safetensors"
            
            if not adapter_config.exists():
                errors.append(f"adapter_config.json not found in {adapter_path}")
            
            if not adapter_weights.exists():
                errors.append(f"adapters.safetensors not found in {adapter_path}")
        
        if errors:
            print("❌ Validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("✅ Fusion inputs validated successfully")
        return True
    
    def _is_huggingface_model_id(self, model_path: str) -> bool:
        """Check if the path looks like a HuggingFace model ID.
        
        Args:
            model_path: Model path or ID to check
            
        Returns:
            True if it looks like a HuggingFace model ID
        """
        # HuggingFace model IDs typically have format: username/model-name or organization/model-name
        # They don't start with / or contain path separators like local paths
        return (
            '/' in model_path and 
            not model_path.startswith('/') and 
            not model_path.startswith('./') and
            not model_path.startswith('../') and
            len(model_path.split('/')) == 2
        )
    
