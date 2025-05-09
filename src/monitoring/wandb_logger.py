import wandb
import yaml
import os
from dotenv import load_dotenv

class WandbLogger:
    """Class to handle Weights & Biases logging."""
    
    def __init__(self, config_path: str, run_name: str = None):
        load_dotenv()  # Load environment variables from .env
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Get API key from environment
        api_key = os.getenv("WANDB_API_KEY")
        if not api_key or len(api_key) != 40:
            raise ValueError(f"Invalid WANDB_API_KEY: must be 40 characters long, got {len(api_key) if api_key else 0}")
        
        # Set API key for W&B
        os.environ["WANDB_API_KEY"] = api_key
        
        # Authenticate with W&B
        wandb.login(key=api_key)
        
        # Initialize W&B
        wandb.init(
            project=self.config['wandb']['project'],
            entity=self.config['wandb']['entity'],
            name=run_name or self.config['wandb']['experiment_name'],
            tags=self.config['wandb']['tags'],
            config=self.config,
            settings=wandb.Settings(code_dir=".")
        )
    
    def log_params(self, params: dict):
        """Log model parameters."""
        wandb.config.update(params)
    
    def log_metrics(self, metrics: dict):
        """Log model metrics."""
        wandb.log(metrics)
    
    def log_model(self, model_path: str, model_name: str):
        """Log model artifact."""
        if self.config['wandb']['log_models']:
            artifact = wandb.Artifact(model_name, type='model')
            if os.path.isdir(model_path):
                artifact.add_dir(model_path)
            else:
                artifact.add_file(model_path)
            wandb.log_artifact(artifact)
    
    def log_dataset(self, dataset_path: str, dataset_name: str):
        """Log dataset artifact."""
        if self.config['wandb']['log_datasets']:
            artifact = wandb.Artifact(dataset_name, type='dataset')
            artifact.add_file(dataset_path)
            wandb.log_artifact(artifact)
    
    def finish(self):
        """Finish the W&B run."""
        wandb.finish()