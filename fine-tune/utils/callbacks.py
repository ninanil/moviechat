from transformers import TrainerCallback
import os
import subprocess

class SaveAllCheckpointsCallback(TrainerCallback):
    def __init__(self, save_dir, save_steps=50):
        self.save_dir = save_dir
        self.save_steps = save_steps

    def on_save(self, args, state, control, **kwargs):
        if state.global_step % self.save_steps == 0:
            self._save_full_checkpoint(state.global_step)

    def _save_full_checkpoint(self, step):
        checkpoint_dir = os.path.join(self.save_dir, f"checkpoint-{step}")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Save checkpoint using DVC
        try:
            result = subprocess.run(['dvc', 'add', checkpoint_dir], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Successfully added {checkpoint_dir} to DVC.")
            else:
                print(f"Error adding {checkpoint_dir} to DVC: {result.stderr}")
                subprocess.run(["dvc", "push"], check=True, cwd='.')
                print(f"Full training checkpoint saved and pushed to DagsHub: {checkpoint_dir}")
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
