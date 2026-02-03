import os
import numpy as np
import torch

class SMPLGenerator:
    def __init__(self, smpl_model_path, model_type='smpl', gender='neutral', device=None):
        """
        Initializes the SMPL model once to avoid reloading overhead.
        """
        try:
            import smplx
        except ImportError:
            raise ImportError("Please install smplx to use this class: pip install smplx")

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(smpl_model_path):
            raise FileNotFoundError(f"SMPL model path not found: {smpl_model_path}")

        print(f"Loading SMPL model from {smpl_model_path} to {self.device}...")
        self.smpl = smplx.create(
            smpl_model_path, 
            model_type=model_type, 
            gender=gender
        ).to(self.device)
        print("SMPL model loaded successfully.")

    def generate(self, amass_npz_path, output_path, frame_index=0):
        """
        Generates a mesh for a specific frame from an AMASS .npz file.
        """
        try:
            import trimesh
        except ImportError:
            raise ImportError("Please install trimesh: pip install trimesh")

        if not os.path.exists(amass_npz_path):
            print(f"Warning: AMASS file not found at {amass_npz_path}")
            return False

        # Load data
        data = np.load(amass_npz_path)
        
        # Extract parameters needed for SMPL
        # AMASS data keys: ['poses', 'betas', 'trans', ...]
        # poses shape: [N, 156] usually (for SMPL)
        all_poses = torch.tensor(data['poses']).float().to(self.device)
        all_trans = torch.tensor(data['trans']).float().to(self.device)
        betas = torch.tensor(data['betas'][:10]).float().unsqueeze(0).to(self.device) # Shape [1, 10]

        # Select specific frame
        if frame_index >= len(all_poses):
            print(f"Frame index {frame_index} out of bounds for {amass_npz_path} (len={len(all_poses)})")
            return False
            
        # Standard SMPL pose splitting:
        # global_orient (root rotation): indices [0:3]
        # body_pose: indices [3:72]
        root_orient = all_poses[frame_index:frame_index+1, :3]
        body_pose = all_poses[frame_index:frame_index+1, 3:72]
        trans = all_trans[frame_index:frame_index+1]

        # Forward pass
        output = self.smpl(
            betas=betas,
            global_orient=root_orient,
            body_pose=body_pose,
            transl=trans
        )
        
        vertices = output.vertices.detach().cpu().numpy()[0]
        faces = self.smpl.faces
        
        # Save mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.export(output_path)
        print(f"Generated mesh: {output_path}")
        return True

    def generate_sequence(self, amass_npz_path, output_dir, start_time=None, stop_time=None, duration=None, start_frame=None, num_frames=None):
        """
        Generates meshes for a sequence of frames from an AMASS .npz file.
        Supports time-based (start_time, stop_time, duration) or frame-based (start_frame, num_frames) slicing.
        """
        try:
            import trimesh
        except ImportError:
            raise ImportError("Please install trimesh: pip install trimesh")

        if not os.path.exists(amass_npz_path):
            print(f"Warning: AMASS file not found at {amass_npz_path}")
            return False

        os.makedirs(output_dir, exist_ok=True)
        data = np.load(amass_npz_path)
        
        # Determine frame rate (default to 60.0 if not found)
        fps = float(data.get('mocap_framerate', 60.0))
        total_frames = len(data['poses'])
        
        # Determine Start Frame
        start_idx = 0
        if start_frame is not None:
            start_idx = int(start_frame)
        elif start_time is not None:
            start_idx = int(start_time * fps)
        
        if start_idx < 0: start_idx = 0
        if start_idx >= total_frames:
            print(f"Start frame {start_idx} exceeds total frames {total_frames}.")
            return False

        # Determine End Frame
        end_idx = total_frames
        if num_frames is not None:
            end_idx = start_idx + int(num_frames)
        elif stop_time is not None:
            end_idx = int(stop_time * fps)
        elif duration is not None:
            end_idx = start_idx + int(duration * fps)
            
        if end_idx > total_frames:
            end_idx = total_frames
        if end_idx < start_idx:
            end_idx = start_idx

        print(f"Generating sequence from frame {start_idx} to {end_idx} (FPS: {fps})...")
        
        # Pre-load data to device
        all_poses = torch.tensor(data['poses']).float().to(self.device)
        all_trans = torch.tensor(data['trans']).float().to(self.device)
        betas = torch.tensor(data['betas'][:10]).float().unsqueeze(0).to(self.device)

        count = 0
        for i in range(start_idx, end_idx):
            output_path = os.path.join(output_dir, f"frame_{i:04d}.obj")
            
            # Extract single frame data
            root_orient = all_poses[i:i+1, :3]
            body_pose = all_poses[i:i+1, 3:72]
            trans = all_trans[i:i+1]

            # Forward pass
            output = self.smpl(
                betas=betas,
                global_orient=root_orient,
                body_pose=body_pose,
                transl=trans
            )
            
            vertices = output.vertices.detach().cpu().numpy()[0]
            faces = self.smpl.faces
            
            # Save mesh
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            mesh.export(output_path)
            count += 1
        
        print(f"Generated {count} meshes in {output_dir}")
        return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate SMPL meshes from AMASS .npz file")
    parser.add_argument("--amass_path", type=str, required=True, help="Path to the specific AMASS .npz file")
    parser.add_argument("--output_dir", type=str, default="output_meshes", help="Directory to save generated meshes")
    parser.add_argument("--smpl_path", type=str, required=True, help="Path to the SMPL model directory")
    
    # Time/Frame control
    parser.add_argument("--start_time", "-ss", type=float, help="Start time in seconds")
    parser.add_argument("--stop_time", "-to", type=float, help="Stop time in seconds")
    parser.add_argument("--duration", "-t", type=float, help="Duration in seconds")
    parser.add_argument("--start_frame", "-sf", type=int, help="Start frame index")
    parser.add_argument("--num_frames", "-n", type=int, help="Number of frames to generate")

    args = parser.parse_args()
    
    try:
        generator = SMPLGenerator(smpl_model_path=args.smpl_path)
        generator.generate_sequence(
            args.amass_path, 
            args.output_dir, 
            start_time=args.start_time,
            stop_time=args.stop_time,
            duration=args.duration,
            start_frame=args.start_frame,
            num_frames=args.num_frames
        )
    except Exception as e:
        print(f"Error: {e}")
