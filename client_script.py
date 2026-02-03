from gradio_client import Client
import os
import json
import argparse
from mesh_generator import SMPLGenerator

def main():
    parser = argparse.ArgumentParser(description="TMR Client & Mesh Generator")
    parser.add_argument("--query", type=str, default="A person is dancing", help="Text query for motion search")
    parser.add_argument("--videos", type=int, default=4, help="Number of results to retrieve")
    parser.add_argument("--amass_root", type=str, required=True, help="Root directory of your local AMASS dataset")
    parser.add_argument("--smpl_path", type=str, required=True, help="Path to your SMPL models directory")
    parser.add_argument("--output_dir", type=str, default="output_meshes", help="Directory to save generated meshes")
    parser.add_argument("--limit", type=int, default=1, help="Limit number of meshes to generate for validation (default: 1)")
    args = parser.parse_args()

    # 1. Initialize API Client
    client_url = "http://127.0.0.1:7860/"
    print(f"Connecting to API at {client_url}...")
    client = Client(client_url)

    # 2. Get Predictions
    print(f"Searching for: '{args.query}'")
    result = client.predict(
        query=args.query,
        gallery="All motions",
        videos=args.videos,
        api_name="/predict"
    )

    # Handle result format (if it's a file path string or direct JSON object)
    if isinstance(result, str) and os.path.exists(result):
        with open(result, 'r') as f:
            data = json.load(f)
    else:
        data = result

    if not data:
        print("No results found.")
        return

    # 3. Initialize Mesh Generator (Loaded ONCE)
    print("\nInitializing SMPL Generator...")
    try:
        generator = SMPLGenerator(smpl_model_path=args.smpl_path)
    except Exception as e:
        print(f"Failed to initialize SMPL Generator: {e}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # 4. Process Results
    items_to_process = data[:args.limit]
    print(f"\nProcessing {len(items_to_process)} results (Limit: {args.limit})...")
    
    for i, item in enumerate(items_to_process):
        print(f"\n[{i+1}/{len(items_to_process)}] {item.get('corresponding text')} (Score: {item.get('score')})")
        
        amass_rel_path = item.get('AMASS path')
        if not amass_rel_path:
            print("  No AMASS path in result.")
            continue

        # Construct full path to .npz file
        if not amass_rel_path.endswith('.npz'):
            amass_rel_path += '.npz'
            
        full_amass_path = os.path.join(args.amass_root, amass_rel_path)
        
        start_t = item.get('start_time')
        end_t = item.get('end_time')

        # Create a descriptive directory name
        # Replace slashes and spaces to make it filesystem-friendly
        safe_name = item.get('AMASS path').replace('/', '_').replace(' ', '_').replace('.npz', '')
        dir_name = f"{safe_name}_{start_t:.2f}_to_{end_t:.2f}"
        sequence_dir = os.path.join(args.output_dir, dir_name)
        
        os.makedirs(sequence_dir, exist_ok=True)

        # Save metadata including query info and ranking
        metadata_path = os.path.join(sequence_dir, "metadata.json")
        metadata = item.copy()
        metadata.update({
            "query": args.query,
            "requested_limit": args.limit,
            "total_videos_api": args.videos,
            "ranking": i + 1
        })
        with open(metadata_path, 'w') as meta_f:
            json.dump(metadata, meta_f, indent=4)
        print(f"  Saved metadata (Rank: {i+1}) to {metadata_path}")

        print(f"  Generating sequence from {start_t}s to {end_t}s...")
        
        # Generate sequence using API-provided timings
        generator.generate_sequence(
            full_amass_path, 
            sequence_dir, 
            start_time=start_t, 
            stop_time=end_t
        )

if __name__ == "__main__":
    main()
