from gradio_client import Client
import os
import json
import argparse
from .mesh_generator import SMPLGenerator
from .video_utils import download_and_trim_video


def find_amass_file(amass_root, rel_path):
    """
    Attempts to find the AMASS file using fuzzy matching to handle
    naming differences (spaces vs underscores, _poses vs _stageii).
    """
    # Normalize path separators
    rel_path = rel_path.replace("\\", "/")

    # 1. Try exact match (with and without extension)
    candidates_to_try = [
        os.path.join(amass_root, rel_path),
        os.path.join(amass_root, rel_path + ".npz"),
    ]

    for p in candidates_to_try:
        if os.path.exists(p):
            return p

    # 2. Fuzzy search in the specific directory
    clean_rel_path = rel_path
    if clean_rel_path.endswith(".npz"):
        clean_rel_path = clean_rel_path[:-4]

    dataset_subdir, filename = os.path.split(clean_rel_path)
    search_dir = os.path.join(amass_root, dataset_subdir)

    if not os.path.isdir(search_dir):
        print(f"  [Warning] Directory not found: {search_dir}")
        return None

    try:
        files = [f for f in os.listdir(search_dir) if f.endswith(".npz")]
    except OSError:
        return None

    # Normalization strategy:
    # 1. Replace spaces with underscores
    # 2. Identify "core" name by stripping common suffixes like _poses

    base_name = filename
    sanitized_name = base_name.replace(" ", "_")

    core_name = sanitized_name
    if core_name.endswith("_poses"):
        core_name = core_name[:-6]
    if core_name.endswith("_stageii"):
        core_name = core_name[:-8]

    for f in files:
        f_name = f[:-4]  # remove .npz

        # Exact match of sanitized name
        if f_name == sanitized_name:
            return os.path.join(search_dir, f)

        # Match core name prefix with allowed suffixes
        if f_name.startswith(core_name):
            suffix = f_name[len(core_name) :]
            if suffix in ["_stageii", "_poses", ""]:
                return os.path.join(search_dir, f)

    return None


def main():
    parser = argparse.ArgumentParser(description="TMR Client & Mesh Generator")
    parser.add_argument(
        "--query",
        type=str,
        default="A person is dancing",
        help="Text query for motion search",
    )
    parser.add_argument(
        "--videos", type=int, default=4, help="Number of results to retrieve"
    )
    parser.add_argument(
        "--amass_root",
        type=str,
        required=True,
        help="Root directory of your local AMASS dataset",
    )
    parser.add_argument(
        "--smpl_path",
        type=str,
        required=True,
        help="Path to your SMPL models directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output_meshes",
        help="Directory to save generated meshes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Limit number of meshes to generate for validation (default: 1)",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Download and trim the reference video from BABEL (requires internet)",
    )
    parser.add_argument(
        "--remote",
        nargs="?",
        const="http://127.0.0.1:7860/",
        help="Use remote API server (default: http://127.0.0.1:7860/ if flag is present without value)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Perform search and display results without generating meshes",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for SMPL inference (default: 32)",
    )
    args = parser.parse_args()

    if not args.remote:
        print("Running in LOCAL mode...")
        # Import core logic here to avoid dependency if not used
        try:
            # Assumes running as module: python -m tmr_api.tools.tmr_client
            from ..search import TMRSearcher
        except ImportError:
            try:
                # Fallback if PYTHONPATH is set correctly to src/
                from tmr_api.search import TMRSearcher
            except ImportError:
                print(
                    "Error: Could not import TMRSearcher. Ensure you are running as a module or have installed the package."
                )
                return

        print("Initializing TMR Model (this may take a few seconds)...")
        searcher = TMRSearcher()

        split_mode = "all"  # Default to all motions
        print(f"Searching for: '{args.query}' (Local)")

        # Local search
        results = searcher.search(args.query, split_mode=split_mode, nmax=args.videos)

        data = []
        for res in results:
            data.append(
                {
                    "score": res["score"],
                    "corresponding text": res["text"],
                    "AMASS path": res["path"],
                    "start_time": res["start"],
                    "end_time": res["end"],
                    "fps": res["fps"],
                    "babel_id": res.get("babel_id"),
                }
            )

    else:
        # 1. Initialize API Client
        client_url = args.remote
        print(f"Connecting to API at {client_url}...")
        client = Client(client_url)

        # 2. Get Predictions
        print(f"Searching for: '{args.query}'")
        result = client.predict(
            query=args.query,
            gallery="All motions",
            videos=args.videos,
            api_name="/predict",
        )

        # Handle result format (if it's a file path string or direct JSON object)
        if isinstance(result, str) and os.path.exists(result):
            with open(result, "r") as f:
                data = json.load(f)
        else:
            data = result

    if not data:
        print("No results found.")
        return

    if not data:
        print("No results found.")
        return

    # 3. Initialize Mesh Generator (Loaded ONCE) - Only if not dry run
    if not args.dry_run:
        print("\nInitializing SMPL Generator...")
        try:
            generator = SMPLGenerator(smpl_model_path=args.smpl_path)
        except Exception as e:
            print(f"Failed to initialize SMPL Generator: {e}")
            return

    os.makedirs(args.output_dir, exist_ok=True)

    # 4. Process Results
    items_to_process = data[: args.limit]
    print(f"\nProcessing {len(items_to_process)} results (Limit: {args.limit})...")

    dataset_counts = {}

    for i, item in enumerate(items_to_process):
        amass_rel_path = item.get("AMASS path")
        score = item.get("score")
        text = item.get("corresponding text")

        # Identify source dataset from path (e.g. "KIT/...", "CMU/...")
        source_dataset = amass_rel_path.split("/")[0] if amass_rel_path else "Unknown"
        dataset_counts[source_dataset] = dataset_counts.get(source_dataset, 0) + 1

        print(
            f"\n[{i+1}/{len(items_to_process)}] Score: {score} | Dataset: {source_dataset}"
        )
        print(f"  Text: {text}")
        print(f"  Path: {amass_rel_path}")

        if args.dry_run:
            continue

        if not amass_rel_path:
            print("  No AMASS path in result.")
            continue

        # Resolve full path using fuzzy matching
        full_amass_path = find_amass_file(args.amass_root, amass_rel_path)

        if not full_amass_path:
            print(
                f"  [Error] Could not find file for {amass_rel_path} in {args.amass_root}"
            )
            continue

        start_t = item.get("start_time")
        end_t = item.get("end_time")

        # Create a descriptive directory name
        # Replace slashes and spaces to make it filesystem-friendly
        safe_name = (
            item.get("AMASS path")
            .replace("/", "_")
            .replace(" ", "_")
            .replace(".npz", "")
        )
        dir_name = f"{safe_name}_{start_t:.2f}_to_{end_t:.2f}"
        sequence_dir = os.path.join(args.output_dir, dir_name)

        os.makedirs(sequence_dir, exist_ok=True)

        if args.save_video:
            babel_id = item.get("BABEL keyid")
            if babel_id:
                video_filename = (
                    f"reference_{babel_id}_{start_t:.2f}_to_{end_t:.2f}.mp4"
                )
                video_path = os.path.join(sequence_dir, video_filename)
                download_and_trim_video(babel_id, start_t, end_t, video_path)
            else:
                print("  [Warning] No BABEL ID found, cannot save video.")

        print(f"  Generating sequence from {start_t}s to {end_t}s...")

        # Generate sequence using API-provided timings
        gen_result = generator.generate_sequence(
            full_amass_path,
            sequence_dir,
            start_time=start_t,
            stop_time=end_t,
            batch_size=args.batch_size,
        )

        if gen_result and gen_result.get("success"):
            # Save metadata including query info, ranking, and file details
            metadata_path = os.path.join(sequence_dir, "metadata.json")

            # Use relative path for AMASS
            rel_amass_path = os.path.relpath(full_amass_path, args.amass_root)
            # Use basename for SMPL model
            smpl_model_name = os.path.basename(args.smpl_path)

            metadata = item.copy()
            metadata.update(
                {
                    "query": args.query,
                    "requested_limit": args.limit,
                    "total_videos_api": args.videos,
                    "ranking": i + 1,
                    "resolved_amass_path": rel_amass_path,
                    "smpl_model": smpl_model_name,
                    "fps": gen_result.get("fps"),
                    "frame_count": gen_result.get("count"),
                }
            )
            with open(metadata_path, "w") as meta_f:
                json.dump(metadata, meta_f, indent=4)
            print(f"  Saved metadata (Rank: {i+1}) to {metadata_path}")
        else:
            print(f"  [Error] Generation failed for {full_amass_path}")

    # Print Summary
    print("\n" + "=" * 30)
    print("DATASET SUMMARY")
    print("=" * 30)
    if not dataset_counts:
        print("No results found.")
    else:
        # Sort by count descending
        sorted_counts = sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)
        for ds, count in sorted_counts:
            print(f"{ds}: {count}")
    print("=" * 30 + "\n")


if __name__ == "__main__":
    main()
