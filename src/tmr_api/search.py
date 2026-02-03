import os
import torch
import numpy as np
from functools import partial
from .load import load_model, load_json, load_unit_motion_embs_splits, load_keyids_splits

class TMRSearcher:
    def __init__(self, data_dir=None, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Model
        self.model = load_model(self.device)
        
        # Load Embeddings
        self.splits = ["train", "val", "test"]
        self.all_unit_motion_embs = load_unit_motion_embs_splits(self.splits, self.device)
        self.all_keyids = load_keyids_splits(self.splits)
        
        # Load Annotations
        module_dir = os.path.dirname(os.path.abspath(__file__))
        annotations_dir = os.path.join(module_dir, "annotations")
        self.h3d_index = load_json(os.path.join(annotations_dir, "humanml3d.json"))
        self.amass_to_babel = load_json(os.path.join(annotations_dir, "amass_to_babel.json"))

    def search(self, query, split_mode="all", nmax=8):
        """
        Performs text-to-motion retrieval.
        query: str
        split_mode: "all" (train+val+test) or "unseen" (test only)
        nmax: int
        """
        if split_mode == "unseen":
            splits = ["test"]
        else:
            splits = ["train", "val", "test"]
            
        unit_motion_embs = torch.cat([self.all_unit_motion_embs[s] for s in splits])
        keyids = np.concatenate([self.all_keyids[s] for s in splits])

        scores = self.model.compute_scores(query, unit_embs=unit_motion_embs)

        sorted_idxs = np.argsort(-scores)
        best_keyids = keyids[sorted_idxs]
        best_scores = scores[sorted_idxs]

        results = []
        for keyid, score in zip(best_keyids, best_scores):
            if len(results) == nmax:
                break
                
            # Retrieve metadata from annotations
            if keyid not in self.h3d_index:
                continue
                
            dico = self.h3d_index[keyid]
            path = dico["path"]
            
            # Basic filtering matching original app.py logic
            if "M" in keyid: continue # No mirrored
            if "humanact12" in path: continue
            if path not in self.amass_to_babel: continue

            ann = dico["annotations"][0]
            
            # Construct result object
            result = {
                "keyid": keyid,
                "score": round(float(score), 2),
                "text": ann["text"],
                "start": ann["start"],
                "end": ann["end"],
                "path": path,
                "babel_id": self.amass_to_babel[path].zfill(6),
                "fps": dico.get("fps", 20.0) # From metadata if available
            }
            results.append(result)
            
        return results
