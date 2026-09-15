#!/usr/bin/env python3
"""
Batch Reconstruction Pipeline:
Iterates over all patient profiles in .data/patients/ and reconstructs high-definition,
smooth 3D models (baseline.glb + face_HD.png) with seamless anisotropic texture blending.
"""

import os
import sys
import shutil
import glob
from pathlib import Path

# Add ai-engine to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from reconstruct_gnm_fullhead import reconstruct_patient_gnm

ROOT_DIR = Path("/home/ubuntu/dr-vantruong-3d-studio")
PATIENTS_DIR = ROOT_DIR / ".data" / "patients"
PUBLIC_DIR = ROOT_DIR / "public" / "models" / "patients"

def sync_outputs(patient_id: str, rec_dir: Path):
    src_glb = rec_dir / "baseline.glb"
    if not src_glb.exists():
        return
    
    # Destination folders
    dest_dirs = [
        PATIENTS_DIR / patient_id / "models",
        PATIENTS_DIR / patient_id / "reconstruction",
        PUBLIC_DIR / patient_id / "reconstruction",
        PUBLIC_DIR / patient_id / "models",
    ]
    for d in dest_dirs:
        d.mkdir(parents=True, exist_ok=True)
        target_glb = d / "baseline.glb"
        if target_glb.resolve() != src_glb.resolve():
            shutil.copy2(src_glb, target_glb)
        if (rec_dir / "face_HD.png").exists():
            target_png = d / "face_HD.png"
            if target_png.resolve() != (rec_dir / "face_HD.png").resolve():
                shutil.copy2(rec_dir / "face_HD.png", target_png)
        if (rec_dir / "baseline.json").exists():
            target_json = d / "baseline.json"
            if target_json.resolve() != (rec_dir / "baseline.json").resolve():
                shutil.copy2(rec_dir / "baseline.json", target_json)
    
    # Also top level model.glb / baseline.glb
    top_glb1 = PATIENTS_DIR / patient_id / "model.glb"
    if top_glb1.resolve() != src_glb.resolve():
        shutil.copy2(src_glb, top_glb1)
    top_glb2 = PUBLIC_DIR / patient_id / "baseline.glb"
    if top_glb2.resolve() != src_glb.resolve():
        shutil.copy2(src_glb, top_glb2)

def main():
    print("=" * 60)
    print("STARTING UNIVERSAL BATCH RECONSTRUCTION ACROSS ALL PATIENTS")
    print("=" * 60)
    
    all_patients = sorted([d.name for d in PATIENTS_DIR.iterdir() if d.is_dir()])
    print(f"Total patient directories found: {len(all_patients)}")
    
    # Prioritize target patient 50469afc-5822-4858-9ff8-d497e9b0a707
    target_pid = "50469afc-5822-4858-9ff8-d497e9b0a707"
    if target_pid in all_patients:
        all_patients.remove(target_pid)
        all_patients.insert(0, target_pid)
    
    succeeded = []
    skipped = []
    failed = []
    
    for i, pid in enumerate(all_patients, 1):
        p_dir = PATIENTS_DIR / pid
        
        # Check potential photos directory
        candidate_photo_dirs = [
            p_dir / "photos",
            p_dir / "frames",
            PUBLIC_DIR / pid / "photos",
            p_dir,
        ]
        
        photo_dir = None
        for cdir in candidate_photo_dirs:
            if cdir.exists():
                imgs = [f for f in cdir.glob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png") and not f.name.startswith("face_HD")]
                if len(imgs) >= 4:
                    photo_dir = cdir
                    break
        
        if not photo_dir:
            skipped.append((pid, "Insufficient photo set (<4 images)"))
            continue
        
        print(f"\n[{i}/{len(all_patients)}] Processing Patient {pid} (photos in {photo_dir.relative_to(ROOT_DIR)})...")
        out_dir = PUBLIC_DIR / pid / "reconstruction"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            res = reconstruct_patient_gnm(pid, photo_dir, out_dir)
            if res.get("ok"):
                sync_outputs(pid, out_dir)
                succeeded.append(pid)
                print(f"--> [SUCCESS] Patient {pid} reconstructed & synced.")
            else:
                failed.append((pid, str(res.get("error"))))
                print(f"--> [FAIL] Patient {pid}: {res.get('error')}")
        except Exception as e:
            failed.append((pid, str(e)))
            print(f"--> [ERROR] Patient {pid}: {e}")
            
    print("\n" + "=" * 60)
    print("UNIVERSAL BATCH RECONSTRUCTION COMPLETED")
    print(f"Successfully processed: {len(succeeded)}")
    print(f"Skipped (no/few photos): {len(skipped)}")
    print(f"Failed: {len(failed)}")
    print("=" * 60)
    
    if failed:
        print("\nFailed cases summary:")
        for pid, err in failed:
            print(f" - {pid}: {err}")

if __name__ == "__main__":
    main()
