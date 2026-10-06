"""Download base SiC and GaN structures from the Materials Project."""
import os
from mp_api.client import MPRester
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("MP_API_KEY")

def fetch_base_structure(material_id, name):
    print(f"Fetching {name} ({material_id})...")
    with MPRester(API_KEY) as mpr:
        doc = mpr.materials.summary.search(material_ids=[material_id])[0]
        structure = doc.structure
        file_path = f"data/raw/{name}.cif"
        structure.to(fmt="cif", filename=file_path)
        print(f"Success! Saved to {file_path}\n")

if __name__ == "__main__":
    fetch_base_structure("mp-11714", "SiC_base")
    fetch_base_structure("mp-804", "GaN_base")