# Reorganize LSA64 dataset into labeled folders
import os
import shutil

# Mapping from https://facundoq.github.io/datasets/lsa64/
mapping = {
    1: "Opaque", 2: "Red", 3: "Green", 4: "Yellow", 5: "Bright", 6: "Light-blue",
    7: "Colors", 8: "Pink", 9: "Women", 10: "Enemy", 11: "Son", 12: "Man",
    13: "Away", 14: "Drawer", 15: "Born", 16: "Learn", 17: "Call", 18: "Skimmer",
    19: "Bitter", 20: "Sweet milk", 21: "Milk", 22: "Water", 23: "Food",
    24: "Argentina", 25: "Uruguay", 26: "Country", 27: "Last name",
    28: "Where", 29: "Mock", 30: "Birthday", 31: "Breakfast", 32: "Photo",
    33: "Hungry", 34: "Map", 35: "Coin", 36: "Music", 37: "Ship", 38: "None",
    39: "Name", 40: "Patience", 41: "Perfume", 42: "Deaf", 43: "Trap", 44: "Rice",
    45: "Barbecue", 46: "Candy", 47: "Chewing-gum", 48: "Spaghetti",
    49: "Yogurt", 50: "Accept", 51: "Thanks", 52: "Shut down", 53: "Appear",
    54: "To land", 55: "Catch", 56: "Help", 57: "Dance", 58: "Bathe",
    59: "Buy", 60: "Copy", 61: "Run", 62: "Realize", 63: "Give", 64: "Find"
}

input_dir = "lsa64_raw/all"
output_dir = "data_lsa64/raw"

os.makedirs(output_dir, exist_ok=True)

videos = [f for f in os.listdir(input_dir) if f.endswith(".mp4")]

for vid in videos:
    # video naming format: <ID>_xxx_xxx.mp4 → extract ID
    id_str = vid.split("_")[0]  
    id_int = int(id_str)        

    if id_int not in mapping:
        print(f"[WARN] ID {id_int} not found in mapping. Skipping: {vid}")
        continue

    label = mapping[id_int]

    # Create label folder
    target_folder = os.path.join(output_dir, label)
    os.makedirs(target_folder, exist_ok=True)

    src = os.path.join(input_dir, vid)
    dst = os.path.join(target_folder, vid)

    shutil.copy(src, dst) 
    print(f"Copied {vid} → {label}/")

print("\nDONE.")
