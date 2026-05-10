import os
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(BASE_DIR, "static", "known_faces")

print("\n🧹 Starting the 'Digital MS Paint' Wash...")

for filename in os.listdir(folder_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        file_path = os.path.join(folder_path, filename)
        
        try:
            # 1. Open the image using Pillow
            img = Image.open(file_path)
            
            # 2. Force convert to pure, flat RGB 
            # (This strips transparency and weird smartphone color profiles)
            rgb_img = img.convert('RGB')
            
            # 3. Create a new filename ending strictly in .jpg
            name = os.path.splitext(filename)[0]
            new_file_path = os.path.join(folder_path, f"{name}.jpg")
            
            # 4. Save it as a clean JPEG (This deletes the bad metadata automatically)
            rgb_img.save(new_file_path, "JPEG", quality=100)
            
            # 5. If the original file was a .png or .jpeg, delete the old version
            if file_path.lower() != new_file_path.lower():
                os.remove(file_path)
                
            print(f"  [+] Washed and saved clean copy: {name}.jpg")
            
        except Exception as e:
            print(f"  [!] Could not wash {filename}: {e}")

print("✨ All photos are now squeaky clean and ready for the AI!\n")