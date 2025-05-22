import zipfile
with zipfile.ZipFile(r"C:\Users\vinee\OneDrive\parking space\archive1.zip") as zip_ref:
    zip_ref.extractall("PKLot_dataset")