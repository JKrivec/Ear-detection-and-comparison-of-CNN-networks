import os
import csv
import shutil


file = open(os.path.join("./annotations/recognition/ids.csv"), "rU")
reader = csv.reader(file, delimiter=',')

# Split into subfolders, only pics from train folder
for row in reader:
	sub_folder, file_name = row[0].split("/")
	if sub_folder == "train":
		new_dir = "./subfoldered_train/" + row[1].zfill(3)
		if not os.path.exists(new_dir):
			os.makedirs(new_dir)
		old_path = "./" + row[0]
		new_path = new_dir + "/" + file_name
		print("copying -> " + old_path + " to ->" + new_path)
		shutil.copy2(old_path, new_path)