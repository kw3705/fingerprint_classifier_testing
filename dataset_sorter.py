import os

input_path = "C:\\Users\\Kevin\\PycharmProjects\\AI_Testing\\unsorted_fingerprint_dataset\\NISTSpecialDatabase4GrayScaleImagesofFIGS\\sd04\\png_txt\\figs_0"
output_path = "C:\\Users\\Kevin\\PycharmProjects\\AI_Testing\\fingerprint_dataset_test"

""" 
    Given the directory at input_path, with pairs of png and txt files that share a name 
    Read the txt file's second line containing the text "Class: C", where C is a character representing the class of the png
    Then, sort the png into the corresponding directory in output_path
"""
def sort_dataset(input_path, output_path):
    for filename in os.listdir(input_path):
        if filename.endswith(".png"):
            # Read the corresponding txt file
            txt_filename = filename.replace(".png", ".txt")
            with open(input_path + "\\" + txt_filename, 'r') as txt_file:
                # Read the second line of the txt file
                txt_file.readline()
                class_name = txt_file.readline().split()[1]
                # Create the directory if it doesn't exist
                if not os.path.exists(output_path + "\\" + class_name):
                    os.makedirs(output_path + "\\" + class_name)
                # Move the png file to the corresponding directory
                os.rename(input_path + "\\" + filename, output_path + "\\" + class_name + "\\" + filename)

def main():
    sort_dataset(input_path, output_path)

if __name__ == "__main__":
    main()