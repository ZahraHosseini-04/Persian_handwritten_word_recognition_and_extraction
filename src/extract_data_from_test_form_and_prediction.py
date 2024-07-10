import glob
import os
from fuzzywuzzy import fuzz
from difflib import SequenceMatcher
import sqlite3
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from utils.config import check_config_file, check_config_keys
from utils.model import build_model
from utils.preprocessing import preprocess

import tkinter as tk



def aruco_extraction(img):
    """
    Extracts the aruco signs from the given image, and selects the boundaries points of the form

    Args:
        img (numpy.ndarray): an image

    Returns:
        numpy.ndarray or None: boundaries of the form
    """
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)

    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()

    # Detect the markers in the image
    marker_corners, marker_ids, rejected_candidates = cv2.aruco.detectMarkers(
        img, dictionary, parameters=parameters
    )

    # Checks how many markers are detected
    if len(marker_corners) != 4:
        print("{} arucos detected instead of 4!".format(len(marker_corners)))
        return None

    # flatten the marker_corners array
    marker_corners = [mc[0] for mc in marker_corners]

    # corners based on the ids [34: top left, 35:top right, 36:bottom right, 33:bottom left]

    # selects the boundaries clock wise(top left point of the top left marker,
    #                                   top right point of the top right marker,
    #                                   bottom right point of the bottom right marker,
    #                                   bottom left point of the bottom left marker)
    boundaries = np.array(
        [
            marker_corners[int(np.where(marker_ids == 34)[0])][3],
            marker_corners[int(np.where(marker_ids == 35)[0])][2],
            marker_corners[int(np.where(marker_ids == 36)[0])][1],
            marker_corners[int(np.where(marker_ids == 33)[0])][0],
        ],
        dtype=np.float32,
    )
    return boundaries


def form_extraction(img, corners, form_width, form_height):
    """
    Applies perspective to the image and extracts the form

    Args:
        img (numpy.ndarray): an image
        corners (numpy.ndarray): position of the corners of the form
        form_width (int): width of the form
        form_height (int): height of the form

    Returns:
        numpy.ndarray: image of the extracted form
    """
    form_points = np.array(
        [(0, 0), (form_width, 0), (form_width, form_height), (0, form_height)]
    ).astype(np.float32)

    # applies perspective tranformation
    perspective_transformation = cv2.getPerspectiveTransform(corners, form_points)
    form = cv2.warpPerspective(
        img, perspective_transformation, (form_width, form_height)
    )
    return form


# def prediction():
#     with open('prediction.py', 'r', encoding='utf-8') as file:
#         code = file.read()

#     exec(code)


def cell_extraction(
    img, img_path, extracted_path, form_width, form_height, cell_width, cell_height
):
    """
    Extracts cells and the saves them based on the type of the given form

    Args:
        img (numpy.ndarray): an image from the test forms
        img_path (str): path of the image
        extracted_path (str): path of the directory for the final data
        form_width (int): width of the form
        form_height (int): height of the form
        cell_width (int): width of each cell
        cell_height (int): height of each cell
    """
    image_name = os.path.basename(img_path)
    directory_name = os.path.splitext(image_name)[0]
    directory = extracted_path + "/" + directory_name
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Folder '{directory}' created.")
    else:
        print(f"Folder '{directory}' already exists.")
    # Locations are guessed TODO: Find a proper way to apply this
    starting_points = {"ID": (45, 315), "First Name": (45, 445), "Last Name": (45, 580)}
    ending_points = {
        "ID": (557, 395),
        "First Name": (575, 525),
        "Last Name": (575, 670),
    }

    number_of_cells = 8

    for i in range(number_of_cells):
        width = (ending_points["ID"][0] - starting_points["ID"][0]) // 8
        x1 = i * width + starting_points["ID"][0]
        y1 = starting_points["ID"][1]
        x2 = x1 + width
        y2 = ending_points["ID"][1]
        cell = img[y1+5:y2-5, x1+5:x2-5]
        cell = cv2.resize(cell, (cell_width, cell_height))
        cv2.imwrite(directory + "/" + "ID" + "_" + str(i) + ".jpg", cell)

    for i in range(number_of_cells):
        width = (ending_points["First Name"][0] - starting_points["First Name"][0]) // 8
        x1 = i * width + starting_points["First Name"][0]
        y1 = starting_points["First Name"][1]
        x2 = x1 + width
        y2 = ending_points["First Name"][1]
        cell = img[y1:y2, x1:x2]
        cell = cv2.resize(cell, (cell_width, cell_height))
        cv2.imwrite(directory + "/" + "FN" + "_" + str(i) + ".jpg", cell)

    for i in range(number_of_cells):
        width = (ending_points["Last Name"][0] - starting_points["Last Name"][0]) // 8
        x1 = i * width + starting_points["Last Name"][0]
        y1 = starting_points["Last Name"][1]
        x2 = x1 + width
        y2 = ending_points["Last Name"][1]
        cell = img[y1:y2, x1:x2]
        cell = cv2.resize(cell, (cell_width, cell_height))
        cv2.imwrite(directory + "/" + "LN" + "_" + str(i) + ".jpg", cell)


    
    


def extract_and_save(
    test_form_path, extracted_path, form_width, form_height, cell_width, cell_height
):
    """
    Extracts forms from the images and then extracts the cells and saves the results.

    Args:
        test_form_path (str): path of the test form
        extracted_path (str): path for saving the extracted cells
        form_width (int): width of the form
        form_height (int): height of the form
        cell_width (int): width of each cell
        cell_height (int): height of each cell
    """
    for image_path in glob.glob(test_form_path + "/*.*"):
        image = cv2.imread(image_path)
        corners = aruco_extraction(image)
        if corners is None:
            print(f"The image {image_path} is dropped.")
            continue
        form = form_extraction(image, corners, form_width, form_height)
        cell_extraction(
            form,
            image_path,
            extracted_path,
            form_width,
            form_height,
            cell_width,
            cell_height,
        )
    # prediction()


def gui(des_test_form_path):
    import tkinter as tk
    from PIL import Image
    import tkinter.filedialog as fd

    
    def save_image():
        
        files_path = fd.askopenfilenames(parent=root, title='عکس فرم ها را انتخاب کنید', filetypes=[("Imagefiles", "*.jpg *.jpeg *.png")])
        for file_path in files_path:
            image_name = os.path.basename(file_path)
            directory_name = os.path.splitext(image_name)[0]
            image = Image.open(file_path)
            
            image.save(f"{des_test_form_path}/{directory_name}.jpg")
        
        
        close_button.pack()
        close_button.place(relx=0.5, rely=0.7,anchor="center")
        
        
    def close_window():
        root.destroy()

    root = tk.Tk()
    root.geometry('250x200')

    upload_button = tk.Button(root, text="بارگذاری تصویر فرم", command=save_image)
    
    upload_button.pack()
    upload_button.place(relx=0.5, rely=0.3,anchor="center")


    close_button = tk.Button(root, text="نمایش خروجی", command=close_window)
    
     

    root.mainloop()


def data_extraction(config):

    required_keys = [
        "test_forms.test_form_path",
        "test_forms.extracted_path",
        "pre_processing.form_width",
        "pre_processing.form_height",
        "pre_processing.cell_width",
        "pre_processing.cell_height",
    ]

    check_config_keys(config, required_keys)

    test_form_path = config["test_forms"].get("test_form_path")
    extracted_path = config["test_forms"].get("extracted_path")

    form_width = config["pre_processing"].get("form_width")
    form_height = config["pre_processing"].get("form_height")

    cell_width = config["pre_processing"].get("cell_width")
    cell_height = config["pre_processing"].get("cell_height")

    if not os.path.exists(extracted_path):
        os.makedirs(extracted_path)
        print(f"Folder '{extracted_path}' created.")
    else:
        print(f"Folder '{extracted_path}' already exists.")

    gui(test_form_path)

    extract_and_save(
        test_form_path, extracted_path, form_width, form_height, cell_width, cell_height
    )


if __name__ == "__main__":
    config_path = "config/config.yaml"
    config = check_config_file(config_path)
    data_extraction(config)
    



# with open('prediction.py', 'r', encoding='utf-8') as file:
#         code = file.read()

# exec(code)



def check_similarity_with_database(input_word, db_words):
    
    list_of_similar = [] #list includes result and similarity score 50
    list_of_input_word = list(input_word)
    list_of_one_difference = []
    list_of_two_difference = []
    
    for word in db_words:
        word_with_space = word[0]
        word_not_space = str(word[0]).replace(" ", "")
        if len(word_not_space) == len(input_word):
            similarity_score = fuzz.ratio(input_word, word_not_space)
            
            if similarity_score >= 50:  # determining the similarity threshold
                list_of_similar.append([word_not_space, similarity_score, word_with_space])
                
    
    for word, sim, word_with_space in list_of_similar:
        
        if sim  == 100:
            return word_with_space
        
        count_of_sim_char = 0 
        for input_word_char, word_char in zip(list_of_input_word, list(word)):
            if input_word_char == word_char:
                count_of_sim_char += 1
        
        if len(input_word) - count_of_sim_char == 1 :
            list_of_one_difference.append(word_with_space)

        elif len(input_word) - count_of_sim_char == 2 :
            list_of_two_difference.append(word_with_space)


    
        
    if  len(list_of_one_difference) != 0 :
        return ' ,'.join(list_of_one_difference)
    elif len(list_of_two_difference) != 0:
        return ' ,'.join(list_of_two_difference)
    else:
        return "The input word was not recognized!!!"



def compar_db(input_word, db, table):
    # connect to SQLite db
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table}")
    db_words = cursor.fetchall()
            
    list_result = check_similarity_with_database(input_word, db_words)

    # close the database connection
    conn.close()
    return list_result



def predict(config):

    required_keys = [
        "test_forms.extracted_path",
        "pre_processing.cell_width",
        "pre_processing.cell_height",
        "pre_processing.num_classes",
        "pre_processing.gaussian_kernel",
        "model.model_numbers_path",
        "model.model_letters_path",
        "inference.threshold",
    ]
    check_config_keys(config, required_keys)

    form_path = config["test_forms"].get("extracted_path")

    cell_width = config["pre_processing"].get("cell_width")
    cell_height = config["pre_processing"].get("cell_height")

    cell_width = config["pre_processing"].get("cell_width")
    cell_height = config["pre_processing"].get("cell_height")

    gaussian_kernel = config["pre_processing"].get("gaussian_kernel")

    num_classes_numbers = config["pre_processing"].get("num_classes_numbers")
    num_classes_letters = config["pre_processing"].get("num_classes_letters")

    model_numbers_path = config["model"].get("model_numbers_path")
    model_letters_path = config["model"].get("model_letters_path")

    threshold = config["inference"].get("threshold")

    model_numbers = build_model(num_classes_numbers, cell_width, cell_height)
    model_numbers.load_weights(model_numbers_path)

    model_letters = build_model(num_classes_letters, cell_width, cell_height)
    model_letters.load_weights(model_letters_path)

    classes_numbers = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9"
    ]
    classes_letters = [
        "ا",
        "ب",
        "پ",
        "ت",
        "ث",
        "ج",
        "چ",
        "ح",
        "خ",
        "د",
        "ذ",
        "ر",
        "ز",
        "ژ",
        "س",
        "ش",
        "ص",
        "ض",
        "ط",
        "ظ",
        "ع",
        "غ",
        "ف",
        "ق",
        "ک",
        "گ",
        "ل",
        "م",
        "ن",
        "و",
        "ه",
        "ی",
    ]
    
    
    all_forms = []
    forms = glob.glob(form_path + "/*")
    for data_path in forms: #With this loop, all the photos in the extracted folder are dumped in test_path.
        form_name = os.path.basename(data_path)
        test_path = sorted(glob.glob(data_path + "/*.jpg"),reverse=True) #The image of each cell is put in this variable.
        first_name = ""
        last_name = ""
        student_id = ""
        
        for path in test_path:
            image_name = os.path.basename(path)
            image = cv2.imread(path)#Includes a photo of the cell.
            image = cv2.resize(image, (cell_width, cell_height))
            image = preprocess(image, gaussian_kernel)
            image = img_to_array(image) / 255.0
            image = np.expand_dims(image, axis=0)

            if "ID" in image_name:
                predicted = model_numbers.predict(image)[0]
                if predicted[np.argmax(predicted)] >=  threshold:
                    student_id += str(classes_numbers[np.argmax(predicted)])
                else:
                    student_id += " "



            elif "FN" in image_name:
                predicted = model_letters.predict(image)[0]
                if predicted[np.argmax(predicted)] <= 0.24:
                    os.remove(path)
                
                else:
                    if predicted[np.argmax(predicted)] >=  threshold:
                        first_name += str(classes_letters[np.argmax(predicted)])
                        
                    else:
                        first_name += " "



            elif "LN" in image_name:
                predicted = model_letters.predict(image)[0]
                if predicted[np.argmax(predicted)] <= 0.24:
                    os.remove(path)
                
                else:
                    if predicted[np.argmax(predicted)] >=  threshold:
                        last_name += str(classes_letters[np.argmax(predicted)])
                        
                    else:
                        last_name += " "



        # Create the dictionary of each form
        form_data ={"form": form_name, "student_id": student_id[::-1], "fname": compar_db(first_name, 'database/fname.db', 'fname'),"lname": compar_db(last_name, 'database/lname.db', 'lname')}
        
        #Add user information to list of all forms
        all_forms.append(form_data)

    gui(all_forms)




def gui(all_forms):

    # Create the main window
    root = tk.Tk()
    root.title("Form Information")


    # Create a LabelFrame for each form
    for data in all_forms:
        frame = tk.LabelFrame(root, text = f"Form - {data['form']}")
        frame.pack(padx=15, pady=15)

        label_id = tk.Label(frame, text = f"Student ID: {data['student_id']}")
        label_id.pack()

        label_fname = tk.Label(frame, text = f"First name: {data['fname']}")
        label_fname.pack()

        label_lname = tk.Label(frame, text = f"Last name: {data['lname']}")
        label_lname.pack()


        # Disable editing for labels
        label_id.config(state=tk.DISABLED)
        label_fname.config(state=tk.DISABLED)
        label_lname.config(state=tk.DISABLED)

    # Start the main loop
    root.mainloop()

if __name__ == "__main__":
    config_path = "config/config.yaml"
    config = check_config_file(config_path)
    predict(config)