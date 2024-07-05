import cv2
import torch
import os
import numpy as np
import requests
import imutils
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from MiDaS.midas_net import MidasNet
from utils.database import Database
from utils.model import Gradient_FusionModel
from utils.func import scale_image, save_orig
import torch

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True



def capture_image():
    print("Capturing image from webcam...")
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to capture image")

    # Save the captured image to disk (optional)
    actual_image_path = "actual_image.png"
    cv2.imwrite(actual_image_path, frame)
    print(f"Image captured and saved to {actual_image_path}")

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return img


def get_depth_maps(img, size, device, depth_model):
    print("Generating depth maps...")
    with torch.no_grad():
        low_img, high_img = scale_image(img, size, device)

        low_dep = depth_model.forward(low_img).unsqueeze(0)
        high_dep = depth_model.forward(high_img).unsqueeze(0)

        low_dep = low_dep.max() - low_dep
        high_dep = high_dep.max() - high_dep

    print("Depth maps generated.")
    return low_dep, high_dep


def process_depth_maps(low_dep, high_dep, fusion_model):
    print("Processing depth maps with fusion model...")
    with torch.no_grad():
        low_dep, high_dep, pred = fusion_model.inference(low_dep, high_dep)
    print("Depth maps processed.")
    return low_dep, high_dep, pred



def capture_image_from_ip_camera(url):
    print("Capturing image from IP camera...")
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = imutils.resize(img, width=1000, height=1800)

    # Save the captured image to disk (optional)
    actual_image_path = 'actual_image.png'
    cv2.imwrite(actual_image_path, img)
    print(f"Image captured and saved to {actual_image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img



if __name__ == "__main__":
    MILVUS_HOST = "localhost"
    MILVUS_PORT = "19530"
    MONGO_URI = "mongodb+srv://kumaranuj9470:m19ubk6YZORm5o3c@cluster0.159tlth.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    # MONGO_URI = "mongodb://0.0.0.0:27017"
    DB_NAME = "anoop"
    MILVUS_COLLECTION_NAME = "rgbimagefeatures2"
    EXPECTED_DIM = 16000
    ip_camera_url = "http://192.168.1.3:8080/shot.jpg"


    #initialize the database
    image_store = Database(
        MILVUS_HOST,
        MILVUS_PORT,
        MONGO_URI,
        DB_NAME,
        MILVUS_COLLECTION_NAME,
        EXPECTED_DIM,
    )
    

    size = 192
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")



    # Load MiDaS model
    print("Loading MiDaS model...")
    depth_model = MidasNet("./MiDaS/model.pt", non_negative=True)
    # depth_model = torch.hub.load("intel-isl/MiDaS","MiDaS_small")
    depth_model.to(device)
    depth_model.eval()
    print("MiDaS model loaded.")

    # Load Fusion model
    print("Loading Fusion model...")
    fusion_model = Gradient_FusionModel(dict_path="./models/model_dict.pt")
    fusion_model.to(device)
    fusion_model.eval()
    print("Fusion model loaded.")



    while True:

        print("Choices to take image input :")
        print("1. Take image from local directory")
        print("2. Take image from webcam :")
        print("3. Take image using IP camera :")

        user_input = input("Enter your choice : ")
        try:
            choice = int(user_input)
        except ValueError:
            print("Please enter a valid number.")
            exit()

        if(choice == 1):
            #  Load image
            img_loc = 'a2.jpg'
            img = cv2.imread(img_loc)

        elif(choice ==2):
            # Capture image from webcam
            img = capture_image()
            img_loc = 'actual_image.png'
            cv2.imwrite(img_loc, img)

        elif(choice ==3):
            ip_camera_url = input("Enter IP cam url : ")
            # # Capture image from IP camera
            img = capture_image_from_ip_camera(ip_camera_url)
            img_loc = 'actual_image.png'

        else:
            print("Invalid input")




        # Get depth maps
        low_dep, high_dep = get_depth_maps(img, size, device, depth_model)
        save_orig(img, "initial_depthmap.png", high_dep)


        # Search for similar image in database
        query_image_path = img_loc
        results, retrieved_tensors = image_store.retrieve_similar_images(query_image_path)



        # Process the first retrieved image
        if retrieved_tensors:
            print(f"ID: {results[0].id}, Score: {results[0].distance}")
            first_retrieved_tensor = retrieved_tensors[0]

            #threshold for similarity
            if(results[0].distance > 0.5):
                print("No matching image found in database")
                actual_image_path = img_loc
                image_store.store_image_features(actual_image_path, high_dep)
                break

            first_retrieved_tensor = first_retrieved_tensor.to(device)

            save_orig(img, "retrieved_depth.png", first_retrieved_tensor)
            print("Depth image saved as retrieved_depth.png")


            # Process depth maps using fusion model
            low_dep, high_dep, pred = process_depth_maps(low_dep, first_retrieved_tensor, fusion_model)

            # Save the result
            depth_image_path = "improved_depthmap.png"
            save_orig(img, depth_image_path, pred)
            print(f"Output saved to {depth_image_path}")




        else:
            print("No matching image found in database")
            actual_image_path = img_loc
            image_store.store_image_features(actual_image_path, high_dep)

        again = input("Do you want to capture another image? (y/n): ")
        if again.lower() != 'y':
            break

