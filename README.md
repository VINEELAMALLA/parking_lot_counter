# parking_lot_counter
🔗 Live Demo: https://vineelamalla-parking-lot-counter-app-c4mzmg.streamlit.app/
 project — a smart parking space detection web application built using Streamlit, Using computer vision and deep learning techniques.
🔍 Project Overview:
 This application automatically detects "occupied" and "available" parking spaces from image using real-time object detection model YOLOv8. Intended to support efficient parking space allocation and facilitate user convenience in crowded urban locations.
🧠 Key Technologies & Tools:
-->YOLOv8 (You Only Look Once): The pretrained object detection YOLOv8 is trained with PKLot dataset to increase the accuracy of the model to classify parking spaces and vehicles.
-->OpenCV: Used for image preprocessing, frame extraction, and drawing annotations.
-->Python: Used as the primary language for logic implementation, model integration, and backend processing.
-->Streamlit: Used to build a interactive web interface for deploying the solution.
⚙️ Technical Approach:
-->The app accepts input in the form of images.
-->Using YOLOv8, the system detects and localizes vehicles within the parking lot.
-->The analysis algorithm checks for overlap between detected vehicles and pre-defined parking space coordinates.
-->Based on this analysis, each parking space is labeled as either "occupied" or "available" and visualized through the Streamlit interface.
-->You can download the output as CSV file.

