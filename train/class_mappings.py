# Stores class mappings for PlantDoc and NewPlantDiseases and the conversion from PlantDoc -> NewPlantDiseases

newplantdiseases_cls_mapping = {'Apple___Apple_scab': 0, 
                                'Apple___Black_rot': 1,
                                'Apple___Cedar_apple_rust': 2, 
                                'Apple___healthy': 3, 
                                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 4, 
                                'Corn_(maize)___Common_rust_': 5, 
                                'Corn_(maize)___Northern_Leaf_Blight': 6, 
                                'Corn_(maize)___healthy': 7, 
                                'Grape___Black_rot': 8, 
                                'Grape___healthy': 11, 
                                'Potato___Early_blight': 12, 
                                'Potato___Late_blight': 13, 
                                'Potato___healthy': 14, 
                                'Tomato___Bacterial_spot': 15, 
                                'Tomato___Early_blight': 16, 
                                'Tomato___Late_blight': 17, 
                                'Tomato___Leaf_Mold': 18, 
                                'Tomato___Septoria_leaf_spot': 19, 
                                'Tomato___Spider_mites Two-spotted_spider_mite': 20,
                                'Tomato___Target_Spot': 21, 
                                'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 22, 
                                'Tomato___Tomato_mosaic_virus': 23, 
                                'Tomato___healthy': 24}

plantdoc_classes = ['Apple Scab Leaf', 
                    'Apple leaf', 
                    'Apple rust leaf', 
                    'Bell_pepper leaf spot', 
                    'Bell_pepper leaf', 
                    'Blueberry leaf', 
                    'Cherry leaf', 
                    'Corn Gray leaf spot', 
                    'Corn leaf blight', 
                    'Corn rust leaf', 
                    'Peach leaf', 
                    'Potato leaf early blight', 
                    'Potato leaf late blight', 
                    'Potato leaf', 
                    'Raspberry leaf', 
                    'Soyabean leaf', 
                    'Soybean leaf', 
                    'Squash Powdery mildew leaf', 
                    'Strawberry leaf', 
                    'Tomato Early blight leaf', 
                    'Tomato Septoria leaf spot', 
                    'Tomato leaf bacterial spot', 
                    'Tomato leaf late blight', 
                    'Tomato leaf mosaic virus', 
                    'Tomato leaf yellow virus', 
                    'Tomato leaf', 
                    'Tomato mold leaf', 
                    'Tomato two spotted spider mites leaf', 
                    'grape leaf black rot', 'grape leaf']

# maps PlantDoc classes to NewPlantDisease classes
cls_mapping = {'Apple Scab Leaf':0, 
                'Apple leaf':3, 
                'Apple rust leaf':2, 
                'Corn Gray leaf spot':4, 
                'Corn leaf blight':6, 
                'Corn rust leaf':5, 
                'Potato leaf early blight':12, 
                'Potato leaf late blight':13, 
                'Potato leaf':14, 
                'Tomato Early blight leaf':16, 
                'Tomato Septoria leaf spot':19, 
                'Tomato leaf bacterial spot':21, 
                'Tomato leaf late blight':17, 
                'Tomato leaf mosaic virus':23, 
                'Tomato leaf yellow virus':22, 
                'Tomato leaf':24, 
                'Tomato mold leaf':18, 
                'Tomato two spotted spider mites leaf':20, 
                'grape leaf black rot':8, 
                'grape leaf':11}

def print_mappings():
    """
    Prints a dict that maps PlantDoc classes to NewPlantDiseases classes
    Only used for checking that mappings point to correct classes
    """
    # check the classes are mapped correctly first by printing a dict with names mapped to names
    keys = list(newplantdiseases_cls_mapping.keys())
    values = list(newplantdiseases_cls_mapping.values())
    print(dict(map(lambda k_v: (k_v[0], keys[values.index(k_v[1])]), cls_mapping.items())))