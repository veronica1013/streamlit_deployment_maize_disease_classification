# Import libraries

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import torch
import numpy as np
import numpy
import pandas as pd
import cv2
import streamlit as st
from torchvision import models, transforms
import torchvision
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
from transformers import ViTModel
from transformers import ViTFeatureExtractor, ViTModel
from PIL import Image
import requests
import torchvision
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import ViTModel, ViTFeatureExtractor
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn

class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=6):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained(
            'google/vit-base-patch16-224-in21k')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:, 0])
        logits = self.classifier(output)
        return logits
# page selector
my_page = st.sidebar.radio('Page Navigation', ['Home', 'Model'])

if my_page == 'Home':
    #st.title('')
    #button = st.button('a button')
    #if button:
    #st.write('clicked')
    #else:
    #st.title('this is a different page')
    #slide = st.slider('this is a slider')
    #slide

    # Horizontal partitions
    icon = st.container()
    header = st.container()
    overview1 = st.container()
    #spacing = st.container()
    overview2 = st.container()
    predictor = st.container()
    view = st.container()
    recommend = st.container()


# add elements to containers
# icon
    #with icon:
    #st.set_page_config(page_title='Mahindy', page_icon='🖖')
# header
    with header:
        #st.title('MAHINDY')
        st.markdown(
            "<h1 style='text-align: center; color: lightgreen;'>MAHINDY</h1>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='text-align: center;'><i>Maize Crop Disease Identification</i></h4>", unsafe_allow_html=True)
        #st.write('__*Maize Crop Disease Identification.*__')
        st.write('')
        st.write('')
    # overview 1
    with overview1:
        col1, col2 = st.columns(2)
        with col1:
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            maize = 'https://th.bing.com/th/id/OIP.n-fH7-heN1AqnBHt4lrwkgHaE8?w=249&h=180&c=7&r=0&o=5&pid=1.7'
            st.image(maize, width=300)
        with col2:
            st.write('')
            st.write('## Overview')
            st.write("Maize is a cereal crop that was first introduced to Kenya in the 16th century by Arab traders. Millions of Kenyans eat maize as part of their daily diet. The average Kenyan consumes 98kg of maize per year, and maize and maize products account for 28% of the country's income. This crop thrives in warm climates with high rainfall and loamy, well-drained soil.Throughout Kenya, maize growing is practiced in the Rift Valley, Central Kenya, and other locations.")

    # overview 2
    with overview2:
        col1, col2 = st.columns(2)
        with col1:
            st.write('## Maize Disease')
            st.write("The growth of the population has outpaced maize output. Drought, low soil fertility, pests, and diseases are all obstacles to maize production. Maize diseases diminish yields by up to 90% in maize-growing regions such as Kenya and other African countries. Maize is susceptible to a number of leaf diseases, including Maize Gray Leaf Spot, Maize Streak Virus, Common Rust, Head Smut, and Northern Blight. Both the quality and quantity of maize crops are affected by the diseases.")
        with col2:
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            maize2 = 'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.bVeDRo86SZKlTjIHV_x8ZQHaE8%26pid%3DApi&f=1'
            st.image(maize2, width=300)
    # predictor
    with predictor:
        st.write(' ')
        st.write('## Identify Disease')
        st.write(' ')
        ### load file
        uploaded_file = st.file_uploader("Choose an image file", type=[
                                         "jpg", "jpeg", "png", "JPG"])

        #view
        if uploaded_file is not None:
            with view:
                col1, col2 = st.columns(2)

                with col1:
                    uploaded_file = Image.open(uploaded_file)
                    st.image(uploaded_file)
                    Genrate_pred = st.button("Generate Prediction")
                if Genrate_pred:
                                                                                
    # prediction = model(inputs)
    # predicted_class = np.argmax(prediction.cpu())
    # st.write('##### Class')
    # st.write("Predicted Label for the image is {}".format(
        # map_dict[predicted_class]))
    # st.write('##### Confidence interval')
                    with col2:
                        map_dict = {
                            0: 'Blight',
                            1: 'Common_rust',
                            2: 'Gray_Leaf_Spot',
                            3: 'SMUT500',
                            4: 'healthy',
                            5: 'maizestreak_aug'}

                        #if uploaded_file is not None:
                        # Loading the Vision Transformer Model
                        # Model path
                        MODEL_PATH = "D:/veronica_moringa/project_dsp/ViT_2nd_adjusted.pt"
                        # load model
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        if torch.cuda.is_available():
                            model.cuda()

                        # Transform the Model
                        data_transforms = transforms.Compose(
                            [
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                            ])

                        feature_extractor = ViTFeatureExtractor.from_pretrained(
                            'google/vit-base-patch16-224-in21k')
                        input = data_transforms(uploaded_file)

                        # PREDICTION
                        with torch.no_grad():
                            inputs = input
                            inputs = torch.tensor(
                                feature_extractor(inputs)['pixel_values'])
                            #
                            inputs = inputs.to(device)
                            prediction = model(inputs)
                            predicted_class = np.argmax(prediction.cpu())
                            predicted_class = predicted_class.detach().numpy()
                            if predicted_class == 0:
                                result = 'Northern Blight'
                            elif predicted_class == 1:
                                result = 'Common Rust'
                            elif predicted_class == 2:
                                result = 'Gray Leaf Spot'
                            elif predicted_class == 3:
                                result = 'Head Smut'
                            elif predicted_class == 4:
                                result = 'Healthy'
                            else:
                                result = 'Maize Streak'
                            st.write('##### Class')
                            st.write("Predicted disease for the image is:", result)
                            st.write('##### Confidence interval')
       # recommend
                with recommend:
                    col1, col2 = st.columns(2)
                    #result = format(map_dict[prediction])
                    if Genrate_pred:
                        #prediction = model(inputs)
                        #predicted_class = np.argmax(prediction.cpu())
                        #predicted_class = predicted_class.detach().numpy()
                        #result = map_dict[predicted_class]
                        if result != 'Healthy':
                            with col1:
                                st.write('#### Causes/Symptoms')
                                st.write(result)
                                #if Genrate_pred:
                                    #prediction = model(inputs)
                                    #predicted_class = np.argmax(prediction.cpu())
                                if result == 'Common Rust':
                                    st.write(
                                        'Common rust is a fungal disease. \n Mostly develops during cold moist weather.')
                                    st.write(
                                        'Symptoms  often appear after silking. \n Small, round to elongate brown pustules form on both leaf surfaces of the plant.')
                                    st.write(
                                        'As the pustules mature they become brown to black. \n If disease is severe, the leaves may yellow and die early.')
                                elif result == 'Northern Blight':
                                    st.write(
                                        'Blight is caused by the fungus Setosphaeria turcica. \n Symptoms usually appear first on the lower leaves.')
                                    st.write(
                                        'Under moist conditions, dark gray spores are produced, \n entire leaves on severely blighted plants can die,')
                                    st.write(
                                        'so individual lesions are not visible. \n Lesions may occur on the outer husk of ears, but the kernels are not infected.')
                                elif result == 'Maize Streak':
                                    st.write(
                                        'Its a viral disease. Symptoms appear on the leaves 3-7 days after inoculation as pale spots or flecks')
                                    st.write(
                                        'Stunting of severely infected maize is observed.')
                                    st.write(
                                        'Fully elongated leaves develop a chlorosis with broken yellow streaks along the veins,')
                                    st.write(
                                        'contrasting with the dark green color of normal foliage. \n hese streaks arise as a consequence of impaired chloroplast formation \n within MSV-infected photosynthesising cells that surround veins.')
                                elif result == 'Head Smut':
                                    st.write(
                                        'Head smut is a fungal disease that affects cereal crops. \n It is caused by the fungus Sphacelotheca reiliana.')
                                    st.write(
                                        'Infection by smut fungi starts in the soil and the fungi grow through the plant during the season.')
                                    st.write(
                                        'The disease is prevalent in the fields that were exposed to crop stress,\n which includes poor nutrient levels, weeds and moisture.')
                                    st.write(
                                        'Leaf-like proliferations develop on the tassel and ears. \n Ears may be aborted and replaced with a proliferation of leafy tissue.')
                                    st.write(
                                        'Plants also may be severely dwarfed. Disease is most common in soils with nitrogen deficiencies.')
                                elif result == 'Gray Leaf Spot':
                                    st.write(
                                        'Gray leaf spot(GLS) is a common fungal disease is a foliar fungal \n disease that affects maize the most')
                                    st.write(
                                        'significant yield-limiting diseases of corn. Gray leaf spot lesions begin as small necrotic pinpoints with \n')
                                    st.write(
                                        'chlorotic halos, these are more visible when leaves are backlit. \n Coloration of initial lesions can range from tan to brown before sporulation begins.')
                                    st.write(
                                        'Because early lesions are ambiguous, they are easily confused with \n other foliar diseases such as anthracnose leaf blight, eyespot, or common rust.')
                                    st.write(
                                        'As infection progresses, lesions begin to take on a more distinct shape. \n Lesion expansion is limited by parallel leaf veins, resulting in the blocky shaped “spots”.')
                                    st.write(
                                        'As sporulation commences, the lesions take on a more gray coloration. \n Entire leaves can be killed when weather conditions are favorable, and rapid disease progression causes lesions to merge.')
                            with col2:
                                st.write('#### Management')
                                #if Genrate_pred:
                                    #prediction = model(inputs)
                                    #predicted_class = np.argmax(prediction.cpu())
                                    
                                if result == 'Common Rust':
                                    st.write(
                                        'The most effective method of controlling the disease \n is to plant resistant hybrids; application of')
                                    st.write(
                                        'appropriate fungicides may provide some degree on control and reduce disease severity; \n fungicides are most effective when the amount of secondary inoculum is still low,')
                                    st.write(
                                        'generally when plants only have a few rust pustules per leaf.')
                                elif result == 'Gray Leaf Spot':
                                    st.write(
                                        'Crop rotation away from corn can reduce disease pressure, \n but multiple years may be necessary in no-till scenarios.')
                                    st.write(
                                        'Planting hybrids with a high level of genetic resistance \n can help reduce the risk of yield loss due to gray leaf spot infection.')
                                    st.write(
                                        'During the growing season, foliar fungicides can be used to manage gray leaf spot outbreaks.')
                                elif result == 'Maize Streak':
                                    st.write(
                                        'Use certified disease free seed from a registered stockist \n and plant at the onset of the rains and seed varieties that offer')
                                    st.write(
                                        'high disease tolerance such as DEKALB. Inspect the field regularly when the maize is small, looking for diseased plants.')
                                    st.write(
                                        'Uproot infected plants when they first show signs of disease. \n This will keep the disease from spreading to healthy plants.')
                                    st.write(
                                        'Plant maize in an open area to avoid shade as leafhoppers prefer shade.')
                                elif result == 'Head Smut':
                                    st.write(
                                        'Field sanitation and crop rotation prove to be of \n considerable help in reducing the incidence of the disease.')
                                    st.write(
                                        'Use of resistant varieties is the best method of management. However, not much work has so far been successfully done in this respect mainly due to obscure nature of the disease.')
                                elif result == 'Northern Blight':
                                    st.write(
                                        'Use resistant hybrids. Fungicides may be warranted on inbreds \n for seed production during the early stages of this disease. Crop rotation and tillage practices may be helpful in some cases')
                        else:
                            st.write(
                                'Your Crop is Healthy! \n Continue maintaning good farming practices and share with others.')
else:
    header2 = st.container()
    model_description1 = st.container()
    model_description2 = st.container()
    model_description3 = st.container()
    model_parameters = st.container()
    model_evaluation = st.container()

    with header2:
        st.markdown(
            "<h1 style='text-align: center; color: lightgreen;'>MODEL</h1>", unsafe_allow_html=True)
        st.write('')
    
    with model_description1:
        st.write("### Model Description")
        st.write("The Vision Transformer(ViT) model was proposed in an Image is Worth 16x16 Words: Transformers for Image Recognition at Scale by Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. It’s the first paper that successfully trains a Transformer encoder on ImageNet, attaining very good results compared to familiar convolutional architectures.")

    with model_description2:
        architecture = 'https://miro.medium.com/max/582/1*LVN1GN8BFqqnhd2KchlnjA.png'
        st.image(architecture, width=700)

    with model_description3:
        st.write("The system utilizes the vision transformer algorithm developed by hugging face using pytorch.")
        st.write("To feed images to the Transformer encoder, each image is split into a sequence of fixed-size non-overlapping patches, which are then linearly embedded. A[CLS] token is added to serve as representation of an entire image, which can be used for classification. The authors also add absolute position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder.")
        st.write("The Vision Transformer was pre-trained using a resolution of 224x224.")
    
    with model_parameters:
        st.write("### Model Parameters")
        st.write("The following are the parameters that were used in this model: \nLearning_rate=0.0001,\n Epochs=5, \nLoss_function=categorical_crossentropy, \nBatch size=20, \nOptimizer=Adam, \nArchitecture=ViTModel")

    with model_evaluation:
        st.write("### Model Evaluation")
        st.write("The model was initially trained with different parameters and below were the best metrics:")
        st.write("Accuracy: 94%, \nAverage Precision: 94%, \nRecall: 93.5%")
