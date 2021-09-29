# Import libraries

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import torch
import numpy as np
import pandas as pd
import cv2
import streamlit as st

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
        st.markdown("<h1 style='text-align: center; color: lightgreen;'>MAHINDY</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'><i>Maize Crop Disease Identification</i></h4>", unsafe_allow_html=True)
        #st.write('__*Maize Crop Disease Identification.*__')
        st.write('')
        st.write('')
    # overview 1
    with overview1:
        col1,col2 = st.columns(2)
        with col1:
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            maize ='https://th.bing.com/th/id/OIP.n-fH7-heN1AqnBHt4lrwkgHaE8?w=249&h=180&c=7&r=0&o=5&pid=1.7'
            st.image(maize, width=300)
        with col2:
            st.write('')
            st.write('## Overview')
            st.write("Maize is a cereal crop that was first introduced to Kenya in the 16th century by Arab traders. Millions of Kenyans eat maize as part of their daily diet. The average Kenyan consumes 98kg of maize per year, and maize and maize products account for 28% of the country's income. This crop thrives in warm climates with high rainfall and loamy, well-drained soil.Throughout Kenya, maize growing is practiced in the Rift Valley, Central Kenya, and other locations.")

    # overview 2
    with overview2:
        col1,col2 = st.columns(2)
        with col1:
            st.write('## Maize Disease')
            st.write("The growth of the population has outpaced maize output. Drought, low soil fertility, pests, and diseases are all obstacles to maize production. Maize diseases diminish yields by up to 90% in maize-growing regions such as Kenya and other African countries. Maize is susceptible to a number of leaf diseases, including Maize Gray Leaf Spot, Maize Streak Virus, Common Rust, Head Smut, and Northern Blight. Both the quality and quantity of maize crops are affected by the diseases.")
        with col2:
            st.write('')
            st.write('')
            st.write('')
            st.write('')
            maize2 = 'https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ftse1.mm.bing.net%2Fth%3Fid%3DOIP.bVeDRo86SZKlTjIHV_x8ZQHaE8%26pid%3DApi&f=1'
            st.image(maize2,width=300)
    # predictor
    with predictor:
        st.write(' ')
        st.write('## Identify Disease')
        st.write(' ')
        #uploaded_file = st.file_uploader("Upload your image")

        # Load the model
        model = tf.keras.models.load_model(
            "D:\\veronica_moringa\\project_dsp\\MobileNet_Epoch_20.pd")
        ### load file
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "JPG"])

            #view
        if uploaded_file is not None:
            with view:
                col1,col2 =st.columns(2)
    
                with col1:
                    map_dict = {0: 'Blight',
                                1: 'Common_rust',
                                2: 'Gray_Leaf_Spot',
                                3: 'SMUT500',
                                4: 'healthy',
                                5: 'maizestreak_aug'}
        #if uploaded_file is not None:
            # Convert the file to an opencv image.
                    file_bytes = np.asarray(
                        bytearray(uploaded_file.read()), dtype=np.uint8)
                    opencv_image = cv2.imdecode(file_bytes, 1)
                    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
                    resized = cv2.resize(opencv_image, (224, 224))
                    # Now do something with the image! For example, let's display it:
                    st.image(opencv_image, channels="RGB")
                    resized = mobilenet_v2_preprocess_input(resized)
                    img_reshape = resized[np.newaxis, ...]
                    Genrate_pred = st.button("Generate Prediction")
                          #st.write('Test')
                    # Convert the file to an opencv image.
                    # file_bytes = np.asarray(bytearray(uploaded_file), dtype=np.uint8)
                    # opencv_image = cv2.imdecode(file_bytes, 1)
    # 
                   # Now do something with the image! For example, let's display it:
                    # st.image(opencv_image, channels="BGR")
             # predictor
                with col2:
                    if Genrate_pred:
                        prediction = model.predict(img_reshape).argmax()
                        st.write('##### Class')
                        st.write("Predicted Label for the image is {}".format(map_dict[prediction]))
                        st.write('##### Confidence interval')
        # recommend
                with recommend:
                    
                    col1,col2 = st.columns(2)
                    #result = format(map_dict[prediction])
                    if Genrate_pred:
                        prediction = model.predict(img_reshape).argmax()
                        result = map_dict[prediction]
                        if result != 'healthy':

                            with col1:

                                st.write('#### Causes/Symptoms')
                                st.write(result)
                                if Genrate_pred:
                                    prediction = model.predict(img_reshape).argmax()

                                    if result == 'Common_rust':
                                        st.write('Common rust is a fungal disease. \n Mostly develops during cold moist weather.')
                                        st.write('Symptoms  often appear after silking. \n Small, round to elongate brown pustules form on both leaf surfaces of the plant.')
                                        st.write('As the pustules mature they become brown to black. \n If disease is severe, the leaves may yellow and die early.')
                                    elif result == 'Blight':
                                        st.write('Blight is caused by the fungus Setosphaeria turcica. \n Symptoms usually appear first on the lower leaves.')
                                        st.write('Under moist conditions, dark gray spores are produced, \n entire leaves on severely blighted plants can die,')
                                        st.write('so individual lesions are not visible. \n Lesions may occur on the outer husk of ears, but the kernels are not infected.')
                                    elif result == 'maizestreak_aug':
                                        st.write('Its a viral disease. Symptoms appear on the leaves 3-7 days after inoculation as pale spots or flecks')
                                        st.write('Stunting of severely infected maize is observed.')
                                        st.write('Fully elongated leaves develop a chlorosis with broken yellow streaks along the veins,')
                                        st.write('contrasting with the dark green color of normal foliage. \n hese streaks arise as a consequence of impaired chloroplast formation \n within MSV-infected photosynthesising cells that surround veins.')
                                    elif result == 'SMUT500':
                                        st.write('Head smut is a fungal disease that affects cereal crops. \n It is caused by the fungus Sphacelotheca reiliana.')
                                        st.write('Infection by smut fungi starts in the soil and the fungi grow through the plant during the season.')
                                        st.write('The disease is prevalent in the fields that were exposed to crop stress,\n which includes poor nutrient levels, weeds and moisture.')                     
                                        st.write('Leaf-like proliferations develop on the tassel and ears. \n Ears may be aborted and replaced with a proliferation of leafy tissue.')
                                        st.write('Plants also may be severely dwarfed. Disease is most common in soils with nitrogen deficiencies.')
                                    elif result == 'Gray_Leaf_Spot':
                                        st.write('Gray leaf spot(GLS) is a common fungal disease is a foliar fungal \n disease that affects maize the most')
                                        st.write('significant yield-limiting diseases of corn. Gray leaf spot lesions begin as small necrotic pinpoints with \n')
                                        st.write('chlorotic halos, these are more visible when leaves are backlit. \n Coloration of initial lesions can range from tan to brown before sporulation begins.')
                                        st.write('Because early lesions are ambiguous, they are easily confused with \n other foliar diseases such as anthracnose leaf blight, eyespot, or common rust.')
                                        st.write('As infection progresses, lesions begin to take on a more distinct shape. \n Lesion expansion is limited by parallel leaf veins, resulting in the blocky shaped “spots”.')
                                        st.write('As sporulation commences, the lesions take on a more gray coloration. \n Entire leaves can be killed when weather conditions are favorable, and rapid disease progression causes lesions to merge.')
                            with col2:
                                st.write('#### Management')
                                if Genrate_pred:
                                    prediction = model.predict(img_reshape).argmax()
                                    if result == 'Common_rust':
                                        st.write('The most effective method of controlling the disease \n is to plant resistant hybrids; application of')
                                        st.write('appropriate fungicides may provide some degree on control and reduce disease severity; \n fungicides are most effective when the amount of secondary inoculum is still low,')
                                        st.write('generally when plants only have a few rust pustules per leaf.')
                                    elif result == 'Gray_Leaf_Spot':
                                        st.write('Crop rotation away from corn can reduce disease pressure, \n but multiple years may be necessary in no-till scenarios.')
                                        st.write('Planting hybrids with a high level of genetic resistance \n can help reduce the risk of yield loss due to gray leaf spot infection.')
                                        st.write('During the growing season, foliar fungicides can be used to manage gray leaf spot outbreaks.')
                                    elif result == 'maizestreak_aug':
                                        st.write('Use certified disease free seed from a registered stockist \n and plant at the onset of the rains and seed varieties that offer')
                                        st.write('high disease tolerance such as DEKALB. Inspect the field regularly when the maize is small, looking for diseased plants.')
                                        st.write('Uproot infected plants when they first show signs of disease. \n This will keep the disease from spreading to healthy plants.')
                                        st.write('Plant maize in an open area to avoid shade as leafhoppers prefer shade.')
                                    elif result == 'SMUT500':
                                        st.write('Field sanitation and crop rotation prove to be of \n considerable help in reducing the incidence of the disease.')
                                        st.write('Use of resistant varieties is the best method of management. However, not much work has so far been successfully done in this respect mainly due to obscure nature of the disease.')
                                    elif result == 'Blight':
                                        st.write('Use resistant hybrids. Fungicides may be warranted on inbreds \n for seed production during the early stages of this disease. Crop rotation and tillage practices may be helpful in some cases')
                        else:
                            st.write('Your Crop is Healthy! \n Continue maintaning good farming practices and share with others.')

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
        st.write("MobileNets is TensorFlow's first mobile computer vision model, and it is based on a streamlined design that leverages depth-wise separable convolutions to generate light weight deep neural networks")

    with model_description2:
        architecture = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTETshb044phh601qhX_8BkJ4mv3wUvWYILM39OmW-5KR3CfezM-mUovBcoaIEov1loNvI&usqp=CAU'
        st.image(architecture, width=700)

    with model_description3:
        st.write("This convolution originated from the idea that a filter’s depth and spatial dimension can be separated- thus, the name separable. Let us take the example of Sobel filter, used in image processing to detect edges.")
        st.write("MobileNets are small, low-latency, low-power models parameterized to meet the resource constraints of a variety of use cases. They can be built upon for classification, detection, embeddings, and segmentation.")

    with model_parameters:
        st.write("### Model Parameters")
        st.write("The following are the parameters that were used in this model: \nLearning_rate=0.0001,\n Epochs=50, \nLoss_function=categorical_crossenthropy, \nFine-tuned layers=10, \nOptimizer=Adagrad, \nArchitecture=ModelNet")

    with model_evaluation:
        st.write("### Model Evaluation")
        st.write("The model was initially trained with different parameters and below were the best metrics:")
        st.write("Accuracy: 84%, \nPrecision: 68%, \nRecall: 60%")

                 


 

    

                                                                                                                                                                                                                                                                                                                                                                    
