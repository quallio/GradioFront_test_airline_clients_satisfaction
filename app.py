import gradio as gr
import pandas as pd
import pickle


# Define params names
PARAMS_NAME = [
    "Age",
    "Class",
    "Wifi",
    "Booking",
    "Seat",
    "Checkin"
]

# Load model
with open("model/rf.pkl", "rb") as f:
    model = pickle.load(f)

# Columnas
COLUMNS_PATH = "model/categories_ohe.pickle"
with open(COLUMNS_PATH, 'rb') as handle:
    ohe_tr = pickle.load(handle)


def predict(*args):
    # Esta es otra opción
    #answer_dict = {
    #    param_name: [param_value]
    #    for param_name, param_value in zip(PARAMS_NAME, args)
    #}

    answer_dict = {}

    for i in range(len(PARAMS_NAME)):
        answer_dict[PARAMS_NAME[i]] = [args[i]]

    single_instance = pd.DataFrame.from_dict(answer_dict)
    
    # Reformat columns
    single_instance_ohe = pd.get_dummies(single_instance)
    single_instance_ohe = single_instance_ohe.reindex(columns=ohe_tr).fillna(0)

    prediction = model.predict(single_instance_ohe)

    response = format(prediction[0], '.2f')

    print("answer_dict : ")
    print(answer_dict)

    return response



with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Satisfacción aerolínea 🛫 🌎 🛬
        """
    )

    with gr.Row():
        with gr.Column():

            gr.Markdown(
                """
                ## Predecir satisfacción del cliente
                """
            )

            Age = gr.Slider(label="Edad", minimum=1, maximum=100, step=1, randomize=True)
            
            Class = gr.Radio(
                label="Class",
                choices=["Bussines", "Eco", "Eco Plus"],
                value="Eco"
                )
            
            Wifi = gr.Slider(label="Servicio del WiFi", minimum=0, maximum=5, step=1, randomize=True)

            Booking = gr.Slider(label="Facilidad de registro", minimum=0, maximum=5, step=1, randomize=True)

            Seat = gr.Dropdown(
                label="Comodidad del asiento",
                choices=[0, 1, 2, 3, 4, 5],
                multiselect=False,
                value=0
                )
            
            Checkin = gr.Dropdown(
                label="Experiencia con el ChekIn",
                choices=[0, 1, 2, 3, 4, 5],
                multiselect=False,
                value=0
                )

        with gr.Column():

            gr.Markdown(
                """
                ## Predicción
                """
            )

            label = gr.Label(label="Score")
            predict_btn = gr.Button(value="Evaluar")
            predict_btn.click(
                predict,
                inputs=[
                   Age,
                   Class,
                   Wifi,
                   Booking,
                   Seat,
                   Checkin,
                ],
                outputs=[label],
            )
        

            
    gr.Markdown(
        """
        <p style='text-align: center'>
            <a href='https://www.escueladedatosvivos.ai/cursos/bootcamp-de-data-science' 
                target='_blank'>Proyecto demo creado para ejercitar Gradio & HuggingFace en el bootcamp de EDVAI 🤗
            </a>
        </p>
        """
    )

demo.launch()
