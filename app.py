import torch
import torchvision
import create_model
import gradio as gr
from timeit import default_timer as timer




def pred(img):
    model, transform = create_model.create_model(101)

    model.load_state_dict(torch.load(f="resnet_model_test.pth",
                          map_location = torch.device("cpu")))
    

    with open("class_names.txt","r") as f:
        class_names = [class_name.strip() for class_name in f.readlines()]


    Results ={}

    img = transform(img).unsqueeze(dim = 0)

    model = model.eval()
    start = timer()

    with torch.inference_mode():
        pred_logits = model(img)

    pred_probs = torch.softmax(pred_logits, dim = 1)

    
    
    for x in range(101):
        Results[class_names[x]] = pred_probs[0][x]

    end_timer = timer()

    pred_time = round(end_timer-start, 2)

    return Results, pred_time



title = "Resnet Test"

demo = gr.Interface(fn = pred,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes= 5, label="Prediction"),
                             gr.Number(label="Prediction Time(seconds)")],
                             title = title)


demo.launch(debug=True)