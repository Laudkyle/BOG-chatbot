import random
import json
import torch



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json','r') as f:
    intents=json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

hidden_size =data["hidden_size"]
input_size = data["input_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size,hidden_size,output_size).to(device)
model.load_state_dict(model_state)
model.eval()


bot_name = "Joe"
print("Lets have a chat")

while True:
    sentence =input("You :")
    if sentence =="quit":
        break
    sentence =tokenize(sentence)
    X =bag_of_words(sentence, all_words)
    X.reshape(1,X.shape[0])
    X = torch.from_numpy(X).to(device)
    output = model(X).to(device)

    # Check if the output tensor has a single row
    if output.shape[0] != 1:
        # Reshape the output tensor to have a single row
        output = output.reshape(1, output.shape[0])

    # Get the predicted class and its probability
    _, pred = torch.max(output, dim=1)
    tag = tags[pred.item()]
    prob = torch.softmax(output, dim=1)[0][pred.item()]


    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag ==  intent["tag"]:
                print(f"{bot_name} : { random.choice(intent['responses'])}")
    else:
        print(f"{bot_name} : I do not understand ....")


