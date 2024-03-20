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
all_words = data[all_words]
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
    X = X.torch.from_numpy()
    output = model(X)

    _,pred = torch.max(output, dim=1)
    tag = tags[pred.item()]

    for intent in intents["intents"]:
        if tag ==  intent["tag"]:
            print(f"{bot_name} : { random.choice(intent['responses'])}")

