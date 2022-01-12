import torch
import torch.nn.functional as F
from torch import nn

class MyAwesomeModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
import torch
import torch.nn.functional as F
from torch import nn

class MyAwesomeModel(nn.Module):
    def __init__(self, hidden_size, output_size, drop_p=0.3):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.num_classes = output_size
        
        self.arch = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16, 
                      kernel_size=3, 
                      padding=1,
                      stride=1),
            # output dim (16, 28, 28)
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, 
                         stride=2),
            # output dim (16, 14, 14)
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, 
                      out_channels=8, 
                      kernel_size=5, 
                      padding=2),
            nn.Dropout2d(p=drop_p),
            # output dim (8, 14, 14)
            nn.MaxPool2d(kernel_size=2,
                         stride=2),
            # output dim (8, 7, 7)
            nn.ReLU(inplace=True)
        )
        
        # fully connected output layers
        # [(W−K+2P)/S]+1
        self.fc1_features = 8*7*7
        self.fc1 = nn.Linear(in_features=self.fc1_features, 
                             out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size,
                             out_features=self.num_classes)
        


    def forward(self, x):

        x = self.arch(x)
        x = x.view(-1, self.fc1_features)
        x = F.relu(self.fc1(x))

        return F.log_softmax(self.fc2(x), dim=1)

def validation(model, testloader, criterion):
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:

        # images = images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy

def train(model, trainloader, testloader, criterion, optimizer=None, epochs=5, print_every=40):
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    steps = 0
    running_loss = 0
    for e in range(epochs):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
            steps += 1
            
            # Flatten images into a 784 long vector
            # images.resize_(images.size()[0], 784)
            
            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()
                
                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                
                running_loss = 0
                
                # Make sure dropout and grads are on for training
                model.train()
