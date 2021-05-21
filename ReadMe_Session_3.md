1. Mention the data representation
	- Created Custom Dataset class 
	- MNIST DATA - Image: 60000 * 28 * 28, Label 60000 * 1 
	- Random Number - 60000 * 1 
2. Mention your data generation strategy
    - Created a Custom Dataset class and in it's \__init__ method initialised all the inputs 
        ```py
        class MNIST_Random(Dataset):
          def __init__(self, train_set):
            self.data = train_set.data
            self.data = self.data.unsqueeze(1)
            self.data = self.data.float()
            
            self.targets = train_set.targets #mnist image true value 
            self.random_input = torch.randint(0,10, (len(train_set.data),)) # random input 
            self.label_2 = self.targets + self.random_input # sum of random input
            
          def __getitem__(self, index):
            img, targets, random_input, sum = self.data[index], int(self.targets[index]), int(self.random_input[index]), int(self.label_2[index])
        
            return img, targets, random_input, sum
          def __len__(self):
            return len(self.data)
        ```
3. Mention how you have combined the two inputs
    - I have concatenated the output of MNIST dataset after passing through series of CNNs with one-hot encoding of random number
        ```py
        def forward(self, x, random):
            random = F.one_hot(random, num_classes=10)
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            mnist_output = self.fc2(x)
            # print(mnist_output.shape,random.shape )
            # Contatenation
            concat = torch.cat([mnist_output, random], dim=-1) # Concatinating the mnist output and random input
            x = self.fc3(concat)
            x = F.relu(x)
            x = self.fc4(concat)
            x = F.relu(x)
            # output = F.log_softmax(mnist_output, dim=1)
            # add_out =  F.log_softmax(x, dim=1)
            return mnist_output, x
        ```
4. Mention how you are evaluating your results
    - I combined the loss of output from MNIST part of the network and output from addition part of the network
    - Used the crossentropy loss as loss function
        ```py
            loss = nn.CrossEntropyLoss()
            l1 = loss(output, target)
            l2 = loss(add_out, rand_sum)
            loss = l1 + l2
        ```
5. Mention "what" results you finally got and how did you evaluate your results
- My code is running but I made some mistake in the code and model could not learn addition. 
6. Mention what loss function you picked and why!
    - I chose Cross Entropy Loss as the loss function because out of both the part of network was giving probailistic output of 10 x 1 and 19 x 1. 
![nn](https://user-images.githubusercontent.com/11247181/119197356-39fa3080-baa5-11eb-87c5-0d6200621f72.png)
