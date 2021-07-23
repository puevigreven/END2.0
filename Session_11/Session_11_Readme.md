# Session 11

## 1. Encoder Feed-forward manual steps
##### Step 1  - Create Embedding and LSTM Objects with appropriate dimension 

```
embedding = nn.Embedding(input_size, hidden_size).to(device)
lstm = nn.LSTM(hidden_size, hidden_size).to(device)
```
##### Embedding Dimensions - 
nn.Embedding(num_embeddings, embedding_dim)
- num_embeddings – size of the dictionary of embeddings => input_lang.n_words
- embedding_dim – the size of each embedding vector => hidden_size = 256 (we have decided to use 256 as dimension)

##### LSTM Dimensions - 
nn.LSTM(input_size, hidden_size) 
- input_size – The number of expected features in the input x => 256 
- hidden_size – The number of features in the hidden state h => 256

##### Step 2 - Initialize encoder_output, hidden and cell_state

```
encoder_outputs = torch.zeros(MAX_LENGTH, 256, device=device)
encoder_hidden = torch.zeros(1, 1, 256, device=device)
encoder_cell_state = torch.zeros( 1, 1, 256, device=device)
```

##### Step 3 - Network 
For every word indices of input sentence, we want to run the encoder function. The encoder function is first embedding layer and then LSTM layer. We are saving encoder_hidden in encoder_outputs because we will need in our decoder. 
```
for i in range(input_tensor.size()[0]):
      embedded_input = embedding(input_tensor[i].view(-1, 1))
      output, (encoder_hidden, encoder_cell_state) = lstm(embedded_input, (encoder_hidden, encoder_cell_state))
      encoder_outputs[i] += encoder_hidden[0,0]
```

##### Complete Code in one place

```
embedding = nn.Embedding(input_size, hidden_size).to(device)
lstm = nn.LSTM(hidden_size, hidden_size).to(device)

encoder_outputs = torch.zeros(MAX_LENGTH, 256, device=device)
encoder_hidden = torch.zeros(1, 1, 256, device=device)
encoder_cell_state = torch.zeros( 1, 1, 256, device=device)
for i in range(input_tensor.size()[0]):
      # print(input_sentence.split(' ')[i]) why will this cause error?
      embedded_input = embedding(input_tensor[i].view(-1, 1))
      output, (encoder_hidden, encoder_cell_state) = lstm(embedded_input, (encoder_hidden, encoder_cell_state))
      encoder_outputs[i] += encoder_hidden[0,0]
```

## Decoder Feed-forward manual steps
In decoder attention mechanism we are using previous hidden output with embedded input to generate attention weights and use these attention weights with encoder outputs to create input for LSTM. 

Step 0 - Create Embedding and LSTM Objects with appropriate dimension
```
embedding = nn.Embedding(output_size, 256).to(device)
attn_weight_layer = nn.Linear(256 * 2, 10).to(device)
input_to_gru_layer = nn.Linear(256 * 2, 256).to(device)
lstm = nn.LSTM(256, 256).to(device)
output_word_layer = nn.Linear(256, output_lang.n_words).to(device)
```

Step 1 - Create SOS token as first input to the decoder 
```
decoder_input = torch.tensor([[SOS_token]], device=device)
decoder_hidden = encoder_hidden
output_size = output_lang.n_words
embedded = embedding(decoder_input)
```

Step 2 - Concat previous hidden with input embedded vector
``` 
self.attn(torch.cat((embedded[0], hidden[0]), 1))
```
Step 3 - Pass the output of previous output to softmax layer, so the sum of weights equals 1 
``` 
attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
```
Step 4 - Performs a batch matrix-matrix product of matrices stored in attn_weights and Encoder outputs.
```
attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
```
Step 5 - Concat Input embedded and attn_applied and pass it to linear layer
```
output = torch.cat((embedded[0], attn_applied[0]), 1)
output = self.attn_combine(output).unsqueeze(0)
```
Step 6 - Pass this output as input to GRU
```
output = F.relu(output)
output, hidden = self.LSTM(output, hidden)
```

Step 7. When doing the second pass on decoder, we take decoder input as decoder output from the previous step. 
```
decoder_input = torch.tensor([[top_index.item()]], device=device) ## Change SOS_indices to top_index.item()
# decoder_input = torch.tensor([[SOS_token]], device=device)
decoder_hidden = encoder_hidden
output_size = output_lang.n_words
embedded = embedding(decoder_input)
attn_weights = attn_weight_layer(torch.cat((embedded[0], decoder_hidden[0]), 1))
attn_weights = F.softmax(attn_weights, dim = 1)
attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
input_to_gru = input_to_gru_layer(torch.cat((embedded[0], attn_applied[0]), 1))
input_to_gru = input_to_gru.unsqueeze(0)
cell_state = torch.zeros( 1, 1, 256, device=device)
output, (decoder_hidden, cell_state) = lstm(input_to_gru, (decoder_hidden, cell_state))
output = F.relu(output)
output = F.softmax(output_word_layer(output[0]), dim = 1)
top_value, top_index = output.data.topk(1)
output_lang.index2word[top_index.item()], attn_weights
```


#### Complete Attention Mechanism Diagram

