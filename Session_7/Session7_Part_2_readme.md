# Session 7 :  Seq2seq 



# Quora Dataset: 

The dataset consists of over 400,000 lines of potential question duplicate pairs. Each line contains IDs for each question in the pair, the full text for each question, and a binary value that indicates whether the line truly contains a duplicate pair. 

### Data Preparation:




    df = pd.read_csv("/content/drive/MyDrive/END2.0/quora_duplicate_questions.tsv", sep='\t')
    df = df[df['is_duplicate']== 1].reset_index()
    df.shape
        

**Dataset Processing**

    Question1 = Field(tokenize = tokenize_en, 
                sequential = True, 
                include_lengths=True,
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)
    
    Question2 = Field(tokenize = tokenize_en, 
                sequential = True, 
                include_lengths=True,
                init_token = '<sos>', 
                eos_token = '<eos>', 
                lower = True)
                
    fields = [('Question1', Question1), ('Question2', Question2)]
    
    
    example = [Example.fromlist([df.question1[i],df.question2[i]], fields) for i in range(df.shape[0])] 
    
    QuoraDataset = Dataset(example, fields)
    
    (train_data, valid_data) = QuoraDataset.split(split_ratio=[70, 30], random_state = random.seed(SEED))
**Example**

1.     vars(train_data[0])
	    		
	
		{'Question1': ['what',
		  'are',
		  'the',
		  'most',
		  'intellectually',
		  'stimulating',
		  'movies',
		  'you',
		  'have',
		  'ever',
		  'seen',
		  '?'],
		 'Question2': ['what',
		  'are',
		  'the',
		  'most',
		  'intellectually',
		  'stimulating',
		  'films',
		  'you',
		  'have',
		  'ever',
		  'watched',
		  '?']}
	

## Question and Answer Dataset

Manually-generated factoid question/answer pairs with difficulty ratings from Wikipedia articles. Dataset includes articles, questions, and answers.

----------
##  Dataset Preparation

    s08 = pd.read_csv("/S08/question_answer_pairs.txt", sep = '\t',encoding = "ISO-8859-1")
    s09 = pd.read_csv("/S09/question_answer_pairs.txt", sep = '\t',encoding = "ISO-8859-1")
    s10 = pd.read_csv("/S10/question_answer_pairs.txt", sep = '\t',encoding = "ISO-8859-1")
    
    df_qa = s08.append(s09).append(s10).reset_index()
Assigning the weights

	model.embedding.weight.data.copy_(pretrained_embeddings)
	> tensor([[ 0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000], [ 0.0000, 0.0000, 0.0000, ..., 0.0000, 0.0000, 0.0000], [ 0.0120, 0.2075, -0.1258, ..., 0.1387, -0.3605, -0.0350], ..., [ 0.0164, -0.1268, 0.1124, ..., -0.0723, 0.4662, -0.3872], [-0.0592, 0.1091, -0.2313, ..., 0.0914, 0.6806, -0.4423], [-0.3185, -0.0888, 0.0675, ..., -0.2871, 0.6534, -0.5551]])

## Dataset Processing
	Question = Field(tokenize = tokenize_en, 
	            sequential = True, 
	            include_lengths=True,
	            init_token = '<sos>', 
	            eos_token = '<eos>', 
	            lower = True)
	
	Answer = Field(tokenize = tokenize_en, 
	            sequential = True, 
	            include_lengths=True,
	            init_token = '<sos>', 
	            eos_token = '<eos>', 
	            lower = True)
	            
	fields = [('Question', Question), ('Answer', Answer)]
	
	example = [Example.fromlist([str(df_qa.Question[i]),str(df_qa.Answer[i])], fields) for i in range(df_qa.shape[0])] 
	
	QnADataset = Dataset(example, fields)
	
	(train_data, valid_data) = QnADataset.split(split_ratio=[70, 30], random_state = random.seed(SEED))

**Example**

    vars(train_data[0])

    {'Answer': ['nan'],
     'Question': ['what',
      'instrument',
      'was',
      'produced',
      'after',
      'the',
      'xylophone',
      'in',
      'the',
      '1920s',
      '?']}




