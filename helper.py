import json


def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def find_all(input_str, search_str):
    l1 = []
    index = 0
    while index < len(input_str):
        i = input_str.find(search_str, index)
        if i == -1:
            return l1
        l1.append(i)
        index = i + 1
    return l1


def create_qa_train_dataset(train):
    output = {}
    output['version'] = 'v1.0'
    output['data'] = []

    for line in train:
        paragraphs = []
    
        context = line[1]
    
        qas = []
        question = line[-1]
        qid = line[0]
        answers = []
        answer = line[2]
        if type(answer) != str or type(context) != str or type(question) != str:
            print(context, type(context))
            print(answer, type(answer))
            print(question, type(question))
            continue
        answer_starts = find_all(context, answer)
        for answer_start in answer_starts:
            answers.append({'answer_start': answer_start, 'text': answer})
        qas.append({'question': question, 'id': qid, 'is_impossible': False, 'answers': answers})
    
        paragraphs.append({'context': context, 'qas': qas})
        output['data'].append({'title': 'None', 'paragraphs': paragraphs})
    return output


def read_squad(path):
    """
    This function extracts questions, contexts, and answers from the JSON files into training and validation sets. 
    """
    # open JSON file and load intro dictionary
    with open(path, 'rb') as f:
        squad_dict = json.load(f)

    # create lists for contexts, questions, and answers
    contexts = []
    questions = []
    answers = []
    # iterate through all data 
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                if 'plausible_answers' in qa.keys():
                    access = 'plausible_answers'
                else:
                    access = 'answers'
                for answer in qa[access]:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    # return formatted data lists
    return contexts, questions, answers


def add_end_idx(answers, contexts):
    """
    This function adds an 'answer_end' value to the dict
    """
    # loop through each answer-context pair
    for answer, context in zip(answers, contexts):
        target_text = answer['text']
        start_idx = answer['answer_start']
        # define the end index based on the start index
        end_idx = start_idx + len(target_text)
        if context[start_idx:end_idx] == target_text:
            answer['answer_end'] = end_idx
        else:
            for n in [1, 2]:
                if context[start_idx-n:end_idx-n] == target_text:
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n
                    

def add_original_answer(encodings, answers, tokenizer):
    """
    This function adds GT answers to the Dataloader, tokenises them and pads to the same length
    """
    true_answers = []
    for answer in answers:
        true_answer = answer.get('text')
        true_answer_tokenised = tokenizer(true_answer, truncation=True, padding='max_length')["input_ids"]
        true_answers.append(true_answer_tokenised)
    return true_answers


def add_token_positions(encodings, answers, orig_answer, tokenizer):
    """
    This function adds start_positions and end_positions to the existing Encoding objects
    """
    # initialize lists to contain the token indices of answer start/end
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        # append start/end token position using char_to_token method
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        # end position cannot be found, char_to_token found space, so shift position until found
        shift = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
            shift += 1
    # update our encodings object with the new token-based start/end positions
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions, 'orig_answer': orig_answer})
