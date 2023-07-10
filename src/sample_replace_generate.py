import argparse
import json
import random
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from sklearn.model_selection import train_test_split

def random_replace(question, classifier, context_nouns):
    pos = classifier(question)
    #print(pos)
    nouns = []
    for p in pos:
        if p['entity_group'] in ['PROPN', 'NOUN']:
            nouns.append(p['word'])
    #print(nouns)
    #print(context_nouns)
    new_question = question.lower().replace(random.choice(nouns), random.choice(context_nouns))
    #print(new_question)
    return new_question

def generate_question_swap(classifier, all_questions, all_types, context, question, ratio):
    # get list of nouns from context
    pos = classifier(context)
    nouns = []
    for p in pos:
        if p['entity_group'] in ['PROPN', 'NOUN']:
            nouns.append(p['word'])

    # generate new questions
    new_questions = set()
    for _ in range(ratio):
        # get label by question id
        try: 
            label = all_questions[question]
            # get a random new question in the same category 
            new = random.choice(all_types[label])
            # replace a random noun in the new question with some keywords in context 
            new_questions.add(random_replace(new, classifier, nouns))
        except:
            # print("Error occurs at", question)
            continue
      
    return new_questions

def generate_question_replace(classifier, context, question, ratio):
    # get NER for context 
    ner = classifier(context)    
    ner_dict = dict()
    ner_list = dict()
    for item in ner:
        ner_dict[item['word']] = item['entity_group']
        if item['entity_group'] not in ner_list:
            ner_list[item['entity_group']] = [item['word']]
        else:
            ner_list[item['entity_group']].append(item['word'])

    # generate questions
    new_questions = set()
    for item in ner_dict:
        # if find NE in question, replace that with a random entity 
        if item in question:
            ner_type = ner_dict[item]
            rand_list = [i for i in ner_list[ner_type] if i != item]
            if len(rand_list) < 1:
                # print("No new question for:", question)
                break
            # add the new question to the list 
            for _ in range(ratio):
                new_questions.add(question.replace(item, random.choice(rand_list)))
            break 

    return new_questions   

def load_and_cache_examples(file_path, split=True):
    
    print("Creating examples from dataset file at", file_path)

    with open(file_path) as file: 
        data = json.load(file)
    if split:
        x_train, x_dev = train_test_split(data,test_size=0.2)
        ex = [x_train, x_dev] 
    else:
        i = 0
        ex = []
        while i+20000 < len(data):
            ex.append(data[i:i+20000])
            i += 20000
        ex.append(data[i:])
            
    print("Done getting examples! \n")

    return ex

def main():
    # parse arguments 
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--ratio",
        default=2,
        type=int,
        required=True,
        help="The ratio of new questions to old questions",
    )
    parser.add_argument(
        "--file_path",
        default='data/mrc/vi/squad_2.0/squad_train-v2.0.json',
        type=str,
        required=True,
        help="The path to file",
    )
    parser.add_argument(
        "--output_dir",
        default='data/mrc/vi/squad_2.0',
        type=str,
        required=True,
        help="The path to save file",
    )
    parser.add_argument(
        "--generate_type",
        default='replace',
        type=str,
        required=True,
        help="Sample generation strategy",
    )
    parser.add_argument("--split", action="store_true", help="Whether to split data into train & dev.")
    parser.add_argument("--context_cache", action="store_true", help="Whether to load context from cache.")

    args = parser.parse_args()

    # get pipeline 
    if args.generate_type=='replace':
        classifier = pipeline("ner", aggregation_strategy='average')
    else:
        classifier = pipeline("token-classification", model = "vblagoje/bert-english-uncased-finetuned-pos", aggregation_strategy="simple")
        print("Downloaded pipeline.")
        with open('data/mrc/vi/squad_2.0/squad_labelled_types.json') as file: 
            all_types = json.load(file)
        with open('data/mrc/vi/squad_2.0/id_labelled.json') as file:
            label_dict = json.load(file)

    with open("data/mrc/vi/squad_2.0/squad_train-v2.0.json") as file:
        all_data = json.load(file)['data']

    # all_data = load_and_cache_examples(file_path=args.file_path, split=args.split)

    # get all context
    print("Load context from data...")
    if args.context_cache:
        with open('data/mrc/vi/squad_2.0/squad_context_{}.json'.format(args.generate_type)) as file: 
            context = json.load(file)
    else:
        all_context = set()
        for data in all_data:
            for paragraph in data['paragraphs']:
                all_context.add(paragraph['context'])
        context = dict()
        if args.generate_type=='replace':
            for cont in tqdm(all_context):
                # replace words in questions
                # get NER for context 
                ner = classifier(cont)    
                ner_dict = dict()
                ner_list = dict()
                for n in ner:
                    ner_dict[n['word']] = n['entity_group']
                    if n['entity_group'] not in ner_list:
                        ner_list[n['entity_group']] = [n['word']]
                    else:
                        ner_list[n['entity_group']].append(n['word'])
                context[cont] = {'ner_dict': ner_dict, 'ner_list': ner_list}
        else:
            for cont in tqdm(all_context):
                # swap questions
                # get list of nouns from context
                pos = classifier(cont)
                nouns = []
                for p in pos:
                    if p['entity_group'] in ['PROPN', 'NOUN']:
                        nouns.append(p['word'])
                context[cont] = nouns 

        with open('data/mrc/vi/squad_2.0/squad_context_{}.json'.format(args.generate_type), "w") as outfile:
            json_object = json.dumps(context, indent=4)
            outfile.write(json_object)

    # save data as splits before adding new:
    if args.split:
        with open(args.output_dir+"/squad_train_new.json", "w") as outfile:
            json_object = json.dumps(all_data[0], indent=4)
            outfile.write(json_object)

        with open(args.output_dir+"/squad_dev_new.json", "w") as outfile:
            json_object = json.dumps(all_data[1], indent=4)
            outfile.write(json_object)

    # generate
    print("\nStart generating new samples...")
    count = 0
    count_all = 0

    # new ids
    unique_id = 300000000000000000000000

    for data in all_data:
        for paragraph in data['paragraphs']:
            cont = paragraph['context']
            new_items = []
            # print(data[0]['question_text'])
            new_questions = []
            for item in paragraph['qas']:
                if args.generate_type=='replace':
                    # replace words from question to make impossible questions
                    # generate questions
                    for n in context[cont]['ner_dict']:
                        # if find NE in question, replace that with a random entity 
                        if n in item['question']:
                            ner_type = context[cont]['ner_dict'][n]
                            rand_list = [i for i in context[cont]['ner_list'][ner_type] if i != n]
                            if len(rand_list) < 1:
                                # print("No new question for:", question)
                                break
                            # add the new question to the list                                 
                            new_questions.append(item['question'].replace(n, random.choice(rand_list)))
                            break  
                else:
                    # swap questions
                    # generate new questions
                    # get label
                    try: 
                        label = label_dict[item['id']]
                        # get a random new question in the same category 
                        new = random.choice(all_types[label])
                        # replace a random noun in the new question with some keywords in context 
                        new_questions.append(random_replace(
                            question=new, classifier=classifier, context_nouns=context[cont]))
                    except:
                        # print("Error occurs at", question)
                        continue

            # add new questions to the database 
            for new in random.choices(new_questions, k=(len(paragraph['qas'])//args.ratio)):
                new_items.append({
                    "id": str(unique_id),
                    "question": new,
                    "answers": [],
                    "is_impossible": True
                })
                unique_id += 1

            paragraph['qas'].extend(new_items)
            count += len(new_items)
            count_all += len(paragraph['qas'])

        print("Generate", count, "from", count_all, "questions.")

    with open("{}/squad_train_aug_{}_05.json".format(args.output_dir, args.generate_type), "w") as outfile:
        json_object = json.dumps({"version": "v2.0", "data": all_data}, indent=4)
        outfile.write(json_object)

if __name__ == "__main__":
    main()
