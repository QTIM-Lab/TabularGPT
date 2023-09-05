import argparse
import pandas as pd
import torch
import csv

from models.tabular_gpt import TabularGPT
from trainers.tabular_gpt_trainer import TrainerWithVal

from preprocessors.titanic_preprocessor import TitanicPreprocessor

def eval_model(device, batch_size):
    train_data = pd.read_csv('./examples/titanic/titanic_train.csv')
    test_data = pd.read_csv('./examples/titanic/titanic_test.csv')

    preprocessing = TitanicPreprocessor()
    test_dataset = preprocessing.preprocess_test(test_data, train_data)
    # assert train and val are TensorDatasets?

    # Instantiate minGPT-TS model
    model_config = TabularGPT.get_default_config()
    model_config.vocab_size = 0 # 0, just because Karpathy did an assert
    model_config.block_size = 8  # just because asserts, does not do anything?
    # 'gpt-nano' = dict(n_layer=3, n_head=3, n_embd=48)
    # 'gpt2' = dict(n_layer=12, n_head=12, n_embd=768)  # 124M params
    # model_config.model_type = 'gpt2'
    model_config.model_type = None
    model_config.n_layer = 16
    model_config.n_head = 16
    model_config.n_embd = 1024
    
    model_config.num_vars = 8
    model_config.out_dim = 1  # state_dim_1, state_dim_2
    
    model_config.embed_vars = {
        '0': 3,  # PClass has 3
        '1': 2,  # Sex has 2
        '3': 2,  # SibSp has 2
        '4': 2,  # Parch has 2
        '6': 2,  # Cabin has 2
        '7': 4  # Embarked has 4 (null is 0)
    }
    # it's normally just 0.1 for all
    model_config.attn_pdrop = 0.0
    model_config.embd_pdrop = 0.0
    model_config.resid_pdrop = 0.0
    model = TabularGPT(model_config,device,output_type="binaryclass").to(device)
    checkpoint = torch.load('experiment_model.pt')
    model.load_state_dict(checkpoint)

    test_outputs = []
    with torch.no_grad():
        for i in range(len(test_dataset)):
            output = 1 if model(test_dataset[i][0].unsqueeze(0).to(device))[0][0,-1,0] > 0 else 0
            test_outputs.append(output)

    passenger_ids = []
    with open('./examples/titanic/titanic_test.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            passenger_ids.append(row['PassengerId'])

    # Make sure the lengths of outputs_list and passenger_ids match
    if len(test_outputs) != len(passenger_ids):
        raise ValueError("The number of outputs and passenger IDs do not match.")
    
    # Create a new CSV file to store the combined data
    output_filename = 'combined_output.csv'

    # Write the combined data to the new CSV file
    with open(output_filename, mode='w', newline='') as file:
        fieldnames = ['PassengerId', 'Survived']  # Define the column headers
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()  # Write the header row

        for passenger_id, output in zip(passenger_ids, test_outputs):
            writer.writerow({'PassengerId': passenger_id, 'Survived': output})


def main():
    print("Eval starting...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    parser = argparse.ArgumentParser(description="Eval TabularGPT.")

    # parser.add_argument('--model', type=str, choices=['model1', 'model2'], required=True,
    #                     help="Select the model to train.")
    # parser.add_argument('--trainer', type=str, choices=['trainer1', 'trainer2'], required=True,
    #                     help="Select the trainer to use for training.")
    # parser.add_argument('--data-path', type=str, required=True,
    #                     help="Path to the training data.")
    # parser.add_argument('--output-dir', type=str, required=True,
    #                     help="Directory to save the trained model.")
    parser.add_argument('--batch-size', type=int, required=True,
                        help="Batch size.")
    # parser.add_argument('--output-dir', type=str, required=True,
    #                     help="Directory to save the trained model.")

    args = parser.parse_args()

    eval_model(device, args.batch_size)

if __name__ == "__main__":
    main()
