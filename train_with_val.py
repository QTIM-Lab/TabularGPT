import argparse
import pandas as pd
import torch

from models.tabular_gpt import TabularGPT
from trainers.tabular_gpt_trainer import TrainerWithVal

from preprocessors.titanic_preprocessor import TitanicPreprocessor

def train_model(device, n_layer, n_head, n_embd, attn_pdrop, embd_pdrop, resid_pdrop, batch_size, learning_rate):
    train_data = pd.read_csv('./examples/titanic/titanic_train.csv')

    preprocessing = TitanicPreprocessor()
    train_dataset, val_dataset = preprocessing.preprocess_train_and_val(train_data)
    # assert train and val are TensorDatasets?



    # Instantiate minGPT-TS model
    model_config = TabularGPT.get_default_config()
    model_config.vocab_size = 0 # 0, just because Karpathy did an assert
    model_config.block_size = 8  # just because asserts, does not do anything?
    # 'gpt-nano' = dict(n_layer=3, n_head=3, n_embd=48)
    # 'gpt2' = dict(n_layer=12, n_head=12, n_embd=768)  # 124M params
    # model_config.model_type = 'gpt2'
    model_config.model_type = None
    model_config.n_layer = n_layer
    model_config.n_head = n_head
    model_config.n_embd = n_embd
    
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
    model_config.attn_pdrop = attn_pdrop
    model_config.embd_pdrop = embd_pdrop
    model_config.resid_pdrop = resid_pdrop
    model = TabularGPT(model_config,device,output_type="binaryclass")

    # Instantiate minGPT trainer
    train_config = TrainerWithVal.get_default_config()
    train_config.batch_size = batch_size
    train_config.learning_rate = learning_rate  # 4e-5 decent for 0.09M params
    train_config.max_iters = 100000
    train_config.num_workers = 0
    train_config.val_interval = 100
    train_config.patience = 20
    trainer = TrainerWithVal(train_config,
                            model,
                            train_dataset,
                            val_dataset,
                            'experiment_model.pt')


    # Load and preprocess your data (data_path) here

    # Train the model
    trainer.run(log_level=1)


def main():
    print("Train with val starting...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    parser = argparse.ArgumentParser(description="Train TabularGPT with val.")

    # parser.add_argument('--model', type=str, choices=['model1', 'model2'], required=True,
    #                     help="Select the model to train.")
    # parser.add_argument('--trainer', type=str, choices=['trainer1', 'trainer2'], required=True,
    #                     help="Select the trainer to use for training.")
    # parser.add_argument('--data-path', type=str, required=True,
    #                     help="Path to the training data.")
    # parser.add_argument('--output-dir', type=str, required=True,
    #                     help="Directory to save the trained model.")
    parser.add_argument('--n-layer', type=int, required=True,
                        help="Number of layers.")
    parser.add_argument('--n-head', type=float, required=True,
                        help="Number of heads.")
    parser.add_argument('--n-embd', type=int, required=True,
                        help="Embedding dim.")
    parser.add_argument('--attn-pdrop', type=float, required=True,
                        help="attention layer dropout.")
    parser.add_argument('--embd-pdrop', type=float, required=True,
                        help="embed layer dropout.")
    parser.add_argument('--resid-pdrop', type=float, required=True,
                        help="resid layer dropout.")
    parser.add_argument('--batch-size', type=int, required=True,
                        help="Batch size.")
    parser.add_argument('--learning-rate', type=float, required=True,
                        help="Learning rate.")

    args = parser.parse_args()

    train_model(device, args.n_layer, args.n_head, args.n_embd, 
                args.attn_pdrop, args.embd_pdrop, args.resid_pdrop, 
                args.batch_size, args.learning_rate)

if __name__ == "__main__":
    main()
