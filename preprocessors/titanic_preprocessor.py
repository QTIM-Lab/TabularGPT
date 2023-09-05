import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from preprocessors.preprocessor import Preprocessor

class TitanicPreprocessor(Preprocessor):
    # def preprocess_train(self, train_data):
    #     # Drop unnecessary columns
    #     train_data = train_data.drop(columns=['PassengerId', 'Name', 'Ticket'])

    #     # Encode 'Pclass' and 'Sex'
    #     self.fit_encoder(train_data['Pclass'], 'Pclass')
    #     train_data['Pclass'] = self.transform_encoder(train_data['Pclass'], 'Pclass')

    #     self.fit_encoder(train_data['Sex'], 'Sex')
    #     train_data['Sex'] = self.transform_encoder(train_data['Sex'], 'Sex')

    #     # Handle missing values in 'Age' by filling with the mean
    #     train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
    #     # z-score age
    #     self.fit_scaler(train_data[['Age']], 'Age')
    #     train_data['Age'] = self.transform_scaler(train_data[['Age']], 'Age')

    #     # z-score fare
    #     self.fit_scaler(train_data[['Fare']], 'Fare')
    #     train_data['Fare'] = self.transform_scaler(train_data[['Fare']], 'Fare')

    #     # Create binary features for 'SibSp' and 'Parch'
    #     train_data['SibSp'] = (train_data['SibSp'] > 0).astype(int)
    #     train_data['Parch'] = (train_data['Parch'] > 0).astype(int)

    #     # Create a binary feature for 'Cabin' presence
    #     train_data['Cabin'] = train_data['Cabin'].notna().astype(int)

    #     # Handle missing values in 'Embarked' by encoding as specified
    #     train_data['Embarked'] = train_data['Embarked'].fillna('0').map({'0': 0, 'C': 1, 'Q': 2, 'S': 3}).astype(int)

    #     # Extract the target variable (Survived)
    #     y_train_data = train_data.pop('Survived')
        
    #     return train_data, y_train_data
    
    def preprocess_train_and_val(self, data, val_size=0.2):
        # Drop unnecessary columns
        data = data.drop(columns=['PassengerId', 'Name', 'Ticket'])

        # Encode 'Pclass' and 'Sex'
        self.fit_encoder(data['Pclass'], 'Pclass')
        data['Pclass'] = self.transform_encoder(data['Pclass'], 'Pclass')

        self.fit_encoder(data['Sex'], 'Sex')
        data['Sex'] = self.transform_encoder(data['Sex'], 'Sex')

        # Handle missing values in 'Age' by filling with the mean
        data['Age'].fillna(data['Age'].mean(), inplace=True)

        # Create binary features for 'SibSp' and 'Parch'
        data['SibSp'] = (data['SibSp'] > 0).astype(int)
        data['Parch'] = (data['Parch'] > 0).astype(int)

        # Create a binary feature for 'Cabin' presence
        data['Cabin'] = data['Cabin'].notna().astype(int)

        # Handle missing values in 'Embarked' by encoding as specified
        data['Embarked'] = data['Embarked'].fillna('0').map({'0': 0, 'C': 1, 'Q': 2, 'S': 3}).astype(int)

        # Extract the target variable (Survived)
        y_data = data.pop('Survived')
        
        # Split the data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(data, y_data, test_size=val_size, random_state=42)

        # z-score age
        self.fit_scaler(x_train[['Age']], 'Age')
        x_train['Age'] = self.transform_scaler(x_train[['Age']], 'Age')
        x_val['Age'] = self.transform_scaler(x_train[['Age']], 'Age')
        
        # z-score fare
        self.fit_scaler(x_train[['Fare']], 'Fare')
        x_train['Fare'] = self.transform_scaler(x_train[['Fare']], 'Fare')
        x_val['Fare'] = self.transform_scaler(x_train[['Fare']], 'Fare')

        # TENSORS

        x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
        x_val_tensor = torch.tensor(x_val.values, dtype=torch.float32)

        # Create the y_train tensor with -1 for all columns except the last column
        y_train_tensor = torch.full((x_train.shape[0], x_train.shape[1]), -float('inf'), dtype=torch.float32)
        # Fill the last column with the actual class indices
        y_train_tensor[:, -1] = torch.tensor(y_train.values, dtype=torch.float32)
        # Create the y_val tensor with -1 for all columns except the last column
        y_val_tensor = torch.full((x_val.shape[0], x_val.shape[1]), -float('inf'), dtype=torch.float32)
        # Fill the last column with the actual class indices
        y_val_tensor[:, -1] = torch.tensor(y_val.values, dtype=torch.float32)

        # DATASET

        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(x_val_tensor, y_val_tensor)
        
        return train_dataset, val_dataset
    
    def preprocess_test(self, test_data, train_data):
        # Drop unnecessary columns
        test_data = test_data.drop(columns=['PassengerId', 'Name', 'Ticket'])

        # Encode 'Pclass' and 'Sex'
        self.fit_encoder(train_data['Pclass'], 'Pclass')
        test_data['Pclass'] = self.transform_encoder(test_data['Pclass'], 'Pclass')

        self.fit_encoder(train_data['Sex'], 'Sex')
        test_data['Sex'] = self.transform_encoder(test_data['Sex'], 'Sex')

        # Handle missing values in 'Age' by filling with the mean
        test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)
        # z-score age
        self.fit_scaler(train_data[['Age']], 'Age')
        test_data['Age'] = self.transform_scaler(test_data[['Age']], 'Age')

        # z-score fare
        self.fit_scaler(train_data[['Fare']], 'Fare')
        test_data['Fare'] = self.transform_scaler(test_data[['Fare']], 'Fare')

        # Create binary features for 'SibSp' and 'Parch'
        test_data['SibSp'] = (test_data['SibSp'] > 0).astype(int)
        test_data['Parch'] = (test_data['Parch'] > 0).astype(int)

        # Create a binary feature for 'Cabin' presence
        test_data['Cabin'] = test_data['Cabin'].notna().astype(int)

        # Handle missing values in 'Embarked' by encoding as specified
        test_data['Embarked'] = test_data['Embarked'].fillna('0').map({'0': 0, 'C': 1, 'Q': 2, 'S': 3}).astype(int)
        
        x_test_tensor = torch.tensor(test_data.values, dtype=torch.float32)
        test_dataset = TensorDataset(x_test_tensor)
        
        return test_dataset
