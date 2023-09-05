from sklearn.preprocessing import LabelEncoder, StandardScaler

class Preprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}

    def fit_scaler(self, data, feature_name):
        # Initialize and fit a scaler for the specified feature
        scaler = StandardScaler()
        self.scalers[feature_name] = scaler
        self.scalers[feature_name].fit(data)

    def transform_scaler(self, data, feature_name):
        # Transform data using the specified scaler
        return self.scalers[feature_name].transform(data)

    def fit_encoder(self, data, feature_name):
        # Initialize and fit an encoder for the specified feature
        encoder = LabelEncoder()
        encoder.fit(data)
        self.encoders[feature_name] = encoder

    def transform_encoder(self, data, feature_name):
        # Transform data using the specified encoder
        return self.encoders[feature_name].transform(data)