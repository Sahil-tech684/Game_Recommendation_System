class Params:
    def __init__(self):
        # TF-IDF parameters
        self.MAX_FEATURES = 5000
        self.STOP_WORDS = 'english'
        
        # Data processing
        self.SAMPLE_SIZE = 5000  # For development
        
params = Params()