class KMeansException(Exception):
    def __init__(self):
        message = 'Not enough pixels left to perform clustering.'
        super(KMeansException, self).__init__(message)
