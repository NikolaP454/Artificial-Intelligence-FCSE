class Dataset:
    
    def get_data(self, type: str='list'):
        '''
        Returns the data in the specified format.
        
        Parameters:
            type (str): The format of the data to be returned. Default is 'list'.
        '''
        
        raise NotImplementedError
    
    def head(self, number_of_rows: int=5):
        '''
        Returns the first n rows of the dataset.
        
        Parameters:
            number_of_rows (int): The number of rows to return. Default is 5.
        '''
        
        raise NotImplementedError
    
    def info(self):
        '''
        Returns the information about the dataset.
        
        Parameters:
            None
        '''
        
        raise NotImplementedError
    