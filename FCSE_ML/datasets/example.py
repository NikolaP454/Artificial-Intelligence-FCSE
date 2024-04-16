import pandas as pd

from core.dataset.dataset import Dataset

class Example(Dataset):
    
    def __init__(self):
        self.columns = ['id', 'name', 'age', 'is_student', 'gpa']
        
        self.data = [
            [1, 'Alice', 20, True, 3.5],
            [2, 'Bob', 21, False, 3.0],
            [3, 'Charlie', 22, True, 3.8],
            [4, 'David', 23, False, 2.5],
            [5, 'Eve', 24, True, 3.9]
        ]
        
        self.data = pd.DataFrame(self.data, columns=self.columns)
    
    def get_data(self, type: str='list'):
        if type == 'list':
            return self.data.values.tolist()
        elif type == 'df':
            return self.data
        else:
            raise ValueError('Invalid type')
        
        
    def head(self, number_of_rows: int=5) -> str:
        return self.data.head(number_of_rows).to_string()
    
    def info(self) -> str:
        return self.data.info()
    
    