import pickle
from utils.path_utils import abs_path_from_project_path


def save_to_pickle(data, filename):
    """
    Save data to a pickle file.
    
    :param data: Data to be saved
    :param filename: Name of the file to save the data to
    """
    
    with open(abs_path_from_project_path(f"saved/{filename}"), 'wb') as file:
        pickle.dump(data, file)

def load_from_pickle(filename):
    """
    Load data from a pickle file.
    
    :param filename: Name of the file to load the data from
    :return: Loaded data
    """
    
    with open(abs_path_from_project_path(f"saved/{filename}"), 'rb') as file:
        return pickle.load(file)