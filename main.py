from utils.custom_exceptions import CustomException

from utils.logger_exceptions import get_logger

logger = get_logger(__name__)

def data_loader(file_path):
    try:
        with open(file_path, 'r') as file:
            data = file.read()
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise CustomException(f"File not found: {file_path}")
    except Exception as e:
        logger.error(f"An error occurred while loading the file: {str(e)}")
        raise CustomException(f"An error occurred while loading the file: {str(e)}")
    
if __name__ == "__main__":
    file_path = "data.txt"
    try:
        data = data_loader(file_path)
        logger.info("Data loaded successfully", extra={"file_path": file_path})
    except CustomException as e:
        logger.error(f"Failed to load data: {str(e)}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")