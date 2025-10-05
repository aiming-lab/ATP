from copy import deepcopy
from abc import ABC

class BaseModel(ABC):
    def __init__(self, config):
        """
        Base model constructor to initialize common attributes.
        :param config: A dictionary containing model configuration parameters.
        """
        self.config = config
        
    def predict(self, question, texts = None, images = None, history = None):
        pass
    
    def predict_message(self, messages):
        return self.predict(question = None, history = messages)
        
    def clean_up(self):
        pass

    def process_message(self, question, texts, images, history):
        if history is not None:
            assert(self.is_valid_history(history))
            messages = deepcopy(history)
        else:
            messages = []
        
        if texts is not None:
            messages.append(self.create_text_message(texts, question))
        if images is not None:
            messages.append(self.create_image_message(images, question))
        
        if question is not None and (texts is None or len(texts) == 0) and (images is None or len(images) == 0):
            messages.append(self.create_ask_message(question))
        
        return messages
    
    def is_valid_history(self, history):
        return True
    
    def create_text_message(self, texts, question):
        """
        Create a text message from the provided texts and question.
        :param texts: List of text strings.
        :param question: The question to be included in the message.
        :return: A dictionary representing the text message.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def create_image_message(self, images, question):
        """
        Create an image message from the provided images and question.
        :param images: List of image paths.
        :param question: The question to be included in the message.
        :return: A dictionary representing the image message.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def create_ask_message(self, question):
        """
        Create a message asking a question.
        :param question: The question to be asked.
        :return: A dictionary representing the ask message.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")