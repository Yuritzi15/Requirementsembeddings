from abc import ABC, abstractmethod
from google import genai
import ollama

#--------------- Abstract product for LLM -----------------
class BaseLLMClient(ABC):
    """ Abstract class for LLM Clients"""
    @abstractmethod
    def query(self, prompt:str)->str:
        pass

#--------------- Abstract Factory -----------------
class LLMFactory(ABC):
    @abstractmethod
    def create_text_generator(self)->BaseLLMClient:
        """create an instance of a text generator"""
        pass


# ------------------ CONCRETE PRODUCTS -------------------
class GeminiClient(BaseLLMClient):
    def __init__(self, api_key:str, model_name:str='gemini-2.5-flash'):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def query(self, prompt:str)->str:
        try:
            response=self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            return f"Error generando texto con Gemini: {str(e)}"

class OllamaClient(BaseLLMClient):
    def __init__(self, model_name:str='llama3'):
        self.model_name=model_name

    def query(self, prompt:str)->str:
        try:
            response=ollama.generate(model=self.model_name, prompt=prompt)
            return response['response']
        except Exception as e:
            return f"Error generando texto con Ollama: {str(e)}, modelo {self.model_name}"

# ------------------ CONCRETE FACTORIES -------------------       
class GeminiFactory(LLMFactory):
    def __init__(self, api_key:str):
        self.api_key=api_key
    
    def create_text_generator(self)-> BaseLLMClient:
        return GeminiClient(api_key=self.api_key)
    
class OllamaFactory(LLMFactory):
    def __init__(self, model_name:str= 'llama3'):
        self.model_name=model_name

    def create_text_generator(self) -> BaseLLMClient:
        return OllamaClient(model_name=self.model_name)