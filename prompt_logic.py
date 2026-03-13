from llms import LLMFactory
from llms import GeminiFactory
from llms import OllamaFactory

def app_logic(factory:LLMFactory):
    text_generator=factory.create_text_generator()
    prompt = "¿Por qué el cielo es azul? Explícalo en una frase."
    print(f"Prompt: {prompt}")
    resultado = text_generator.query(prompt)
    print(f"Respuesta de la IA:\n{resultado}\n")
    print("-" * 40)

    # --- Ejecución ---
if __name__ == "__main__":
    
    print("--- USANDO GEMINI ---")
    MI_API_KEY = "AIzaSyAm6kU34gkjUZqDgKX8rfQCWVUIG33_waI" 
    gemini_factory = GeminiFactory(api_key=MI_API_KEY)
    # Ejecutamos la app con Gemini
    #app_logic(gemini_factory)

    print("--- USANDO OLLAMA ---")
    # Instanciamos la fábrica de Ollama (asegúrate de tener llama3 descargado)
    ollama_factory = OllamaFactory(model_name="llama3")
    # Ejecutamos la misma app, pero ahora usará Ollama local
    app_logic(ollama_factory)