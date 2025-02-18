from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from collections import deque
import sys
import os
import certifi
import ssl
import json

class MemoriaChatbot:
    def __init__(self, tamanho_max=5):
        self.historico = deque(maxlen=tamanho_max)
        self.conhecimento_base = []
        self.memoria_longa = []  # Nova lista para memória de longo prazo
        
    def adicionar_conversa(self, entrada, resposta, importante=False):
        self.historico.append({"entrada": entrada, "resposta": resposta})
        if importante:
            self.memoria_longa.append({"entrada": entrada, "resposta": resposta})
def obter_contexto_completo(self):
        # Combina conhecimento base com histórico
        contexto = []
        
        # Adiciona conhecimento base
        if self.conhecimento_base:
            contexto.append("Conhecimento Base:")
            contexto.extend(self.conhecimento_base)
        
        # Adiciona memória de longo prazo
        if self.memoria_longa:
            contexto.append("\nAprendizado Importante:")
            for mem in self.memoria_longa[-3:]:  # Últimas 3 memórias importantes
                contexto.append(f"P: {mem['entrada']}")
                contexto.append(f"R: {mem['resposta']}")
        
        # Adiciona histórico recente
        contexto.append("\nConversa Atual:")
        for item in self.historico:
            contexto.append(f"Humano: {item['entrada']}")
            contexto.append(f"Bot: {item['resposta']}")
            
        return "\n".join(contexto)
      
def processar_resposta(chatbot, entrada, memoria, temperatura=0.7):
    """Processa a entrada e gera uma resposta mais elaborada"""
    
    # Análise de relevância da entrada
    palavras_chave = ['porque', 'como', 'explique', 'qual', 'por que', 'define']
    entrada_importante = any(palavra in entrada.lower() for palavra in palavras_chave)
    
    # Obtém contexto completo
    contexto_completo = memoria.obter_contexto_completo()
    
    # Prepara o prompt com instruções específicas
    prompt = f"""
    Baseado no seguinte contexto e conhecimento:
    {contexto_completo}
    
    Por favor, responda à seguinte pergunta de forma {
        'detalhada e explicativa' if entrada_importante else 'simples e direta'
    }:
    
    Humano: {entrada}
    Bot:"""
    
    respostas = []
    for temp in [temperatura, temperatura + 0.2]:
        resposta = chatbot(
            prompt,
            max_length=300 if entrada_importante else 150,
            num_return_sequences=2,
            temperature=temp,
            pad_token_id=tokenizer.eos_token_id,
            top_p=0.9,
            repetition_penalty=1.2
        )
        respostas.extend([r['generated_text'] for r in resposta])
    
    # Escolhe a melhor resposta
    melhor_resposta = max(respostas, key=len)
    
    # Atualiza a memória
    memoria.adicionar_conversa(entrada, melhor_resposta, importante=entrada_importante)
    
    return melhor_resposta

def carregar_conhecimento_arquivo(pasta="conhecimento"):
    """Carrega conhecimento de arquivos txt e json"""
    conhecimento = []
    
    # Cria pasta se não existir
    if not os.path.exists(pasta):
        os.makedirs(pasta)
        print(f"Pasta '{pasta}' criada!")
        return conhecimento
    for arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, arquivo)
        try:
            if arquivo.endswith('.txt'):
                with open(caminho, 'r', encoding='utf-8') as f:
                    conhecimento.extend(f.readlines())
                print(f"📚 Aprendi com: {arquivo}")
            
            elif arquivo.endswith('.json'):
                with open(caminho, 'r', encoding='utf-8') as f:
                    dados = json.load(f)
                    if isinstance(dados, list):
                        conhecimento.extend(dados)
                    elif isinstance(dados, dict):
                        for tema, info in dados.items():
                            conhecimento.append(f"{tema}: {info}")
                print(f"📚 Aprendi com: {arquivo}")
                
        except Exception as e:
            print(f"⚠️ Erro ao ler {arquivo}: {e}")

    return [info.strip() for info in conhecimento if info.strip()]

def salvar_conhecimento(conhecimento, arquivo="conhecimento/memoria.json"):
    """Salva o conhecimento atual em um arquivo"""
    os.makedirs(os.path.dirname(arquivo), exist_ok=True)
    try:
        with open(arquivo, 'w', encoding='utf-8') as f:
             json.dump(conhecimento, f, ensure_ascii=False, indent=2)
        print("💾 Conhecimento salvo com sucesso!")
    except Exception as e:
        print(f"⚠️ Erro ao salvar conhecimento: {e}")

def iniciar_chatbot():
    try:
        # Configuração SSL
        cert_path = certifi.where()
        os.environ['SSL_CERT_FILE'] = cert_path
        os.environ['REQUESTS_CA_BUNDLE'] = cert_path
        ssl._create_default_https_context = ssl._create_unverified_context

        print(f"Usando certificado em: {cert_path}")
        print("Inicializando o chatbot... Por favor, aguarde...")

        # Base de conhecimento
        memoria=MemoriaChatbot(tamanho_max=10)
        memoria.conhecimento = carregar_conhecimento_arquivo()
        
        # Carrega o modelo GPT2 em português
        modelo = "pierreguillou/gpt2-small-portuguese"
        tokenizer = AutoTokenizer.from_pretrained(modelo, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(modelo, trust_remote_code=True)
        chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)
        
        print("\nChatbot pronto! Comandos disponíveis:")
        print("'aprender: [texto]' - Para ensinar nova informação")
        print("'memoria' - Para ver o conhecimento atual")
        print("'salvar' - Para salvar o conhecimento atual")
        print("'carregar' - Para recarregar conhecimento dos arquivos")
        print("'sair' - Para encerrar a conversa\n")
        
        while True:
            try:
                entrada = input("\nVocê: ").strip()
                
                # Verifica comandos especiais
                if entrada.lower() == "sair":
                    salvar_conhecimento(conhecimento)
                    print("Chatbot: Conhecimento salvo. Até mais!")
                    break
                    
                elif entrada.lower() == "memoria":
                    if conhecimento:
                        print("\n📚 Conhecimento atual:")
                        for idx, info in enumerate(conhecimento, 1):
                            print(f"{idx}. {info}")
                    else:
                        print("Chatbot: Ainda não aprendi nada novo.")
                    continue
                
                elif entrada.lower() == "salvar":
                    salvar_conhecimento(conhecimento)
                    continue
                
                elif entrada.lower() == "carregar":
                    conhecimento = carregar_conhecimento_arquivo()
                    print("Chatbot: Conhecimento recarregado!")
                    continue
                    
                elif entrada.lower().startswith("aprender:"):
                    nova_info = entrada[9:].strip()
                    if nova_info:
                        conhecimento.append(nova_info)
                        print("Chatbot: Aprendi essa nova informação! 📚")
                    else:
                        print("Chatbot: Por favor, forneça alguma informação para eu aprender.")
                    continue

                resposta = processar_resposta(chatbot, entrada, memoria, temperatura=0.7)
                print(f"Chatbot: {resposta}")
                
            except KeyboardInterrupt:
                 print("\nChatbot: Salvando memória antes de encerrar...")
                 salvar_conhecimento(memoria.conhecimento_base)
                 break
                
    except Exception as e:
        print(f"Erro ao iniciar o chatbot: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    iniciar_chatbot()