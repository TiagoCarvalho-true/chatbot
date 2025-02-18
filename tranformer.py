from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import sys
import os
import certifi
import ssl
import json

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
        conhecimento = carregar_conhecimento_arquivo()
        
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

                # Gera resposta considerando o conhecimento atual
                contexto = "\n".join(conhecimento)
                prompt = f"{contexto}\nPergunta: {entrada}\nResposta:"
                
                resposta = chatbot(
                    prompt,
                    max_length=150,  # Aumentado para respostas mais completas
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )[0]['generated_text']
                
                print(f"Chatbot: {resposta}")
                
            except KeyboardInterrupt:
                print("\nChatbot: Conversa interrompida pelo usuário.")
                break
                
    except Exception as e:
        print(f"Erro ao iniciar o chatbot: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    iniciar_chatbot()