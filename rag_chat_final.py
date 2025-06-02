import os
from loguru import logger
from langchain_community.vectorstores import FAISS
import re
import gradio as gr

logger.add("log/kyrgyz_laws_rag.log", format="{time} {level} {message}", level="DEBUG", rotation="100 KB", compression="zip")

def get_index_db():
    logger.debug('...get_index_db')
    from langchain_huggingface import HuggingFaceEmbeddings

    model_id = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    embeddings = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={'device': 'cuda'},  
    )

    db_file_name = 'db/laws_db'
    file_path = db_file_name + "/index.faiss"
    
    if os.path.exists(file_path):
        logger.debug('Загрузка существующей базы')
        db = FAISS.load_local(db_file_name, embeddings, allow_dangerous_deserialization=True)
    else:
        logger.debug('Создание новой базы знаний из .txt файлов')
        from langchain_community.document_loaders import TextLoader

        dir = 'laws'
        documents = []
        loaded_files = []
        
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".txt"):
                    try:
                        logger.debug(f'Загрузка файла: {file}')
                        file_path = os.path.join(root, file)
                        loader = TextLoader(file_path, encoding='utf-8')
                        file_docs = loader.load()
                        
        
                        for doc in file_docs:
                            doc.metadata['source_file'] = file
                            doc.metadata['law_name'] = file.replace('.txt', '').replace('_', ' ')
                        
                        documents.extend(file_docs)
                        loaded_files.append(file)
                    except Exception as e:
                        logger.error(f'Ошибка загрузки файла {file}: {e}')

        logger.info(f'Загружено файлов: {len(loaded_files)}')
        logger.info(f'Файлы: {loaded_files}')

        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  
            chunk_overlap=150,  
            separators=["\n\n", "\n", ". ", " "],  
            keep_separator=True
        )
        source_chunks = text_splitter.split_documents(documents)
        
        logger.info(f'Создано чунков: {len(source_chunks)}')

        db = FAISS.from_documents(source_chunks, embeddings)
        db.save_local(db_file_name)

    return db

def get_message_content(topic, db, k):
    logger.debug('...get_message_content')
    

    try:
        docs = db.max_marginal_relevance_search(topic, k=k, fetch_k=k*2)
    except:

        docs = db.similarity_search(topic, k=k)
    

    sources_content = {}
    for doc in docs:
        source = doc.metadata.get('source_file', 'unknown')
        law_name = doc.metadata.get('law_name', source)
        
        if source not in sources_content:
            sources_content[source] = {
                'law_name': law_name,
                'chunks': []
            }
        sources_content[source]['chunks'].append(doc.page_content)
    

    message_content = ""
    for i, (source, data) in enumerate(sources_content.items()):
        message_content += f"\n=== ЗАКОН: {data['law_name']} ===\n"
        for j, chunk in enumerate(data['chunks']):

            clean_chunk = re.sub(r'\n+', ' ', chunk.strip())
            clean_chunk = re.sub(r'\s+', ' ', clean_chunk)
            message_content += f"{clean_chunk}\n"
        message_content += "\n"
    
    logger.debug(f"Найдено релевантных источников: {len(sources_content)}")
    return message_content.strip()

def get_model_response(topic, message_content):
    logger.debug('...get_model_response')

    from langchain_ollama import ChatOllama
    llm = ChatOllama(
        model="llama3.2:3b-instruct-fp16", 
        temperature=0.1,  
        top_p=0.9,
        repeat_penalty=1.1
    )

    rag_prompt = """Ты юридический эксперт по законам Кыргызской Республики. Отвечай ТОЛЬКО на русском языке.

КОНТЕКСТ ИЗ ЗАКОНОВ КР:
{context}

ВОПРОС: {question}

ИНСТРУКЦИИ:
1. Отвечай СТРОГО на основе предоставленного контекста
2. Если информации недостаточно - так и скажи
3. Указывай конкретные статьи/пункты законов при ссылке
4. Используй только русский язык и только кириллицу
5. Будь точным и кратким (3-5 предложений)
6. Не добавляй информацию, которой нет в контексте

ОТВЕТ:"""

    from langchain_core.messages import HumanMessage
    prompt = rag_prompt.format(context=message_content, question=topic)
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        
        answer = response.content.strip()
        

        english_patterns = [
            r'\b(according to|based on|however|therefore|thus|themselves|moreover|furthermore)\b',
            r'\b(Article|Section|Chapter|Paragraph)\b'
        ]
        
        for pattern in english_patterns:
            answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)

        answer = re.sub(r'\s+', ' ', answer).strip()
        
        return answer
        
    except Exception as e:
        logger.error(f"Ошибка получения ответа от модели: {e}")
        return "Извините, произошла ошибка при обработке вашего запроса."

def validate_answer(answer, context):
    """Проверка качества ответа"""
    if len(answer.strip()) < 20:
        return False, "Ответ слишком короткий"
    

    english_words = re.findall(r'\b[A-Za-z]+\b', answer)
    if len(english_words) > 1:  
        return False, f"Найдены английские слова: {english_words}"
    
    return True, "OK"

def interactive_chat():
    """Интерактивный режим чата"""
    print("🏛️  Чат-бот эксперт по законам Кыргызской Республики")
    print("📖 Введите 'выход' для завершения\n")
    
    db = get_index_db()
    
    while True:
        topic = input("❓ Ваш юридический вопрос: ").strip()
        
        if topic.lower() in ['выход', 'exit', 'quit', 'q']:
            print("👋 До свидания!")
            break
            
        if not topic:
            print("❌ Пожалуйста, введите вопрос\n")
            continue
            
        try:
            print("🔍 Поиск в законах КР...")
            

            for k in [6, 4, 8]:
                message_content = get_message_content(topic, db, k)
                answer = get_model_response(topic, message_content)
                
                is_valid, validation_msg = validate_answer(answer, message_content)
                
                if is_valid:
                    break
                else:
                    logger.warning(f"Ответ не прошел валидацию (k={k}): {validation_msg}")
            
            print(f"\n📋 Ответ юридического эксперта:")
            print(f"{'='*50}")
            print(answer)
            print(f"{'='*50}\n")
            
        except Exception as e:
            logger.error(f"Ошибка обработки вопроса: {e}")
            print("❌ Произошла ошибка. Попробуйте переформулировать вопрос.\n")


db_instance = None

def initialize_db():
    """Инициализация базы данных при запуске приложения"""
    global db_instance
    if db_instance is None:
        db_instance = get_index_db()
    return db_instance

def process_question(question, history):
    """Функция для обработки вопросов в Gradio интерфейсе"""
    if not question.strip():
        return history + [("", "❌ Пожалуйста, введите вопрос")]
    
    try:

        history = history + [(question, "🔍 Поиск в законах КР...")]
        yield history
        
  
        db = initialize_db()
        

        answer = None
        for k in [6, 4, 8]:
            message_content = get_message_content(question, db, k)
            temp_answer = get_model_response(question, message_content)
            
            is_valid, validation_msg = validate_answer(temp_answer, message_content)
            
            if is_valid:
                answer = temp_answer
                break
            else:
                logger.warning(f"Ответ не прошел валидацию (k={k}): {validation_msg}")
        
        if answer is None:
            answer = "Извините, не удалось получить корректный ответ. Попробуйте переформулировать вопрос."
        
 
        history[-1] = (question, f"📋 **Ответ юридического эксперта:**\n\n{answer}")
        yield history
        
    except Exception as e:
        logger.error(f"Ошибка обработки вопроса: {e}")
        error_msg = "❌ Произошла ошибка при обработке вашего запроса. Попробуйте переформулировать вопрос."
        history[-1] = (question, error_msg)
        yield history

def create_gradio_interface():
    """Создание Gradio интерфейса"""
    

    initialize_db()
    
    with gr.Blocks(
        title="Эксперт по законам КР",
        theme=gr.themes.Soft(),
        css="""
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(90deg, #1e40af, #3b82f6);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .chatbot {
            height: 500px;
        }
        """
    ) as interface:
        

        gr.HTML("""
        <div class="header">
            <h1>Юридический эксперт по законам Кыргызской Республики</h1>
            <p>Задайте ваш вопрос о законодательстве КР и получите экспертный ответ</p>
        </div>
        """)
        
   
        chatbot = gr.Chatbot(
            label="Диалог с экспертом",
            elem_classes=["chatbot"],
            placeholder="Здесь будет отображаться диалог с юридическим экспертом",
            show_copy_button=True
        )
        

        msg = gr.Textbox(
            label="Ваш юридический вопрос",
            placeholder="Например: Какие права имеет работник при увольнении?",
            lines=2,
            max_lines=5
        )
        
    
        with gr.Row():
            submit_btn = gr.Button("Отправить", variant="primary", scale=2)
            clear_btn = gr.Button("🗑️ Очистить чат", variant="secondary", scale=1)
        
    
        gr.Examples(
            examples=[
                ["Какие документы нужны для регистрации ООО?"],
                ["Какой минимальный размер оплаты труда в КР?"],
                ["Какие права имеет потребитель при покупке товара?"],
                ["Какой срок исковой давности по гражданским делам?"]
            ],
            inputs=msg,
            label="💡 Примеры вопросов"
        )
        

        gr.Markdown("""
        ---
        ### Информация:
        - Система анализирует законы Кыргызской Республики и предоставляет ответы на основе официальных документов
        - Ответы носят информационный характер и не заменяют профессиональную юридическую консультацию
        - Для получения точной правовой помощи обратитесь к квалифицированному юристу
        """)
        

        def clear_chat():
            return []
        
        def submit_and_clear(message, history):
  
            for updated_history in process_question(message, history):
                yield updated_history, ""
        

        submit_btn.click(
            submit_and_clear,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        msg.submit(
            submit_and_clear,
            inputs=[msg, chatbot],
            outputs=[chatbot, msg]
        )
        
        clear_btn.click(
            clear_chat,
            outputs=[chatbot]
        )
    
    return interface

if __name__ == "__main__":

    print("Юридический эксперт по законам КР")
    print("Выберите режим запуска:")
    print("1 - Веб-интерфейс Gradio")
    print("2 - Интерактивный чат в консоли")
    print("3 - Одиночный вопрос в консоли")
    
    mode = input("Введите номер режима (1-3): ").strip()
    
    if mode == "1" or mode == "":
    
        interface = create_gradio_interface()
        print("🌐 Запуск веб-интерфейса...")
        print("📱 Интерфейс будет доступен по адресу: http://localhost:7860")
        print("🔗 Или по адресу: http://127.0.0.1:7860")
        print("⏹️  Для остановки нажмите Ctrl+C")
        
        interface.launch(
            server_name="127.0.0.1",  
            server_port=7860,        
            share=True,              
            debug=False,
            show_error=True,
            inbrowser=True           
        )
    elif mode == "2":
        interactive_chat()
    else:
        db = get_index_db()
        NUMBER_RELEVANT_CHUNKS = 6  # Увеличил для лучшего контекста
        topic = input("Введите ваш юридический вопрос: ")
        message_content = get_message_content(topic, db, NUMBER_RELEVANT_CHUNKS)
        answer = get_model_response(topic, message_content)
        print("\n📋 Ответ модели:")
        print(f"{'='*50}")
        print(answer)
        print(f"{'='*50}")