import os
import sys

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from llama_index.llms import OpenAI
from llama_index.memory import ChatMemoryBuffer
from llama_index import (
    SimpleDirectoryReader, 
    VectorStoreIndex, 
    StorageContext, 
    ServiceContext,
    load_index_from_storage,
    set_global_service_context
) 

os.environ['OPENAI_API_KEY'] = "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"

app = Flask(__name__)
CORS(app)

index = None
chat_engine = None

EXAM_MENTOR = """
You are PsychMentor, embodying a friendly teacher persona with a short, succinct communication style. 
Upon the chatbot feature's activation, you initiate conversations, ready to delve into the MRCPsych syllabus. 
Facing multiple inquiries, you calmly dissect them, addressing each logically and thoroughly. 
A hint of lightheartedness enlivens the discourse, balanced with the seriousness of the psychiatric field. 
You elucidate concepts through clinical scenarios and vignettes, handling sensitive topics with empathy, 
yet a clinical focus. As discussions wind down, you offer to consolidate the learned material into a summary, 
ensuring a comprehensive review for the aspiring specialists.
"""


def initialize_index(index_dir):
    global index
    storage_context = StorageContext.from_defaults(persist_dir=index_dir)
    if os.path.exists(index_dir):
        index = load_index_from_storage(storage_context)
    else:
        documents = SimpleDirectoryReader("./data").load_data()
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        storage_context.persist(index_dir)


def initialize_llm():
  llm = OpenAI("gpt-3.5-turbo-0613", temperature=0.0)
  service_context = ServiceContext.from_defaults(llm=llm)
  set_global_service_context(service_context)


def initialize_chat_engine():
    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    global chat_engine
    global index
    chat_engine = index.as_chat_engine(
        chat_mode="openai", 
        verbose=True,
        system_prompt=EXAM_MENTOR,
        memory=memory
    )


@app.route("/test/query", methods=["GET"])
def test_query_index():
  global index
  query_text = request.args.get("text", None)
  if query_text is None:
    return "No text found, please include a ?text=blah parameter in the URL", 400
  query_engine = index.as_query_engine()
  response = query_engine.query(query_text)
  return str(response), 200


@app.route("/query", methods=["POST"])
@cross_origin()
def query_index():
  global index
  data = request.get_json()
  query_text = data["text"]
  if query_text is None:
    return jsonify({"No text found"}), 400
  query_engine = index.as_query_engine()
  response = query_engine.query(query_text)
  return jsonify({"response": str(response)}), 200


@app.route("/chat", methods=["POST"])
@cross_origin()
def chat():
    global index
    global chat_engine
    data = request.get_json()
    query_text = data["text"]
    if query_text is None:
        return jsonify({"No text found"}), 400
    response = chat_engine.chat(query_text)
    return jsonify({"response": str(response)}), 200


if __name__ == "__main__":
    # init global index
    print("Initializing the index...")
    index_dir = "./index"
    initialize_index(index_dir)

    # configure llm
    print("Initializing GPT...")
    initialize_llm()

    # start chat engine
    args = sys.argv
    if len(args) >= 2:
       del args[0]
       if args[0] == "--chat":
            initialize_chat_engine()
           
    # setup server
    # query example: http://localhost:5601/test/query?text=What is Piaget's model?
    print("Starting the server...")
    app.run(host="0.0.0.0", port=5601)
