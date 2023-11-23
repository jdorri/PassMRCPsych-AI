import os
import sys

from flask import Flask, request, jsonify, render_template
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

app = Flask(__name__)
CORS(app)

index = None
chat_engine = None

EXAM_MENTOR = """
You are PsychMentor, an LLM-powered tutor helping students 
pass their Royal College of Psychiatrists (MRCPsych) exams.

You begin each chat with a friendly message, for example: 
"Hi and welcome to PassMRCPsych! Is there anything I can 
help you learn today?"

Be thorough and accurate, but also make sure your responses 
are clear and easy to understand.

When approriate, offer to explore clinical scenarios and 
vignettes so the students can apply their clinical knowledge. 

Don't be afraid to use a hint of lightheartedness to balance 
the seriousness of the psychiatric field! 

Remember to check if the student has any further questions, or
whether they'd like help on another area of the syllabus.  

And at the end of the chat you should put a suitable emoji, and 
also include emojis wherever they seem appropriate in the text.
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
  llm = OpenAI("gpt-4-0613", temperature=0.7)
  service_context = ServiceContext.from_defaults(llm=llm)
  set_global_service_context(service_context)


def initialize_chat_engine():
    memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
    global chat_engine
    global index
    chat_engine = index.as_chat_engine(
        chat_mode="openai", 
        verbose=False,
        system_prompt=EXAM_MENTOR,
        memory=memory,
        streaming=True
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

@app.route("/")
@cross_origin()
def index():
    return render_template("index.html")


if __name__ == "__main__":
    print("Initializing the index...")
    index_dir = "./index"
    initialize_index(index_dir)

    print("Initializing GPT...")
    initialize_llm()

    args = sys.argv
    if len(args) >= 2:
       del args[0]
       if args[0] == "--chat":
            initialize_chat_engine()

    print("Starting the server...")
    app.run(host="0.0.0.0", port=5601)

else:
  index_dir = "./index"
  initialize_index(index_dir)
  initialize_llm()
  initialize_chat_engine()
