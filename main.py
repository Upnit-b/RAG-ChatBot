from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import *
from src.prompts import *
from store import docs

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def create_app():
    app = Flask(__name__)

    load_dotenv()

    retriever = docs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    llm = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")

    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | chat_prompt
        | llm
        | StrOutputParser()
    )

    @app.route("/")
    def index():
        return render_template("chat.html")

    @app.route("/get", methods=["GET", "POST"])
    def chat():
        msg = request.form["msg"]
        response = rag_chain.invoke(msg)
        print("Response:", response)
        return str(response)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8080, debug=True)
