from flask import Flask, render_template, request, session, redirect
from flask_session import Session
import os
#from dotenv import load_dotenv
from main import create_graph as langgraph_app  # Replace with your actual graph object

#load_dotenv()

app = Flask(__name__)
#app.secret_key = os.getenv("secret_key")
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
graph = langgraph_app()


@app.route("/", methods=["GET", "POST"])
def chat():
    
    if "messages" not in session:
        session["messages"] =[]
    
    if request.method == "POST":
        user_input = request.form.get("query","").strip()
        if user_input.strip().lower() == "exit":
            print("Bye")
            session.clear()

        session["messages"].append({"role": "user", "content": user_input})

        # Construct state with the current messages
        state = {"messages": session["messages"], "message_type": None}

        updated_state = graph.invoke(state)
        if updated_state.get("messages") and len(updated_state["messages"]) > 0:
            session["messages"].append(
                {"role": "assistant", "content": updated_state["messages"][-1].content}
            )
            session.modified = True
        
        return redirect("/")
    # Render with full chat history
    return render_template(
        "index.html",
        messages=session.get("messages", []),
        history=session.get("messages", [])
    )


@app.route("/reset", methods=["POST"])
def reset():
    session.clear()
    return redirect("/")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)