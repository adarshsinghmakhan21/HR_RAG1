"""
STEP 4 — Flask Deployment
Serves the HR RAG assistant as a REST API + web UI.

Run:
python app.py

Open:
http://127.0.0.1:5000
"""

import os
import sys

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS


# ─────────────────────────────────────────────────────────────
# Project Import Setup
# ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)

from step3_rag_engine import HRPolicyRAG


# ─────────────────────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────────────────────

app = Flask(__name__)

CORS(app)

print("\n[Flask] Initializing HR RAG engine...\n")

rag = HRPolicyRAG()

print("[Flask] Ready to serve ✅")


# ─────────────────────────────────────────────────────────────
# HTML UI
# ─────────────────────────────────────────────────────────────

HTML_UI = """
<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Kit Hub Skill AI Assistant</title>

<style>

*{
    margin:0;
    padding:0;
    box-sizing:border-box;
}

body{
    font-family:system-ui;
    background:#020617;
    color:white;
    overflow:hidden;
}

/* MAIN LAYOUT */
.main{
    display:flex;
    height:100vh;
}

/* SIDEBAR */
.sidebar{
    width:320px;
    background:#08112b;
    border-right:1px solid #1e293b;
    padding:24px;
    display:flex;
    flex-direction:column;
    gap:20px;
    overflow-y:auto;
}

/* BRAND */
.brand{
    display:flex;
    align-items:center;
    gap:16px;
}

.logo{
    width:70px;
    height:70px;
    border-radius:20px;
    object-fit:cover;
    background:white;
    padding:6px;
}

.brand-title{
    font-size:24px;
    font-weight:700;
}

.brand-sub{
    color:#94a3b8;
    margin-top:4px;
}

/* QUESTIONS */
.questions{
    display:flex;
    flex-direction:column;
    gap:12px;
}

.q-btn{
    background:#0b1736;
    border:1px solid #1e293b;
    color:white;
    padding:16px;
    border-radius:18px;
    cursor:pointer;
    text-align:left;
    transition:0.3s;
    font-size:15px;
    font-weight:600;
}

.q-btn:hover{
    background:#172554;
    transform:translateY(-2px);
}

/* CONTENT */
.content{
    flex:1;
    display:flex;
    flex-direction:column;
    background:
    radial-gradient(circle at bottom right,#6d28d9 0%,transparent 30%),
    #020617;
}

/* TOPBAR */
.topbar{
    height:80px;
    border-bottom:1px solid #1e293b;
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding:0 32px;
}

.topbar h2{
    font-size:22px;
}

.status{
    color:#22c55e;
    font-weight:600;
}

/* HERO */
.hero{
    flex:1;
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    text-align:center;
    padding:40px;
}

.hero-logo{
    width:130px;
    height:130px;
    border-radius:30px;
    object-fit:cover;
    background:white;
    padding:10px;
    margin-bottom:30px;
    display:block;
}

.hero h1{
    font-size:72px;
    margin-bottom:18px;
    font-weight:800;
}

.hero p{
    color:#94a3b8;
    font-size:24px;
    max-width:900px;
    line-height:1.7;
}

/* CHAT */
.chat-box{
    padding:24px;
    height:450px;
    overflow-y:auto;
    display:flex;
    flex-direction:column;
    gap:18px;
}

/* MESSAGE */
.msg{
    max-width:75%;
    padding:18px;
    border-radius:22px;
    line-height:1.8;
    white-space:pre-wrap;
    animation:fadeIn 0.3s ease;
}

.user{
    background:#2563eb;
    align-self:flex-end;
}

.bot{
    background:#111827;
    border:1px solid #1e293b;
}

@keyframes fadeIn{
    from{
        opacity:0;
        transform:translateY(10px);
    }
    to{
        opacity:1;
        transform:translateY(0);
    }
}

/* INPUT AREA */
.input-area{
    padding:24px 32px;
    border-top:1px solid #1e293b;
    display:flex;
    gap:16px;
    background:#08112b;
}

.input-area input{
    flex:1;
    background:#0f172a;
    border:1px solid #1e293b;
    color:white;
    padding:22px;
    border-radius:24px;
    font-size:18px;
    outline:none;
}

.input-area input:focus{
    border-color:#2563eb;
}

.send-btn{
    background:linear-gradient(135deg,#2563eb,#7c3aed);
    border:none;
    color:white;
    padding:0 34px;
    border-radius:20px;
    cursor:pointer;
    font-size:18px;
    font-weight:700;
    transition:0.3s;
}

.send-btn:hover{
    transform:scale(1.05);
}

/* TYPING */
.typing{
    color:#94a3b8;
    padding:10px;
    font-size:14px;
}

/* SCROLL */
::-webkit-scrollbar{
    width:8px;
}

::-webkit-scrollbar-thumb{
    background:#334155;
    border-radius:10px;
}

</style>
</head>

<body>

<div class="main">

    <!-- SIDEBAR -->
    <div class="sidebar">

        <div class="brand">

            <img src="/static/logo.jpeg" class="logo">

            <div>
                <div class="brand-title">Kit Hub Skill</div>
                <div class="brand-sub">AI HR Assistant</div>
            </div>

        </div>

        <div>

            <h3 style="margin-bottom:16px;color:#94a3b8">
                Suggested Questions
            </h3>

            <div class="questions">

                <button class="q-btn"
                onclick="quickQuestion('How many leave days do employees get?')">
                How many leave days do employees get?
                </button>

                <button class="q-btn"
                onclick="quickQuestion('Can employees work from home?')">
                Can employees work from home?
                </button>

                <button class="q-btn"
                onclick="quickQuestion('What are office timings?')">
                What are office timings?
                </button>

                <button class="q-btn"
                onclick="quickQuestion('Explain maternity leave policy')">
                Explain maternity leave policy
                </button>

                <button class="q-btn"
                onclick="quickQuestion('What employee benefits are available?')">
                What employee benefits are available?
                </button>

                <button class="q-btn"
                onclick="quickQuestion('What is the attendance policy?')">
                What is the attendance policy?
                </button>

                <button class="q-btn"
                onclick="quickQuestion('Explain overtime policy')">
                Explain overtime policy
                </button>

                <button class="q-btn"
                onclick="quickQuestion('What is the notice period?')">
                What is the notice period?
                </button>

                <button class="q-btn"
                onclick="quickQuestion('Can employees claim internet reimbursement?')">
                Can employees claim internet reimbursement?
                </button>

                <button class="q-btn"
                onclick="quickQuestion('Explain remote work policy')">
                Explain remote work policy
                </button>

                <button class="q-btn"
                onclick="quickQuestion('What is the dress code policy?')">
                What is the dress code policy
                </button>

                <button class="q-btn"
                onclick="quickQuestion('Explain travel reimbursement policy')">
                Explain travel reimbursement policy
                </button>

            </div>

        </div>

    </div>

    <!-- CONTENT -->
    <div class="content">

        <!-- TOPBAR -->
        <div class="topbar">

            <h2>Kit Hub Skill HR Intelligence Platform</h2>

            <div class="status">● AI Online</div>

        </div>

        <!-- HERO -->
        <div class="hero" id="hero">

            <img src="/static/logo.jpeg" class="hero-logo">

            <h1>Welcome to Kit Hub Skill AI</h1>

            <p>
                Ask HR policy questions and get intelligent
                AI-powered answers instantly.
            </p>

        </div>

        <!-- CHAT -->
        <div class="chat-box" id="chat"></div>

        <!-- INPUT -->
        <div class="input-area">

            <input
            type="text"
            id="q"
            placeholder="Ask Kit Hub Skill AI..."
            onkeydown="if(event.key==='Enter') sendQuestion()">

            <button class="send-btn" onclick="sendQuestion()">
                Send
            </button>

        </div>

    </div>

</div>

<script>

function quickQuestion(q){
    document.getElementById("q").value = q;
    sendQuestion();
}

function addMessage(text, cls){

    const msg = document.createElement("div");
    msg.className = "msg " + cls;
    msg.innerText = text;

    document.getElementById("chat").appendChild(msg);

    document.getElementById("chat").scrollTop =
    document.getElementById("chat").scrollHeight;
}

async function sendQuestion(){

    const input = document.getElementById("q");

    const query = input.value.trim();

    if(!query) return;

    document.getElementById("hero").style.display = "none";

    addMessage(query,"user");

    input.value = "";

    const typing = document.createElement("div");
    typing.className = "typing";
    typing.id = "typing";
    typing.innerText = "AI is typing...";
    document.getElementById("chat").appendChild(typing);

    try{

        const response = await fetch("/ask",{
            method:"POST",
            headers:{
                "Content-Type":"application/json"
            },
            body:JSON.stringify({
                query:query
            })
        });

        const data = await response.json();

        document.getElementById("typing").remove();

        let answer =
        "📌 " + data.answer +
        "\\n\\n🎯 Intent: " + data.intent +
        "\\n📄 Sources: " + data.sources.join(", ");

        addMessage(answer,"bot");

    }catch(err){

        document.getElementById("typing").remove();

        addMessage("Error: " + err,"bot");
    }
}

</script>

</body>
</html>
"""


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():

    return render_template_string(HTML_UI)


@app.route("/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json()

        query = data.get("query", "").strip()

        if not query:

            return jsonify({
                "answer": "Please enter a question.",
                "intent": "empty_query",
                "sources": []
            })

        result = rag.ask(query)

        if isinstance(result, dict):

            answer = result.get("answer", "No answer found")
            intent = result.get("intent", "general")
            sources = result.get("sources", [])

        else:

            answer = str(result)
            intent = "general"
            sources = []

        return jsonify({
            "answer": answer,
            "intent": intent,
            "sources": sources
        })

    except Exception as e:

        print("\n[ERROR]")
        print(str(e))

        return jsonify({
            "answer": "Internal server error",
            "intent": "error",
            "sources": [],
            "error": str(e)
        }), 500


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("      HR Policy Assistant — Flask Server")
    print("      Open: http://127.0.0.1:5000")
    print("=" * 60 + "\n")

    app.run(
        debug=False,
        host="0.0.0.0",
        port=5000
    )