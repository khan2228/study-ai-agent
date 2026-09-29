import streamlit as st

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent


# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="Study AI Agent",
    page_icon="🤖",
    layout="centered"
)


# =========================
# TOOL 1: CALCULATOR
# =========================

@tool
def calculator(expression: str) -> str:
    """Solve mathematical expressions, equations, derivatives and integrals."""

    try:
        from sympy import sympify, solve, diff, integrate

        expression = expression.strip()

        # Equation
        if "=" in expression:
            left, right = expression.split("=", 1)
            equation = sympify(left) - sympify(right)
            result = solve(equation)
            return f"Solution: {result}"

        # Derivative
        if expression.lower().startswith("derivative"):
            expr = expression.split(":", 1)[1].strip()
            parsed_expr = sympify(expr)
            x = list(parsed_expr.free_symbols)[0]
            return f"Derivative: {diff(parsed_expr, x)}"

        # Integral
        if expression.lower().startswith("integral"):
            expr = expression.split(":", 1)[1].strip()
            parsed_expr = sympify(expr)
            x = list(parsed_expr.free_symbols)[0]
            return f"Integral: {integrate(parsed_expr, x)}"

        # Normal calculation
        result = sympify(expression)
        return f"Answer: {result}"

    except Exception as e:
        return f"Could not solve: {e}"


# =========================
# TOOL 2: QUIZ GENERATOR
# =========================

@tool
def quiz_generator(topic: str) -> str:
    """Generate a quiz about any topic."""

    return f"""
Create a short educational quiz about {topic}.

Requirements:
- Create 5 multiple-choice questions.
- Give 4 options for each question.
- Clearly show the correct answer.
- Keep the questions suitable for a student.
"""


# =========================
# GEMINI MODEL
# =========================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


# =========================
# CREATE AGENT
# =========================

agent = create_agent(
    model=llm,
    tools=[
        calculator,
        quiz_generator
    ],
    system_prompt=(
        "You are a helpful general-purpose AI study assistant. "
        "Answer questions about any academic or general topic. "
        "Explain difficult concepts in simple language when appropriate. "
        "Use the calculator tool for mathematical calculations. "
        "Use the quiz_generator tool when the user asks for a quiz. "
        "Do not say that you only know predefined topics. "
        "Answer normally using your knowledge."
    )
)


# =========================
# MEMORY
# =========================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []


# =========================
# HEADER
# =========================

st.title("🤖 Study AI Agent")

st.caption(
    "General-purpose AI assistant powered by LangChain + Gemini"
)


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header("🛠️ Tools")

    st.write("🧮 Calculator")
    st.write("📝 Quiz Generator")
    st.write("🧠 General AI Questions")

    st.divider()

    st.write("💡 You can ask:")

    st.write("• Explain Psychology")
    st.write("• What is Agentic AI?")
    st.write("• Explain Indus Valley Civilization")
    st.write("• Calculate 125 * 89")
    st.write("• Make a quiz about Networking")

    st.divider()

    if st.button("🗑️ New Chat"):
        st.session_state.chat_history = []
        st.session_state.agent_messages = []
        st.rerun()


# =========================
# SHOW CHAT HISTORY
# =========================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================
# CHAT INPUT
# =========================

question = st.chat_input("Ask anything...")


# =========================
# PROCESS QUESTION
# =========================

if question:

    # Show user message
    with st.chat_message("user"):
        st.markdown(question)

    # Save user message for UI
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    # Save user message for agent memory
    st.session_state.agent_messages.append({
        "role": "user",
        "content": question
    })

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking... 🤔"):

            result = agent.invoke({
                "messages": st.session_state.agent_messages
            })

            response = result["messages"][-1].content

            # Handle Gemini list response
            if isinstance(response, list):

                text_parts = []

                for item in response:

                    if isinstance(item, dict):

                        if item.get("type") == "text":
                            text_parts.append(
                                item.get("text", "")
                            )

                    elif isinstance(item, str):
                        text_parts.append(item)

                response = "\n".join(text_parts)

            st.markdown(response)

    # Save complete agent memory
    st.session_state.agent_messages = result["messages"]

    # Save clean response for UI
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response
    })