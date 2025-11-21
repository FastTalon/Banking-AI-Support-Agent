import streamlit as st
import os
import random
import re
from datetime import datetime  # ✅ NEW: for timestamps
from typing import Optional     # ✅ NEW: for type hints

# 🚨 CRITICAL FIX FOR SQLITE3/CHROMA CONFLICT 🚨
# This hot-swaps the incompatible system 'sqlite3' with the
# working version from 'pysqlite3-binary' (which is in requirements.txt).
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# 🛠️ CRITICAL FIX FOR BASETOOL IMPORT PATH 🛠️
# BaseTool moved from 'crewai_tools' to 'crewai' in newer versions.
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool  # ✅ keep this import
from langchain_openai import ChatOpenAI

# -------------------------------
# 1. LOAD OPENAI API KEY
# -------------------------------
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("🚨 OPENAI_API_KEY missing in Streamlit Cloud Secrets!")
    st.stop()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# -------------------------------
# 2. SESSION STATE DATABASE
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "support_tickets" not in st.session_state:
    st.session_state.support_tickets = {
        "650932": "Resolved",
        "123456": "Pending Review"
    }

# ✅ NEW: logs for LLMOps
if "logs" not in st.session_state:
    st.session_state.logs = []

# ✅ NEW: track last generated ticket for logging/debugging
if "last_generated_ticket" not in st.session_state:
    st.session_state.last_generated_ticket = None

# -------------------------------
# 3. CUSTOM TOOLS
# -------------------------------
class NegativeFeedbackTool(BaseTool):
    name: str = "Negative Feedback Ticket Generator"
    description: str = "Creates a 6-digit ticket and sets status to Pending."

    def _run(self) -> str:
        ticket = str(random.randint(100000, 999999))
        while ticket in st.session_state.support_tickets:
            ticket = str(random.randint(100000, 999999))

        st.session_state.support_tickets[ticket] = "Pending"

        # ✅ NEW: store last generated ticket for logging/debugging
        st.session_state.last_generated_ticket = ticket

        return f"NEW_TICKET_GENERATED:{ticket}"

class QueryStatusTool(BaseTool):
    name: str = "Ticket Status Checker"
    description: str = "Checks ticket status."

    def _run(self, ticket_number: str) -> str:
        status = st.session_state.support_tickets.get(ticket_number, "Not Found")
        return f"Ticket {ticket_number} status: {status}"

ticket_tool = NegativeFeedbackTool()
query_tool = QueryStatusTool()

# -------------------------------
# 4. AGENTS
# -------------------------------
classifier = Agent(
    role="Message Classifier",
    goal="Classify as Positive Feedback, Negative Feedback, or Query.",
    backstory="You ONLY output one category.",
    llm=llm,
    verbose=False,
    allow_delegation=False
)

feedback_handler = Agent(
    role="Feedback Handler",
    goal="Positive feedback → thank user. Negative → create ticket.",
    backstory="Handles all feedback with empathy.",
    tools=[ticket_tool],
    llm=llm,
    verbose=False,
    allow_delegation=False
)

query_handler = Agent(
    role="Query Handler",
    goal="Extract 6-digit ticket and check status.",
    backstory="Specialist in database lookups.",
    tools=[query_tool],
    llm=llm,
    verbose=False,
    allow_delegation=False
)

# -------------------------------
# 5. LLMOps HELPERS (EVALUATION + LOGGING)
# -------------------------------
def evaluate_interaction(
    user_prompt: str,
    classification: str,
    route: str,
    final_response: str
) -> str:
    """
    Uses the LLM as a QA evaluator to score the response quality.
    Returns a human-readable multi-line string with simple metrics.
    """
    evaluation_prompt = f"""
You are an evaluation agent for a banking customer support assistant.

Evaluate the following interaction and provide compact metrics.

User message:
{user_prompt}

System classification:
{classification}

Selected route/agent:
{route}

Assistant final response:
{final_response}

Rate the assistant's response on:
- Overall helpfulness (1-10)
- Empathy/tonality (1-10)
- Clarity (1-10)
- Appropriateness for banking support (Yes/No)

Return your evaluation in EXACTLY this format (no extra text):

Overall: <number>/10
Empathy: <number>/10
Clarity: <number>/10
Appropriate: <Yes or No>
ShortComment: <one short sentence>
"""
    result = llm.invoke(evaluation_prompt).content
    return result


def log_interaction(
    user_prompt: str,
    classification: str,
    route: str,
    ticket: Optional[str],
    final_response: str,
    success: bool,
    error: Optional[str]
) -> None:
    """
    Appends a structured log entry to st.session_state.logs with
    evaluation metrics and debug info.
    """
    try:
        evaluation = evaluate_interaction(
            user_prompt=user_prompt,
            classification=classification,
            route=route,
            final_response=final_response
        )
    except Exception as e:
        evaluation = f"Evaluation failed: {e}"

    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_prompt": user_prompt,
        "classification": classification,
        "route": route,
        "ticket": ticket,
        "response": final_response,
        "success": success,
        "error": error,
        "evaluation": evaluation,
    }

    st.session_state.logs.append(log_entry)

# -------------------------------
# 6. STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="AI Banking Support", layout="wide")
st.title("🏦 AI Banking Customer Support Agent")
st.subheader("Powered by CrewAI Multi-Agent Workflow + LLMOps Dashboard")

# Sidebar shows ticket DB
with st.sidebar:
    st.title("🗄️ Ticket Database")
    st.json(st.session_state.support_tickets)

# Chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# -------------------------------
# 7. MAIN INTERACTION
# -------------------------------
if prompt := st.chat_input("How can we help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_box = st.empty()

        # ✅ NEW: tracking variables for logging / LLMOps
        classification = "Unclassified"
        route = "Unknown"
        ticket_used: Optional[str] = None
        final_response = ""
        success = True
        error_msg: Optional[str] = None

        try:
            # Step 1: Classification
            status_box.info("Step 1: Classifying message...")

            classify_task = Task(
                description=f"Classify: '{prompt}'",
                agent=classifier,
                expected_output="Positive Feedback, Negative Feedback, or Query."
            )

            classification = str(Crew(
                agents=[classifier],
                tasks=[classify_task],
                process=Process.sequential,
                verbose=False
            ).kickoff()).strip()

            status_box.info(f"Result: **{classification}**")

            # Step 2: Routing
            if "Query" in classification:
                route = "Query Handler"
                status_box.info("Step 2: Checking ticket status...")
                match = re.search(r'\b\d{6}\b', prompt)
                ticket = match.group(0) if match else "No ticket found"
                ticket_used = ticket

                query_task = Task(
                    description=f"Check ticket {ticket}.",
                    agent=query_handler,
                    expected_output="Return ticket status."
                )
                final_response = str(Crew(
                    agents=[query_handler],
                    tasks=[query_task],
                    process=Process.sequential,
                    verbose=False
                ).kickoff()).strip()

            elif "Positive Feedback" in classification:
                route = "Positive Feedback Handler"
                status_box.info("Step 2: Generating thank-you message...")
                final_response = llm.invoke(
                    f"Write a warm thank-you message for: '{prompt}'"
                ).content

            elif "Negative Feedback" in classification:
                route = "Negative Feedback Handler"
                status_box.info("Step 2: Logging feedback and creating ticket...")

                feedback_task = Task(
                    description=f"Negative feedback: '{prompt}'. Use ticket generator.",
                    agent=feedback_handler,
                    expected_output="Apologize and include ticket number."
                )
                final_response = str(Crew(
                    agents=[feedback_handler],
                    tasks=[feedback_task],
                    process=Process.sequential,
                    verbose=False
                ).kickoff()).strip()

                # Capture last generated ticket for logging
                ticket_used = st.session_state.last_generated_ticket

            else:
                route = "Fallback"
                final_response = "I couldn't classify your request—please try again."

        except Exception as e:
            success = False
            error_msg = str(e)
            final_response = (
                "⚠️ An internal error occurred while processing your request. "
                "Our team has been notified."
            )

        status_box.empty()
        st.markdown(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})

        # ✅ NEW: log + evaluate this interaction for LLMOps
        log_interaction(
            user_prompt=prompt,
            classification=classification,
            route=route,
            ticket=ticket_used,
            final_response=final_response,
            success=success,
            error=error_msg,
        )

# -------------------------------
# 8. LLMOps DASHBOARD (LOGS + DEBUGGING)
# -------------------------------
st.divider()
st.subheader("🧪 LLMOps: Evaluation, Logs & Debugging")

if st.session_state.logs:
    latest = st.session_state.logs[-1]

    # Latest evaluation summary
    with st.expander("📊 Latest Interaction Evaluation", expanded=True):
        st.markdown(f"**Timestamp (UTC):** {latest['timestamp']}")
        st.markdown(f"**User prompt:** {latest['user_prompt']}")
        st.markdown(f"**Classification:** `{latest['classification']}`")
        st.markdown(f"**Route / Agent:** `{latest['route']}`")
        if latest["ticket"]:
            st.markdown(f"**Ticket:** `{latest['ticket']}`")
        st.markdown("**Assistant Response:**")
        st.markdown(latest["response"])
        st.markdown("**Evaluation (QA-based scoring):**")
        st.code(latest["evaluation"])

    # Full historical logs
    with st.expander("🕒 Historical Logs (All Interactions)"):
        st.dataframe(st.session_state.logs, use_container_width=True)

    # Debugging trace panel
    with st.expander("🐞 Debug Trace (Last Interaction)"):
        st.markdown("**Raw Log Entry:**")
        st.json(latest)
        if latest["error"]:
            st.error(f"Error captured: {latest['error']}")
else:
    st.info("No interactions logged yet. Start chatting to see LLMOps metrics and logs here.")








