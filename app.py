import streamlit as st
import os
import random
import re
# 🚨 CRITICAL FIX FOR SQLITE3/CHROMA CONFLICT 🚨
# This hot-swaps the incompatible system 'sqlite3' with the
# working version from 'pysqlite3-binary' (which is in requirements.txt).
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# 🛠️ CRITICAL FIX FOR BASETOOL IMPORT PATH 🛠️
# BaseTool moved from 'crewai_tools' to 'crewai' in newer versions.
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool # <--- CHANGE THIS LINE!
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
# 5. STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="AI Banking Support", layout="wide")
st.title("🏦 AI Banking Customer Support Agent")
st.subheader("Powered by CrewAI Multi-Agent Workflow")

# Sidebar shows ticket DB
with st.sidebar:
    st.title("🗄️ Ticket Database")
    st.json(st.session_state.support_tickets)

# Chat history
for m in st.session_state.messages:
    with st.chat_message(m["role"], icon="👤" if m["role"] == "user" else "🤖"):
        st.markdown(m["content"])

# -------------------------------
# 6. MAIN INTERACTION
# -------------------------------
if prompt := st.chat_input("How can we help you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_box = st.empty()

        # Step 1: Classification
        status_box.info("Step 1: Classifying message...")

        classify_task = Task(
            description=f"Classify: '{prompt}'",
            agent=classifier,
            expected_output="Positive Feedback, Negative Feedback, or Query."
        )

        classification = Crew(
            agents=[classifier],
            tasks=[classify_task],
            process=Process.sequential,
            verbose=False
        ).kickoff().strip()

        status_box.info(f"Result: **{classification}**")

        # Step 2: Routing
        final_response = ""

        if "Query" in classification:
            status_box.info("Step 2: Checking ticket status...")
            match = re.search(r'\b\d{6}\b', prompt)
            ticket = match.group(0) if match else "No ticket found"

            query_task = Task(
                description=f"Check ticket {ticket}.",
                agent=query_handler,
                expected_output="Return ticket status."
            )
            final_response = Crew(
                agents=[query_handler],
                tasks=[query_task],
                process=Process.sequential,
                verbose=False
            ).kickoff()

        elif "Positive Feedback" in classification:
            status_box.info("Step 2: Generating thank-you message...")
            final_response = llm.invoke(
                f"Write a warm thank-you message for: '{prompt}'"
            ).content

        elif "Negative Feedback" in classification:
            status_box.info("Step 2: Logging feedback...")
            feedback_task = Task(
                description=f"Negative feedback: '{prompt}'. Use ticket generator.",
                agent=feedback_handler,
                expected_output="Apologize and include ticket number."
            )
            final_response = Crew(
                agents=[feedback_handler],
                tasks=[feedback_task],
                process=Process.sequential,
                verbose=False
            ).kickoff()

        else:
            final_response = "I couldn't classify your request—please try again."

        status_box.empty()
        st.markdown(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})

