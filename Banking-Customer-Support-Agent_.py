import streamlit as st
import os
import random
import re
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool
from langchain_openai import ChatOpenAI

# --- 1. SECRETS & CONFIGURATION ---
# IMPORTANT: This retrieves the API key from the Streamlit Cloud Secrets Manager.
try:
    # Ensure the environment variable is set for crewAI/LangChain
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("🚨 OPENAI_API_KEY not found in st.secrets! Please add it to your Streamlit Cloud secrets.")
    st.stop()

# Initialize the LLM (using gpt-4o for robust classification and generation)
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- 2. SESSION STATE & DATABASE SIMULATION ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize simulated support database (as per Word doc's requirement for ticket tracking)
if "support_tickets" not in st.session_state:
    st.session_state.support_tickets = {
        # Pre-populate with example data
        "650932": "Resolved",
        "123456": "Pending Review"
    }

# --- 3. CUSTOM TOOLS (Database Interactions) ---

class NegativeFeedbackTool(BaseTool):
    """Generates a new 6-digit ticket and saves it to the database (session state)."""
    name: str = "Negative Feedback Ticket Generator"
    description: str = "Generates a new unique 6-digit ticket, sets its status to 'Pending', saves it to the database, and returns the ticket number."

    def _run(self) -> str:
        # Generate a unique 6-digit number
        ticket_number = str(random.randint(100000, 999999))
        while ticket_number in st.session_state.support_tickets:
            ticket_number = str(random.randint(100000, 999999))
            
        # Add new ticket to our simulated database, fulfilling the 'Insert into support database' requirement
        st.session_state.support_tickets[ticket_number] = "Pending"
        
        # Return the new ticket number for the Feedback Handler Agent to use in the response
        return f"NEW_TICKET_GENERATED:{ticket_number}"

class QueryStatusTool(BaseTool):
    """Checks the status of a specific support ticket against the database."""
    name: str = "Ticket Status Checker"
    description: str = "Checks the status of a specific 6-digit support ticket number provided by the user against the database."

    def _run(self, ticket_number: str) -> str:
        # Query our simulated database (table: support_tickets)
        status = st.session_state.support_tickets.get(ticket_number, "Not Found")
        
        if status == "Not Found":
            return f"The ticket number {ticket_number} was not found in our system."
        
        # Return the status for the Query Handler Agent to use in the response
        return f"Ticket {ticket_number} status: {status}"

# Instantiate tools
ticket_tool = NegativeFeedbackTool()
query_tool = QueryStatusTool()

# --- 4. AGENT DEFINITIONS (Part 1: Multi-Agent Design) ---

# Classifier Agent
classifier = Agent(
    role='Message Classifier and Router',
    goal="Accurately categorize the user's message into one of three distinct categories: 'Positive Feedback', 'Negative Feedback', or 'Query'.",
    backstory="You are the system's first line of defense, an expert at discerning user intent and sentiment. Your **SOLE** output must be one of the three category names only.",
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Feedback Handler Agent
feedback_handler = Agent(
    role='Feedback Specialist and Formal Logger',
    goal="Handle all feedback messages. For positive feedback, generate a warm thank you. For negative feedback, use the provided tool to create a new ticket and include the new ticket number in an empathetic apology.",
    backstory="You are an empathetic customer support agent responsible for logging complaints formally and thanking users for compliments.",
    tools=[ticket_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# Query Handler Agent
query_handler = Agent(
    role='Ticket Status Checker Agent',
    goal="Search the user's message for a 6-digit ticket number, check its status using the tool, and report the status clearly.",
    backstory="You are a database specialist who efficiently finds ticket statuses. You must return a concise update based on the status found by your tool.",
    tools=[query_tool],
    llm=llm,
    verbose=True,
    allow_delegation=False
)

# --- 5. STREAMLIT UI & APP LOGIC (Part 2: UI Design) ---

st.set_page_config(page_title="AI Banking Support", layout="wide")
st.title("🏦 AI Banking Customer Support Agent System")
st.subheader("Multi-Agent Workflow based on Application Outline")

# Sidebar for database view (Fulfills UI requirement for database interaction/view)
with st.sidebar:
    st.title("🗄️ Support Ticket Database")
    st.caption("This data is stored in session state and updates in real-time.")
    st.json(st.session_state.support_tickets)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], icon="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("How can we help you today? (Try: 'I love your new app' or 'My ticket 650932 is resolved?')"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", icon="👤"):
        st.markdown(prompt)

    # --- Agent Execution Logic ---
    with st.chat_message("assistant", icon="🤖"):
        agent_status = st.empty()
        
        # 1. Run the Classifier Agent
        agent_status.info("Step 1: Running Classifier Agent...")
        
        classify_task = Task(
            description=f"Classify this user message: '{prompt}'",
            agent=classifier,
            expected_output="One of: 'Positive Feedback', 'Negative Feedback', or 'Query'. Only output the category name.",
            async_execution=False # Must complete before routing
        )
        
        try:
            # Note: We suppress verbose output here to keep the UI clean, 
            # but the agent's internal logs are still captured.
            classification = Crew(agents=[classifier], tasks=[classify_task], process=Process.sequential, verbose=False).kickoff()
        except Exception as e:
            st.error(f"Classification failed: {e}")
            agent_status.empty()
            st.session_state.messages.append({"role": "assistant", "content": "I apologize, an error occurred during classification."})
            st.experimental_rerun()
            return
            
        agent_status.info(f"Classification result: **{classification.strip()}**")
        
        final_response = ""
        
        # 2. Route to the appropriate Handler Agent
        
        if "Query" in classification:
            # --- Query Handler Workflow (Checks ticket status) ---
            agent_status.info("Step 2: Routing to Query Handler Agent...")
            
            # Use regex to robustly find a 6-digit number for the Query Handler to focus on
            match = re.search(r'\b\d{6}\b', prompt)
            ticket_to_check = match.group(0) if match else "No 6-digit ticket found"

            query_task = Task(
                description=f"The user's query is: '{prompt}'. If a 6-digit ticket number is present, check its status using the tool. If not, state that no ticket was found. The specific ticket to check is: {ticket_to_check}",
                agent=query_handler,
                expected_output="A concise response to the user reporting the status of the ticket or stating it was not found."
            )
            final_response = Crew(agents=[query_handler], tasks=[query_task], process=Process.sequential, verbose=False).kickoff()

        elif "Positive Feedback" in classification:
            # --- Positive Feedback Handler Workflow (Responds with thanks) ---
            agent_status.info("Step 2: Routing to Feedback Handler (Positive) Agent...")
            
            # Simple LLM invocation for personalized response (no tool needed as per outline)
            prompt_for_llm = f"Generate a warm, personalized thank-you message for this positive feedback: '{prompt}'"
            final_response = llm.invoke(prompt_for_llm).content

        elif "Negative Feedback" in classification:
            # --- Negative Feedback Handler Workflow (Generates new ticket) ---
            agent_status.info("Step 2: Routing to Feedback Handler (Negative) Agent...")
            
            # This task ensures the tool is used to generate and log the new ticket, 
            # and then the agent incorporates the result into the final response.
            feedback_task = Task(
                description=f"The user has negative feedback: '{prompt}'. **CRITICAL**: Use the 'Negative Feedback Ticket Generator' tool, then formulate an empathetic apology that **MUST** include the newly generated ticket number from the tool's output.",
                agent=feedback_handler,
                expected_output="An empathetic, apologetic message that includes the generated ticket number."
            )
            final_response = Crew(agents=[feedback_handler], tasks=[feedback_task], process=Process.sequential, verbose=False).kickoff()
        
        else:
            final_response = "I'm sorry, the classifier returned an ambiguous result. Could you please rephrase your request? (Expected: Positive Feedback, Negative Feedback, or Query)."

        # Display the final response and update history
        agent_status.empty() # Clear the status message
        st.markdown(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})