from dotenv import load_dotenv
load_dotenv()

## db llm, tools, create_agent , system_prompt

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import streamlit as st

db = SQLDatabase.from_uri("sqlite:///my_tasks.db")

db.run(
    """
    CREATE TABLE IF NOT EXISTS tasks(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        description TEXT,
        status TEXT CHECK (status IN ('pending','in_progress','completed')) DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        deadline DATE NOT NULL
        );

    """   
)

# print("DB table created successfully")
## llm tools memory system prompts

model= ChatGroq(model="openai/gpt-oss-20b",temperature=0,model_kwargs={"include_reasoning": False},streaming=True)
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools =toolkit.get_tools()
system_prompt = """
You are a proactive, professional, and friendly Personal Assistant (PA) helping the user manage their tasks, schedule, and deadlines using a live SQLite database.

### Database Schema
Table: `tasks`
- `id` (INTEGER, PRIMARY KEY, AUTOINCREMENT)
- `title` (TEXT, NOT NULL)
- `description` (TEXT)
- `status` (TEXT CHECK (status IN ('pending','in_progress','completed')) DEFAULT 'pending')
- `deadline` (DATE): Format `YYYY-MM-DD` (can be NULL if no due date).
- `created_at` (TIMESTAMP DEFAULT CURRENT_TIMESTAMP)

### Persona & Conversation Rules
1. **Act Like a Real PA:** Talk naturally and conversationally (e.g., *"You've got 3 things due today. Here they are:"* or *"Got it, I've scheduled 'Review Report' for today."*).
2. **Handling Time & "Today's" Tasks:** When the user asks for "today's tasks" or tasks due on a specific day, write SQL queries filtering by the deadline column using SQLite date functions (e.g., `WHERE date(deadline) = date('now')`).
3. **Query Limits & Sorting:** Unless looking up a specific date or ID, limit general `SELECT` queries to a maximum of 10 results, using `ORDER BY deadline ASC, created_at DESC` (so urgent upcoming deadlines show first).
4. **Post-Action Confirmation:** After creating, updating, or deleting a task, immediately run a `SELECT` query to show the updated list or item so the user gets real-time confirmation.
5. **Output Formatting:** Present lists of tasks in a clean, structured Markdown table.
6. **CRUD & Scheduling Operations:**
   - **Create (with deadline):** `INSERT INTO tasks (title, description, status, deadline) VALUES (...)`
   - **Read (General):** `SELECT * FROM tasks ORDER BY deadline ASC LIMIT 10`
   - **Read (Today):** `SELECT * FROM tasks WHERE date(deadline) = date('now')`
   - **Update:** `UPDATE tasks SET status = '<status>', deadline = '<date>' WHERE id = ?`
   - **Delete:** `DELETE FROM tasks WHERE id = ? OR title = ?`
7. **Safety:** Never execute an `UPDATE` or `DELETE` query without a proper `WHERE` clause.
"""

@st.cache_resource
def get_agent():
    agent = create_agent(
        model=model,
        tools =tools,
        checkpointer=InMemorySaver(),
        system_prompt=system_prompt
    )
    return agent

agent=get_agent()

st.subheader("Task Bot ,Manage your To-Do")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    

prompt = st.chat_input("Ask me to manage your tasks")

if prompt:
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # Stream AI response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        for chunk, metadata in agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            {"configurable": {"thread_id": "1"}},
            stream_mode="messages"
        ):
            # Only display output from the AI model
            if metadata.get("langgraph_node") == "model":
                content = chunk.content

                if isinstance(content, str):
                    token = content
                else:
                    token = "".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict)
                        and block.get("type") == "text"
                    )

                if token:
                    full_response += token
                    placeholder.markdown(full_response + "▌")

        placeholder.markdown(full_response)

    # Save conversation
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response
    })

        

