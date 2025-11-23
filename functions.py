import sqlite3
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

DATABASE_FILE = "tickets.db"

def init_db():
    """Initialize a simple local SQLite database."""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tickets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            status TEXT DEFAULT 'open'
        )
    ''')
    conn.commit()
    conn.close()

def create_ticket(title):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO tickets (title) VALUES (?)", (title,))
    ticket_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return json.dumps({"ticket_id": ticket_id, "status": "created"})
    
def cancel_ticket(ticket_id):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
    if cursor.fetchone() is None:
        conn.close()
        return json.dumps({"error": f"Ticket ID {ticket_id} not found."})

    cursor.execute("UPDATE tickets SET status = 'closed' WHERE id = ?", (ticket_id,))
    conn.commit()
    conn.close()
    
    return json.dumps({"ticket_id": ticket_id, "status": "closed"})

def get_tickets(status=None):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    if status:
        cursor.execute("SELECT * FROM tickets WHERE status = ?", (status,))
    else:
        cursor.execute("SELECT * FROM tickets")
    rows = cursor.fetchall()
    conn.close()
    
    results = [{"id": r[0], "title": r[1], "status": r[2]} for r in rows]
    return json.dumps(results)

available_functions = {
    "create_ticket": create_ticket,
    "cancel_ticket": cancel_ticket,
    "get_tickets": get_tickets
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "create_ticket",
            "description": "Create a new support ticket in the database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "The title or description of the issue."}
                },
                "required": ["title"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_ticket",
            "description": "Cancel an existing support ticket by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "integer",
                        "description": "The unique ID of the ticket to cancel."
                    }
                },
                "required": ["ticket_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_tickets",
            "description": "Get a list of support tickets from the database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string", 
                        "enum": ["open", "closed"],
                        "description": "Filter by status (e.g. 'open')"
                    }
                }
            }
        }
    }
]

def get_response(user_input):
    init_db()
    messages = [{"role": "system", "content": "You are a helpful support ticket database assistant with abilities such as \'Create a support ticket\', \'Cancel a support ticket\' and \'List all support tickets\'"}]
    
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gemini-2.0-flash",
        messages=messages,
        tools=tools
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"  [System] Calling tool: {function_name} with {function_args}")
            
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            })

        second_response = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=messages
        )
        final_reply = second_response.choices[0].message.content
        return final_reply
        messages.append({"role": "assistant", "content": final_reply})
    
    else:
        reply = response_message.content
        return reply
        messages.append({"role": "assistant", "content": reply})

