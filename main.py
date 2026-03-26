from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import anthropic
import requests
import os
import json
import uuid
import time

app = FastAPI()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

HA_SYSTEMS = {
    "home": {
        "url": os.environ.get("HOME_HA_URL", ""),
        "token": os.environ.get("HOME_HA_TOKEN", "")
    },
    "farm": {
        "url": os.environ.get("FARM_HA_URL", ""),
        "token": os.environ.get("FARM_HA_TOKEN", "")
    }
}

sessions = {}

CONTEXT_TTL = 1800  # seconds — refresh HA context every 30 minutes
context_cache = {
    "home": {"summary": None, "timestamp": 0},
    "farm": {"summary": None, "timestamp": 0}
}


def build_system_context(system: str) -> str:
    cache = context_cache.get(system)
    if cache and cache["summary"] and (time.time() - cache["timestamp"]) < CONTEXT_TTL:
        return cache["summary"]

    states = ha_request(system, "GET", "states")
    if isinstance(states, dict) and "error" in states:
        return f"[{system.capitalize()} system unavailable: {states['error']}]"

    domains: dict = {}
    for state in states:
        domain = state["entity_id"].split(".")[0]
        domains.setdefault(domain, []).append(state["entity_id"])

    lines = [f"### {system.capitalize()} System"]
    for domain, entity_ids in sorted(domains.items()):
        lines.append(f"\n**{domain}** ({len(entity_ids)})")
        for eid in entity_ids:
            lines.append(f"  - {eid}")

    summary = "\n".join(lines)
    context_cache[system]["summary"] = summary
    context_cache[system]["timestamp"] = time.time()
    return summary


def get_system_prompt() -> str:
    home_context = build_system_context("home")
    farm_context = build_system_context("farm")
    return f"""You are an expert Home Assistant advisor with access to two systems: 'home' and 'farm'.

Below is a current snapshot of both systems. Use these entity IDs directly for simple requests. \
Use the available tools when you need live state/attribute data, or to make changes.

{home_context}

{farm_context}

Guidelines:
- Use entity IDs from the snapshot above when possible — call get_entities only when you need current state or attributes
- Confirm which system you are working with before making any changes
- When creating automations, explain what you are creating before doing it
- Be practical and specific"""

def ha_request(system: str, method: str, endpoint: str, data: dict = None):
    config = HA_SYSTEMS.get(system)
    if not config or not config["url"]:
        return {"error": f"System '{system}' not configured"}
    url = f"{config['url']}/api/{endpoint}"
    headers = {
        "Authorization": f"Bearer {config['token']}",
        "Content-Type": "application/json"
    }
    try:
        if method == "GET":
            response = requests.get(url, headers=headers, timeout=10)
        else:
            response = requests.post(url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}

tools = [
    {
        "name": "get_entities",
        "description": "Get all entity states from a Home Assistant system including lights, switches, sensors, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "system": {"type": "string", "enum": ["home", "farm"], "description": "Which HA system to query"}
            },
            "required": ["system"]
        }
    },
    {
        "name": "get_services",
        "description": "Get all available services from a Home Assistant system.",
        "input_schema": {
            "type": "object",
            "properties": {
                "system": {"type": "string", "enum": ["home", "farm"], "description": "Which HA system to query"}
            },
            "required": ["system"]
        }
    },
    {
        "name": "get_automations",
        "description": "Get all automations from a Home Assistant system.",
        "input_schema": {
            "type": "object",
            "properties": {
                "system": {"type": "string", "enum": ["home", "farm"], "description": "Which HA system to query"}
            },
            "required": ["system"]
        }
    },
    {
        "name": "call_service",
        "description": "Call a Home Assistant service to control devices or trigger actions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "system": {"type": "string", "enum": ["home", "farm"]},
                "domain": {"type": "string", "description": "Service domain e.g. light, switch, automation"},
                "service": {"type": "string", "description": "Service name e.g. turn_on, turn_off, toggle"},
                "data": {"type": "object", "description": "Service data e.g. {\"entity_id\": \"light.living_room\"}"}
            },
            "required": ["system", "domain", "service"]
        }
    },
    {
        "name": "create_automation",
        "description": "Create a new automation in Home Assistant.",
        "input_schema": {
            "type": "object",
            "properties": {
                "system": {"type": "string", "enum": ["home", "farm"]},
                "automation": {"type": "object", "description": "Full automation config including alias, trigger, condition, action"}
            },
            "required": ["system", "automation"]
        }
    }
]

def process_tool_call(tool_name: str, tool_input: dict):
    system = tool_input.get("system")
    if tool_name == "get_entities":
        return ha_request(system, "GET", "states")
    elif tool_name == "get_services":
        return ha_request(system, "GET", "services")
    elif tool_name == "get_automations":
        states = ha_request(system, "GET", "states")
        if isinstance(states, list):
            return [s for s in states if s["entity_id"].startswith("automation.")]
        return states
    elif tool_name == "call_service":
        domain = tool_input["domain"]
        service = tool_input["service"]
        data = tool_input.get("data", {})
        return ha_request(system, "POST", f"services/{domain}/{service}", data)
    elif tool_name == "create_automation":
        automation = tool_input["automation"]
        return ha_request(system, "POST", "config/automation/config", automation)
    return {"error": f"Unknown tool: {tool_name}"}


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HA Assistant</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: sans-serif; background: #1a1a2e; color: #eee; height: 100vh; display: flex; flex-direction: column; }
        h1 { padding: 16px; background: #16213e; border-bottom: 1px solid #333; font-size: 1.1rem; }
        #chat { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
        .message { max-width: 80%; padding: 10px 14px; border-radius: 12px; line-height: 1.5; white-space: pre-wrap; }
        .user { align-self: flex-end; background: #0f3460; }
        .assistant { align-self: flex-start; background: #16213e; border: 1px solid #333; }
        .thinking { align-self: flex-start; color: #888; font-style: italic; padding: 10px 14px; }
        #input-area { display: flex; gap: 8px; padding: 16px; background: #16213e; border-top: 1px solid #333; }
        #input { flex: 1; padding: 10px 14px; border-radius: 8px; border: 1px solid #333; background: #0f0f1a; color: #eee; font-size: 1rem; }
        button { padding: 10px 20px; border-radius: 8px; border: none; background: #e94560; color: white; cursor: pointer; font-size: 1rem; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
    </style>
</head>
<body>
    <h1>Home Assistant AI</h1>
    <div id="chat"></div>
    <div id="input-area">
        <input type="text" id="input" placeholder="Ask about your home or farm..." />
        <button id="send-btn" onclick="sendMessage()">Send</button>
    </div>
    <script>
        let sessionId = localStorage.getItem('ha_session_id');

        function addMessage(text, cls) {
            const chat = document.getElementById('chat');
            const div = document.createElement('div');
            div.className = cls;
            div.textContent = text;
            chat.appendChild(div);
            chat.scrollTop = chat.scrollHeight;
            return div;
        }

        async function sendMessage() {
            const input = document.getElementById('input');
            const btn = document.getElementById('send-btn');
            const message = input.value.trim();
            if (!message) return;

            input.value = '';
            btn.disabled = true;
            addMessage(message, 'message user');
            const thinking = addMessage('Thinking...', 'thinking');

            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({message, session_id: sessionId})
                });
                const data = await res.json();
                sessionId = data.session_id;
                localStorage.setItem('ha_session_id', sessionId);
                thinking.remove();
                addMessage(data.response, 'message assistant');
            } catch (err) {
                thinking.textContent = 'Error: ' + err.message;
            }

            btn.disabled = false;
            input.focus();
        }

        document.getElementById('input').addEventListener('keydown', e => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
        });
    </script>
</body>
</html>"""

class Prompt(BaseModel):
    message: str
    session_id: str = None

@app.get("/", response_class=HTMLResponse)
def root():
    return HTML

@app.post("/chat")
def chat(prompt: Prompt):
    session_id = prompt.session_id or str(uuid.uuid4())
    messages = list(sessions.get(session_id, []))
    messages.append({"role": "user", "content": prompt.message})

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=get_system_prompt(),
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            text = next((b.text for b in response.content if hasattr(b, "text")), "No response")
            messages.append({"role": "assistant", "content": text})
            sessions[session_id] = messages
            return {"response": text, "session_id": session_id}

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = process_tool_call(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            return {"response": "Unexpected error occurred", "session_id": session_id}
