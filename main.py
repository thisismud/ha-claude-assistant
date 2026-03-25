from fastapi import FastAPI
from pydantic import BaseModel
import anthropic
import requests
import os
import json

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
                "automation": {"type": "object", "description": "Full automation config object including alias, trigger,
condition, action"}
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

SYSTEM_PROMPT = """You are an expert Home Assistant advisor with access to two systems: 'home' and 'farm'.

You can query entities, services, and automations, and you can make changes including calling services and creating
automations.

Guidelines:
- Always fetch current entity data before making changes or giving advice so you use correct entity IDs
- Confirm which system you're working with before making any changes
- When creating automations, explain what you're creating before doing it
- Be practical and specific — use real entity IDs from the user's system"""

class Prompt(BaseModel):
    message: str

@app.get("/")
def root():
    return {"status": "AI assistant running"}

@app.post("/chat")
def chat(prompt: Prompt):
    messages = [{"role": "user", "content": prompt.message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages
        )

        if response.stop_reason == "end_turn":
            text = next((b.text for b in response.content if hasattr(b, "text")), "No response")
            return {"response": text}

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
            return {"response": "Unexpected error occurred"}
