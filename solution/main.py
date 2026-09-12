"""
Customer Support AI Agent — Solution
==========================================
Run locally (after filling in config values):
  uv run main.py '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'

Deploy to AgentCore:
  agentcore deploy

Invoke deployed agent:
  agentcore invoke '{"prompt": "Hello", "customer_id": "CUST-123", "session_id": "s1"}'
"""

# ── Imports ───────────────────────────────────────────────────────────────────
from strands import Agent, tool
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from bedrock_agentcore.memory import MemoryClient
from strands.models import BedrockModel
from strands.tools.mcp.mcp_client import MCPClient
from mcp.client.streamable_http import streamable_http_client
import argparse, json
import os, asyncio, boto3
from strands.hooks import (
    HookProvider, AfterInvocationEvent, HookRegistry, MessageAddedEvent,
)
import logging
import uuid
from typing import Dict
from bedrock_agentcore.tools.code_interpreter_client import code_session
import urllib.request
from bs4 import BeautifulSoup


logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("CSAI_Agent")

# ── App Initialisation ────────────────────────────────────────────────────────
app = BedrockAgentCoreApp()

# Suppress interactive tool-consent prompts (required in headless deployments).
os.environ["BYPASS_TOOL_CONSENT"] = "true"

# ── Configuration ──────────────────────────────────────────────────────────────
GATEWAY_URL = "https://customersupportgateway-2e9pfx19oe.gateway.bedrock-agentcore.us-east-1.amazonaws.com/mcp"
KB_ID       = "ONG4BY7VCI"
REGION      = "us-east-1"
MEMORY_ID   = "CustomerSupportMemory-WXmjx15eyQ"


# ── Model and Clients ────────────────────────────────────────────────────────
model_id = "global.amazon.nova-2-lite-v1:0"

model = BedrockModel(model_id=model_id)
memory_client = MemoryClient(region_name=REGION)
_bedrock_runtime = boto3.client("bedrock-agent-runtime", region_name=REGION)


# ── Namespace Helper ─────────────────────────────────────────────────────────
def get_namespaces(mem_client: MemoryClient, memory_id: str) -> Dict:
    """Return a dict mapping strategy type → namespace template string."""
    strategies = mem_client.get_memory_strategies(memory_id)
    return {strategy["type"]: strategy["namespaces"][0] for strategy in strategies}


# ── Memory Hook ───────────────────────────────────────────────────────────────
class MemoryHook(HookProvider):
    """Long-term memory hook for the customer support agent."""

    def __init__(
        self,
        actor_id: str,
        session_id: str,
        memory_client: MemoryClient,
        memory_id: str,
    ):
        self.actor_id = actor_id
        self.session_id = session_id
        self.memory_client = memory_client
        self.memory_id = memory_id
        self.namespaces = get_namespaces(memory_client, memory_id)

    def retrieve_customer_context(self, event: MessageAddedEvent):
        """Retrieve relevant memories and prepend them to the user message."""
        message = event.message

        if message.get("role") != "user":
            return
        content = message.get("content", [])
        if not content or "text" not in content[0]:
            return

        original_text = content[0]["text"]
        all_context = []

        for strategy_type, namespace_template in self.namespaces.items():
            namespace = namespace_template.format(actorId=self.actor_id)
            try:
                memories = self.memory_client.retrieve_memories(
                    memory_id=self.memory_id,
                    namespace=namespace,
                    query=original_text,
                    top_k=5,
                )
            except Exception as e:
                logger.warning(f"Memory retrieval failed for {namespace}: {e}")
                continue

            for m in memories:
                text = m.get("content", {}).get("text", "")
                if text:
                    all_context.append(f"[{strategy_type}] {text}")

        if all_context:
            context_block = "\n".join(all_context)
            content[0]["text"] = f"Customer Context:\n{context_block}\n\n{original_text}"

    def save_support_interaction(self, event: AfterInvocationEvent):
        """Save the completed turn to memory after the agent responds."""
        messages = event.agent.messages
        customer_query = None
        agent_response = None

        for msg in reversed(messages):
            content = msg.get("content", [])
            if not content or "text" not in content[0]:
                continue
            text = content[0]["text"]
            role = msg.get("role")
            if role == "assistant" and agent_response is None:
                agent_response = text
            elif role == "user" and customer_query is None:
                customer_query = text
            if customer_query and agent_response:
                break

        if customer_query and agent_response:
            try:
                self.memory_client.create_event(
                    memory_id=self.memory_id,
                    actor_id=self.actor_id,
                    session_id=self.session_id,
                    messages=[(customer_query, "USER"), (agent_response, "ASSISTANT")],
                )
            except Exception as e:
                logger.warning(f"Failed to save interaction to memory: {e}")

    def register_hooks(self, registry: HookRegistry) -> None:  # type: ignore
        """Register both memory callbacks."""
        registry.add_callback(MessageAddedEvent, self.retrieve_customer_context)
        registry.add_callback(AfterInvocationEvent, self.save_support_interaction)


# ── Knowledge Base Tool ───────────────────────────────────────────────────────
@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the Amazon product catalog and support knowledge base.
    Use this for product specifications, return policies, warranty
    information, loyalty program details, and order status definitions.
    """
    if not KB_ID or KB_ID == "<kbid>":
        return "Knowledge base not configured."

    try:
        resp = _bedrock_runtime.retrieve(
            knowledgeBaseId=KB_ID,
            retrievalQuery={"text": query},
        )
    except Exception as e:
        logger.warning(f"Knowledge base retrieval failed: {e}")
        return f"Knowledge base search failed: {e}"

    results = resp.get("retrievalResults", [])
    if not results:
        return "No relevant information found in the knowledge base."

    chunks = [r["content"]["text"] for r in results if r.get("content", {}).get("text")]
    return "\n---\n".join(chunks)


# ── Loyalty Discount Tool (Code Interpreter) ─────────────────────────────────
@tool
def calculate_loyalty_discount(
    loyalty_points: int,
    tier: str,
    order_total: float,
    product_category: str = "standard",
) -> str:
    """
    Calculate the loyalty discount for a customer order using the
    AgentCore Code Interpreter. Runs exact arithmetic in a secure sandbox.
    """
    code = f"""
import json

earn_rates = {{"standard": 1, "device": 2, "fresh": 5}}
tier_rates = {{"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}}

loyalty_points = {loyalty_points}
tier = "{tier}"
order_total = {order_total}
product_category = "{product_category}"

POINTS_PER_DOLLAR = 100
max_points_by_order_cap = int(order_total * 0.5 * POINTS_PER_DOLLAR)
points_redeemed = min(loyalty_points, max_points_by_order_cap)
points_redeemed = (points_redeemed // 500) * 500
points_redeemed = max(points_redeemed, 0)
points_discount_value = points_redeemed / POINTS_PER_DOLLAR

subtotal_after_points = order_total - points_discount_value

tier_rate = tier_rates.get(tier, 0.0)
tier_discount = subtotal_after_points * tier_rate

final_total = round(subtotal_after_points - tier_discount, 2)
total_savings = round(order_total - final_total, 2)

earn_rate = earn_rates.get(product_category, 1)
points_earned = int(round(final_total * earn_rate))
remaining_points = loyalty_points - points_redeemed + points_earned

result = {{
    "order_total": order_total,
    "points_redeemed": points_redeemed,
    "points_discount_value": round(points_discount_value, 2),
    "tier": tier,
    "tier_discount_rate": tier_rate,
    "tier_discount_value": round(tier_discount, 2),
    "final_total": final_total,
    "total_savings": total_savings,
    "points_earned": points_earned,
    "remaining_points": remaining_points,
}}
print(json.dumps(result))
"""

    try:
        with code_session(REGION) as code_client:
            response = code_client.invoke(
                "executeCode",
                {"language": "python", "code": code, "clearContext": True},
            )
            for event in response["stream"]:
                if "result" in event:
                    for item in event["result"].get("content", []):
                        if item.get("type") == "text":
                            return item["text"]
            return json.dumps({"error": "No output returned from Code Interpreter."})

    except Exception as e:
        logger.warning(f"Code Interpreter unavailable, using fallback: {e}")
        tier_rates = {"Silver": 0.00, "Gold": 0.10, "Platinum": 0.15}
        tier_rate = tier_rates.get(tier, 0.0)
        tier_discount = order_total * tier_rate
        final_total = round(order_total - tier_discount, 2)
        return json.dumps({
            "fallback": True,
            "tier": tier,
            "tier_discount_rate": tier_rate,
            "tier_discount_value": round(tier_discount, 2),
            "final_total": final_total,
            "note": "Code Interpreter unavailable — only tier discount applied.",
        })


# ── Web Fetching Tool (Lightweight Web Scraping) ──────────────────────────────
@tool
def fetch_web_page(url: str) -> str:
    """
    Fetch and return the HTML page title and clean body text content for a given URL.
    Use this tool whenever you need to visit or fetch content from web pages.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
            }
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode("utf-8", errors="ignore")
            soup = BeautifulSoup(html, "html.parser")
            
            title = soup.title.string.strip() if soup.title and soup.title.string else "No title found"
            
            for element in soup(["script", "style", "nav", "footer"]):
                element.extract()
                
            text = soup.get_text(separator=" ", strip=True)[:2000]
            return f"Page Title: {title}\n\nContent Sample:\n{text}"
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return f"Error fetching web page: {e}"


# ── Agent Entrypoint ─────────────────────────────────────────────────────────
@app.entrypoint
async def invoke(payload, context=None):
    """Main handler called by AgentCore for every incoming request."""
    try:
        user_input = payload.get("prompt", "")
        actor_id = payload.get("customer_id", "anonymous")
        session_id = payload.get("session_id") or str(uuid.uuid4())

        memory_hook = MemoryHook(
            actor_id=actor_id,
            session_id=session_id,
            memory_client=memory_client,
            memory_id=MEMORY_ID,
        )

        tools = [
            search_knowledge_base,
            calculate_loyalty_discount,
            fetch_web_page,
        ]

        gateway_client = MCPClient(lambda: streamable_http_client(GATEWAY_URL))

        with gateway_client:
            gateway_tools = gateway_client.list_tools_sync()
            tools.extend(gateway_tools)

            agent = Agent(
                model=model,
                tools=tools,
                hooks=[memory_hook],
                system_prompt=(
                    "You are a helpful customer support agent for an "
                    "e-commerce company. Use your tools to look up orders "
                    "and customers, process refunds, search the knowledge "
                    "base for product info and policies, calculate loyalty "
                    "discounts, and fetch web pages when a question needs "
                    "current information from a website URL. Be "
                    "concise, friendly, and accurate — never guess at order "
                    "or refund details, always look them up."
                ),
            )

            result = await agent.invoke_async(user_input)

        return result.message["content"][0]["text"]

    except Exception as e:
        logger.exception("Agent invocation failed")
        return f"I'm sorry, I ran into an error processing your request: {e}"


# ── CLI entry point ──────────────────────────────────────────────────────────
def main():
    """Run one invocation from the command line for local testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("payload", type=str)
    args = parser.parse_args()

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    response = loop.run_until_complete(invoke(json.loads(args.payload)))
    print(response)

if __name__ == "__main__":
    app.run()