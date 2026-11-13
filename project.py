# Level 2: Custom State & Domain Routing
# 🎯 The Goal:

# In real production applications, agents rarely communicate using only chat messages. 
# You need to track structured domain data—like User IDs, Ticket Status, 
# and Issue Categories—directly in the state.

# In this problem, you will learn:

# How to add custom fields to MessagesState (customer_id, issue_type, resolved).
# How the supervisor updates structured state (issue_type and resolved).
# How agents and routing functions read from this custom state.

# 📋 The Scenario: Customer Support Helpdesk

# A customer reaches out with a problem. Depending on what they need, 
# the ticket must be routed to the right specialist:

# billing_agent: Handles payments and refunds.
# Tool: check_invoice(invoice_id: str)
# Tool: process_refund(invoice_id: str, amount: float)
# tech_agent: Handles server status and account password resets.
# Tool: check_server_status()
# Tool: reset_password(customer_id: str)

from dotenv import load_dotenv
load_dotenv()

from typing import Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_agent
from operator import add
from pydantic import BaseModel

class SupportState(MessagesState):
    next:str
    customer_id: str
    issue_type: Literal["billing","technical","general","unknown"]
    resolved:bool

INVOICES = {
    "INV_101": {"amount": 50.0, "status": "paid", "customer_id": "CUST_1"},
    "INV_102": {"amount": 120.0, "status": "paid", "customer_id": "CUST_2"},
}
SERVERS = {"us-east": "ALL SYSTEMS OPERATIONAL", "eu-west": "DEGRADED PERFORMANCE"}

@tool
def check_invoice(invoice_id:str) -> str:
    """check the invoice by its ID and return its details"""
    inv = INVOICES.get(invoice_id.upper())
    if inv:
        return f"Invoice {invoice_id}: Amount ${inv["amount"]}, Status: {inv["status"]}"
    return f"Invoice {invoice_id} not found"

@tool
def process_refund(invoice_id:str, amount:float)->str:
    """Process refund for a invoice"""
    inv = INVOICES.get(invoice_id.upper())
    if inv:
        return f"SUCCESS: Refund of {amount}$ has been processed for Invoice{invoice_id}"
    return f"FAILED: Invoice {invoice_id} not found"

@tool
def check_server_status()->str:
    """check the health of all servers"""
    return f"SERVER STATUS REPORT: {SERVERS}"

@tool
def reset_password(customer_id:str)->str:
    """send the reset password link to the customer"""
    return f"SENT: reset_password link has been sent to customer {customer_id}"

llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0)

billing_agent = create_agent(
    model = llm,
    tools = [check_invoice, process_refund],
    system_prompt=(
        "you're a billing agent and your task is to help the customers with all billing related issues"
    )
)

tech_agent = create_agent(
    model = llm,
    tools = [check_server_status,reset_password],
    system_prompt=(
        "You're a tech agent and your task is to help users with server related queries like its status" \
        "and password related issues like reseting passwords."
    )
)

class SupervisorDecision(BaseModel):
    next: Literal["billing_agent","tech_agent","FINISH"]
    issue_type:Literal["billing","technical","general","unknown"]
    resolved:bool

supervisor_llm = llm.with_structured_output(SupervisorDecision)

def supervisor(state:SupportState)->dict:
    prompt = (
        f""""You are an expert customer support supervisor managing two specialist agents:\n"
        "  - billing_agent: Handles invoices, payments, and refunds.\n"
        "  - tech_agent   : Handles server status and account password resets.\n\n"
        "Your job is to analyze the conversation and produce three decisions:\n"
        "1. 'issue_type': Classify as 'billing', 'technical', 'general', or 'unknown'.\n"
        "2. 'resolved': Set to True ONLY if the customer's request has been completely addressed "
        "   by a specialist or if no action is needed. Otherwise, False.\n"
        "3. 'next':\n"
        "   - If not resolved and it's a billing issue -> 'billing_agent'\n"
        "   - If not resolved and it's a tech issue -> 'tech_agent'\n"
        "   - If resolved -> 'FINISH'\n"
        
        messages: {state["messages"]}"""
    )
    response: SupervisorDecision = supervisor_llm.invoke(prompt)
    return {
        "next": response.next,
        "issue_type":response.issue_type,
        "resolved":response.resolved
    }

def billing_agent_node(state:SupportState)->dict:
    result = billing_agent.invoke({"messages": state["messages"]})
    last_message = result["messages"][-1]
    agent_message = AIMessage(content=last_message.content, name= "BillingAgent")
    return{
        "messages":[agent_message]
    }

def tech_agent_node(state:SupportState)->dict:
    result = tech_agent.invoke({"messages": state["messages"]})
    last_message = result["messages"][-1]
    agent_message = AIMessage(content=last_message.content,name = "TechAgent")
    return{
        "messages":[agent_message]
    }

def route(state:SupportState) -> Literal["billing_agent", "tech_agent","__end__"]:
    if state['resolved'] is True or state["next"] == "FINISH":
        return END
    return state["next"]

builder = StateGraph(SupportState)
builder.add_node("supervisor",supervisor)
builder.add_node("billing_agent",billing_agent_node)
builder.add_node("tech_agent",tech_agent_node)

builder.add_edge(START,"supervisor")
builder.add_conditional_edges("supervisor",route)
builder.add_edge("billing_agent","supervisor")
builder.add_edge("tech_agent","supervisor")

graph = builder.compile()

initial_input = {
    "messages":[HumanMessage(content="Hi, my customer ID is CUST_1. I need a refund of $50 for invoice INV_101.")]
}

initial_input2 = {
    "messages":[HumanMessage(content="Users are reporting errors in Europe. Can you check our servers?")]
}

result1 = graph.invoke(initial_input)
result2 = graph.invoke(initial_input2)

for msg in result1["messages"]:
    msg.pretty_print()
    print()
for msg in result2["messages"]:
    msg.pretty_print()
    print()



