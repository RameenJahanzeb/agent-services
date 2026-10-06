import os
import re
from enum import Enum
from typing import Any, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from contact_tools import (
    create_contact,
    delete_contact,
    list_contacts,
    update_contact,
)


class Operation(str, Enum):
    CREATE = "CREATE"
    READ = "READ"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    UNKNOWN = "UNKNOWN"


class IntentExtraction(BaseModel):
    operation: Operation = Field(description="Exactly one supported contact operation")
    name: str | None = Field(
        default=None, description="CREATE: contact's complete name, or null if absent"
    )
    phone: str | None = Field(
        default=None, description="CREATE: explicitly provided phone, or null if absent"
    )
    search: str | None = Field(
        default=None, description="READ: search term, or null when requesting all contacts"
    )
    all_contacts: bool = Field(
        default=False, description="READ: true when the user wants the full contact list"
    )
    identifier: str | None = Field(
        default=None, description="UPDATE/DELETE: existing contact name or partial name"
    )
    new_name: str | None = Field(
        default=None, description="UPDATE: explicitly requested replacement name"
    )
    new_phone: str | None = Field(
        default=None, description="UPDATE: explicitly requested replacement phone"
    )


class ReplyResult(BaseModel):
    reply: str = Field(description="Short, friendly response to show the user")


class ChatState(TypedDict, total=False):
    message: str
    operation: str
    extraction: dict[str, Any]
    result: dict[str, Any]
    success: bool
    reply: str


def _get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured in agent-service/.env.")
    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        google_api_key=api_key,
        temperature=0,
    )


def _phone_error(phone: str | None) -> str | None:
    if not phone or not phone.strip():
        return "Phone number is missing."
    digits = re.sub(r"\D", "", phone)
    if len(digits) < 7:
        return "Phone number is too short; please provide at least 7 digits."
    if len(digits) > 15:
        return "Phone number is too long; please provide no more than 15 digits."
    if not re.fullmatch(r"\+?[0-9().\-\s]{7,20}", phone.strip()):
        return "Phone number contains unsupported characters."
    return None


def build_graph():
    llm = _get_llm()
    understand = llm.with_structured_output(IntentExtraction)
    generate_reply = llm.with_structured_output(ReplyResult)

    async def understand_message(state: ChatState) -> dict[str, Any]:
        result = await understand.ainvoke(
            [
                (
                    "system",
                    "Classify the user's message into exactly one operation: CREATE "
                    "(add a contact), READ (view or search contacts), UPDATE "
                    "(change a contact), DELETE (remove a contact), or UNKNOWN "
                    "(unclear or unrelated). Use the primary requested action. "
                    "In the same response, extract only fields relevant to that "
                    "operation. CREATE uses name and phone; READ uses search and "
                    "all_contacts; UPDATE uses identifier, new_name, and/or new_phone; "
                    "DELETE uses identifier. For UNKNOWN, leave extraction fields "
                    "empty. Never infer or invent missing values; use null for values "
                    "not clearly provided.",
                ),
                ("human", state["message"]),
            ]
        )
        values = result.model_dump()
        return {
            "operation": result.operation.value,
            "extraction": {
                key: value for key, value in values.items() if key != "operation"
            },
        }

    async def execute_operation(state: ChatState) -> dict[str, Any]:
        operation = Operation(state["operation"])
        data = state.get("extraction", {})

        if operation == Operation.CREATE:
            name = data.get("name")
            phone = data.get("phone")
            issues = []
            if not isinstance(name, str) or not name.strip():
                issues.append("Contact name is missing.")
            elif len(name.strip()) > 100:
                issues.append("Contact name must be 100 characters or fewer.")
            phone_issue = _phone_error(phone)
            if phone_issue:
                issues.append(phone_issue)
            if issues:
                return {"result": {"status": "invalid", "details": issues}, "success": False}
            result = await create_contact(name.strip(), phone.strip())
            return {"result": result, "success": result["status"] == "success"}

        if operation == Operation.READ:
            search = None if data.get("all_contacts") else data.get("search")
            result = await list_contacts(search=search)
            return {"result": result, "success": result["status"] == "success"}

        identifier = data.get("identifier")
        if not isinstance(identifier, str) or not identifier.strip():
            return {
                "result": {"status": "invalid", "details": ["Contact name is missing."]},
                "success": False,
            }

        if operation == Operation.UPDATE:
            new_name = data.get("new_name")
            new_phone = data.get("new_phone")
            issues = []
            if not new_name and not new_phone:
                issues.append("No replacement name or phone number was provided.")
            if new_name is not None and (
                not isinstance(new_name, str)
                or not new_name.strip()
                or len(new_name.strip()) > 100
            ):
                issues.append("The replacement name must contain 1 to 100 characters.")
            if new_phone is not None:
                phone_issue = _phone_error(new_phone)
                if phone_issue:
                    issues.append(phone_issue)
            if issues:
                return {"result": {"status": "invalid", "details": issues}, "success": False}
            result = await update_contact(identifier.strip(), new_name, new_phone)
            return {"result": result, "success": result["status"] == "success"}

        result = await delete_contact(identifier.strip())
        return {"result": result, "success": result["status"] == "success"}

    async def respond(state: ChatState) -> dict[str, Any]:
        result = await generate_reply.ainvoke(
            [
                (
                    "system",
                    "Write one short, friendly, accurate message for the user. "
                    "Use only the operation and result provided; never claim a change "
                    "succeeded unless the result says status=success. For invalid "
                    "results, clearly request the missing/corrected information. For "
                    "ambiguous or not-found results, ask the user to clarify the name. "
                    "For a READ, summarize the returned contacts naturally and avoid "
                    "dumping metadata. For UNKNOWN, ask what they want to do with their "
                    "contacts. Do not expose internal errors or raw JSON.",
                ),
                (
                    "human",
                    f"Operation: {state.get('operation', 'UNKNOWN')}\n"
                    f"Success: {state.get('success', False)}\n"
                    f"Result: {state.get('result', {})}",
                ),
            ]
        )
        return {"reply": result.reply, "success": state.get("success", False)}

    builder = StateGraph(ChatState)
    builder.add_node("understand", understand_message)
    builder.add_node("execute", execute_operation)
    builder.add_node("respond", respond)
    builder.add_edge(START, "understand")
    builder.add_conditional_edges(
        "understand",
        lambda state: state["operation"],
        {
            Operation.CREATE.value: "execute",
            Operation.READ.value: "execute",
            Operation.UPDATE.value: "execute",
            Operation.DELETE.value: "execute",
            Operation.UNKNOWN.value: "respond",
        },
    )
    builder.add_edge("execute", "respond")
    builder.add_edge("respond", END)
    return builder.compile()
