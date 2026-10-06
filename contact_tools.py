import os
from typing import Any

import httpx


def _api_url(path: str = "") -> str:
    base_url = os.getenv("CONTACT_API_BASE_URL", "http://localhost:3000").rstrip("/")
    return f"{base_url}/api/contacts{path}"


def _http_error(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        body = response.text
    if isinstance(body, dict):
        detail = body
    else:
        detail = body
    return {
        "status": "error",
        "http_status": response.status_code,
        "detail": detail,
    }


async def _get_contacts(client: httpx.AsyncClient) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    try:
        response = await client.get(_api_url())
    except httpx.HTTPError as error:
        return None, {"status": "error", "detail": f"Could not reach contacts API: {error}"}
    if not response.is_success:
        return None, _http_error(response)
    try:
        contacts = response.json()["contacts"]
    except (ValueError, KeyError, TypeError):
        return None, {"status": "error", "detail": "Contacts API returned an invalid list response."}
    if not isinstance(contacts, list):
        return None, {"status": "error", "detail": "Contacts API returned an invalid contacts list."}
    return contacts, None


def _match_contact(
    contacts: list[dict[str, Any]], identifier: str
) -> tuple[dict[str, Any] | None, str | None]:
    term = identifier.strip().casefold()
    exact = [contact for contact in contacts if str(contact.get("name", "")).casefold() == term]
    if len(exact) == 1:
        return exact[0], None
    if len(exact) > 1:
        return None, "ambiguous"

    partial = [
        contact
        for contact in contacts
        if term in str(contact.get("name", "")).casefold()
    ]
    if len(partial) == 1:
        return partial[0], None
    if len(partial) > 1:
        return None, "ambiguous"
    return None, "not_found"


async def create_contact(name: str, phone: str) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(_api_url(), json={"name": name, "phone": phone})
    except httpx.HTTPError as error:
        return {"status": "error", "detail": f"Could not reach contacts API: {error}"}
    if not response.is_success:
        return _http_error(response)
    try:
        return {"status": "success", "contact": response.json()["contact"]}
    except (ValueError, KeyError, TypeError):
        return {"status": "error", "detail": "Contacts API returned an invalid create response."}


async def list_contacts(search: str | None = None) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        contacts, error = await _get_contacts(client)
    if error:
        return error
    if search:
        term = search.casefold()
        contacts = [
            contact
            for contact in contacts
            if term in str(contact.get("name", "")).casefold()
        ]
    return {"status": "success", "contacts": contacts, "search": search}


async def update_contact(
    identifier: str, new_name: str | None = None, new_phone: str | None = None
) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        contacts, error = await _get_contacts(client)
        if error:
            return error
        contact, match_status = _match_contact(contacts, identifier)
        if not contact:
            return {"status": match_status, "identifier": identifier}
        body = {
            "name": new_name.strip() if new_name else contact.get("name"),
            "phone": new_phone.strip() if new_phone else contact.get("phone"),
        }
        try:
            response = await client.put(
                _api_url(f"/{contact['id']}"),
                json=body,
            )
        except (httpx.HTTPError, KeyError) as error:
            return {"status": "error", "detail": f"Could not update contact: {error}"}
    if not response.is_success:
        return _http_error(response)
    try:
        return {"status": "success", "contact": response.json()["contact"]}
    except (ValueError, KeyError, TypeError):
        return {"status": "error", "detail": "Contacts API returned an invalid update response."}


async def delete_contact(identifier: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        contacts, error = await _get_contacts(client)
        if error:
            return error
        contact, match_status = _match_contact(contacts, identifier)
        if not contact:
            return {"status": match_status, "identifier": identifier}
        try:
            response = await client.delete(_api_url(f"/{contact['id']}"))
        except (httpx.HTTPError, KeyError) as error:
            return {"status": "error", "detail": f"Could not delete contact: {error}"}
    if not response.is_success:
        return _http_error(response)
    return {"status": "success", "name": contact.get("name")}
