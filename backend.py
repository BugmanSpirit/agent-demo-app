from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterator, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
import google.generativeai as genai
import httpx
from pydantic import BaseModel, Field


load_dotenv()


PARALLEL_API_URL = "https://api.parallel.ai/v1beta/search"
GEMINI_MODEL = "gemini-2.5-flash"
MAX_SEARCH_STEPS = 25
DEFAULT_PORT = 8000


SEARCH_TOOL_DESCRIPTION = """# Web Search Tool

**Purpose:** Perform web searches and return LLM-friendly results.

**Usage:**
- objective: Natural-language description of your research goal (max 200 characters)

**Best Practices:**
- Be specific about what information you need
- Mention if you want recent/current data
- Keep objectives concise but descriptive"""

SEARCH_OBJECTIVE_DESCRIPTION = (
    "Natural-language description of your research goal (max 200 characters)"
)

DEFAULT_SYSTEM_PROMPT_TEMPLATE = """You are a simple search agent. Your mission is to comprehensively fulfill the user's search objective by conducting 1 up to 3 searches from different angles until you have gathered sufficient information to provide a complete answer. The current date is {current_date}

**Research Philosophy:**
- Each search should explore a unique angle or aspect of the topic
- NEVER try to OPEN an article, the excerpts provided should be enough

**Key Parameters:**
- objective: Describe what you're trying to accomplish. This helps the search engine understand intent and provide relevant results.

**Output:**
After doing the searches required, write up your 'search report' that answers the initial search query. Even if you could not answer the question ensure to always provide a final report! Please do NOT use markdown tables. """


PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def check_api_keys() -> None:

    if not PARALLEL_API_KEY or not GEMINI_API_KEY:
        print("⚠️  WARNING: Missing API keys!")
        print(f"   PARALLEL_API_KEY: {'✓ Set' if PARALLEL_API_KEY else '✗ Missing'}")
        print(f"   GEMINI_API_KEY: {'✓ Set' if GEMINI_API_KEY else '✗ Missing'}")
        print("\n   Please set these in your .env file or environment variables.")
        print("   The server will start but API calls will fail.\n")


check_api_keys()


if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


app = FastAPI(
    title="Web Search Agent",
    description="AI-powered web search using Google Gemini and Parallel Search API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchRequest(BaseModel):
    query: str = Field(..., description="User's search query")
    systemPrompt: Optional[str] = Field(
        None, description="Optional custom system prompt"
    )


def search_web(objective: str) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json", "x-api-key": PARALLEL_API_KEY}
    payload = {
        "objective": objective,
        "processor": "base",
        "max_results": 10,
        "max_chars_per_result": 2500,
    }

    with httpx.Client(timeout=30.0) as client:
        response = client.post(PARALLEL_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()


def create_search_tool() -> genai.protos.Tool:
    return genai.protos.Tool(
        function_declarations=[
            genai.protos.FunctionDeclaration(
                name="search",
                description=SEARCH_TOOL_DESCRIPTION,
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "objective": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description=SEARCH_OBJECTIVE_DESCRIPTION,
                        )
                    },
                    required=["objective"],
                ),
            )
        ]
    )


def get_default_system_prompt() -> str:
    current_date = datetime.now().strftime("%Y-%m-%d")
    return DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(current_date=current_date)


async def stream_research(
    query: str, system_prompt: Optional[str] = None
) -> AsyncGenerator[str, None]:
    import asyncio

    system_instruction = system_prompt or get_default_system_prompt()

    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        tools=[create_search_tool()],
        system_instruction=system_instruction,
    )

    chat = model.start_chat(enable_automatic_function_calling=False)
    step_count = 0

    try:
        response = await asyncio.to_thread(
            lambda: chat.send_message(query, stream=True)
        )

        for chunk in response:
            step_count += 1

            if step_count >= MAX_SEARCH_STEPS:
                yield f"data: {json.dumps({'type': 'finish', 'finishReason': 'max_steps'})}\n\n"
                break

            if chunk.candidates and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:

                    if hasattr(part, "text") and part.text:
                        yield f"data: {json.dumps({'type': 'text-delta', 'text': part.text})}\n\n"

                    elif hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        yield f"data: {json.dumps({'type': 'tool-call', 'toolName': fc.name, 'args': dict(fc.args)})}\n\n"

                        if fc.name == "search":
                            objective = fc.args.get("objective", "")
                            search_result = search_web(objective)
                            yield f"data: {json.dumps({'type': 'tool-result', 'toolName': 'search', 'output': search_result})}\n\n"

                            function_response = genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=fc.name, response={"result": search_result}
                                )
                            )

                            async for item in process_response_stream(
                                chat, function_response, step_count
                            ):
                                yield item

        finish_reason = (
            str(response.candidates[0].finish_reason)
            if response.candidates and response.candidates[0].finish_reason
            else "completed"
        )
        yield f"data: {json.dumps({'type': 'finish', 'finishReason': finish_reason})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'error': {'message': str(e)}})}\n\n"

    yield "data: [DONE]\n\n"


async def process_response_stream(
    chat: Any, function_response: Any, initial_step_count: int
) -> AsyncGenerator[str, None]:
    import asyncio

    step_count = initial_step_count
    next_response = await asyncio.to_thread(
        lambda: chat.send_message(function_response, stream=True)
    )

    for next_chunk in next_response:
        step_count += 1
        if step_count >= MAX_SEARCH_STEPS:
            yield f"data: {json.dumps({'type': 'finish', 'finishReason': 'max_steps'})}\n\n"
            break

        if next_chunk.candidates and next_chunk.candidates[0].content.parts:
            for next_part in next_chunk.candidates[0].content.parts:
                if hasattr(next_part, "text") and next_part.text:
                    yield f"data: {json.dumps({'type': 'text-delta', 'text': next_part.text})}\n\n"

                elif hasattr(next_part, "function_call") and next_part.function_call:

                    fc2 = next_part.function_call
                    yield f"data: {json.dumps({'type': 'tool-call', 'toolName': fc2.name, 'args': dict(fc2.args)})}\n\n"

                    if fc2.name == "search":
                        objective2 = fc2.args.get("objective", "")
                        search_result2 = search_web(objective2)
                        yield f"data: {json.dumps({'type': 'tool-result', 'toolName': 'search', 'output': search_result2})}\n\n"

                        function_response2 = genai.protos.Part(
                            function_response=genai.protos.FunctionResponse(
                                name=fc2.name, response={"result": search_result2}
                            )
                        )

                        async for item in process_response_stream(
                            chat, function_response2, step_count
                        ):
                            yield item


@app.get("/", response_class=FileResponse)
async def serve_index() -> FileResponse:
    index_path = Path(__file__).parent / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.post("/agents/cerebras-search")
async def research_endpoint(request: SearchRequest) -> StreamingResponse:
    if not request.query:
        raise HTTPException(status_code=400, detail="Query is required")

    if not PARALLEL_API_KEY or not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server configuration error: Missing API keys. Please check server logs.",
        )

    async def event_generator() -> AsyncIterator[str]:

        async for chunk in stream_research(request.query, request.systemPrompt):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@app.get("/health")
async def health_check() -> Dict[str, str]:
    return {
        "status": "healthy",
        "gemini_configured": "yes" if GEMINI_API_KEY else "no",
        "parallel_configured": "yes" if PARALLEL_API_KEY else "no",
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", str(DEFAULT_PORT)))
    print(f"\n🚀 Starting Web Search Agent")
    print(f"   Server: http://localhost:{port}")
    print(f"   Health: http://localhost:{port}/health")
    print(f"   Press CTRL+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
